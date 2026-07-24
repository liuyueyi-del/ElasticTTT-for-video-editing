import os
import math
import json
import csv
import copy
import re
import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, inject_adapter_in_model

from diffsynth.core import UnifiedDataset, ModelConfig, load_state_dict
from diffsynth.core.data.operators import (
    LoadVideo,
    LoadGIF,
    LoadImage,
    ImageCropAndResize,
    ToList,
    ToAbsolutePath,
    RouteByType,
    RouteByExtensionName,
)
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.parsers import add_general_config, add_video_size_config
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__, log_level="INFO")


def _flush_gpu():
    """Release cached GPU memory (call between heavy model stages)."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Standard-library logger that works even when Accelerator is not initialised
# (e.g. during --inference_only runs where _train_one is never called).
import logging as _std_logging
_std_logger = _std_logging.getLogger(__name__ + ".std")
if not _std_logger.handlers:
    _sh = _std_logging.StreamHandler()
    _sh.setFormatter(_std_logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    _std_logger.addHandler(_sh)
    _std_logger.setLevel(_std_logging.INFO)

_DATASET_PROMPT_COLUMNS = (
    "edited_prompt",
    "background_editing",
    "subject_editing",
    "color_editing",
    "removal",
    "addition",
    "source_blend",
    "target_blend",
)


def _sanitize_filename(text, max_len=80):
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] if text else "sample"


def _load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "Failed to load config; please install omegaconf or pyyaml."
            ) from exc
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}


def _apply_config_to_args(args, config: Dict):
    train_data = config.get("train_data") or {}
    validation_data = config.get("validation_data") or {}

    # Command-line arguments take priority over YAML config.
    # We only apply a YAML value when the current arg value is still equal to
    # the parser default (meaning the user did NOT explicitly pass it on the
    # command line).  _parser_defaults is attached to args in __main__.
    _defaults = getattr(args, "_parser_defaults", {})

    def _cli_was_set(dest):
        """Return True if the user explicitly passed this arg on the command line."""
        default_val = _defaults.get(dest)
        current_val = getattr(args, dest, default_val)
        return current_val != default_val

    if config.get("output_dir") and not _cli_was_set("output_path"):
        args.output_path = config.get("output_dir")
    if config.get("dataset_csv") and not _cli_was_set("dataset_csv"):
        args.dataset_csv = config.get("dataset_csv")
    if config.get("dataset_root") and not _cli_was_set("dataset_root"):
        args.dataset_root = config.get("dataset_root")

    for key, value in config.items():
        if key in ("train_data", "validation_data", "output_dir", "dataset_csv", "dataset_root"):
            continue
        if hasattr(args, key) and value is not None and not _cli_was_set(key):
            setattr(args, key, value)

    # Backward-compatible keys
    if config.get("adam_weight_decay") is not None and hasattr(args, "weight_decay"):
        if config.get("weight_decay") is None:
            args.weight_decay = config.get("adam_weight_decay")
    if config.get("gradient_checkpointing") is not None and hasattr(args, "use_gradient_checkpointing"):
        args.use_gradient_checkpointing = config.get("gradient_checkpointing")

    if train_data:
        if train_data.get("n_sample_frames") is not None:
            args.num_frames = int(train_data["n_sample_frames"])
        if train_data.get("sample_start_idx") is not None:
            args.sample_start_idx = int(train_data["sample_start_idx"])
        if train_data.get("sample_frame_rate") is not None:
            args.sample_frame_rate = int(train_data["sample_frame_rate"])
        if train_data.get("width") is not None:
            args.width = int(train_data["width"])
        if train_data.get("height") is not None:
            args.height = int(train_data["height"])

    _coerce_args(args)
    return train_data, validation_data


def _coerce_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_int(value):
    if isinstance(value, str) and value.strip() != "":
        return int(float(value))
    return int(value)


def _coerce_float(value):
    if isinstance(value, str) and value.strip() != "":
        return float(value)
    return float(value)


def _coerce_args(args):
    float_keys = [
        "learning_rate",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "weight_decay",
        "max_grad_norm",
        "target_noise_scale",
        "flow_shift",
        "flow_align_ratio",
        "flow_align_cfg_scale",
        "flow_align_gamma",
        "fl_mask_ratio",
        "bg_mask_ratio",
        "latent_scale",
        "loss_mae_weight",
        "loss_contrastive_weight",
        "contrastive_temperature",
        "lora_alpha",
        "reference_strength",
        "reference_init_strength",
        "reference_fg_init_strength",
        "reference_bg_init_strength",
        "guidance_scale",
        "source_cfg_scale",
        "inference_shift",
        "box_scale",
    ]
    int_keys = [
        "gradient_accumulation_steps",
        "train_batch_size",
        "max_train_steps",
        "checkpointing_steps",
        "validation_steps",
        "num_train_timesteps",
        "num_frames",
        "sample_start_idx",
        "sample_frame_rate",
        "num_inference_steps",
        "inference_seed",
        "inference_video_length",
        "fps",
        "reference_start_idx",
        "reference_frame_rate",
        "lora_rank",
        "num_epochs",
        "feather_radius",
    ]
    bool_keys = [
        "use_contrastive_loss",
        "use_lora",
        "run_inference_after_train",
        "inference_only",
        "disable_flow_mask",
        "use_extra_text_token",
        "extra_text_token_trainable",
        "use_reference_noise",
        "scale_lr",
        "use_gradient_checkpointing",
        "use_gradient_checkpointing_offload",
        "find_unused_parameters",
        "reference_strict_timestep_align",
    ]
    for key in float_keys:
        if hasattr(args, key):
            val = getattr(args, key)
            if isinstance(val, str) and val.strip() == "":
                continue
            if isinstance(val, (str, int, float)):
                setattr(args, key, _coerce_float(val))
    for key in int_keys:
        if hasattr(args, key):
            val = getattr(args, key)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                continue
            if isinstance(val, (str, int, float)):
                setattr(args, key, _coerce_int(val))
    for key in bool_keys:
        if hasattr(args, key):
            val = getattr(args, key)
            if isinstance(val, str):
                setattr(args, key, _coerce_bool(val))


def _load_dataset_csv_entries(
    csv_path: str,
    dataset_root: Optional[str] = None,
    prompt_tail_columns: int = 5,
):
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    if dataset_root:
        dataset_root = Path(dataset_root).expanduser().resolve()
    else:
        dataset_root = csv_path.parent

    entries = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        tail_cols = int(prompt_tail_columns or 0)
        if tail_cols > 0 and len(fieldnames) > 0:
            prompt_cols = fieldnames[-tail_cols:]
        else:
            # Auto-detect: collect all columns that come *after* the source/mask
            # prompt columns, treating them as editing / validation targets.
            # Priority: after masked_prompt > after original_prompt > after prompt.
            # The "edited_prompt" column used by demo/datasets.csv is detected
            # naturally as the column that follows "original_prompt".
            if "masked_prompt" in fieldnames:
                prompt_cols = fieldnames[fieldnames.index("masked_prompt") + 1 :]
            elif "original_prompt" in fieldnames:
                prompt_cols = fieldnames[fieldnames.index("original_prompt") + 1 :]
            elif "prompt" in fieldnames:
                prompt_cols = fieldnames[fieldnames.index("prompt") + 1 :]
            else:
                prompt_cols = []
            # Exclude any leftover structural columns that should not be prompts.
            _structural = {"video", "name", "video_path", "masked_prompt", "original_prompt", "prompt"}
            prompt_cols = [c for c in prompt_cols if c not in _structural]

        for row_index, row in enumerate(reader, start=1):
            name = (row.get("name") or row.get("video") or row.get("video_path") or "").strip()
            if not name:
                continue
            video_path = Path(name)
            if not video_path.is_absolute():
                video_path = dataset_root / name
            if not video_path.is_file():
                raise FileNotFoundError(f"Video not found: {video_path}")

            original_prompt = (row.get("original_prompt") or row.get("prompt") or "").strip()
            masked_prompt = (row.get("masked_prompt") or "").strip()
            if not original_prompt:
                raise ValueError(f"Row {row_index} missing original_prompt: {name}")

            validation_prompts = []
            for col in prompt_cols:
                value = (row.get(col) or "").strip()
                if value:
                    validation_prompts.append(value)

            entries.append(
                {
                    "row_index": row_index,
                    "name": name,
                    "video_path": str(video_path),
                    "prompt": original_prompt,
                    "masked_prompt": masked_prompt,
                    "validation_prompts": validation_prompts,
                }
            )
    if not entries:
        raise ValueError(f"No valid rows found in dataset CSV: {csv_path}")
    return entries


def text_augmentation(
    text,
    gpt2_model,
    gpt2_tokenizer,
    max_length: int = 100,
    num_return_sequences: int = 1,
    do_sample: bool = True,
):
    if gpt2_model is None or gpt2_tokenizer is None:
        return text
    if gpt2_tokenizer.pad_token_id is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    input_ids = gpt2_tokenizer.encode(text, return_tensors="pt")
    attention_mask = input_ids.ne(gpt2_tokenizer.pad_token_id)
    output = gpt2_model.generate(
        input_ids,
        max_length=int(max_length),
        num_return_sequences=int(num_return_sequences),
        do_sample=bool(do_sample),
        attention_mask=attention_mask,
        pad_token_id=gpt2_tokenizer.pad_token_id,
    )
    augmented_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return augmented_text


def text_masking(text, mask_ratio: float = 0.15):
    words = text.split()
    if not words:
        return text
    num_masks = max(1, int(len(words) * mask_ratio))
    mask_indices = torch.randperm(len(words))[:num_masks]
    for idx in mask_indices:
        words[idx] = "[MASK]"
    return " ".join(words)


def contrastive_loss(features, labels, temperature: float = 0.5):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    labels = labels.to(features.device)

    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    diag = torch.eye(mask.size(0), device=mask.device, dtype=torch.bool)
    positive_mask = mask & ~diag
    negative_mask = ~mask

    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    positive_sim = similarity_matrix[positive_mask].unsqueeze(1)
    negative_sim = similarity_matrix[negative_mask].view(positive_sim.size(0), -1)

    logits = torch.cat([positive_sim, negative_sim], dim=1)
    labels = torch.zeros(logits.size(0), device=features.device, dtype=torch.long)

    return F.cross_entropy(logits / temperature, labels)


def transfer_coordinates(motions: list, flow_size, target_size):
    w_scale = target_size[0] / flow_size[0]
    h_scale = target_size[1] / flow_size[1]
    new_motions = []
    for motion in motions:
        motion[1] = motion[1] * w_scale
        motion[3] = motion[3] * w_scale
        motion[2] = motion[2] * h_scale
        motion[4] = motion[4] * h_scale
        new_motions.append(motion)
    return new_motions


def apply_tube_mask_to_video(video_data, mask_ratio=0.5, tube_length=4, patch_size=(16, 16)):
    frames, channels, height, width = video_data.shape
    num_patches_per_frame = (height // patch_size[0]) * (width // patch_size[1])
    num_masks_per_frame = int(mask_ratio * num_patches_per_frame)

    mask_per_frame = torch.hstack(
        [
            torch.ones(num_patches_per_frame - num_masks_per_frame),
            torch.zeros(num_masks_per_frame),
        ]
    )

    mask = torch.zeros_like(video_data)

    for frame_idx in range(0, frames - tube_length + 1, tube_length):
        tube_mask = torch.zeros((tube_length, channels, height, width))
        for frame in range(tube_length):
            frame_mask = torch.zeros((height, width))
            for i in range(height // patch_size[0]):
                for j in range(width // patch_size[1]):
                    patch_mask = torch.from_numpy(
                        np.random.choice(mask_per_frame, size=1)
                    )[0]
                    frame_mask[
                        i * patch_size[0] : (i + 1) * patch_size[0],
                        j * patch_size[1] : (j + 1) * patch_size[1],
                    ] = patch_mask
            frame_mask = frame_mask.repeat(channels, 1, 1)
            tube_mask[frame, :channels, ...] = frame_mask

        mask[frame_idx : frame_idx + tube_length] = tube_mask

    mask = mask.to(video_data.device)
    masked_video = video_data * mask
    return masked_video


def apply_flow_tube_mask_to_video(
    video_data,
    flow,
    fl_mask_ratio=0.7,
    bg_mask_ratio=0,
    tube_length=4,
    patch_size=(16, 16),
):
    frames, channels, height, width = video_data.shape
    assert flow is not None
    mask = torch.zeros_like(video_data)
    for start_idx in range(0, frames - tube_length + 1, tube_length):
        tube_mask = torch.zeros((tube_length, channels, height, width))
        for frame in range(tube_length):
            frame_index = start_idx + frame
            flow_box = flow[frame_index]
            _, x, y, w, h = flow_box
            num_patches_per_frame = (height // patch_size[0]) * (
                width // patch_size[1]
            )
            num_flow_patches_per_frame = (h // patch_size[0]) * (
                w // patch_size[1]
            )
            # Guard: flow box smaller than one patch → treat entire frame as bg.
            if num_flow_patches_per_frame == 0:
                frame_mask = torch.ones((height, width))
                frame_mask = frame_mask.repeat(channels, 1, 1)
                tube_mask[frame, :channels, ...] = frame_mask
                continue
            num_bg_patches_per_frame = num_patches_per_frame - num_flow_patches_per_frame
            num_bg_masks_per_frame = int(bg_mask_ratio * num_bg_patches_per_frame)
            num_flow_masks_per_frame = int(fl_mask_ratio * num_flow_patches_per_frame)
            flow_mask_per_frame = torch.hstack(
                [
                    torch.ones(num_flow_patches_per_frame - num_flow_masks_per_frame),
                    torch.zeros(num_flow_masks_per_frame),
                ]
            )

            bg_mask_per_frame = torch.hstack(
                [
                    torch.ones(num_bg_patches_per_frame - num_bg_masks_per_frame),
                    torch.zeros(num_bg_masks_per_frame),
                ]
            )
            frame_mask = torch.zeros((height, width))
            for i in range(height // patch_size[0]):
                for j in range(width // patch_size[1]):
                    cur_y = i * patch_size[0]
                    cur_x = j * patch_size[1]
                    if abs(cur_y - y) <= h // 2 and abs(cur_x - x) <= w // 2:
                        patch_mask = torch.from_numpy(
                            np.random.choice(flow_mask_per_frame, size=1)
                        )[0]
                    else:
                        patch_mask = torch.from_numpy(
                            np.random.choice(bg_mask_per_frame, size=1)
                        )[0]
                    frame_mask[
                        i * patch_size[0] : (i + 1) * patch_size[0],
                        j * patch_size[1] : (j + 1) * patch_size[1],
                    ] = patch_mask
            frame_mask = frame_mask.repeat(channels, 1, 1)
            tube_mask[frame, :channels, ...] = frame_mask

        mask[start_idx : start_idx + tube_length] = tube_mask

    mask = mask.to(video_data.device)
    masked_video = video_data * mask
    return masked_video


_YOLO_MODEL = None


def _get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    try:
        from ultralytics import YOLO
    except Exception as exc:
        logger.warning("YOLO not available for flow mask: %s", exc)
        return None
    try:
        _YOLO_MODEL = YOLO("yolov8n.pt")
    except Exception as exc:
        logger.warning("Failed to load YOLO weights: %s", exc)
        return None
    return _YOLO_MODEL


def estimate_motion(frames=None, video_path: Optional[str] = None):
    model = _get_yolo_model()
    if model is None:
        return [], None

    track_history = defaultdict(lambda: [])
    flow_img_size = None
    frame_count = 0
    frame_shape = None

    if frames is None and video_path:
        try:
            import cv2
        except Exception as exc:
            logger.warning("cv2 unavailable for flow mask: %s", exc)
            return [], None
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_shape = frame.shape[:2]
            results = model.track(frame, persist=True)
            boxes = results[0].boxes
            if boxes is None or boxes.xywh is None or len(boxes) == 0:
                frame_count += 1
                continue
            xywh = boxes.xywh
            if xywh.numel() == 0:
                frame_count += 1
                continue
            annotated_frame = results[0].plot()
            flow_img_size = annotated_frame.shape[:2]
            ids = boxes.id
            if ids is None:
                xywh_np = xywh.cpu().numpy()
                if xywh_np.ndim == 1:
                    xywh_np = xywh_np.reshape(1, -1)
                areas = xywh_np[:, 2] * xywh_np[:, 3]
                box = xywh_np[int(areas.argmax())]
                track_history[0].append([frame_count, *box])
            else:
                track_ids = ids.int().cpu().tolist()
                xywh_np = xywh.cpu().numpy()
                for box, track_id in zip(xywh_np, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append([frame_count, x, y, w, h])
            frame_count += 1
        cap.release()
    else:
        frames = frames or []
        for frame in frames:
            frame_np = np.array(frame)
            frame_shape = frame_np.shape[:2]
            results = model.track(frame_np, persist=True)
            boxes = results[0].boxes
            if boxes is None or boxes.xywh is None or len(boxes) == 0:
                frame_count += 1
                continue
            xywh = boxes.xywh
            if xywh.numel() == 0:
                frame_count += 1
                continue
            flow_img_size = frame_np.shape[:2]
            ids = boxes.id
            if ids is None:
                xywh_np = xywh.cpu().numpy()
                if xywh_np.ndim == 1:
                    xywh_np = xywh_np.reshape(1, -1)
                areas = xywh_np[:, 2] * xywh_np[:, 3]
                box = xywh_np[int(areas.argmax())]
                track_history[0].append([frame_count, *box])
            else:
                track_ids = ids.int().cpu().tolist()
                xywh_np = xywh.cpu().numpy()
                for box, track_id in zip(xywh_np, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append([frame_count, x, y, w, h])
            frame_count += 1

    if flow_img_size is None and frame_shape is not None:
        flow_img_size = frame_shape

    ret_tracks = []
    for _, items in track_history.items():
        if not items:
            continue
        frame_map = {int(item[0]): item[1:] for item in items}
        min_frame = min(frame_map)
        first_box = frame_map[min_frame]
        last_box = first_box
        full = []
        for f in range(0, min_frame):
            full.append([f, *first_box])
        for f in range(min_frame, frame_count):
            if f in frame_map:
                last_box = frame_map[f]
            full.append([f, *last_box])
        if len(full) == frame_count:
            ret_tracks.append(np.asarray(full).astype(int))
    if not ret_tracks and frame_count > 0 and flow_img_size is not None:
        h, w = flow_img_size
        x = w / 2.0
        y = h / 2.0
        bw = w * 0.5
        bh = h * 0.5
        fallback = [[f, x, y, bw, bh] for f in range(frame_count)]
        ret_tracks.append(np.asarray(fallback).astype(int))
    return ret_tracks, flow_img_size


class LoadVideoStride:
    def __init__(
        self,
        num_frames=81,
        sample_start_idx=0,
        sample_frame_rate=1,
        frame_processor=lambda x: x,
        time_division_factor=4,
        time_division_remainder=1,
    ):
        self.num_frames = num_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.frame_processor = frame_processor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self._fallback = LoadVideo(
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            frame_processor=frame_processor,
        )
        self._decord_failed = False

    def __call__(self, data: str):
        if self._decord_failed:
            return self._fallback(data)
        try:
            import decord
        except Exception:
            self._decord_failed = True
            return self._fallback(data)
        decord.bridge.set_bridge("torch")
        try:
            vr = decord.VideoReader(data)
            sample_index = list(
                range(self.sample_start_idx, len(vr), self.sample_frame_rate)
            )[: self.num_frames]
            if not sample_index:
                raise ValueError("No frames sampled.")
            frames = vr.get_batch(sample_index).cpu().numpy()
            images = [self.frame_processor(Image.fromarray(frame)) for frame in frames]
            return images
        except Exception:
            self._decord_failed = True
            return self._fallback(data)


class SingleVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, prompt, masked_prompt, validation_prompts):
        self.data = {
            "video": video_path,
            "prompt": prompt,
            "masked_prompt": masked_prompt,
            "validation_prompts": validation_prompts,
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data.copy()


def _load_reference_video_frames(
    video_path: str,
    width: Optional[int],
    height: Optional[int],
    frame_num: int,
    start_idx: int,
    frame_rate: int,
):
    if not video_path:
        return None
    max_pixels = 1024 * 1024 if height is None or width is None else None
    frame_processor = ImageCropAndResize(height, width, max_pixels, 16, 16)
    loader = LoadVideoStride(
        num_frames=frame_num,
        sample_start_idx=start_idx,
        sample_frame_rate=frame_rate,
        frame_processor=frame_processor,
    )
    return loader(video_path)


def _find_latest_lora(output_dir):
    if not output_dir or not os.path.isdir(output_dir):
        return None
    candidates = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".safetensors")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


class WanTTTTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_alpha=None,
        lora_init_weights="gaussian",
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_lora=True,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        device="cpu",
        height=None,
        width=None,
        num_frames=81,
        sample_start_idx=0,
        sample_frame_rate=1,
        use_flow_mask=True,
        fl_mask_ratio=0.75,
        bg_mask_ratio=0.75,
        mask_patch_size: Tuple[int, int] = (2, 2),
        latent_scale=1.0,
        noise_scheduler_type="flow",
        prediction_type="epsilon",
        target_noise_scale=0.01,
        flow_align_ratio=0.2,
        flow_align_cfg_scale=7.5,
        flow_align_gamma=1.0,
        use_contrastive_loss=True,
        loss_mae_weight=0.1,
        loss_contrastive_weight=0.1,
        gpt2_model_name="gpt2",
        text_mask_ratio=0.15,
        text_augmentation_max_length=100,
        contrastive_temperature=0.5,
        use_extra_text_token=True,
        extra_text_token_trainable=True,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is disabled; enabling it to reduce memory usage."
            )
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"
            )
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )

        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        if preset_lora_path is not None and preset_lora_model is not None:
            self.pipe.load_lora(getattr(self.pipe, preset_lora_model), preset_lora_path)

        if use_lora and lora_base_model is not None:
            if (not hasattr(self.pipe, lora_base_model)) or getattr(self.pipe, lora_base_model) is None:
                raise ValueError(f"No {lora_base_model} model found in the pipeline.")
            base_model = getattr(self.pipe, lora_base_model)
            target_modules = self.parse_lora_target_modules(base_model, lora_target_modules)
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank if lora_alpha is None else lora_alpha,
                init_lora_weights=lora_init_weights,
                target_modules=target_modules,
            )
            base_model = inject_adapter_in_model(lora_config, base_model)
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = base_model.load_state_dict(state_dict, strict=False)
                if len(load_result[1]) > 0:
                    logger.warning(
                        "LoRA key mismatch: unexpected keys %s", load_result[1]
                    )
            setattr(self.pipe, lora_base_model, base_model)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        self.use_flow_mask = use_flow_mask
        self.fl_mask_ratio = fl_mask_ratio
        self.bg_mask_ratio = bg_mask_ratio
        self.mask_patch_size = mask_patch_size
        self.latent_scale = latent_scale

        self.noise_scheduler_type = noise_scheduler_type
        self.prediction_type = prediction_type
        self.target_noise_scale = target_noise_scale
        self.flow_align_ratio = float(flow_align_ratio)
        self.flow_align_cfg_scale = flow_align_cfg_scale
        self.flow_align_gamma = flow_align_gamma

        self.use_contrastive_loss = use_contrastive_loss
        self.loss_mae_weight = loss_mae_weight
        self.loss_contrastive_weight = loss_contrastive_weight

        self.gpt2_model_name = gpt2_model_name
        self.text_mask_ratio = text_mask_ratio
        self.text_augmentation_max_length = text_augmentation_max_length
        self.contrastive_temperature = contrastive_temperature

        if use_extra_text_token:
            text_dim = self.pipe.dit.text_embedding[0].in_features
            token_device = self.pipe.dit.patch_embedding.weight.device
            token_dtype = self.pipe.dit.text_embedding[0].weight.dtype
            extra_text_token = torch.nn.Parameter(
                torch.zeros(1, int(text_dim), device=token_device, dtype=token_dtype)
            )
            if not extra_text_token_trainable:
                extra_text_token.requires_grad_(False)
            self.pipe.dit.extra_text_token = extra_text_token

        self._motion_cache = {}

    def get_optimizer_groups(self, learning_rate: float, extra_text_token_lr: Optional[float] = None):
        extra_token_params = []
        extra_text_token = getattr(self.pipe.dit, "extra_text_token", None)
        if extra_text_token is not None and extra_text_token.requires_grad:
            extra_token_params.append(extra_text_token)
        extra_token_param_ids = {id(p) for p in extra_token_params}
        params_to_optimize = [
            p for p in self.parameters() if p.requires_grad and id(p) not in extra_token_param_ids
        ]
        optimizer_groups = []
        if params_to_optimize:
            optimizer_groups.append({"params": params_to_optimize, "lr": learning_rate})
        if extra_token_params:
            token_lr = learning_rate if extra_text_token_lr is None else float(extra_text_token_lr)
            optimizer_groups.append({"params": extra_token_params, "lr": token_lr})
        if not optimizer_groups:
            raise ValueError("No trainable parameters found. Check trainable_modules or LoRA settings.")
        return optimizer_groups, params_to_optimize + extra_token_params

    def _encode_prompts(self, prompts, device):
        ids, mask = self.pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def _apply_extra_text_vector(self, contexts):
        extra_text_token = getattr(self.pipe.dit, "extra_text_token", None)
        if extra_text_token is None:
            return contexts
        vec = extra_text_token.to(device=contexts.device, dtype=contexts.dtype)
        return contexts + vec

    def _get_prompt(self, data):
        prompt = data.get("prompt")
        if not prompt:
            prompt = data.get("original_prompt")
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ""
        return (prompt or "").strip()

    def _get_masked_prompt(self, data):
        masked_prompt = data.get("masked_prompt")
        if isinstance(masked_prompt, (list, tuple)):
            masked_prompt = masked_prompt[0] if masked_prompt else None
        if masked_prompt is None:
            return ""
        return str(masked_prompt).strip()

    def _get_validation_prompts(self, data, fallback):
        prompts = []
        if "validation_prompts" in data:
            raw = data.get("validation_prompts")
            if isinstance(raw, (list, tuple)):
                prompts.extend([str(v).strip() for v in raw if str(v).strip()])
            elif isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("["):
                    try:
                        parsed = json.loads(raw)
                        prompts.extend([str(v).strip() for v in parsed if str(v).strip()])
                    except Exception:
                        pass
                if not prompts and raw:
                    prompts.extend([v.strip() for v in raw.split("|") if v.strip()])
        if not prompts:
            for key in _DATASET_PROMPT_COLUMNS:
                value = (data.get(key) or "").strip()
                if value:
                    prompts.append(value)
        if not prompts and fallback:
            prompts = list(fallback)
        return prompts

    def _prepare_video_tensor(self, data, weight_dtype):
        video = None
        video_path = None
        for key in ("video", "name", "video_path"):
            if key in data:
                video = data[key]
                if isinstance(video, str):
                    video_path = video
                break
        if video is None and "pixel_values" in data:
            video = data["pixel_values"]
        if video is None:
            raise ValueError("No video data found in dataset entry.")

        if isinstance(video, str):
            frames = self._load_video_frames(video, weight_dtype)
            video_tensor = self.pipe.preprocess_video(frames, torch_dtype=weight_dtype, device=self.pipe.device)
            return video_tensor, frames, video

        if isinstance(video, list):
            frames = video
            video_tensor = self.pipe.preprocess_video(frames, torch_dtype=weight_dtype, device=self.pipe.device)
            return video_tensor, frames, video_path

        if torch.is_tensor(video):
            if video.ndim == 4:
                if video.shape[0] in (1, 3, 4):
                    video = video.unsqueeze(0)
                else:
                    video = video.permute(1, 0, 2, 3).unsqueeze(0)
            if video.ndim != 5:
                raise ValueError(f"Unexpected video tensor shape: {video.shape}")
            return video.to(device=self.pipe.device, dtype=weight_dtype), None, video_path

        raise ValueError("Unsupported video input format.")

    def _load_video_frames(self, video_path, weight_dtype):
        try:
            import decord
        except Exception:
            decord = None
        if decord is not None:
            decord.bridge.set_bridge("torch")
            if self.width is not None and self.height is not None:
                vr = decord.VideoReader(video_path, width=self.width, height=self.height)
            else:
                vr = decord.VideoReader(video_path)
            sample_index = list(
                range(self.sample_start_idx, len(vr), self.sample_frame_rate)
            )[: self.num_frames]
            if not sample_index:
                raise ValueError("No frames sampled from video.")
            frames = vr.get_batch(sample_index).cpu().numpy()
            return [Image.fromarray(frame) for frame in frames]

        # Fallback to imageio via LoadVideo
        max_pixels = 1024 * 1024 if self.height is None or self.width is None else None
        frame_processor = ImageCropAndResize(
            self.height, self.width, max_pixels, 16, 16
        )
        fallback = LoadVideo(
            num_frames=self.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            frame_processor=frame_processor,
        )
        return fallback(video_path)

    def _get_motion_list(self, frames, video_path, latents):
        if not self.use_flow_mask:
            return None
        cache_key = video_path or id(frames)
        if cache_key in self._motion_cache:
            return self._motion_cache[cache_key]

        all_motions_list, flow_img_size = estimate_motion(frames, video_path)
        track_lengths = [len(motions) for motions in all_motions_list]
        motion_list = None
        if not track_lengths or max(track_lengths) == 0:
            self._motion_cache[cache_key] = None
            return None

        main_obj_id = track_lengths.index(max(track_lengths))
        max_frames = track_lengths[main_obj_id]
        sample_index = list(
            range(self.sample_start_idx, max_frames, self.sample_frame_rate)
        )[: self.num_frames]
        if not sample_index:
            raise ValueError("No frames sampled for motion estimation.")
        motion_list = [all_motions_list[main_obj_id][index] for index in sample_index]

        if flow_img_size is not None:
            target_h = latents.shape[2]
            target_w = latents.shape[3]
            motion_list = transfer_coordinates(motion_list, flow_img_size, [target_h, target_w])

        self._motion_cache[cache_key] = motion_list
        return motion_list

    def forward(
        self,
        data,
        noise_scheduler,
        gpt2_model=None,
        gpt2_tokenizer=None,
        validation_prompts=None,
        weight_dtype=None,
    ):
        device = self.pipe.device
        weight_dtype = weight_dtype or self.pipe.torch_dtype

        video_tensor, frames, video_path = self._prepare_video_tensor(data, weight_dtype)
        if video_tensor.size(0) != 1:
            raise ValueError("This training script currently supports batch_size=1.")
        video = video_tensor[0]
        with torch.no_grad():
            latents_list = self.pipe.vae.encode(
                video_tensor, device=device, tiled=False
            )
        latents = latents_list[0].to(dtype=self.pipe.torch_dtype, device=device)

        motion_list = None
        if self.use_flow_mask and frames is not None:
            motion_list = self._get_motion_list(frames, video_path, latents)

        latent_length = latents.shape[1]
        if motion_list is not None:
            if len(motion_list) < latent_length:
                motion_list = motion_list + [motion_list[-1]] * (
                    latent_length - len(motion_list)
                )
            elif len(motion_list) > latent_length:
                motion_list = motion_list[:latent_length]
            masked_latents = apply_flow_tube_mask_to_video(
                latents.permute(1, 0, 2, 3),
                motion_list,
                fl_mask_ratio=self.fl_mask_ratio,
                bg_mask_ratio=self.bg_mask_ratio,
                tube_length=latent_length,
                patch_size=self.mask_patch_size,
            )
        else:
            masked_latents = apply_tube_mask_to_video(
                latents.permute(1, 0, 2, 3),
                mask_ratio=self.bg_mask_ratio,
                tube_length=latent_length,
                patch_size=self.mask_patch_size,
            )
        masked_latents = masked_latents.permute(1, 0, 2, 3).contiguous()
        masked_latents = masked_latents.to(dtype=self.pipe.torch_dtype, device=device)

        latents = latents * self.latent_scale
        masked_latents = masked_latents * self.latent_scale

        combined_latents = torch.stack([latents, masked_latents], dim=0)

        noise = torch.randn_like(combined_latents)
        if self.noise_scheduler_type == "flow":
            schedule_timesteps = noise_scheduler.timesteps.to(device=combined_latents.device)
            schedule_sigmas = noise_scheduler.sigmas.to(device=combined_latents.device)
            step_idx = torch.randint(
                0, schedule_timesteps.numel(), (1,), device=combined_latents.device
            )
            timestep_single = schedule_timesteps[step_idx].reshape(1)
            sigma_single = schedule_sigmas[step_idx].reshape(1)
            timesteps = timestep_single.repeat(combined_latents.shape[0])
            sigmas = sigma_single.repeat(combined_latents.shape[0])
            if hasattr(noise_scheduler, "scale_noise"):
                noisy_latents = noise_scheduler.scale_noise(
                    combined_latents, timesteps, noise=noise
                )
            else:
                # diffsynth FlowMatchScheduler.add_noise expects a single timestep
                noisy_latents = noise_scheduler.add_noise(
                    combined_latents, noise, timestep_single[0]
                )
        else:
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (1,), device=combined_latents.device
            )
            timesteps = timesteps.repeat(combined_latents.shape[0]).long()
            noisy_latents = noise_scheduler.add_noise(combined_latents, noise, timesteps)

        validation_prompts = self._get_validation_prompts(data, validation_prompts or [])
        flow_align_ratio = float(self.flow_align_ratio)
        flow_align_ratio = max(0.0, min(1.0, flow_align_ratio))
        if flow_align_ratio > 0.0 and self.noise_scheduler_type != "flow":
            flow_align_ratio = 0.0

        use_flow_align = False
        source_text = self._get_prompt(data)
        target_text = source_text
        if flow_align_ratio > 0.0 and validation_prompts:
            if torch.rand(1, device=combined_latents.device).item() < flow_align_ratio:
                use_flow_align = True
                target_text = validation_prompts[
                    torch.randint(0, len(validation_prompts), (1,)).item()
                ]

        original_text = target_text
        augmented_text = text_augmentation(
            original_text,
            gpt2_model,
            gpt2_tokenizer,
            max_length=self.text_augmentation_max_length,
        )
        masked_text = self._get_masked_prompt(data)
        if not masked_text:
            masked_text = text_masking(augmented_text, mask_ratio=self.text_mask_ratio)

        text_inputs = [original_text, augmented_text, masked_text]
        source_text_index = None
        if use_flow_align:
            source_text_index = len(text_inputs)
            text_inputs.append(source_text)

        with torch.no_grad():
            text_outputs = self._encode_prompts(text_inputs, device)
        text_outputs = self._apply_extra_text_vector(text_outputs)
        main_text_outputs = text_outputs[:3]
        encoder_hidden_states = main_text_outputs[0].unsqueeze(0).repeat(
            combined_latents.shape[0], 1, 1
        )

        model_pred = self.pipe.model_fn(
            dit=self.pipe.dit,
            latents=noisy_latents,
            timestep=timesteps,
            context=encoder_hidden_states,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        if use_flow_align:
            with torch.no_grad():
                source_context = text_outputs[source_text_index].unsqueeze(0).repeat(
                    combined_latents.shape[0], 1, 1
                )
                source_pred = self.pipe.model_fn(
                    dit=self.pipe.dit,
                    latents=noisy_latents,
                    timestep=timesteps,
                    context=source_context,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                )
            target_pred = model_pred.detach()
            cfg_pred = source_pred + self.flow_align_cfg_scale * (target_pred - source_pred)
            sigma_t = sigmas.view(-1, 1, 1, 1, 1)
            est_target = noisy_latents - sigma_t * cfg_pred
            est_source = noisy_latents - sigma_t * source_pred
            target = (cfg_pred - source_pred) + self.flow_align_gamma * (est_target - est_source)
        else:
            if self.noise_scheduler_type == "flow":
                target = noise - combined_latents
            else:
                if self.prediction_type == "epsilon":
                    target = noise
                elif self.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(combined_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")
            if self.target_noise_scale != 0.0:
                target = target + self.target_noise_scale * torch.randn_like(target)

        loss_noise = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.noise_scheduler_type == "flow":
            sigma_t = sigmas[1].view(1, 1, 1, 1, 1)
            latent_rec = noisy_latents[1].unsqueeze(0) - sigma_t * model_pred[1].unsqueeze(0)
        else:
            latent_rec = noise_scheduler.step(
                model_pred[1].unsqueeze(0), timesteps[0], noisy_latents[1].unsqueeze(0)
            ).pred_original_sample
        latent_rec = (
            latent_rec.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, latent_rec.shape[1], latent_rec.shape[3], latent_rec.shape[4])
        )
        loss_mae = F.mse_loss(
            latent_rec.float(),
            combined_latents[0].permute(1, 0, 2, 3).float(),
            reduction="mean",
        )

        loss_contrastive = torch.tensor(0.0, device=combined_latents.device)
        if self.use_contrastive_loss:
            text_dtype = self.pipe.dit.text_embedding[0].weight.dtype
            text_features = self.pipe.dit.text_embedding(
                main_text_outputs.to(device=combined_latents.device, dtype=text_dtype)
            ).mean(dim=1)
            video_features = self.pipe.dit.encode(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )
            if isinstance(video_features, list):
                video_features = torch.stack(video_features, dim=0)
            combined_features = torch.cat([text_features, video_features], dim=0)
            labels = torch.cat(
                [
                    torch.arange(3, device=combined_latents.device),
                    torch.arange(video_features.shape[0], device=combined_latents.device),
                ],
                dim=0,
            )
            loss_contrastive = contrastive_loss(
                combined_features, labels, temperature=self.contrastive_temperature
            )

        loss = (
            loss_noise
            + self.loss_mae_weight * loss_mae
            + self.loss_contrastive_weight * loss_contrastive
        )
        return loss


def _infer_vae_compression(args) -> Tuple[int, int]:
    """Infer (spatial, temporal) VAE downsampling factors from model identifiers.

    The mask building below has to compute latent dimensions *before* the
    pipeline is loaded, so we cannot just query ``pipe.vae``. Instead we
    inspect every place the user could have specified a model:
      - ``model_id_with_origin_paths`` (e.g. ``"Wan-AI/Wan2.2-TI2V-5B:..."``)
      - ``model_paths``                (raw filesystem paths / JSON list)
      - ``tokenizer_path``             (sometimes lives in the model dir)

    Mapping (verified against ``diffsynth/configs/model_configs.py`` and
    ``diffsynth/models/wan_video_vae.py``):

      - ``Wan2.2_VAE`` / ``Wan2.2-TI2V-5B``  → spatial 16×, temporal 4× (z=48)
      - everything else (Wan2.1_VAE, all other Wan2.2-A14B variants) →
        spatial 8×, temporal 4× (z=16)

    Temporal compression is 4× for every Wan VAE shipped so far, but we
    return it as a tuple so callers do not hard-code ``// 4`` either.
    """
    sources = []
    for attr in ("model_id_with_origin_paths", "model_paths", "tokenizer_path"):
        v = getattr(args, attr, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            sources.extend(str(x) for x in v)
        else:
            sources.append(str(v))
    haystack = " ".join(sources).lower()
    # Only Wan2.2-TI2V-5B ships the 48-channel / 16×-spatial VAE.
    # Wan2.2-T2V-A14B / I2V-A14B / S2V-14B / Animate-14B all reuse Wan2.1_VAE.
    if "wan2.2_vae" in haystack or "ti2v-5b" in haystack:
        return (16, 4)
    return (8, 4)


def _build_inference_pipe(args, device):
    tmp = DiffusionTrainingModule()
    model_configs = tmp.parse_model_configs(
        args.model_paths, args.model_id_with_origin_paths, device=device
    )
    tokenizer_config = (
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"
        )
        if args.tokenizer_path is None
        else ModelConfig(args.tokenizer_path)
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    return pipe


def _load_extra_text_token(pipe, lora_path):
    if not lora_path:
        return
    state_dict = load_state_dict(lora_path, torch_dtype=pipe.torch_dtype, device=pipe.device)
    extra_key = None
    for key in state_dict:
        if key.endswith("extra_text_token"):
            extra_key = key
            break
    if extra_key is None:
        return
    extra_tensor = state_dict[extra_key].to(device=pipe.device, dtype=pipe.torch_dtype)
    pipe.dit.extra_text_token = torch.nn.Parameter(extra_tensor)


def _feather_mask(mask: torch.Tensor, feather_radius: int) -> torch.Tensor:
    """Apply Gaussian blur feathering to a 5-D mask ``(1,1,T,H,W)``.

    Args:
        mask: Binary or soft mask of shape ``(1, 1, T, H, W)``.
        feather_radius: Gaussian blur kernel radius.
            0 = no-op (returns the mask unchanged).

    Returns:
        Feathered mask clamped to [0, 1], same shape as input.
    """
    if feather_radius <= 0:
        return mask
    kernel_size = feather_radius * 2 + 1
    sigma = float(feather_radius) / 2.0
    # Build 2D Gaussian kernel
    ax = torch.arange(kernel_size, dtype=torch.float32) - feather_radius
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel_2d = kernel.view(1, 1, kernel_size, kernel_size)
    # Apply per-frame 2D spatial blur
    pad = feather_radius
    blurred_frames = []
    for t_idx in range(mask.shape[2]):
        frame = mask[:, :, t_idx, :, :]  # [1, 1, H, W]
        frame_padded = torch.nn.functional.pad(
            frame, (pad, pad, pad, pad), mode="replicate"
        )
        frame_blurred = torch.nn.functional.conv2d(
            frame_padded, kernel_2d, padding=0
        )
        blurred_frames.append(frame_blurred)
    mask = torch.stack(blurred_frames, dim=2)  # [1, 1, T, H, W]
    return mask.clamp(0.0, 1.0)


def _build_reference_noise_mask_latents(
    reference_frames,
    latent_length: int,
    latent_h: int,
    latent_w: int,
    feather_radius: int = 3,
    box_scale: float = 1.0,
):
    """Build a soft latent-space mask from YOLO tracking boxes.

    Args:
        feather_radius: Gaussian blur kernel radius for soft edges.
            0 = hard binary mask, >0 = soft feathered edges.
        box_scale: Scale factor for the YOLO box size.
            1.0 = original box, >1.0 = enlarge box, <1.0 = shrink box.
            Center position stays the same, width and height are multiplied.
    """
    if not reference_frames or latent_length <= 0 or latent_h <= 0 or latent_w <= 0:
        return None
    all_motions_list, flow_img_size = estimate_motion(frames=reference_frames, video_path=None)
    if not all_motions_list or flow_img_size is None:
        return None

    track_lengths = [len(motions) for motions in all_motions_list]
    if not track_lengths or max(track_lengths) <= 0:
        return None
    main_obj_id = int(track_lengths.index(max(track_lengths)))
    motion_list = list(all_motions_list[main_obj_id])
    if not motion_list:
        return None

    if len(motion_list) != latent_length:
        indices = np.linspace(0, len(motion_list) - 1, latent_length).round().astype(int)
        motion_list = [motion_list[i] for i in indices]

    motion_list = transfer_coordinates(motion_list, flow_img_size, [latent_h, latent_w])

    box_scale = max(0.0, float(box_scale))
    mask = torch.zeros((1, 1, latent_length, latent_h, latent_w), dtype=torch.float32)
    for t, flow_box in enumerate(motion_list[:latent_length]):
        _, x, y, w, h = flow_box
        half_w = float(w) * box_scale / 2.0
        half_h = float(h) * box_scale / 2.0
        x1 = max(0, int(math.floor(float(x) - half_w)))
        x2 = min(latent_w, int(math.ceil(float(x) + half_w)))
        y1 = max(0, int(math.floor(float(y) - half_h)))
        y2 = min(latent_h, int(math.ceil(float(y) + half_h)))
        if x2 > x1 and y2 > y1:
            mask[:, :, t, y1:y2, x1:x2] = 1.0
    if float(mask.sum().item()) <= 0:
        return None

    mask = _feather_mask(mask, feather_radius)
    return mask


def _save_mask_visualizations(
    pixel_mask: torch.Tensor,
    latent_mask: torch.Tensor,
    video_path: str,
    sampled_indices: List[int],
    save_dir: str,
):
    """Save pixel-space overlay images and latent-space grayscale images.

    Outputs are written into *save_dir*/semantic_mask_vis/:
      - ``pixel_XX.png``  – video frame with red mask overlay
      - ``latent_XX.png`` – latent-space mask as grayscale

    Args:
        pixel_mask: ``(1, 1, T_sampled, H_pixel, W_pixel)`` after temporal slice.
        latent_mask: ``(1, 1, T_lat, H_lat, W_lat)`` after downsample + feather.
        video_path: Path to the reference video (for loading overlay frames).
        sampled_indices: Which video frame indices were selected.
        save_dir: Root directory under which ``semantic_mask_vis/`` is created.
    """
    vis_dir = os.path.join(save_dir, "semantic_mask_vis")
    os.makedirs(vis_dir, exist_ok=True)

    # --- Pixel-space overlays ---
    # Load the matching frames from the source video.
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        valid_indices = [i for i in sampled_indices if 0 <= i < len(vr)]
        raw_frames = [Image.fromarray(vr[i].asnumpy()) for i in valid_indices]
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        valid_indices = [i for i in sampled_indices if 0 <= i < total]
        raw_frames = []
        for i in valid_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                raw_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

    pm = pixel_mask[0, 0].cpu().numpy()  # (T_sampled, H, W)
    for t_idx in range(min(len(raw_frames), pm.shape[0])):
        frame = raw_frames[t_idx].convert("RGBA")
        mask_arr = (pm[t_idx] * 255).astype(np.uint8)
        # Resize mask to frame size if they differ
        if (mask_arr.shape[0], mask_arr.shape[1]) != (frame.height, frame.width):
            mask_img = Image.fromarray(mask_arr, mode="L").resize(
                (frame.width, frame.height), Image.NEAREST
            )
        else:
            mask_img = Image.fromarray(mask_arr, mode="L")
        # Red overlay with alpha
        overlay_color = Image.new("RGBA", frame.size, (255, 0, 0, 120))
        overlay = Image.composite(overlay_color, frame, mask_img)
        overlay.convert("RGB").save(os.path.join(vis_dir, f"pixel_{t_idx:03d}.png"))

    _std_logger.info("Saved %d pixel-space mask overlays to %s/", len(raw_frames), vis_dir)

    # --- Original pixel-space binary mask (pure black/white) ---
    # Save the SAM2 mask at original video resolution as a clean B/W image,
    # so the user can inspect exactly what SAM2 segmented before any
    # downsampling or feathering was applied.
    for t_idx in range(min(len(raw_frames), pm.shape[0])):
        mask_bw = (pm[t_idx] * 255).astype(np.uint8)
        if (mask_bw.shape[0], mask_bw.shape[1]) != (raw_frames[t_idx].height, raw_frames[t_idx].width):
            mask_bw_img = Image.fromarray(mask_bw, mode="L").resize(
                (raw_frames[t_idx].width, raw_frames[t_idx].height), Image.NEAREST
            )
        else:
            mask_bw_img = Image.fromarray(mask_bw, mode="L")
        mask_bw_img.save(os.path.join(vis_dir, f"mask_{t_idx:03d}.png"))

    _std_logger.info(
        "Saved %d original pixel-space mask images to %s/",
        min(len(raw_frames), pm.shape[0]), vis_dir,
    )


def _build_semantic_mask_latents(
    semantic_mask_config_path: str,
    video_path: str,
    original_prompt: str,
    modified_prompt: str,
    latent_length: int,
    latent_h: int,
    latent_w: int,
    num_frames: int,
    sample_start_idx: int = 0,
    sample_frame_rate: int = 1,
    feather_radius: int = 3,
    cache_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    grounding_phrases: Optional[List[str]] = None,
):
    """Build a latent-space mask using the LLM + Grounding-DINO + SAM2 pipeline.

    The semantic mask pipeline produces a pixel-space binary mask of shape
    ``(1, 1, T_all, H_pixel, W_pixel)`` covering **every** frame of the
    source video.  Before downsampling we select only the frames that the
    inference / training path actually uses (controlled by *sample_start_idx*,
    *sample_frame_rate* and *num_frames*) so that the temporal axis of the
    mask is aligned with the video frames fed into the diffusion model.

    Args:
        semantic_mask_config_path: Path to the semantic mask YAML config.
        video_path: Path to the reference video.
        original_prompt: Source/original prompt describing the reference video.
        modified_prompt: Target/edited prompt describing the desired output.
        latent_length: Temporal length in latent space.
        latent_h: Spatial height in latent space.
        latent_w: Spatial width in latent space.
        num_frames: Number of video frames actually used by inference.
        sample_start_idx: First frame index used when sampling the video.
        sample_frame_rate: Stride between sampled frames.
        feather_radius: Gaussian blur radius for soft edges (0 = hard mask).
        cache_dir: Optional directory for caching the mask ``.pt`` file.
        save_dir: Optional directory to save mask visualisation images.
        grounding_phrases: Optional explicit list of object names to mask
            (e.g. ``["jeep", "road"]``).  When provided the VLM step is
            skipped entirely and these phrases are fed directly to
            Grounding-DINO.  If ``None`` or empty the VLM is used as
            usual to infer the phrases from the two prompts.

    Returns:
        ``torch.Tensor`` of shape ``(1, 1, T_lat, H_lat, W_lat)`` or ``None``.
    """
    if latent_length <= 0 or latent_h <= 0 or latent_w <= 0:
        return None

    # ------ Build a deterministic cache key ------
    import hashlib
    _phrases_key = ",".join(sorted(grounding_phrases)) if grounding_phrases else ""
    _hash_src = (
        f"{Path(video_path).name}|{original_prompt}|{modified_prompt}"
        f"|{latent_length}x{latent_h}x{latent_w}"
        f"|s{sample_start_idx}_r{sample_frame_rate}_n{num_frames}"
        f"|phrases:{_phrases_key}"
    )
    _hash_hex = hashlib.sha256(_hash_src.encode("utf-8")).hexdigest()[:16]

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        video_stem = Path(video_path).stem
        cache_name = f"semantic_mask_{_sanitize_filename(video_stem)}_{_hash_hex}.pt"
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.isfile(cache_path):
            _std_logger.info("Loading cached semantic mask from %s", cache_path)
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached.shape == (1, 1, latent_length, latent_h, latent_w):
                return cached
            _std_logger.info(
                "Cached mask shape %s != expected (%d,%d,%d); regenerating.",
                list(cached.shape), latent_length, latent_h, latent_w,
            )

    # ------ Import and run the semantic mask pipeline ------
    _repo_root = str(Path(__file__).resolve().parents[3])
    import importlib.util
    _svm_path = os.path.join(_repo_root, "semantic_video_mask.py")
    if not os.path.isfile(_svm_path):
        raise FileNotFoundError(
            f"semantic_video_mask.py not found at {_svm_path}. "
            "Ensure it exists at the DiffSynth-Studio repo root."
        )
    spec = importlib.util.spec_from_file_location("semantic_video_mask", _svm_path)
    svm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svm_module)

    cfg = svm_module.load_config(semantic_mask_config_path)
    cfg["video_path"] = video_path
    cfg["original_prompt"] = original_prompt
    cfg["modified_prompt"] = modified_prompt

    # ------ Explicit grounding phrases (bypass VLM when provided) ------
    if grounding_phrases:
        cfg["grounding_phrases"] = list(grounding_phrases)
        _std_logger.info(
            "Using explicit grounding phrases (VLM skipped): %s", grounding_phrases
        )
    else:
        cfg.pop("grounding_phrases", None)

    # ------ Pre-compute sampled frame indices and inject into SAM2 config ------
    # This tells SAM2 to process only the frames the diffusion model will use,
    # instead of propagating through the full video (which can be 300+ frames
    # and exceed system memory for long videos).
    try:
        try:
            import decord as _decord
            _decord.bridge.set_bridge("native")
            _vr = _decord.VideoReader(video_path)
            _total_video_frames = len(_vr)
            del _vr
        except Exception:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(video_path)
            _total_video_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            _cap.release()
        sampled_indices = list(
            range(sample_start_idx, _total_video_frames, sample_frame_rate)
        )[:num_frames]
        if not sampled_indices:
            sampled_indices = list(range(min(num_frames, _total_video_frames)))
    except Exception as _exc:
        _std_logger.warning("Could not pre-compute sampled frame indices: %s", _exc)
        sampled_indices = None

    if sampled_indices is not None:
        # Deep-copy the sam2 sub-config so we don't mutate the loaded YAML.
        cfg["sam2"] = dict(cfg.get("sam2") or {})
        cfg["sam2"]["sampled_frame_indices"] = sampled_indices
        _std_logger.info(
            "SAM2 will process %d/%s sampled frames (start=%d, rate=%d)",
            len(sampled_indices),
            _total_video_frames if sampled_indices is not None else "?",
            sample_start_idx,
            sample_frame_rate,
        )

    # Disable the built-in mask/visualisation saving inside svm_module.run():
    # - save_visualization: would re-extract all frames at full resolution as
    #   overlay PNGs – extremely slow and redundant (_save_mask_visualizations
    #   below handles visualisation separately).
    # - save_masks: saves large .pt tensors (T×H×W) that are never read back by
    #   the training pipeline; the in-memory result["masks"] is used directly.
    cfg["save_visualization"] = False
    cfg["save_masks"] = False

    _std_logger.info(
        "Running semantic mask pipeline: video=%s, src='%s', tgt='%s'",
        video_path, original_prompt, modified_prompt,
    )
    result = svm_module.run(cfg)
    pixel_mask = result["masks"]  # (1, 1, T_sampled_or_all, H_pixel, W_pixel)

    if pixel_mask is None or float(pixel_mask.sum().item()) <= 0:
        _std_logger.warning("Semantic mask pipeline returned an empty mask.")
        return None

    # ------ Temporal alignment: select only the sampled frames ------
    # When SAM2 already processed only the sampled frames (sampled_indices was
    # injected above), T_sampled == num_frames and this slice is a no-op.
    # We keep it as a safety net in case SAM2 still returns extra frames.
    total_mask_frames = pixel_mask.shape[2]
    if sampled_indices is not None and total_mask_frames == len(sampled_indices):
        # SAM2 returned exactly the sampled frames – no extra slicing needed.
        _std_logger.info(
            "Semantic mask has %d sampled frames (start=%d, rate=%d)",
            total_mask_frames, sample_start_idx, sample_frame_rate,
        )
    else:
        # Fallback: SAM2 returned all frames; slice to sampled subset.
        post_sampled = list(
            range(sample_start_idx, total_mask_frames, sample_frame_rate)
        )[:num_frames]
        if not post_sampled:
            post_sampled = list(range(min(num_frames, total_mask_frames)))
        pixel_mask = pixel_mask[:, :, post_sampled, :, :]
        _std_logger.info(
            "Semantic mask temporal slice: %d/%d frames selected (start=%d, rate=%d)",
            pixel_mask.shape[2], total_mask_frames, sample_start_idx, sample_frame_rate,
        )

    # ------ Downsample to latent space (approximate) ------
    mask = F.interpolate(
        pixel_mask.float(),
        size=(latent_length, latent_h, latent_w),
        mode="nearest",
    )
    mask = mask.clamp(0.0, 1.0)

    # ------ Apply Gaussian feathering ------
    mask = _feather_mask(mask, feather_radius)

    # ------ Save visualisation images ------
    if save_dir:
        try:
            _save_mask_visualizations(
                pixel_mask=pixel_mask,
                latent_mask=mask,
                video_path=video_path,
                sampled_indices=sampled_indices,
                save_dir=save_dir,
            )
        except Exception as exc:
            _std_logger.warning("Failed to save mask visualizations: %s", exc)

    # ------ Cache the result ------
    if cache_path:
        torch.save(mask, cache_path)
        _std_logger.info("Cached semantic mask to %s", cache_path)

    return mask


def _run_inference_for_entry(args, validation_data, lora_path, entry_video, entry_source_prompt=None):
    prompts = validation_data.get("prompts") or []
    if getattr(args, "use_lora", True):
        print("Loading LoRA from:", lora_path)
    else:
        print("LoRA disabled; running base model inference.")
    if getattr(args, "use_lora", True) and not lora_path:
        raise RuntimeError(f"No LoRA checkpoint found in {getattr(args, 'output_path', 'unknown')}. Check that output_path matches the directory used during training.")
    if not prompts:
        _std_logger.warning("No validation prompts provided; skip inference.")
        return

    output_dir = validation_data.get("inference_output_dir") or args.inference_output_dir
    if output_dir:
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(args.output_path, output_dir)
    else:
        output_dir = os.path.join(args.output_path, "inference")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    height = int(validation_data.get("height", args.height or 480))
    width = int(validation_data.get("width", args.width or 832))
    video_length = int(
        validation_data.get(
            "video_length",
            validation_data.get("num_frames", args.inference_video_length or args.num_frames),
        )
    )
    num_inference_steps = int(
        validation_data.get("num_inference_steps", args.num_inference_steps)
    )
    guidance_scale = float(
        validation_data.get("guidance_scale", args.guidance_scale)
    )
    source_cfg_scale = float(
        validation_data.get("source_cfg_scale", args.source_cfg_scale)
    )
    source_prompt = validation_data.get("source_prompt", args.source_prompt)
    if (source_prompt is None or str(source_prompt).strip() == "") and entry_source_prompt:
        source_prompt = entry_source_prompt
    fps = int(validation_data.get("fps", args.fps))
    inference_shift = float(
        validation_data.get("inference_shift", args.inference_shift or args.flow_shift)
    )
    inference_seed = validation_data.get("inference_seed", args.inference_seed)

    use_reference_noise = bool(
        validation_data.get("use_reference_noise", args.use_reference_noise)
    )
    reference_video = validation_data.get("reference_video", args.reference_video)
    if not reference_video and entry_video:
        reference_video = entry_video
    reference_strength = validation_data.get("reference_strength", args.reference_strength)
    reference_init_strength = validation_data.get(
        "reference_init_strength", args.reference_init_strength
    )
    reference_fg_init_strength = validation_data.get(
        "reference_fg_init_strength",
        args.reference_fg_init_strength,
    )
    reference_bg_init_strength = validation_data.get(
        "reference_bg_init_strength",
        args.reference_bg_init_strength,
    )
    reference_strict_timestep_align = bool(
        validation_data.get(
            "reference_strict_timestep_align",
            args.reference_strict_timestep_align,
        )
    )
    reference_start_idx = int(
        validation_data.get("reference_start_idx", args.reference_start_idx)
    )
    reference_frame_rate = int(
        validation_data.get("reference_frame_rate", args.reference_frame_rate)
    )

    input_video = None
    denoising_strength = None
    dual_region_noise_enabled = False
    reference_noise_mask_latents = None
    latent_h = latent_w = latent_length = feather_radius = 0
    if use_reference_noise and reference_video:
        input_video = _load_reference_video_frames(
            reference_video,
            width=width,
            height=height,
            frame_num=video_length,
            start_idx=reference_start_idx,
            frame_rate=reference_frame_rate,
        )
        # Use actual frame count when video is shorter than requested, so noise and
        # input_latents match (avoids RuntimeError: tensor size mismatch in add_noise).
        if input_video and len(input_video) < video_length:
            video_length = len(input_video)
        if reference_init_strength is not None:
            denoising_strength = float(reference_init_strength)
        elif reference_strength is not None:
            denoising_strength = float(reference_strength)
        else:
            denoising_strength = 1.0

        if reference_fg_init_strength is not None and reference_bg_init_strength is not None:
            fg_strength = max(0.0, min(1.0, float(reference_fg_init_strength)))
            bg_strength = max(0.0, min(1.0, float(reference_bg_init_strength)))
            reference_fg_init_strength = fg_strength
            reference_bg_init_strength = bg_strength
            semantic_mask_cfg = getattr(args, "semantic_mask_config", None)
            if input_video:
                # Compute latent dimensions from video dimensions without loading the pipeline
                latent_h = height // 8
                latent_w = width // 8
                latent_length = (video_length - 1) // 4 + 1
                feather_radius = int(
                    validation_data.get("feather_radius", args.feather_radius)
                )

                if not semantic_mask_cfg:
                    # ---- YOLO bounding-box mask path (no per-prompt variation) ----
                    box_scale = float(
                        validation_data.get("box_scale", args.box_scale)
                    )
                    reference_noise_mask_latents = _build_reference_noise_mask_latents(
                        reference_frames=input_video,
                        latent_length=int(latent_length),
                        latent_h=int(latent_h),
                        latent_w=int(latent_w),
                        feather_radius=feather_radius,
                        box_scale=box_scale,
                    )
            if not semantic_mask_cfg:
                # For YOLO path, evaluate mask validity once here.
                if reference_noise_mask_latents is not None:
                    denoising_strength = max(fg_strength, bg_strength)
                    dual_region_noise_enabled = True
                else:
                    _std_logger.warning(
                        "Dual-region reference noise requested but no valid YOLO mask; "
                        "fallback to global denoising_strength.",
                    )
                    reference_fg_init_strength = None
                    reference_bg_init_strength = None
        elif (
            reference_fg_init_strength is not None
            or reference_bg_init_strength is not None
        ):
            _std_logger.warning(
                "Both reference_fg_init_strength and reference_bg_init_strength are required; fallback to global denoising_strength."
            )
            reference_fg_init_strength = None
            reference_bg_init_strength = None

    # Pre-compute shared semantic mask arguments (used inside the loop).
    _semantic_mask_cfg = getattr(args, "semantic_mask_config", None)
    _use_per_prompt_mask = (
        _semantic_mask_cfg
        and use_reference_noise
        and reference_video
        and input_video
        and reference_fg_init_strength is not None
        and reference_bg_init_strength is not None
    )

    _original_prompt = entry_source_prompt or source_prompt or ""

    for idx, prompt in enumerate(prompts):
        seed = None
        if inference_seed is not None:
            seed = int(inference_seed) + idx

        cur_mask = reference_noise_mask_latents
        cur_dual = dual_region_noise_enabled
        cur_fg = reference_fg_init_strength
        cur_bg = reference_bg_init_strength
        cur_denoising = denoising_strength

        # ----------------------------------------------------------------
        # Step A: SAM2 semantic mask (runs before Wan is loaded).
        # ----------------------------------------------------------------
        if _use_per_prompt_mask:
            _std_logger.info(
                "[%d/%d] Running SAM2 mask for prompt: '%s'",
                idx + 1, len(prompts), prompt,
            )
            prompt_vis_dir = os.path.join(
                output_dir, f"semantic_mask_vis_{idx:02d}_{_sanitize_filename(prompt)}"
            )
            # Resolve explicit grounding phrases: per-prompt > per-entry > global arg.
            _raw_phrases = (
                validation_data.get("grounding_phrases")
                or getattr(args, "grounding_phrases", None)
            )
            if isinstance(_raw_phrases, str):
                _entry_phrases = [p.strip() for p in _raw_phrases.split(",") if p.strip()]
            elif _raw_phrases:
                _entry_phrases = [str(p).strip() for p in _raw_phrases if str(p).strip()]
            else:
                _entry_phrases = None

            cur_mask = _build_semantic_mask_latents(
                semantic_mask_config_path=_semantic_mask_cfg,
                video_path=reference_video,
                original_prompt=_original_prompt,
                modified_prompt=prompt,
                latent_length=int(latent_length),
                latent_h=int(latent_h),
                latent_w=int(latent_w),
                num_frames=video_length,
                sample_start_idx=reference_start_idx,
                sample_frame_rate=reference_frame_rate,
                feather_radius=feather_radius,
                cache_dir=getattr(args, "semantic_mask_cache_dir", None),
                save_dir=prompt_vis_dir,
                grounding_phrases=_entry_phrases,
            )
            if cur_mask is not None:
                cur_dual = True
                cur_denoising = max(float(cur_fg), float(cur_bg))
            else:
                _std_logger.warning(
                    "Semantic mask empty for prompt '%s'; fallback to global denoising.",
                    prompt,
                )
                cur_dual = False
                cur_fg = None
                cur_bg = None

            # Release SAM2 / DINO / LLM GPU memory before loading Wan.
            _std_logger.info("[%d/%d] Clearing GPU memory after SAM2 ...", idx + 1, len(prompts))
            _flush_gpu()

        # ----------------------------------------------------------------
        # Step B: Load Wan pipeline and run inference.
        # ----------------------------------------------------------------
        _std_logger.info("[%d/%d] Loading Wan pipeline ...", idx + 1, len(prompts))
        pipe = _build_inference_pipe(args, device)
        if lora_path:
            pipe.load_lora(getattr(pipe, args.lora_base_model), lora_path, alpha=1)
            _load_extra_text_token(pipe, lora_path)

        video = pipe(
            prompt=prompt,
            negative_prompt="",
            source_prompt=source_prompt,
            input_video=input_video,
            denoising_strength=cur_denoising,
            height=height,
            width=width,
            num_frames=video_length,
            cfg_scale=guidance_scale,
            source_cfg_scale=source_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=inference_shift,
            reference_noise_mask_latents=(cur_mask if cur_dual else None),
            reference_fg_init_strength=(cur_fg if cur_dual else None),
            reference_bg_init_strength=(cur_bg if cur_dual else None),
            reference_strict_timestep_align=(
                reference_strict_timestep_align if cur_dual else False
            ),
            seed=seed,
            rand_device=pipe.device,
        )
        filename = f"{idx:02d}_{_sanitize_filename(prompt)}.mp4"
        save_video(video, os.path.join(output_dir, filename), fps=fps, quality=5)

        # Release Wan pipeline GPU memory before the next iteration.
        if idx < len(prompts) - 1:
            _std_logger.info("[%d/%d] Clearing GPU memory after Wan inference ...", idx + 1, len(prompts))
            del pipe
            _flush_gpu()


def _parse_mask_patch_size(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        cleaned = value.replace("x", ",")
        parts = [p for p in cleaned.split(",") if p]
        nums = [int(p.strip()) for p in parts]
        if len(nums) == 1:
            return (nums[0], nums[0])
        if len(nums) == 2:
            return (nums[0], nums[1])
    raise ValueError(f"Invalid mask patch size: {value}")


def _parse_validation_prompts(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
        return [v.strip() for v in value.split("|") if v.strip()]
    return []


def _parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _format_lr_tag(lr_value: float) -> str:
    formatted = format(float(lr_value), ".0e")
    return formatted.replace("-", "m").replace("+", "")


def _maybe_update_data_file_keys(args):
    if not args.dataset_metadata_path or not os.path.isfile(args.dataset_metadata_path):
        return
    if args.data_file_keys != "image,video":
        return
    try:
        import pandas as pd
    except Exception:
        return
    df = pd.read_csv(args.dataset_metadata_path, nrows=1)
    cols = set(df.columns)
    if "video" in cols:
        args.data_file_keys = "video"
    elif "name" in cols:
        args.data_file_keys = "name"
    elif "video_path" in cols:
        args.data_file_keys = "video_path"


def build_dataset(args):
    _maybe_update_data_file_keys(args)
    frame_processor = ImageCropAndResize(
        args.height, args.width, args.max_pixels, 16, 16
    )
    if args.sample_start_idx != 0 or args.sample_frame_rate != 1:
        video_loader = LoadVideoStride(
            num_frames=args.num_frames,
            sample_start_idx=args.sample_start_idx,
            sample_frame_rate=args.sample_frame_rate,
            frame_processor=frame_processor,
        )
    else:
        video_loader = LoadVideo(
            args.num_frames, 4, 1, frame_processor=frame_processor
        )

    main_data_operator = RouteByType(
        operator_map=[
            (
                str,
                ToAbsolutePath(args.dataset_base_path)
                >> RouteByExtensionName(
                    operator_map=[
                        (("jpg", "jpeg", "png", "webp"), LoadImage() >> frame_processor >> ToList()),
                        (("gif",), LoadGIF(
                            args.num_frames, 4, 1, frame_processor=frame_processor
                        )),
                        (
                            ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
                            video_loader,
                        ),
                    ]
                ),
            )
        ]
    )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=main_data_operator,
    )
    if dataset.load_from_cache:
        raise ValueError("Cached dataset (.pth) is not supported in this script.")
    return dataset


def _build_noise_scheduler(args):
    if args.noise_scheduler_type == "flow":
        try:
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=args.num_train_timesteps, shift=args.flow_shift
            )
            scheduler.set_timesteps(args.num_train_timesteps, shift=args.flow_shift)
            return scheduler
        except Exception as exc:
            logger.warning("Falling back to FlowMatchScheduler: %s", exc)
            scheduler = FlowMatchScheduler("Wan")
            scheduler.set_timesteps(
                args.num_train_timesteps, training=True, shift=args.flow_shift
            )
            return scheduler
    return DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )


def create_parser():
    default_cfg = str(Path(__file__).resolve().parents[3] / "train_mllm_ti2v.yaml")
    parser = argparse.ArgumentParser(description="Wan2.1 TTT LoRA training.")
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to a YAML config file.")
    parser = add_general_config(parser)
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
            break
    parser = add_video_size_config(parser)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max training steps (overrides num_epochs).")
    parser.add_argument("--validation_steps", type=int, default=100, help="Validation interval in steps.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Accelerate checkpointing interval.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", help="Mixed precision mode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--scale_lr", default=False, action="store_true", help="Scale learning rate by batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--prediction_type", type=str, default="epsilon")
    parser.add_argument("--noise_scheduler_type", type=str, default="flow", choices=["flow", "ddpm"])
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--target_noise_scale", type=float, default=0.01)
    parser.add_argument("--flow_align_ratio", type=float, default=0.2)
    parser.add_argument("--flow_align_cfg_scale", type=float, default=7.5)
    parser.add_argument("--flow_align_gamma", type=float, default=1.0)
    parser.add_argument("--fl_mask_ratio", type=float, default=0.75)
    parser.add_argument("--bg_mask_ratio", type=float, default=0.75)
    parser.add_argument("--mask_patch_size", type=str, default="2,2")
    parser.add_argument("--latent_scale", type=float, default=1.0)
    parser.add_argument("--disable_flow_mask", default=False, action="store_true")
    parser.add_argument(
        "--disable_contrastive_loss",
        dest="use_contrastive_loss",
        action="store_false",
        help="Disable contrastive loss.",
    )
    parser.set_defaults(use_contrastive_loss=True)
    parser.add_argument("--loss_mae_weight", type=float, default=0.1)
    parser.add_argument("--loss_contrastive_weight", type=float, default=0.1)
    parser.add_argument("--gpt2_model_name", type=str, default="gpt2")
    parser.add_argument("--text_mask_ratio", type=float, default=0.15)
    parser.add_argument("--text_augmentation_max_length", type=int, default=100)
    parser.add_argument("--contrastive_temperature", type=float, default=0.5)
    parser.add_argument(
        "--disable_extra_text_token",
        dest="use_extra_text_token",
        action="store_false",
        help="Disable extra text token.",
    )
    parser.add_argument(
        "--freeze_extra_text_token",
        dest="extra_text_token_trainable",
        action="store_false",
        help="Do not train extra text token.",
    )
    parser.set_defaults(use_extra_text_token=True, extra_text_token_trainable=True)
    parser.add_argument("--extra_text_token_lr", type=float, default=None)
    parser.add_argument("--validation_prompts", type=str, default=None)
    parser.add_argument("--sample_start_idx", type=int, default=0)
    parser.add_argument("--sample_frame_rate", type=int, default=1)
    parser.add_argument(
        "--disable_lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA training.",
    )
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_init_weights", type=str, default="gaussian")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--source_cfg_scale", type=float, default=0.0)
    parser.add_argument("--source_prompt", type=str, default=None)
    parser.add_argument("--inference_seed", type=int, default=33)
    parser.add_argument("--inference_shift", type=float, default=None)
    parser.add_argument("--inference_video_length", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--reference_video", type=str, default=None)
    parser.add_argument("--reference_strength", type=float, default=None)
    parser.add_argument("--reference_init_strength", type=float, default=None)
    parser.add_argument("--reference_fg_init_strength", type=float, default=None)
    parser.add_argument("--reference_bg_init_strength", type=float, default=None)
    parser.add_argument(
        "--reference_strict_timestep_align",
        type=_coerce_bool,
        default=True,
    )
    parser.add_argument("--feather_radius", type=int, default=3,
                        help="Gaussian blur radius for soft mask edges (0=hard).")
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Scale factor for YOLO box size (1.0=original, >1=larger, <1=smaller).")
    parser.add_argument("--reference_start_idx", type=int, default=0)
    parser.add_argument("--reference_frame_rate", type=int, default=1)
    parser.add_argument("--use_reference_noise", default=False, action="store_true")
    parser.add_argument("--dataset_csv", type=str, default=None, help="CSV dataset for per-video training.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Root folder for CSV videos.")
    parser.add_argument(
        "--csv_prompt_tail_columns",
        type=int,
        default=0,
        help=(
            "Use the last N CSV columns as validation prompts (<=0 = auto-detect: "
            "columns after 'edited_prompt', 'masked_prompt', 'original_prompt', or 'prompt')."
        ),
    )
    parser.add_argument(
        "--sweep_train_steps",
        type=str,
        default="40,50,60,70,80",
        help="Comma-separated max_train_steps sweep values for dataset_csv mode.",
    )
    parser.add_argument(
        "--sweep_learning_rates",
        type=str,
        default="3e-5,1e-4",
        help="Comma-separated learning_rate sweep values for dataset_csv mode.",
    )
    parser.add_argument("--run_inference_after_train", default=False, action="store_true")
    parser.add_argument("--inference_only", default=False, action="store_true")
    parser.add_argument(
        "--csv_start_index",
        type=int,
        default=1,
        help="1-based index of the first CSV entry to process (skip entries before this index).",
    )
    parser.add_argument("--inference_output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--semantic_mask_config",
        type=str,
        default=None,
        help="Path to the semantic mask YAML config (LLM+DINO+SAM2). "
             "When set, uses semantic segmentation masks instead of YOLO boxes.",
    )
    parser.add_argument(
        "--semantic_mask_cache_dir",
        type=str,
        default=None,
        help="Directory to cache/load pre-computed semantic masks (.pt). "
             "Avoids re-running the full LLM+DINO+SAM2 pipeline on every run.",
    )
    parser.add_argument(
        "--grounding_phrases",
        type=str,
        default=None,
        help="Comma-separated object names to mask, e.g. \"river,trees\". "
             "When set, the VLM step is skipped and these phrases are fed "
             "directly to Grounding-DINO. Can also be a list in the YAML config.",
    )
    return parser


def _train_one(args, dataset, validation_prompts):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=args.dataset_num_workers,
    )

    model = WanTTTTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_init_weights=args.lora_init_weights,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_lora=args.use_lora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        device=accelerator.device,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        sample_start_idx=args.sample_start_idx,
        sample_frame_rate=args.sample_frame_rate,
        use_flow_mask=not args.disable_flow_mask,
        fl_mask_ratio=args.fl_mask_ratio,
        bg_mask_ratio=args.bg_mask_ratio,
        mask_patch_size=_parse_mask_patch_size(args.mask_patch_size),
        latent_scale=args.latent_scale,
        noise_scheduler_type=args.noise_scheduler_type,
        prediction_type=args.prediction_type,
        target_noise_scale=args.target_noise_scale,
        flow_align_ratio=args.flow_align_ratio,
        flow_align_cfg_scale=args.flow_align_cfg_scale,
        flow_align_gamma=args.flow_align_gamma,
        use_contrastive_loss=args.use_contrastive_loss,
        loss_mae_weight=args.loss_mae_weight,
        loss_contrastive_weight=args.loss_contrastive_weight,
        gpt2_model_name=args.gpt2_model_name,
        text_mask_ratio=args.text_mask_ratio,
        text_augmentation_max_length=args.text_augmentation_max_length,
        contrastive_temperature=args.contrastive_temperature,
        use_extra_text_token=args.use_extra_text_token,
        extra_text_token_trainable=args.extra_text_token_trainable,
    )

    noise_scheduler = _build_noise_scheduler(args)

    # Avoid Hugging Face hub when contrastive loss is off (text_augmentation is a no-op if None).
    if args.use_contrastive_loss:
        gpt2_model = GPT2LMHeadModel.from_pretrained(args.gpt2_model_name)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model_name)
        gpt2_model.eval().requires_grad_(False)
    else:
        gpt2_model = None
        gpt2_tokenizer = None
        logger.info("Skipping GPT-2 load (use_contrastive_loss=false); no Hub access for tokenizer.")

    learning_rate = args.learning_rate
    if args.scale_lr:
        learning_rate = (
            learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer_groups, clip_params = model.get_optimizer_groups(
        learning_rate, extra_text_token_lr=args.extra_text_token_lr
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        max_train_steps = args.num_epochs * num_update_steps_per_epoch
    else:
        max_train_steps = args.max_train_steps
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    global_step = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        model.train()
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = model(
                    batch,
                    noise_scheduler=noise_scheduler,
                    gpt2_model=gpt2_model,
                    gpt2_tokenizer=gpt2_tokenizer,
                    validation_prompts=validation_prompts,
                    weight_dtype=weight_dtype,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                model_logger.on_step_end(
                    accelerator, model, args.save_steps, loss=loss.detach().item()
                )

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info("Saved state to %s", save_path)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        is_last_epoch = (epoch == num_train_epochs - 1) or (global_step >= max_train_steps)
        if args.save_steps is None and is_last_epoch:
            model_logger.on_epoch_end(accelerator, model, epoch)
        if global_step >= max_train_steps:
            break

    model_logger.on_training_end(accelerator, model, args.save_steps)
    return _find_latest_lora(args.output_path)


def _run_from_dataset_csv(args, train_data_cfg, validation_data_cfg):
    entries = _load_dataset_csv_entries(
        args.dataset_csv,
        args.dataset_root,
        prompt_tail_columns=args.csv_prompt_tail_columns,
    )
    step_candidates = _parse_int_list(args.sweep_train_steps)
    lr_candidates = _parse_float_list(args.sweep_learning_rates)
    if not step_candidates:
        step_candidates = [int(args.max_train_steps)] if args.max_train_steps else [50]
    if not lr_candidates:
        lr_candidates = [float(args.learning_rate)]

    csv_start_index = getattr(args, "csv_start_index", 1) or 1
    for run_index, entry in enumerate(entries, start=1):
        if run_index < csv_start_index:
            _std_logger.info("Skipping entry %d (%s) [csv_start_index=%d]", run_index, entry["name"], csv_start_index)
            continue
        name_stem = Path(entry["name"]).stem
        entry_dir = os.path.join(args.output_path, f"{run_index:03d}_{_sanitize_filename(name_stem)}")
        for max_steps in step_candidates:
            for lr in lr_candidates:
                combo_tag = f"steps{int(max_steps)}_lr{_format_lr_tag(lr)}"
                output_dir = os.path.join(entry_dir, combo_tag)
                local_args = copy.deepcopy(args)
                local_args.output_path = output_dir
                local_args.max_train_steps = int(max_steps)
                local_args.learning_rate = float(lr)

                validation_data = copy.deepcopy(validation_data_cfg) if validation_data_cfg else {}
                if entry["validation_prompts"]:
                    validation_data["prompts"] = entry["validation_prompts"]
                elif validation_data.get("prompts") is None:
                    validation_data["prompts"] = []

                if validation_data.get("reference_video") is None:
                    validation_data["reference_video"] = entry["video_path"]
                if validation_data.get("video_length") is None:
                    validation_data["video_length"] = local_args.num_frames
                if validation_data.get("width") is None:
                    validation_data["width"] = local_args.width
                if validation_data.get("height") is None:
                    validation_data["height"] = local_args.height

                # Persist sweep combination and key inputs for each run.
                os.makedirs(output_dir, exist_ok=True)
                run_cfg_path = os.path.join(output_dir, "run_config.json")
                with open(run_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "dataset_csv": str(Path(args.dataset_csv).expanduser().resolve()),
                            "dataset_video": entry["video_path"],
                            "dataset_name": entry["name"],
                            "prompt": entry["prompt"],
                            "masked_prompt": entry["masked_prompt"],
                            "validation_prompts": validation_data.get("prompts", []),
                            "sweep": {
                                "max_train_steps": int(max_steps),
                                "learning_rate": float(lr),
                                "combo_tag": combo_tag,
                            },
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                dataset = SingleVideoDataset(
                    entry["video_path"],
                    entry["prompt"],
                    entry["masked_prompt"],
                    validation_data.get("prompts", []),
                )

                lora_path = None
                if not local_args.inference_only:
                    lora_path = _train_one(local_args, dataset, validation_data.get("prompts", []))
                    # Release training-model GPU memory before inference (which may load SAM2 + Wan).
                    _std_logger.info("Training done; clearing GPU memory before validation inference ...")
                    _flush_gpu()
                if local_args.run_inference_after_train or local_args.inference_only:
                    if lora_path is None:
                        lora_path = local_args.lora_checkpoint or _find_latest_lora(local_args.output_path)
                    _run_inference_for_entry(
                        local_args, validation_data, lora_path, entry["video_path"], entry["prompt"]
                    )


def main(args):
    config = _load_config(args.config) if args.config else {}
    train_data_cfg, validation_data_cfg = _apply_config_to_args(args, config) if config else ({}, {})

    if args.dataset_csv:
        _run_from_dataset_csv(args, train_data_cfg, validation_data_cfg)
        return

    validation_prompts = _parse_validation_prompts(args.validation_prompts)
    dataset = build_dataset(args)
    lora_path = None
    if not args.inference_only:
        lora_path = _train_one(args, dataset, validation_prompts)
        # Release training-model GPU memory before inference (which may load SAM2 + Wan).
        _std_logger.info("Training done; clearing GPU memory before validation inference ...")
        _flush_gpu()
    if args.run_inference_after_train or args.inference_only:
        validation_data = copy.deepcopy(validation_data_cfg) if validation_data_cfg else {}
        if validation_prompts and not validation_data.get("prompts"):
            validation_data["prompts"] = validation_prompts
        entry_video = train_data_cfg.get("video_path") if train_data_cfg else None
        if validation_data.get("reference_video") is None and entry_video:
            validation_data["reference_video"] = entry_video
        if args.use_lora and lora_path is None:
            lora_path = args.lora_checkpoint or _find_latest_lora(args.output_path)
        entry_source_prompt = train_data_cfg.get("prompt") if train_data_cfg else None
        _run_inference_for_entry(
            args, validation_data, lora_path, entry_video, entry_source_prompt
        )


if __name__ == "__main__":
    parser = create_parser()
    # Capture parser defaults so _apply_config_to_args can tell whether a
    # value was explicitly passed on the command line or is just the default.
    _parser_defaults = {a.dest: a.default for a in parser._actions}
    args = parser.parse_args()
    args._parser_defaults = _parser_defaults
    main(args)
