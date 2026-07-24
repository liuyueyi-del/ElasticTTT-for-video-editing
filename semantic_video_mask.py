#!/usr/bin/env python3
"""
Semantic Video Mask Generator
=============================
Given a reference video, its original prompt, and a new (modified) prompt, this
script:
  1. Uses an LLM to identify which visual objects/concepts differ between the
     two prompts and outputs grounding phrases.
  2. Uses Grounding-DINO to detect those objects in selected video key-frames.
  3. Uses SAM-2 (video predictor) to segment and track the detected objects
     across the entire video.

The result is a per-frame binary mask (or soft mask) for every region that
should be modified, saved as a `.pt` or `.npy` file and, optionally, as
overlay visualisation images.

Usage
-----
    python semantic_video_mask.py --config configs/semantic_mask_config.yaml

All tuneable parameters (model names, thresholds, paths, prompts, …) live in
the YAML config file so that nothing needs to be hard-coded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make the local SAM-2 repo importable (kept outside of DiffSynth's deps).
# ---------------------------------------------------------------------------
_SAM2_REPO = os.environ.get("SAM2_REPO", os.path.expanduser("~/sam2"))
if _SAM2_REPO not in sys.path:
    sys.path.insert(0, _SAM2_REPO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return a plain dict."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except ImportError:
        pass
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name.lower().strip(), torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Module 1 – LLM semantic-diff extraction
# ═══════════════════════════════════════════════════════════════════════════

def _build_user_text(original_prompt: str, modified_prompt: str) -> str:
    """Build the text portion of the user message (without the image)."""
    return (
        "Look at the provided image (a frame from the source video).\n\n"
        f"Source Prompt: \"{original_prompt}\"\n"
        f"Target Prompt: \"{modified_prompt}\"\n\n"
        "Identify the objects or regions **visible in this image** that need "
        "to be changed, replaced, or removed in order to match the target "
        "prompt.  Output ONLY the names of those source-video objects as a "
        "comma-separated list of short visual grounding phrases (nouns or "
        "noun phrases) suitable for an object-detection model.\n\n"
        "Example:\n"
        "  Source Prompt: \"A person walking on a street.\"\n"
        "  Target Prompt: \"A person riding a bike on a street.\"\n"
        "  Output: person\n\n"
        "Example:\n"
        "  Source Prompt: \"A jeep is moving on the road.\"\n"
        "  Target Prompt: \"A horse is moving on the road.\"\n"
        "  Output: jeep\n\n"
        "Example:\n"
        "  Source Prompt: \"A cat sitting on a red sofa.\"\n"
        "  Target Prompt: \"A dog sitting on a blue sofa.\"\n"
        "  Output: cat, sofa\n\n"
        f"Source Prompt: \"{original_prompt}\"\n"
        f"Target Prompt: \"{modified_prompt}\"\n"
        "Output:"
    )


def _extract_first_frame(video_path: str) -> Image.Image:
    """Return the first frame of *video_path* as a PIL Image."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        return Image.fromarray(vr[0].asnumpy())
    except ImportError:
        pass
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read the first frame from {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def extract_grounding_phrases_with_vlm(
    original_prompt: str,
    modified_prompt: str,
    video_path: str,
    llm_cfg: Dict[str, Any],
) -> List[str]:
    """
    Use a Vision-Language Model (Qwen3-VL) to identify the regions in the
    source video that need to be edited.

    A frame from the reference video is fed as image input so the VLM can
    *see* what is actually in the scene, producing much more accurate
    grounding phrases than a text-only LLM.

    Returns a deduplicated list of short phrases, e.g. ``["jeep"]``.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_name = llm_cfg.get("model_name", "Qwen/Qwen3-VL-32B-Instruct")
    device = llm_cfg.get("device", "cuda")
    dtype = _resolve_torch_dtype(llm_cfg.get("torch_dtype", "bfloat16"))
    max_new_tokens = int(llm_cfg.get("max_new_tokens", 150))
    do_sample = bool(llm_cfg.get("do_sample", False))
    system_prompt = llm_cfg.get(
        "system_prompt",
        (
            "You are an AI assistant that analyzes text prompts for video "
            "editing. Given a source prompt, a target prompt, and an image "
            "from the source video, identify the objects or regions IN THE "
            "SOURCE VIDEO that need to be changed, replaced, or removed to "
            "match the target prompt. Output ONLY a comma-separated list of "
            "short visual grounding phrases (nouns or noun phrases) that "
            "refer to objects visible in the SOURCE video. Do NOT output "
            "new/target objects. Do NOT output any explanation."
        ),
    )

    logger.info("Loading VLM: %s (dtype=%s, device=%s)", model_name, dtype, device)
    _local_only = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true") \
                  or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in ("1", "true")
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=_local_only)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=dtype, device_map="auto", local_files_only=_local_only,
    )
    model.eval()

    # Extract the first frame from the reference video.
    logger.info("Extracting first frame from %s for VLM input …", video_path)
    first_frame = _extract_first_frame(video_path)

    user_text = _build_user_text(original_prompt, modified_prompt)

    # Build multi-modal message: system (text-only) + user (image + text).
    # Qwen3-VL processor.apply_chat_template requires *all* content fields
    # to be lists of typed dicts, including the system message.
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first_frame},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Use the processor's chat-template path (handles vision tokens).
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    # Decode only the newly generated tokens.
    generated_ids = [
        out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)
    ]
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    logger.info("VLM raw output: %s", generated_text)

    # ---- Post-process: parse comma-separated phrases ----
    if "Output:" in generated_text:
        generated_text = generated_text.split("Output:")[-1].strip()

    # Take only the first line (the VLM sometimes adds explanations after).
    first_line = generated_text.split("\n")[0].strip()
    phrases = [p.strip().strip('"').strip("'") for p in first_line.split(",")]
    phrases = list(dict.fromkeys(p for p in phrases if p))  # deduplicate, keep order

    logger.info("Extracted grounding phrases: %s", phrases)

    # Free GPU memory occupied by the VLM before loading detection models.
    del model, processor
    torch.cuda.empty_cache()

    return phrases


# ═══════════════════════════════════════════════════════════════════════════
# Module 2 – Grounding-DINO object detection
# ═══════════════════════════════════════════════════════════════════════════

def extract_video_frames(video_path: str,
                         frame_indices: Optional[List[int]] = None,
                         ) -> Tuple[List[Image.Image], int]:
    """
    Extract PIL frames from *video_path* at the requested indices.

    Returns ``(frames, total_frame_count)``.
    If *frame_indices* is ``None``, only the first frame is returned.
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if frame_indices is None:
            frame_indices = [0]
        frame_indices = [i for i in frame_indices if 0 <= i < total]
        frames = []
        for idx in frame_indices:
            arr = vr[idx].asnumpy()  # (H, W, 3) uint8
            frames.append(Image.fromarray(arr))
        return frames, total
    except ImportError:
        pass

    # Fallback: OpenCV
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_indices is None:
        frame_indices = [0]
    frames = []
    for idx in sorted(set(frame_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, total


def detect_objects_grounding_dino(
    frames: List[Image.Image],
    frame_indices: List[int],
    grounding_phrases: List[str],
    gdino_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run Grounding-DINO on *frames* for every phrase in *grounding_phrases*.

    Returns a list of detections, each being a dict::

        {
            "frame_idx": int,
            "box": [x1, y1, x2, y2],    # absolute pixel coords
            "label": str,
            "score": float,
        }
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = gdino_cfg.get("model_id", "IDEA-Research/grounding-dino-tiny")
    box_threshold = float(gdino_cfg.get("box_threshold", 0.3))
    text_threshold = float(gdino_cfg.get("text_threshold", 0.25))
    device = gdino_cfg.get("device", "cuda")

    logger.info("Loading Grounding-DINO: %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    # Build a single text query containing all phrases separated by " . "
    # (Grounding-DINO convention for multiple labels).
    text_query = " . ".join(grounding_phrases) + " ."

    all_detections: List[Dict[str, Any]] = []

    for frame, fidx in zip(frames, frame_indices):
        w, h = frame.size
        inputs = processor(images=frame, text=text_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu()       # (N, 4) – xyxy
        scores = results["scores"].cpu()     # (N,)
        labels = results["labels"]           # list[str]

        for box, score, label in zip(boxes, scores, labels):
            det = {
                "frame_idx": fidx,
                "box": box.tolist(),          # [x1, y1, x2, y2]
                "label": label,
                "score": float(score),
            }
            all_detections.append(det)
            logger.info(
                "  Frame %d: detected '%s' (%.2f) at %s",
                fidx, label, score, [round(v, 1) for v in det["box"]],
            )

    # Free Grounding-DINO from GPU before loading SAM2.
    del model, processor
    torch.cuda.empty_cache()

    return all_detections


# ═══════════════════════════════════════════════════════════════════════════
# Module 3 – SAM-2 video segmentation
# ═══════════════════════════════════════════════════════════════════════════

def _load_sam2_predictor(sam2_cfg: Dict[str, Any]):
    """
    Build and return a ``SAM2VideoPredictor`` from the local SAM-2 repo.

    Supports loading from a HuggingFace model_id **or** from an explicit
    (config_file, checkpoint) pair.
    """
    from sam2.build_sam import (
        build_sam2_video_predictor,
        build_sam2_video_predictor_hf,
    )

    device = sam2_cfg.get("device", "cuda")
    config_file = sam2_cfg.get("config_file")
    checkpoint = sam2_cfg.get("checkpoint")

    if config_file and checkpoint:
        logger.info("Loading SAM2 from local checkpoint: %s + %s", config_file, checkpoint)
        predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint,
            device=device,
        )
    else:
        model_id = sam2_cfg.get("model_id", "facebook/sam2.1-hiera-large")
        logger.info("Loading SAM2 from HuggingFace: %s", model_id)
        predictor = build_sam2_video_predictor_hf(model_id, device=device)

    return predictor


def segment_video_with_sam2(
    video_path: str,
    detections: List[Dict[str, Any]],
    sam2_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use SAM-2 to segment and track the detected objects through the video.

    Parameters
    ----------
    video_path : str
        Path to the reference video (MP4).
    detections : list[dict]
        Output of :func:`detect_objects_grounding_dino`.
    sam2_cfg : dict
        ``sam2`` section of the YAML config.  If the key
        ``"sampled_frame_indices"`` is present (a list of absolute frame
        indices injected by the training script), SAM-2 will **only**
        process those frames instead of the full video, matching the
        number of frames used during inference.

    Returns
    -------
    dict with keys:
        ``"masks"``
            ``torch.Tensor`` of shape ``(1, 1, T, H, W)`` (float32, 0/1),
            where T equals ``len(sampled_frame_indices)`` when that key is
            provided, otherwise the total number of frames in the video.
        ``"per_object"``
            ``dict[int, dict]`` mapping *obj_id* → ``{"label": str,
            "masks": np.ndarray (T, H, W)}``.
        ``"video_height"``, ``"video_width"``, ``"num_frames"``
            Video metadata.
    """
    import shutil
    import tempfile

    if not detections:
        raise ValueError("No detections provided – nothing to segment.")

    predictor = _load_sam2_predictor(sam2_cfg)

    # ------------------------------------------------------------------
    # Determine whether to run on a sampled subset of frames or the full
    # video.  When sampled_frame_indices is provided (injected by the
    # training script from the inference num_frames setting), we extract
    # only those frames into a temporary directory so that SAM-2 never
    # loads the portions of the video that will not be used.
    # ------------------------------------------------------------------
    sampled_frame_indices: Optional[List[int]] = sam2_cfg.get("sampled_frame_indices")

    if sampled_frame_indices is not None:
        logger.info(
            "SAM2: using %d sampled frames (inference subset) instead of full video",
            len(sampled_frame_indices),
        )

        # Extract only the required frames from the source video.
        frames, _total = extract_video_frames(video_path, sampled_frame_indices)
        if not frames:
            raise RuntimeError(
                f"Could not extract sampled frames {sampled_frame_indices} "
                f"from {video_path}"
            )

        # Build original-frame-index → local-index mapping.
        orig_to_local: Dict[int, int] = {
            orig: local for local, orig in enumerate(sampled_frame_indices)
        }

        # Write frames into a temp directory named 00000.jpg, 00001.jpg, …
        # SAM-2's init_state accepts a directory of JPEG frames.
        tmpdir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            for local_idx, frame in enumerate(frames):
                frame.convert("RGB").save(
                    os.path.join(tmpdir, f"{local_idx:05d}.jpg"), quality=95
                )
            logger.info("Wrote %d frames to temp dir %s", len(frames), tmpdir)

            # Remap detection frame indices from original video space to the
            # local (0-based) space within the sampled subset.
            remapped_detections: List[Dict[str, Any]] = []
            for det in detections:
                orig_fidx = det["frame_idx"]
                if orig_fidx in orig_to_local:
                    new_det = dict(det)
                    new_det["frame_idx"] = orig_to_local[orig_fidx]
                    remapped_detections.append(new_det)
                else:
                    # Detection landed on a non-sampled frame; remap to the
                    # nearest sampled frame index.
                    nearest = min(sampled_frame_indices,
                                  key=lambda x: abs(x - orig_fidx))
                    new_det = dict(det)
                    new_det["frame_idx"] = orig_to_local[nearest]
                    remapped_detections.append(new_det)
                    logger.warning(
                        "Detection at orig frame %d remapped to nearest "
                        "sampled frame %d (local idx %d)",
                        orig_fidx, nearest, orig_to_local[nearest],
                    )

            effective_detections = remapped_detections
            sam2_source = tmpdir
            num_frames = len(sampled_frame_indices)

            # ---- 1. Initialise SAM-2 on the extracted frames ----
            inference_state = predictor.init_state(video_path=sam2_source)
            video_h = inference_state["video_height"]
            video_w = inference_state["video_width"]
            logger.info(
                "SAM2 video state: %d sampled frames, %dx%d",
                num_frames, video_w, video_h,
            )

            # ---- 2. Feed bounding-box prompts ----
            obj_id_to_label: Dict[int, str] = {}
            for obj_id, det in enumerate(effective_detections):
                box = det["box"]
                frame_idx = det["frame_idx"]
                label = det["label"]
                obj_id_to_label[obj_id] = label
                logger.info(
                    "  Adding SAM2 box prompt obj_id=%d label='%s' on local frame %d: %s",
                    obj_id, label, frame_idx,
                    [round(v, 1) for v in box],
                )
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box,
                    normalize_coords=True,
                )

            # ---- 3. Propagate masks through the sampled frames only ----
            logger.info("Propagating masks through %d sampled frames …", num_frames)
            per_object_frames: Dict[int, Dict[int, np.ndarray]] = {
                oid: {} for oid in obj_id_to_label
            }
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
                for i, oid in enumerate(obj_ids):
                    mask_np = (masks[i, 0].cpu().numpy() > 0.0).astype(np.uint8)
                    if oid in per_object_frames:
                        per_object_frames[oid][frame_idx] = mask_np

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    else:
        # ------------------------------------------------------------------
        # Original path: run SAM-2 on the full video.
        # ------------------------------------------------------------------
        # ---- 1. Initialise inference state from the video file ----
        inference_state = predictor.init_state(video_path=video_path)
        num_frames = inference_state["num_frames"]
        video_h = inference_state["video_height"]
        video_w = inference_state["video_width"]
        logger.info(
            "SAM2 video state: %d frames, %dx%d", num_frames, video_w, video_h
        )

        # ---- 2. Feed bounding-box prompts from Grounding-DINO ----
        obj_id_to_label = {}
        for obj_id, det in enumerate(detections):
            box = det["box"]
            frame_idx = det["frame_idx"]
            label = det["label"]
            obj_id_to_label[obj_id] = label
            logger.info(
                "  Adding SAM2 box prompt obj_id=%d label='%s' on frame %d: %s",
                obj_id, label, frame_idx,
                [round(v, 1) for v in box],
            )
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
                normalize_coords=True,
            )

        # ---- 3. Propagate masks through the entire video ----
        logger.info("Propagating masks through %d frames …", num_frames)
        per_object_frames = {oid: {} for oid in obj_id_to_label}
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
            for i, oid in enumerate(obj_ids):
                mask_np = (masks[i, 0].cpu().numpy() > 0.0).astype(np.uint8)
                if oid in per_object_frames:
                    per_object_frames[oid][frame_idx] = mask_np

    # ---- 4. Assemble per-object mask arrays (T, H, W) ----
    per_object_result: Dict[int, Dict[str, Any]] = {}
    for oid, frame_dict in per_object_frames.items():
        arr = np.zeros((num_frames, video_h, video_w), dtype=np.uint8)
        for fidx, m in frame_dict.items():
            if m.shape == (video_h, video_w):
                arr[fidx] = m
            else:
                # Resize if SAM2 returned a different resolution.
                from PIL import Image as _PILImage
                m_resized = np.array(
                    _PILImage.fromarray(m).resize((video_w, video_h),
                                                   _PILImage.NEAREST)
                )
                arr[fidx] = m_resized
        per_object_result[oid] = {
            "label": obj_id_to_label.get(oid, "unknown"),
            "masks": arr,
        }

    # ---- 5. Merge into a single composite mask (1, 1, T, H, W) ----
    composite = np.zeros((num_frames, video_h, video_w), dtype=np.float32)
    for oid, info in per_object_result.items():
        composite = np.maximum(composite, info["masks"].astype(np.float32))
    # Shape: (1, 1, T, H, W) – compatible with _build_reference_noise_mask_latents
    composite_tensor = torch.from_numpy(composite).unsqueeze(0).unsqueeze(0)

    # Free SAM2 from GPU.
    del predictor
    torch.cuda.empty_cache()

    return {
        "masks": composite_tensor,
        "per_object": per_object_result,
        "video_height": video_h,
        "video_width": video_w,
        "num_frames": num_frames,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output / saving helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_results(
    result: Dict[str, Any],
    output_dir: str,
    save_masks: bool = True,
    save_visualization: bool = True,
    mask_format: str = "torch",
    video_path: Optional[str] = None,
) -> None:
    """Persist mask tensors and optional overlay visualisations."""
    os.makedirs(output_dir, exist_ok=True)

    composite: torch.Tensor = result["masks"]      # (1,1,T,H,W)
    per_object = result["per_object"]
    num_frames = result["num_frames"]

    # ---- Save composite mask ----
    if save_masks:
        if mask_format == "numpy":
            np.save(os.path.join(output_dir, "composite_mask.npy"),
                    composite.numpy())
            logger.info("Saved composite_mask.npy")
        else:
            torch.save(composite, os.path.join(output_dir, "composite_mask.pt"))
            logger.info("Saved composite_mask.pt")

        # Also save per-object masks.
        for oid, info in per_object.items():
            label_safe = info["label"].replace(" ", "_").replace("/", "_")
            fname = f"obj{oid}_{label_safe}"
            if mask_format == "numpy":
                np.save(os.path.join(output_dir, f"{fname}.npy"), info["masks"])
            else:
                torch.save(
                    torch.from_numpy(info["masks"].astype(np.float32)),
                    os.path.join(output_dir, f"{fname}.pt"),
                )
            logger.info("Saved %s mask for obj %d ('%s')", mask_format, oid, info["label"])

    # ---- Save per-object metadata ----
    meta = {
        "num_frames": num_frames,
        "video_height": result["video_height"],
        "video_width": result["video_width"],
        "objects": {
            str(oid): info["label"] for oid, info in per_object.items()
        },
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Saved metadata.json")

    # ---- Save overlay visualisations ----
    if save_visualization and video_path:
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)

        frames, _ = extract_video_frames(
            video_path,
            frame_indices=list(range(num_frames)),
        )

        # A simple colour palette for different objects.
        palette = [
            (255, 0, 0, 100),
            (0, 255, 0, 100),
            (0, 0, 255, 100),
            (255, 255, 0, 100),
            (255, 0, 255, 100),
            (0, 255, 255, 100),
        ]

        for fidx, frame in enumerate(frames):
            overlay = frame.convert("RGBA")
            for oid, info in per_object.items():
                mask_arr = info["masks"][fidx]  # (H, W) uint8
                if mask_arr.max() == 0:
                    continue
                colour = palette[oid % len(palette)]
                colour_layer = Image.new("RGBA", frame.size, colour)
                mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                overlay = Image.composite(colour_layer, overlay, mask_img)

            overlay.convert("RGB").save(
                os.path.join(vis_dir, f"{fidx:05d}.png")
            )

        logger.info("Saved %d overlay frames to %s/", len(frames), vis_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry-point
# ═══════════════════════════════════════════════════════════════════════════

def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full pipeline and return the result dict.

    If ``cfg["grounding_phrases"]`` is set (a list of strings **or** a
    comma-separated string such as ``"jeep, road"``), Step 1 (VLM
    semantic-diff extraction) is skipped entirely and those phrases are
    used directly as the Grounding-DINO input.  This is useful when you
    already know which objects should be masked and want to avoid loading
    the large VLM model.
    """
    video_path = cfg["video_path"]
    original_prompt = cfg["original_prompt"]
    modified_prompt = cfg["modified_prompt"]
    output_dir = cfg.get("output_dir", "./outputs/semantic_masks")

    llm_cfg = cfg.get("llm", {})
    gdino_cfg = cfg.get("grounding_dino", {})
    sam2_cfg = cfg.get("sam2", {})

    # ------------------------------------------------------------------
    # Step 1: Grounding-phrase resolution
    #   Option A – explicit phrases supplied by the caller: skip the VLM.
    #   Option B – no phrases supplied: run the VLM to infer them.
    # ------------------------------------------------------------------
    _explicit = cfg.get("grounding_phrases")
    if _explicit:
        # Normalise: accept either a list or a comma-separated string.
        if isinstance(_explicit, str):
            grounding_phrases = [p.strip() for p in _explicit.split(",") if p.strip()]
        else:
            grounding_phrases = [str(p).strip() for p in _explicit if str(p).strip()]

    if _explicit and grounding_phrases:
        logger.info("=" * 60)
        logger.info("Step 1: Using explicit grounding phrases (VLM skipped)")
        logger.info("=" * 60)
        logger.info("Explicit grounding phrases: %s", grounding_phrases)
    else:
        # ------------------------------------------------------------------
        # Step 1: LLM semantic-diff extraction (original path)
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Step 1: LLM semantic-diff extraction")
        logger.info("=" * 60)
        grounding_phrases = extract_grounding_phrases_with_vlm(
            original_prompt, modified_prompt, video_path, llm_cfg
        )
        if not grounding_phrases:
            raise RuntimeError(
                "LLM returned no grounding phrases. Check prompts or lower the "
                "constraints in the system prompt."
            )

    # ------------------------------------------------------------------
    # Step 2: Grounding-DINO detection
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Grounding-DINO object detection")
    logger.info("=" * 60)
    detection_indices = gdino_cfg.get("detection_frame_indices", [0])
    frames, total_frames = extract_video_frames(video_path, detection_indices)
    logger.info(
        "Extracted %d key-frame(s) from video (%d total frames)",
        len(frames), total_frames,
    )

    detections = detect_objects_grounding_dino(
        frames, detection_indices, grounding_phrases, gdino_cfg
    )
    if not detections:
        raise RuntimeError(
            "Grounding-DINO found no objects for the phrases: "
            f"{grounding_phrases}. Try lowering box_threshold / text_threshold."
        )

    # ------------------------------------------------------------------
    # Step 3: SAM-2 video segmentation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: SAM-2 video segmentation & tracking")
    logger.info("=" * 60)
    result = segment_video_with_sam2(video_path, detections, sam2_cfg)

    logger.info(
        "Composite mask shape: %s  (non-zero ratio: %.4f)",
        list(result["masks"].shape),
        result["masks"].sum().item()
        / max(result["masks"].numel(), 1),
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Saving results to %s", output_dir)
    logger.info("=" * 60)
    save_results(
        result,
        output_dir=output_dir,
        save_masks=cfg.get("save_masks", True),
        save_visualization=cfg.get("save_visualization", True),
        mask_format=cfg.get("mask_format", "torch"),
        video_path=video_path,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Video Mask Generator – LLM + Grounding-DINO + SAM2"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/semantic_mask_config.yaml",
        help="Path to the YAML configuration file.",
    )
    # Allow quick overrides from the command line.
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--original_prompt", type=str, default=None)
    parser.add_argument("--modified_prompt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--grounding_phrases", type=str, default=None,
        help=(
            "Comma-separated list of object names to mask, e.g. "
            "\"jeep,road\".  When provided the VLM (Step 1) is skipped "
            "and these phrases are fed directly to Grounding-DINO."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides take precedence.
    if args.video_path:
        cfg["video_path"] = args.video_path
    if args.original_prompt:
        cfg["original_prompt"] = args.original_prompt
    if args.modified_prompt:
        cfg["modified_prompt"] = args.modified_prompt
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.grounding_phrases:
        cfg["grounding_phrases"] = args.grounding_phrases

    run(cfg)


if __name__ == "__main__":
    main()
