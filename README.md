# FINAL VERSION OF ELASTICTTT will be released here！
# READY SOON




# Preview：

This project performs per-video test-time LoRA training and prompt-based video editing with Wan2.2-TI2V-5B. Semantic editing masks are generated with Qwen3-VL, Grounding-DINO, and SAM2.

## Requirements

- Linux
- Python 3.10 or later
- CUDA GPU with BF16 support
- Conda

## Environment setup

```bash
source /home/liuyueyi/miniforge3/etc/profile.d/conda.sh
conda activate diffsynth

cd /home/liuyueyi/DiffSynth-Studio/wan_ttt_cfg_edit_final

export PYTHONPATH="$PWD"
export SAM2_REPO="$PWD/third_party/sam2_repo"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

To install the dependencies in a new environment:

```bash
pip install -e .
pip install -r requirements-ttt.txt
```

## Dataset

The launch command reads `datasets.csv` and the corresponding video files from:

```text
https://huggingface.co/datasets/liuyueyi-8/ElasticTTT-video-editing-dataset
```

Video paths in `datasets.csv` are resolved relative to this dataset root.

## Configuration

- `train_mllm_ti2v.yaml`: model, training, inference, and LoRA settings
- `configs/semantic_mask_config.yaml`: Qwen3-VL, Grounding-DINO, and SAM2 settings

## Launch

Run the following command from the project directory:

```bash
accelerate launch \
  examples/wanvideo/model_training/train_ttt_cfg_edit.py \
  --config train_mllm_ti2v.yaml \
  --output_path /home/outputs \
  --dataset_csv /home/datasets.csv \
  --dataset_root /home/test_datasets \
  --semantic_mask_config configs/semantic_mask_config.yaml \
  --sweep_train_steps 70 \
  --sweep_learning_rates 3e-5
```

