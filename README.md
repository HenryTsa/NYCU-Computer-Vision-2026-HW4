# NYCU Deep Vision 2026 HW4

- **Student ID**: Your Student ID
- **Name**: Your Name

---

## Introduction

This project implements an image restoration model based on **PromptIR** to remove rain and snow degradation from images. A single unified model is trained to handle both degradation types simultaneously.

Key contributions:
- PromptIR backbone with channel-wise self-attention (O(C²), resolution-free)
- Multi-term combined loss: Charbonnier + SSIM + Edge (Sobel) + Frequency (FFT)
- Exponential Moving Average (EMA) of model weights for stable inference
- 4-fold Test-Time Augmentation (TTA) using shape-preserving flips
- Rain sample loss weighting (1.5×) to handle structured rain artifacts
- Best validation PSNR: **29.45 dB**

---

## Environment Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow matplotlib
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (recommended)

---

## Usage

### Dataset Structure

Place the dataset under `./hw4_realse_dataset/`:

```
hw4_realse_dataset/
├── train/
│   ├── degraded/    # rain-1.png, snow-1.png, ...
│   └── clean/       # rain_clean-1.png, snow_clean-1.png, ...
└── test/
    └── degraded/    # rain-1.png, snow-1.png, ... (no ground truth)
```

### Training

```bash
python solution.py --mode train \
    --data_root ./hw4_realse_dataset \
    --gpu 0 \
    --epochs 200 \
    --batch_size 4 \
    --patch_size 128 \
    --accum_steps 4
```

### Inference

```bash
python solution.py --mode infer \
    --data_root ./hw4_realse_dataset \
    --ckpt ./checkpoints_v6/best.pth \
    --gpu 0 \
    --output_dir ./output_v6
```

### Train then Infer (All-in-one)

```bash
python solution.py --mode all \
    --data_root ./hw4_realse_dataset \
    --gpu 0
```

### Resume from Checkpoint

```bash
python solution.py --mode all \
    --data_root ./hw4_realse_dataset \
    --gpu 0 \
    --resume
```

**Output:** `./output_v6/pred.npz` — keys are filenames (e.g. `rain-1.png`), values are `(3, H, W)` uint8 numpy arrays.

---

## Performance Snapshot

<!-- Insert a screenshot of the leaderboard here -->

| Model | Val PSNR |
|---|---|
| Baseline (dim=48, Charbonnier loss) | 29.40 dB |
| Main Model (dim=64, Combined loss + EMA) | 29.45 dB |
