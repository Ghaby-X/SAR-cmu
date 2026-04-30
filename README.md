# SAR Depth Estimation Pipeline

A real-time person detection and metric depth estimation system for Search-and-Rescue (SAR) drone video. Combines a fine-tuned YOLOv8n detector with Depth Anything V2 monocular depth estimation, calibrated to real-world distances using ground-truth images taken at known distances.

## Overview

Given a drone video feed, the pipeline:
1. Detects people using a YOLOv8n model fine-tuned on the SARD dataset
2. Estimates a relative depth map using Depth Anything V2 (ViT-S)
3. Converts relative depth to metric distance (metres) via a per-device calibration curve
4. Applies temporal smoothing to stabilise depth estimates across frames
5. Outputs an annotated video with bounding boxes, depth labels, and a live HUD

## Project Structure

```
pipeline/           — Modular processing pipeline
├── config.py       — All tuneable parameters (PipelineConfig dataclass)
├── detector.py     — YOLOv8 wrapper
├── depth.py        — Depth Anything V2 wrapper
├── smoother.py     — Per-track temporal depth smoothing
├── calibration.py  — Load calibration, fit models, apply to raw depth
├── metrics.py      — Per-frame and aggregate metrics
├── visualiser.py   — Frame annotation (bboxes, depth labels, HUD, overlay)
├── io.py           — VideoReader / VideoWriter wrappers
└── run.py          — Entry point and CLI
scripts/
├── detect_depth.py     — Calibration script: runs detector + depth on ground-truth images
└── annotate_bbox.py    — Manual bbox annotation tool for distant/small targets
notebooks/              — Exploration and baseline comparison notebooks
data/                   — SARD dataset (COCO format)
weights/                — Fine-tuned YOLO weights
checkpoints/            — Depth Anything V2 checkpoints
results/                — Pipeline outputs (videos, CSVs, metrics JSON)
calibration.json        — Per-device calibration curves
```

## Setup

**Dependencies:**
- Python 3.10+
- PyTorch (with CUDA recommended)
- `ultralytics` (YOLOv8)
- `opencv-python`, `numpy`, `scipy`, `matplotlib`
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — clone alongside this repo or install as a package

**Checkpoints:**

Download the Depth Anything V2 ViT-S checkpoint and place it at:
```
checkpoints/depth_anything_v2_vits.pth
```

**Weights:**

Fine-tuned YOLOv8n weights trained on the SARD dataset should be at:
```
weights/yolov8n/best.pt
```

## Usage

```bash
# Basic run
python -m pipeline.run --input video/ajara.mp4 --device drone

# Full options
python -m pipeline.run \
    --input video/ajara.mp4 \
    --weights weights/yolov8n/best.pt \
    --encoder vits \            # vits | vitb | vitl
    --method roi_median \       # depth sampling method
    --smoothing ema \           # none | ema | kalman
    --skip 2 \                  # process every 2nd frame
    --device drone \            # calibration device key
    --calib-mode poly3 \        # linear | affine | inverse | power | poly2 | poly3
    --depth-overlay \           # blend depth map over video
    --cpu \                     # force CPU (use if GPU OOM)
    --outdir results/pipeline/drone
```

### Outputs

Each run saves three files to `--outdir`:

| File | Content |
|---|---|
| `<stem>_out.mp4` | Annotated video with bboxes, depth labels, HUD |
| `<stem>_metrics.csv` | Per-frame: detection count, stage latencies, per-detection conf/depth/std |
| `<stem>_metrics.json` | Aggregated: FPS, per-stage latency stats, depth stats |

## Models

### Detection — YOLOv8n (fine-tuned)
- Trained on the [SARD](https://universe.roboflow.com/sard/search-and-rescue_dataset) dataset (single class: `human`)
- Default confidence threshold: 0.25

### Depth Estimation — Depth Anything V2 (ViT-S)
- Outputs **relative** depth — no absolute unit, not comparable across scenes
- Converted to metric depth via per-device calibration (see below)

## Depth Sampling

Five ROI sampling methods are available via `--method`:

| Method | Description |
|---|---|
| `center` | Single pixel at bbox centre. Fast, noisy. |
| `mean` | Mean over full bbox ROI. |
| `median` | Median over full bbox ROI. |
| `roi_median` | Shrink bbox inward by `roi_shrink` fraction, then median. Reduces background bleed. **Default.** |
| `roi_percentile` | Same shrink, then lower percentile. Biases toward closest surface. |

## Calibration

Depth Anything V2 produces relative depth with no absolute scale. A calibration curve maps raw model output to metric distances using ground-truth images taken at known distances:

```
drone_person_L3H0.jpg  →  L=3m horizontal, H=0m height  →  gt = 3.0m
```

### Calibration results (full range: 1–24.5m)

| Mode | Train MAE | Test MAE | Notes |
|---|---|---|---|
| `power` | 3.489m | 0.849m | Fits short range well, struggles at extremes |
| `inverse` | 4.106m | 2.999m | Collapses at long range |
| `poly2` | 2.058m | 2.324m | Underfits the curve |
| **`poly3`** | **0.788m** | **0.280m** | Best — captures full 1–24.5m range. **Default.** |

Calibration is stored per camera device in `calibration.json`. To add a new device, collect ground-truth images at known distances and run:

```bash
python scripts/detect_depth.py
```

For images where auto-detection fails (small/distant subjects), annotate bboxes manually first:

```bash
python scripts/annotate_bbox.py --images images/depth/drone_person_L18.5H0.jpg
```

## Temporal Smoothing

Three smoothing modes are available via `--smoothing`:

| Mode | Description |
|---|---|
| `none` | No smoothing — raw calibrated depth |
| `ema` | Exponential moving average (`alpha=0.3`). **Default.** |
| `kalman` | Weighted rolling buffer with exponential decay. More lag-resistant than EMA. |

## Limitations

- **No tracker**: track IDs are detection indices per frame. Adding ByteTrack/SORT would improve smoother continuity when detections reorder or disappear.
- **Calibration sample size**: 13 images across 8 distances. More samples at longer ranges and varying heights would improve accuracy.
- **Single global calibration per device**: assumes a consistent depth-to-distance relationship across scenes for a given camera, which varies with lighting and scene content.
