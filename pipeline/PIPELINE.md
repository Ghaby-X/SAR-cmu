# SAR Video Processing Pipeline

## Overview

This pipeline performs real-time person detection and metric depth estimation on drone video footage for Search-and-Rescue (SAR) applications. It combines a fine-tuned YOLOv8n detector with Depth Anything V2 monocular depth estimation, calibrated to real-world distances using ground-truth images taken at known distances.

---

## Architecture

The pipeline is structured around strict separation of concerns. Each module has a single responsibility and no module imports from another except `run.py`, which orchestrates everything.

```
pipeline/
├── config.py        — All tuneable parameters in one PipelineConfig dataclass
├── detector.py      — YOLOv8 wrapper: detection only
├── depth.py         — Depth Anything V2 wrapper: depth map + ROI sampling
├── smoother.py      — Per-track temporal depth smoothing
├── calibration.py   — Load calibration, fit models, apply to raw depth
├── metrics.py       — Timer context manager + per-frame/aggregate metrics
├── visualiser.py    — Frame annotation: bboxes, depth labels, HUD, depth overlay
├── io.py            — VideoReader / VideoWriter wrappers
└── run.py           — Entry point: wires all modules, CLI
```

### Data flow per frame

```
VideoReader
    │
    ├─► detector.py       → [{class, conf, bbox}]
    ├─► depth.py          → depth_map (H×W float32, relative)
    │       └─► sample_depth(bbox) → (raw_depth, roi_std)
    ├─► calibration.py    → raw_depth × power law → metric depth (metres)
    ├─► smoother.py       → temporally smoothed depth per track
    ├─► metrics.py        → record timings + depth values
    ├─► visualiser.py     → annotated frame
    └─► VideoWriter       → output video
```

---

## Models

### Detection — YOLOv8n (fine-tuned)
- Weights: `weights/yolov8n/best.pt`
- Trained on the SARD dataset (single class: `human`)
- Input: BGR video frame
- Output: bounding boxes, confidence scores, class IDs

### Depth Estimation — Depth Anything V2 (ViT-S)
- Checkpoint: `checkpoints/depth_anything_v2_vits.pth`
- Outputs **relative** depth — values have no unit and are not directly comparable across scenes
- Converted to metric depth via per-device calibration

---

## Depth Sampling

Raw depth is not sampled at a single pixel. Five methods are available (configured via `--method`):

| Method | Description |
|---|---|
| `center` | Single pixel at bbox centre. Fast, noisy. |
| `mean` | Mean over full bbox ROI. |
| `median` | Median over full bbox ROI. |
| `roi_median` | Shrink bbox inward by `roi_shrink` fraction, then median. Reduces background bleed at edges. **Default.** |
| `roi_percentile` | Same shrink, then lower percentile. Biases toward closest surface. |

The ROI standard deviation is also recorded as a depth confidence proxy — high std means mixed depth region (e.g. person overlapping background).

---

## Calibration

Depth Anything V2 outputs relative depth with no absolute scale. Calibration maps raw model output to metric distances (metres).

### Ground-truth collection
Images were taken at known distances with filenames encoding ground truth:
```
drone_person_L3H0.jpg  →  L=3m (horizontal), H=0m (height)  →  gt = √(3²+0²) = 3.0m
```

### Calibration models compared — short range (1–4.5m)

An initial calibration was performed on 10 drone images covering 1–4.5m (5 train / 5 test, stratified by distance):

| Mode | Formula | Train MAE | Test MAE | Test RMSE |
|---|---|---|---|---|
| `linear` | `gt = a × raw` | 1.640m | 1.671m | — |
| `affine` | `gt = a × raw + b` | 0.510m | 0.561m | 0.573m |
| `inverse` | `gt = a / raw + b` | 0.302m | 0.316m | 0.323m |
| `power` | `gt = a × raw^b` | 0.304m | 0.298m | 0.319m |
| `poly2` | `gt = a×raw² + b×raw + c` | 0.292m | 0.234m | 0.276m |
| `poly3` | `gt = a×raw³ + b×raw² + c×raw + d` | 0.298m | 0.237m | 0.265m |

### Extended calibration — full range (1–24.5m)

Three additional images were captured at 11.5m, 18.5m, and 24.5m to extend the calibration range. At 18.5m and 24.5m the person occupies only ~30×90px, making automatic detection unreliable — both images were **manually annotated** using `scripts/annotate_bbox.py`.

#### Split strategy

The initial stratified 75/25 split placed all long-range samples (11.5m, 18.5m, 24.5m) in the test set because each distance had only one sample. This meant the models were never trained on long-range data, causing power/inverse to extrapolate wildly beyond their training range.

The final split was redesigned to ensure long-range coverage in training:
- **Train (8 samples):** one image per distance across the full range (1m, 1.5m, 2m, 3m, 4.5m, 11.5m, 18.5m, 24.5m)
- **Test (5 samples):** duplicate short-range images (1m, 1.5m, 2m, 3m, 4.5m)

#### Results with full-range training

| Mode | Train MAE | Test MAE | Test RMSE | Notes |
|---|---|---|---|---|
| `power` | 3.489m | 0.849m | — | Fits short range well, struggles at extremes |
| `inverse` | 4.106m | 2.999m | — | Collapses — bias term dominates |
| `poly2` | 2.058m | 2.324m | — | Underfits the curve |
| **`poly3`** | **0.788m** | **0.280m** | — | Best — captures full 1–24.5m range |

**`poly3` is the default.** With long-range samples in training, it correctly maps:
- raw ≈ 1.0 → ~7m
- raw ≈ 0.6 → ~11.5m
- raw ≈ 0.04 → ~21m

#### Why power/inverse fail at full range

Power and inverse were fit on 1–4.5m data in the initial calibration. When raw values from 18–24m images (raw≈0.006–0.04) were encountered at inference, the power law produced values in the millions of metres. The fix was two-fold:
1. Include long-range samples in training so the fit covers the full range
2. Clamp raw input to `[raw_min, raw_max]` and output to `[0.1m, 30m]` in `apply_calibration` to prevent blow-up on out-of-distribution inputs

### Manual annotation fallback

For images where automatic detection fails (small/distant objects), bboxes can be annotated manually:

```bash
python scripts/annotate_bbox.py --images images/depth/drone_person_L18.5H0.jpg \
                                          images/depth/drone_person_L24.5H0.jpg
```

Annotations are saved to `annotations.json` and used automatically during calibration when YOLO fails to detect.

### Per-device, per-mode calibration

All modes are stored per camera device in `calibration.json`. The pipeline selects the correct device entry and mode at runtime:

```bash
python -m pipeline.run --input video/ajara.mp4 --device drone                      # uses poly3 (default)
python -m pipeline.run --input video/ajara.mp4 --device drone --calib-mode power
python -m pipeline.run --input video/ajara.mp4 --device drone --calib-mode poly2
python -m pipeline.run --input video/ajara.mp4 --device drone --calib-mode inverse
```

If `--device` is omitted, the first available entry is used as fallback. If `--calib-mode` is omitted, the `default_mode` stored in `calibration.json` is used (`poly3` for drone, trained on 1–24.5m).

---

## Temporal Smoothing

Raw depth estimates are noisy frame-to-frame. Three smoothing modes are available (`--smoothing`):

| Mode | Description |
|---|---|
| `none` | No smoothing — raw calibrated depth |
| `ema` | Exponential moving average: `α × new + (1−α) × prev`. Controlled by `ema_alpha` (default 0.3). **Default.** |
| `kalman` | Weighted rolling buffer with exponentially decaying weights. More lag-resistant than EMA. |

Smoothing operates per track ID in the model's raw (pre-calibration) space. Calibration is applied once after smoothing.

---

## Metrics & Outputs

Every run produces three output files in `--outdir`:

| File | Content |
|---|---|
| `<stem>_out.mp4` | Annotated video with bboxes, depth labels, HUD |
| `<stem>_metrics.csv` | Per-frame: n_detections, det_ms, depth_ms, smooth_ms, total_ms, per-detection conf/depth/std |
| `<stem>_metrics.json` | Aggregated: FPS, per-stage latency (mean/std/min/max), depth stats, MAE/RMSE if GT provided |

### Quantified metrics

**Speed / Latency** (measured with `Timer` context manager on every frame):
- Detection latency (ms)
- Depth inference latency (ms)
- Smoothing latency (ms)
- Total pipeline latency (ms)
- Throughput (FPS)

**Depth quality**:
- ROI depth std-dev (confidence proxy per detection)
- Raw vs smoothed depth delta (temporal stability)

**Accuracy** (when ground-truth depth is available per frame):
- MAE (m), RMSE (m) on calibrated depth

---

## Usage

```bash
# Basic run
python -m pipeline.run --input video/ajara.mp4 --device drone

# Full options
python -m pipeline.run \
    --input video/ajara.mp4 \
    --weights weights/yolov8n/best.pt \
    --encoder vits \               # vits | vitb | vitl
    --method roi_median \          # depth sampling method
    --smoothing ema \              # none | ema | kalman
    --skip 2 \                     # process every 2nd frame
    --device drone \               # calibration device key
    --calib-mode poly3 \           # linear | affine | inverse | power | poly2 | poly3
    --depth-overlay \              # blend depth map over video
    --cpu \                        # force CPU (use if GPU OOM)
    --outdir results/pipeline/drone
```

---

## Limitations & Future Work

- **No tracker**: track IDs are currently detection indices per frame. When detections reorder or disappear, the smoother loses continuity. Adding ByteTrack/SORT would fix this.
- **Calibration sample size**: 10 images across 5 distances. More samples, especially at longer ranges (>5m) and varying heights, would improve accuracy.
- **Single global calibration per device**: the power law fit assumes the depth-to-distance relationship is consistent across all scenes for a given camera. In practice it varies with scene content and lighting.
- **Detection accuracy metrics**: precision/recall/mAP require running on the labelled test set, not raw video. A separate eval script reusing `detector.py` is needed.
