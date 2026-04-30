from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PipelineConfig:
    # --- paths ---
    yolo_weights: str = str(ROOT / "weights/yolov8n/best.pt")
    depth_encoder: str = "vits"                          # vits | vitb | vitl
    checkpoint_dir: Path = ROOT / "checkpoints"
    calib_path: Path = ROOT / "calibration.json"
    output_dir: Path = ROOT / "results/pipeline"

    # --- detection ---
    det_conf: float = 0.25
    det_classes: list = field(default_factory=lambda: [0])  # 0 = human

    # --- depth sampling ---
    depth_method: str = "roi_median"   # center | mean | median | roi_median | roi_percentile
    roi_shrink: float = 0.2            # shrink bbox by this fraction on each side
    percentile: float = 25.0           # used when method=roi_percentile

    # --- temporal smoothing ---
    smoothing: str = "ema"             # none | ema | kalman
    ema_alpha: float = 0.3
    kalman_window: int = 5

    # --- video ---
    frame_skip: int = 1                # process every Nth frame
    depth_skip: int = 1                # run depth every Nth processed frame (reuse otherwise)
    save_video: bool = True
    save_csv: bool = True
    save_metrics: bool = True
