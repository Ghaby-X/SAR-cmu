import sys
import numpy as np
import torch
from pathlib import Path

# Depth Anything V2 lives two levels up from pipeline/
_DA2_PATH = Path(__file__).resolve().parents[2] / "depth_anything_v2"
if str(_DA2_PATH) not in sys.path:
    sys.path.insert(0, str(_DA2_PATH))
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def set_device(device: str):
    global DEVICE
    DEVICE = device

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_depth_model(encoder: str = "vits", checkpoint_dir: Path = None) -> DepthAnythingV2:
    checkpoint_dir = checkpoint_dir or (Path(__file__).resolve().parents[1] / "checkpoints")
    model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
    ckpt = Path(checkpoint_dir) / f"depth_anything_v2_{encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    return model.to(DEVICE).eval()


def generate_depth_map(model: DepthAnythingV2, frame_bgr: np.ndarray) -> np.ndarray:
    """Returns float32 relative depth map (same H×W as input)."""
    return model.infer_image(frame_bgr)


def sample_depth(depth_map: np.ndarray, bbox: tuple, method: str = "roi_median",
                 roi_shrink: float = 0.2, percentile: float = 25.0) -> tuple[float, float]:
    """
    Sample depth within bbox.
    Returns (depth_value, std_dev_in_roi).
    Methods: center | mean | median | roi_median | roi_percentile
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape

    if method == "center":
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        val = float(depth_map[cy, cx])
        return val, 0.0

    if method in ("roi_median", "roi_percentile"):
        dx = int((x2 - x1) * roi_shrink)
        dy = int((y2 - y1) * roi_shrink)
        x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy

    roi = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        cy, cx = depth_map.shape[0] // 2, depth_map.shape[1] // 2
        return float(depth_map[cy, cx]), 0.0

    std = float(np.std(roi))
    if method == "mean":
        return float(roi.mean()), std
    if method == "roi_percentile":
        return float(np.percentile(roi, percentile)), std
    # median / roi_median
    return float(np.median(roi)), std
