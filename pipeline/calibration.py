import json
import numpy as np
from pathlib import Path


def load_calibration(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_device_calib(calib: dict, device: str = None, mode: str = None) -> dict:
    if "scale" in calib or "a" in calib:
        entry = calib
    elif device and device in calib:
        entry = calib[device]
    else:
        entry = next(iter(calib.values()))

    resolved_mode = mode or entry.get("default_mode") or entry.get("mode", "power")

    if "fits" in entry:
        fit = entry["fits"][resolved_mode]
        if resolved_mode in ("poly2", "poly3"):
            return {"coeffs": fit["coeffs"], "mode": resolved_mode,
                    "raw_min": fit.get("raw_min", 1e-6), "raw_max": fit.get("raw_max", 1e6),
                    "max_depth_m": fit.get("max_depth_m", 50.0)}
        return {"a": fit["a"], "b": fit["b"], "mode": resolved_mode,
                "raw_min": fit.get("raw_min", 1e-6), "raw_max": fit.get("raw_max", 1e6),
                "max_depth_m": fit.get("max_depth_m", 50.0)}

    return {"a": entry["a"], "b": entry.get("b", 0.0), "mode": resolved_mode}


def apply_calibration(raw_depth: float, calib_entry: dict) -> float:
    """Apply calibration to raw relative depth. Supports linear/affine/inverse/power/poly2/poly3."""
    mode = calib_entry.get("mode", "power")

    # clamp raw to training range to prevent extrapolation blow-up
    raw_min = calib_entry.get("raw_min", 1e-6)
    raw_max = calib_entry.get("raw_max", 1e6)
    raw_depth = float(np.clip(raw_depth, max(raw_min, 1e-6), raw_max))

    if mode in ("poly2", "poly3"):
        result = float(np.polyval(calib_entry["coeffs"], raw_depth))
    elif mode == "power":
        result = calib_entry["a"] * (raw_depth ** calib_entry["b"])
    elif mode == "inverse":
        result = calib_entry["a"] / raw_depth + calib_entry.get("b", 0.0)
    else:  # linear / affine
        result = calib_entry["a"] * raw_depth + calib_entry.get("b", 0.0)

    # clamp output to sane metric range
    return float(np.clip(result, 0.1, calib_entry.get("max_depth_m", 50.0)))


def fit_calibration(pred: list[float], gt: list[float], mode: str = "power") -> dict:
    """
    Fit calibration from paired (pred, gt) depth samples.
    mode:
      'linear'  : gt = scale * raw
      'affine'  : gt = a * raw + b
      'inverse' : gt = a / raw + b
      'power'   : gt = a * raw^b   (best fit for drone monocular depth)
    """
    p = np.array(pred, dtype=float)
    g = np.array(gt,   dtype=float)

    if mode == "linear":
        scale = float(np.dot(p, g) / np.dot(p, p))
        fitted = scale * p
        result = {"a": scale, "b": 0.0}

    elif mode == "affine":
        A = np.column_stack([p, np.ones_like(p)])
        c, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        fitted = c[0] * p + c[1]
        result = {"a": float(c[0]), "b": float(c[1])}

    elif mode == "inverse":
        A = np.column_stack([1.0 / p, np.ones_like(p)])
        c, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        fitted = c[0] / p + c[1]
        result = {"a": float(c[0]), "b": float(c[1])}

    elif mode == "power":
        A = np.column_stack([np.log(p), np.ones_like(p)])
        c, _, _, _ = np.linalg.lstsq(A, np.log(g), rcond=None)
        result = {"a": float(np.exp(c[1])), "b": float(c[0])}
        fitted = result["a"] * np.power(p, result["b"])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    residuals = g - fitted
    result["mae_m"]  = float(np.mean(np.abs(residuals)))
    result["rmse_m"] = float(np.sqrt(np.mean(residuals ** 2)))
    result["mode"]   = mode
    result["n"]      = len(pred)
    return result


def save_calibration(calib: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
