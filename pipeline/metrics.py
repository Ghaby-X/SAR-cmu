import time
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import json
import csv
from pathlib import Path


class Timer:
    """Accumulates timing samples per named stage."""

    def __init__(self):
        self._samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        yield
        self._samples[stage].append((time.perf_counter() - t0) * 1000)  # ms

    def last(self, stage: str) -> float:
        s = self._samples.get(stage)
        return s[-1] if s else 0.0

    def summary(self) -> dict:
        return {
            stage: {
                "mean_ms": float(np.mean(v)),
                "std_ms":  float(np.std(v)),
                "min_ms":  float(np.min(v)),
                "max_ms":  float(np.max(v)),
                "n":       len(v),
            }
            for stage, v in self._samples.items()
        }


class MetricsCollector:
    """Collects per-frame metrics and computes aggregate statistics."""

    def __init__(self, timer: Timer):
        self.timer = timer
        self._rows: list[dict] = []

    def record(self, frame_idx: int, detections: list[dict], depth_stats: dict):
        """
        detections: list of {class_id, conf, bbox, raw_depth, smoothed_depth, depth_std}
        depth_stats: {gt_depth_m} if ground truth available, else {}
        """
        row = {
            "frame": frame_idx,
            "n_detections": len(detections),
            "det_ms": self.timer.last("detection"),
            "depth_ms": self.timer.last("depth"),
            "smooth_ms": self.timer.last("smoothing"),
            "total_ms": self.timer.last("detection") + self.timer.last("depth") + self.timer.last("smoothing"),
        }
        for i, d in enumerate(detections):
            row[f"det{i}_conf"] = round(d["conf"], 4)
            row[f"det{i}_raw_depth_m"] = round(d["raw_depth"], 4)
            row[f"det{i}_smooth_depth_m"] = round(d["smoothed_depth"], 4)
            row[f"det{i}_depth_std"] = round(d["depth_std"], 4)
            if "gt_depth_m" in depth_stats:
                err = abs(d["smoothed_depth"] - depth_stats["gt_depth_m"])
                row[f"det{i}_abs_err_m"] = round(err, 4)
        self._rows.append(row)

    def aggregate(self, total_wall_time_s: float) -> dict:
        if not self._rows:
            return {}
        n_frames = len(self._rows)
        total_dets = sum(r["n_detections"] for r in self._rows)
        all_depths = [r[k] for r in self._rows for k in r if k.endswith("_smooth_depth_m")]
        all_errors = [r[k] for r in self._rows for k in r if k.endswith("_abs_err_m")]

        agg = {
            "frames_processed": n_frames,
            "total_detections": total_dets,
            "avg_detections_per_frame": round(total_dets / n_frames, 3),
            "throughput_fps": round(n_frames / total_wall_time_s, 2) if total_wall_time_s > 0 else 0,
            "latency": self.timer.summary(),
            "depth": {
                "mean_m": round(float(np.mean(all_depths)), 4) if all_depths else None,
                "std_m":  round(float(np.std(all_depths)), 4) if all_depths else None,
            },
        }
        if all_errors:
            agg["depth_error"] = {
                "mae_m":  round(float(np.mean(all_errors)), 4),
                "rmse_m": round(float(np.sqrt(np.mean(np.array(all_errors) ** 2))), 4),
            }
        return agg

    def save_csv(self, path: Path):
        if not self._rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._rows[0].keys())
        # union of all keys (rows may have different det columns)
        all_keys = sorted({k for r in self._rows for k in r})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._rows)

    def save_json(self, path: Path, total_wall_time_s: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.aggregate(total_wall_time_s), f, indent=2)
