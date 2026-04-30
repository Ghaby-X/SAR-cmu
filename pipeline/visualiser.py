import cv2
import numpy as np
import matplotlib
_CMAP_DEPTH = matplotlib.colormaps.get_cmap("RdYlGn_r")   # red=close, green=far
_CMAP_OVERLAY = matplotlib.colormaps.get_cmap("Spectral_r")


def _depth_to_bgr(depth: float, min_d: float, max_d: float) -> tuple:
    """Map depth to BGR colour using RdYlGn_r (red=close, green=far)."""
    norm = np.clip((depth - min_d) / max(max_d - min_d, 1e-6), 0, 1)
    r, g, b, _ = _CMAP_DEPTH(norm)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_detections(frame: np.ndarray, detections: list[dict], calib_scale: float = 1.0) -> np.ndarray:
    """
    Draw depth-coloured bboxes with corner accents, filled label background,
    and per-detection confidence + depth labels.
    Colour = depth (red=close, green=far). Thickness = confidence.
    """
    if not detections:
        return frame

    depths = [d["smoothed_depth"] for d in detections]
    min_d, max_d = min(depths), max(depths)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        depth_m = d["smoothed_depth"]
        conf    = d["conf"]
        colour  = _depth_to_bgr(depth_m, min_d, max_d)
        thick   = 1 + int(conf * 1)          # 1–2px based on confidence

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)

        # corner accents
        cl = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px + dx*cl, py), colour, thick + 1)
            cv2.line(frame, (px, py), (px, py + dy*cl), colour, thick + 1)

        # label background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_top  = d["class_name"].upper()
        label_mid  = f"{depth_m:.2f}m"
        (lw1, lh1), _ = cv2.getTextSize(label_top, font, 0.55, 2)
        (lw2, lh2), _ = cv2.getTextSize(label_mid, font, 0.65, 2)
        box_w = max(lw1, lw2) + 10
        box_h = lh1 + lh2 + 14
        cv2.rectangle(frame, (x1, y1 - box_h), (x1 + box_w, y1), colour, -1)
        cv2.putText(frame, label_top, (x1 + 5, y1 - lh2 - 8),
                    font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, label_mid, (x1 + 5, y1 - 4),
                    font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # confidence below box
        label_bot = f"conf:{conf:.0%}  std:{d.get('depth_std', 0):.2f}"
        cv2.putText(frame, label_bot, (x1 + 2, y2 + 14),
                    font, 0.35, colour, 1, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray, metrics: dict) -> np.ndarray:
    """Semi-transparent HUD bar at top + depth scale bar on right."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # semi-transparent dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    fps      = metrics.get("fps", 0)
    det_ms   = metrics.get("det_ms", 0)
    depth_ms = metrics.get("depth_ms", 0)
    total_ms = metrics.get("total_ms", 0)
    n_dets   = metrics.get("n_dets", 0)
    frame_id = metrics.get("frame_id", 0)
    min_d    = metrics.get("min_depth")
    avg_d    = metrics.get("avg_depth")
    max_d    = metrics.get("max_depth")

    cv2.putText(frame, f"FRAME: {frame_id:05d}  |  FPS: {fps:.1f}  |  det:{det_ms:.0f}ms  depth:{depth_ms:.0f}ms  total:{total_ms:.0f}ms",
                (12, 22), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"DETECTIONS: {n_dets}",
                (12, 46), font, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

    if min_d is not None:
        cv2.putText(frame, f"DEPTH: min={min_d:.2f}m  avg={avg_d:.2f}m  max={max_d:.2f}m",
                    (12, 68), font, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
        _draw_depth_scale(frame, min_d, max_d)

    return frame


def _draw_depth_scale(frame: np.ndarray, min_d: float, max_d: float):
    h, w = frame.shape[:2]
    bar_w, bar_h = 20, 140
    x0, y0 = w - bar_w - 15, 90
    for i in range(bar_h):
        colour = _depth_to_bgr(min_d + (i / bar_h) * (max_d - min_d), min_d, max_d)
        cv2.line(frame, (x0, y0 + i), (x0 + bar_w, y0 + i), colour, 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{min_d:.1f}m", (x0 - 2, y0 - 4),   font, 0.38, (255,255,255), 1)
    cv2.putText(frame, f"{max_d:.1f}m", (x0 - 2, y0 + bar_h + 12), font, 0.38, (255,255,255), 1)


def depth_overlay(frame: np.ndarray, depth_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    d_norm  = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    d_colour = (_CMAP_OVERLAY(d_norm)[:, :, :3] * 255).astype(np.uint8)
    d_bgr   = cv2.cvtColor(d_colour, cv2.COLOR_RGB2BGR)
    d_bgr   = cv2.resize(d_bgr, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, d_bgr, alpha, 0)
