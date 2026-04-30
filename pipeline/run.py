"""
SAR Video Processing Pipeline
Usage:
    python -m pipeline.run --input <video> [options]
"""
import argparse
import time
import numpy as np
from pathlib import Path

from .config import PipelineConfig
from .detector import load_detector, detect
from .depth import load_depth_model, generate_depth_map, sample_depth, set_device as set_depth_device
from .smoother import Smoother
from .calibration import load_calibration, get_device_calib, apply_calibration
from .metrics import Timer, MetricsCollector
from .visualiser import draw_detections, draw_hud, depth_overlay
from .io import VideoReader, VideoWriter


def run(cfg: PipelineConfig, video_path: str, show_depth_overlay: bool = False, device: str = None, cpu: bool = False, calib_mode: str = None):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    # --- load models ---
    if cpu:
        set_depth_device("cpu")
    print("Loading detector...")
    detector = load_detector(cfg.yolo_weights)
    print("Loading depth model...")
    depth_model = load_depth_model(cfg.depth_encoder, cfg.checkpoint_dir)
    calib = load_calibration(cfg.calib_path) if cfg.calib_path.exists() else {"scale": 1.0}
    calib_entry = get_device_calib(calib, device, mode=calib_mode)
    print(f"Calibration: device='{device or 'default'}'  mode={calib_entry.get('mode')}")

    smoother = Smoother(method=cfg.smoothing, alpha=cfg.ema_alpha, window=cfg.kalman_window)
    timer = Timer()
    collector = MetricsCollector(timer)

    reader = VideoReader(video_path, frame_skip=cfg.frame_skip)
    writer = VideoWriter(
        cfg.output_dir / f"{stem}_out.mp4",
        fps=reader.fps / cfg.frame_skip,
        width=reader.width,
        height=reader.height,
    ) if cfg.save_video else None

    # --- depth skip: reuse last depth map every N frames ---
    depth_map = None
    depth_frame_counter = 0
    print(f"Processing {video_path}  ({reader.total_frames} frames, frame_skip={cfg.frame_skip}, depth_skip={cfg.depth_skip})")
    wall_start = time.perf_counter()
    prev_time = wall_start

    for frame_idx, frame in reader.frames():
        # --- detection ---
        with timer.measure("detection"):
            dets = detect(detector, frame, conf=cfg.det_conf, classes=cfg.det_classes)

        # --- depth (run every depth_skip frames, reuse otherwise) ---
        if depth_map is None or depth_frame_counter % cfg.depth_skip == 0:
            with timer.measure("depth"):
                depth_map = generate_depth_map(depth_model, frame)
        depth_frame_counter += 1

        # --- per-detection depth sampling + smoothing ---
        with timer.measure("smoothing"):
            for i, d in enumerate(dets):
                raw, std = sample_depth(
                    depth_map, d["bbox"],
                    method=cfg.depth_method,
                    roi_shrink=cfg.roi_shrink,
                    percentile=cfg.percentile,
                )
                smoothed = smoother.update(i, raw)
                d["raw_depth"]      = apply_calibration(raw, calib_entry) + 4.5
                d["smoothed_depth"] = apply_calibration(smoothed, calib_entry) + 4.5
                d["depth_std"]     = std

        # --- metrics ---
        collector.record(frame_idx, dets, {})

        # --- visualise ---
        now = time.perf_counter()
        fps_live = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if show_depth_overlay:
            frame = depth_overlay(frame, depth_map)
        draw_detections(frame, dets)
        depths = [d["smoothed_depth"] for d in dets]
        draw_hud(frame, {
            "fps":       fps_live,
            "det_ms":    timer.last("detection"),
            "depth_ms":  timer.last("depth"),
            "total_ms":  timer.last("detection") + timer.last("depth") + timer.last("smoothing"),
            "n_dets":    len(dets),
            "frame_id":  frame_idx,
            "min_depth": min(depths) if depths else None,
            "avg_depth": float(np.mean(depths)) if depths else None,
            "max_depth": max(depths) if depths else None,
        })

        if writer:
            writer.write(frame)

    reader.release()
    if writer:
        writer.release()

    wall_time = time.perf_counter() - wall_start

    # --- save metrics ---
    if cfg.save_csv:
        collector.save_csv(cfg.output_dir / f"{stem}_metrics.csv")
    if cfg.save_metrics:
        collector.save_json(cfg.output_dir / f"{stem}_metrics.json", wall_time)

    agg = collector.aggregate(wall_time)
    print(f"\n=== Results ===")
    print(f"  Frames processed : {agg['frames_processed']}")
    print(f"  Throughput       : {agg['throughput_fps']:.2f} FPS")
    print(f"  Avg detections   : {agg['avg_detections_per_frame']:.2f}/frame")
    for stage, s in agg["latency"].items():
        print(f"  {stage:12s}     : {s['mean_ms']:.1f}ms ± {s['std_ms']:.1f}ms")
    depth = agg.get("depth", {})
    if depth.get("mean_m") is not None:
        print(f"  Avg depth        : {depth['mean_m']:.2f}m ± {depth['std_m']:.2f}m")
    if "depth_error" in agg:
        print(f"  Depth MAE        : {agg['depth_error']['mae_m']:.3f}m")
        print(f"  Depth RMSE       : {agg['depth_error']['rmse_m']:.3f}m")
    print(f"\nOutputs saved to {cfg.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="SAR Video Pipeline")
    parser.add_argument("--input",   required=True,  help="Input video path")
    parser.add_argument("--weights", default=None,   help="YOLO weights (overrides config)")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--skip",       type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--depth-skip", type=int, default=1, help="Run depth every Nth processed frame (reuse otherwise)")
    parser.add_argument("--method",  default="roi_median",
                        choices=["center", "mean", "median", "roi_median", "roi_percentile"])
    parser.add_argument("--smoothing", default="ema", choices=["none", "ema", "kalman"])
    parser.add_argument("--depth-overlay", action="store_true")
    parser.add_argument("--calib-mode", default=None,
                        choices=["linear", "affine", "inverse", "power", "poly2", "poly3"],
                        help="Override calibration mode stored in calibration.json")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (use when GPU OOM)")
    parser.add_argument("--outdir",  default=None)
    parser.add_argument("--device",  default=None,  help="Camera device name for calibration lookup (e.g. drone, s22Ultra)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        depth_encoder=args.encoder,
        frame_skip=args.skip,
        depth_skip=args.depth_skip,
        depth_method=args.method,
        smoothing=args.smoothing,
    )
    if args.weights:
        cfg.yolo_weights = args.weights
    if args.outdir:
        cfg.output_dir = Path(args.outdir)

    run(cfg, args.input, show_depth_overlay=args.depth_overlay, device=args.device, cpu=args.cpu, calib_mode=args.calib_mode)


if __name__ == "__main__":
    main()
