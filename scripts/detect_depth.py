import sys
import os
import re
import json
import glob
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "depth_anything_v2"))
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
IMAGES_DIR = Path(__file__).resolve().parents[1] / "images" / "depth"
CALIB_OUTPUT = Path(__file__).resolve().parents[1] / "calibration.json"

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #

def parse_file_name(filename: str) -> dict | None:
    """
    Parse '[device]_[item]_L{length}H{height}[_suffix].ext'
    Returns dict with device, item, length_m (distance to object), height_m (camera height).
    Returns None if pattern doesn't match.
    """
    stem = Path(filename).stem
    pattern = r'^(?P<device>[^_]+)_(?P<item>.+?)_L(?P<length>[\d.]+)H(?P<height>[\d.]+)(?:_\w+)?$'
    match = re.match(pattern, stem, re.IGNORECASE)
    if not match:
        return None
    return {
        'device':    match.group('device'),
        'item':      match.group('item'),
        'length_m':  float(match.group('length')),
        'height_m':  float(match.group('height')),
        'path':      str(filename),
    }


def parse_files(directory: str | Path = IMAGES_DIR) -> list[dict]:
    """Return parsed metadata for all images in directory."""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.webp')
    files = []
    for ext in exts:
        files.extend(glob.glob(str(Path(directory) / ext)))
    records = []
    for f in sorted(files):
        meta = parse_file_name(f)
        if meta:
            records.append(meta)
        else:
            print(f"[warn] skipping unrecognised filename: {Path(f).name}")
    return records


# --------------------------------------------------------------------------- #
# Depth model
# --------------------------------------------------------------------------- #

def load_depth_model(encoder: str = 'vits') -> DepthAnythingV2:
    """Load relative Depth Anything V2 (outputs relative depth, calibrated to metres via scale factor)."""
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    ckpt = CHECKPOINT_DIR / f"depth_anything_v2_{encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
    return model.to(DEVICE).eval()


def generate_depth_map(model: DepthAnythingV2, image_bgr: np.ndarray) -> np.ndarray:
    """Run model on a BGR image; returns float32 depth map in metres."""
    return model.infer_image(image_bgr)


# --------------------------------------------------------------------------- #
# Object detection (YOLO)
# --------------------------------------------------------------------------- #

# Synonyms for items that don't match YOLO class names exactly
ITEM_SYNONYMS = {
    'bag':    ['backpack', 'handbag', 'suitcase'],
    'bucket': ['bowl', 'vase'],
    'person': ['person'],
}

def _matching_class_ids(yolo_model: YOLO, item: str) -> set[int]:
    """Return YOLO class IDs whose name matches or is synonymous with item."""
    item_lower = item.lower()
    synonyms = ITEM_SYNONYMS.get(item_lower, [item_lower])
    matched = set()
    for cls_id, cls_name in yolo_model.names.items():
        if any(s in cls_name.lower() or cls_name.lower() in s for s in synonyms):
            matched.add(cls_id)
    return matched


def detect_object(image_bgr: np.ndarray, yolo_model: YOLO, item: str = '', conf: float = 0.25) -> tuple | None:
    """
    Run YOLO on image; return (x1, y1, x2, y2) of best detection.
    If item is given, prefer detections whose class matches the item name.
    Falls back to highest-confidence detection if no class match found.
    """
    results = yolo_model(image_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    xyxy    = boxes.xyxy.cpu().numpy().astype(int)

    if item:
        target_ids = _matching_class_ids(yolo_model, item)
        mask = np.array([c in target_ids for c in cls_ids])
        if mask.any():
            best = int(confs[mask].argmax())
            x1, y1, x2, y2 = xyxy[mask][best]
            matched_cls = yolo_model.names[cls_ids[mask][best]]
            print(f"    detected '{matched_cls}' (matched '{item}')")
            return (x1, y1, x2, y2)
        print(f"    [warn] no '{item}' class match, using highest-confidence detection")

    best = int(confs.argmax())
    x1, y1, x2, y2 = xyxy[best]
    print(f"    detected '{yolo_model.names[cls_ids[best]]}'")
    return (x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# Depth sampling
# --------------------------------------------------------------------------- #

def get_object_depth(depth_map: np.ndarray, bbox: tuple, method: str = 'median') -> float:
    """
    Sample depth within bbox.
    method: 'center' | 'mean' | 'median'
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return float(depth_map[depth_map.shape[0] // 2, depth_map.shape[1] // 2])
    if method == 'center':
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        return float(depth_map[cy, cx])
    if method == 'mean':
        return float(roi.mean())
    return float(np.median(roi))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def _process_records(
    records: list[dict],
    depth_model: DepthAnythingV2,
    yolo_model: YOLO,
    depth_method: str,
) -> list[dict]:
    """Run depth model + YOLO on each record; return list of result dicts."""
    results = []
    for rec in records:
        img = cv2.imread(rec['path'])
        if img is None:
            print(f"[warn] could not read {rec['path']}")
            continue
        depth_map = generate_depth_map(depth_model, img)
        bbox = detect_object(img, yolo_model, item=rec['item'])
        if bbox is None:
            print(f"[warn] no detection in {Path(rec['path']).name}, skipping")
            continue
        pred_depth = get_object_depth(depth_map, bbox, method=depth_method)
        gt_depth = float(np.sqrt(rec['length_m'] ** 2 + rec['height_m'] ** 2))
        results.append({**rec, 'pred_depth': pred_depth, 'gt_depth': gt_depth,
                        'depth_map': depth_map, 'bbox': bbox})
    return results


def run_calibration(
    records: list[dict],
    depth_model: DepthAnythingV2,
    yolo_model: YOLO,
    depth_method: str = 'median',
    test_split: float = 0.25,
    seed: int = 42,
) -> dict:
    """
    Split records into train/test, fit scale on train, evaluate on test.
    Returns calibration dict with per-split metrics.
    """
    import math, random
    random.seed(seed)

    processed = _process_records(records, depth_model, yolo_model, depth_method)
    if len(processed) < 2:
        raise RuntimeError("Need at least 2 valid samples to calibrate.")

    shuffled = processed.copy()
    random.shuffle(shuffled)
    n_test = max(1, math.floor(len(shuffled) * test_split))
    test_set  = shuffled[:n_test]
    train_set = shuffled[n_test:]

    print(f"\nSplit: {len(train_set)} train / {len(test_set)} test")

    # fit scale on train only
    train_pred = np.array([r['pred_depth'] for r in train_set])
    train_gt   = np.array([r['gt_depth']   for r in train_set])
    scale = float(np.dot(train_pred, train_gt) / np.dot(train_pred, train_pred))

    def _metrics(results, label):
        pred = np.array([r['pred_depth'] for r in results])
        gt   = np.array([r['gt_depth']   for r in results])
        res  = gt - scale * pred
        mae  = float(np.mean(np.abs(res)))
        rmse = float(np.sqrt(np.mean(res ** 2)))
        print(f"  [{label}] MAE={mae:.3f}m  RMSE={rmse:.3f}m")
        for r in results:
            print(f"    {Path(r['path']).name}: pred={r['pred_depth']:.3f}m  "
                  f"calibrated={scale*r['pred_depth']:.3f}m  gt={r['gt_depth']:.3f}m")
        return mae, rmse

    train_mae, train_rmse = _metrics(train_set, 'TRAIN')
    test_mae,  test_rmse  = _metrics(test_set,  'TEST')

    calib = {
        'scale': scale,
        'depth_method': depth_method,
        'train': {
            'n': len(train_set), 'mae_m': train_mae, 'rmse_m': train_rmse,
            'samples': [{'file': Path(r['path']).name, 'split': 'train',
                         'predicted_m': r['pred_depth'], 'gt_m': r['gt_depth']} for r in train_set],
        },
        'test': {
            'n': len(test_set), 'mae_m': test_mae, 'rmse_m': test_rmse,
            'samples': [{'file': Path(r['path']).name, 'split': 'test',
                         'predicted_m': r['pred_depth'], 'gt_m': r['gt_depth']} for r in test_set],
        },
    }

    with open(CALIB_OUTPUT, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f"\nCalibration saved to {CALIB_OUTPUT}")
    print(f"  scale={scale:.4f}")

    # tag each processed record with its split for visualisation
    train_paths = {r['path'] for r in train_set}
    for r in processed:
        r['split'] = 'TRAIN' if r['path'] in train_paths else 'TEST'

    calib['_processed'] = processed
    return calib


def apply_calibration(predicted_depth: float, calib: dict) -> float:
    """Convert raw model depth to calibrated real-world depth in metres."""
    return calib['scale'] * predicted_depth


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

def display_pipeline(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    bbox: tuple | None = None,
    pred_depth: float | None = None,
    gt_depth: float | None = None,
    title: str = '',
    split: str = '',
    save_path: str | None = None,
):
    """Show original image + colourised depth map side by side."""
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_colour = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(depth_colour, (x1, y1), (x2, y2), (0, 255, 0), 2)

    split_colour = {'TRAIN': 'steelblue', 'TEST': 'tomato'}.get(split, 'gray')
    split_label  = f'[{split}]  ' if split else ''

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_edgecolor(split_colour)
    fig.patch.set_linewidth(4)

    axes[0].imshow(img_rgb)
    axes[0].set_title(f'{split_label}{title}', color=split_colour, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(depth_colour)
    depth_title = 'Depth map'
    if pred_depth is not None:
        depth_title += f'\npred={pred_depth:.2f}m'
    if gt_depth is not None:
        depth_title += f'  gt={gt_depth:.2f}m'
    axes[1].set_title(depth_title)
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Depth calibration using Depth Anything V2 + YOLO')
    parser.add_argument('--images', type=str, default=str(IMAGES_DIR), help='Path to calibration images')
    parser.add_argument('--encoder', type=str, default='vits', choices=list(MODEL_CONFIGS))
    parser.add_argument('--yolo', type=str, default='yolov8n.pt', help='YOLO weights')
    parser.add_argument('--method', type=str, default='median', choices=['center', 'mean', 'median'])
    parser.add_argument('--visualise', action='store_true', help='Show depth maps for each image')
    parser.add_argument('--outdir', type=str, default=None, help='Save visualisations to this directory')
    args = parser.parse_args()

    print("Loading models...")
    depth_model = load_depth_model(encoder=args.encoder)
    yolo_model = YOLO(args.yolo)

    print(f"Parsing images from {args.images}")
    records = parse_files(args.images)
    if not records:
        print("No valid calibration images found.")
        sys.exit(1)
    print(f"Found {len(records)} calibration images.")

    calib = run_calibration(records, depth_model, yolo_model, depth_method=args.method)

    if args.visualise or args.outdir:
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
        for rec in calib['_processed']:
            img = cv2.imread(rec['path'])
            if img is None:
                continue
            save_path = None
            if args.outdir:
                save_path = str(Path(args.outdir) / (Path(rec['path']).stem + '_depth.png'))
            display_pipeline(
                img, rec['depth_map'],
                bbox=rec['bbox'],
                pred_depth=rec['pred_depth'],
                gt_depth=rec['gt_depth'],
                title=Path(rec['path']).stem,
                split=rec['split'],
                save_path=save_path,
            )
