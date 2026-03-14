#!/usr/bin/env python3
"""
Resize images in-place for YOLO or COCO formatted datasets.
YOLO labels (normalized) are unaffected. COCO bbox coordinates are scaled.

Usage:
    python resize.py --data_dir <path> --format yolo|coco --size 1280
"""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_images(image_dir, size):
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    total = len(images)
    orig_w, orig_h = None, None
    skipped = 0
    for i, img_path in enumerate(images, 1):
        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] {i*100//total}%")
        try:
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
                img.resize((size, size), Image.LANCZOS).save(img_path)
        except Exception as e:
            print(f"  WARNING: skipped {img_path.name} ({e})")
            skipped += 1
    return orig_w, orig_h, total, skipped


def scale_coco_annotations(ann_file, orig_w, orig_h, size):
    with open(ann_file) as f:
        coco = json.load(f)
    sx, sy = size / orig_w, size / orig_h
    for img in coco["images"]:
        img["width"] = size
        img["height"] = size
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x * sx, y * sy, w * sx, h * sy]
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
    with open(ann_file, "w") as f:
        json.dump(coco, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--format",   required=True, choices=["yolo", "coco"])
    parser.add_argument("--size",     type=int, default=1280)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name in ("train", "val", "test"))

    total_resized, total_skipped = 0, 0
    for split_dir in splits:
        image_dir = split_dir / "images"
        if not image_dir.exists():
            continue
        print(f"\nResizing {split_dir.name} to {args.size}x{args.size}...")
        orig_w, orig_h, n, skipped = resize_images(image_dir, args.size)
        total_resized += n - skipped
        total_skipped += skipped
        print(f"  Resized: {n - skipped}, Skipped: {skipped}")

        if args.format == "coco" and orig_w:
            ann_file = split_dir / "annotations.json"
            if ann_file.exists():
                print(f"  Scaling COCO annotations ({orig_w}x{orig_h} -> {args.size}x{args.size})...")
                scale_coco_annotations(ann_file, orig_w, orig_h, args.size)

    print(f"\nDone. Total resized: {total_resized}, Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
