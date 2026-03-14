#!/usr/bin/env python3
"""
Check image sizes in a dataset directory (YOLO or COCO).
Reports unique resolutions found and flags any inconsistencies.

Usage:
    python verify_sizes.py --data_dir <path>
"""

import argparse
from collections import Counter
from pathlib import Path
from PIL import Image


def check_split(image_dir, sample=None):
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if sample:
        images = images[:sample]
    sizes = Counter()
    for p in images:
        with Image.open(p) as img:
            sizes[img.size] += 1
    return sizes, len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sample", type=int, default=None, help="Check only first N images per split")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name in ("train", "val", "test"))

    for split_dir in splits:
        image_dir = split_dir / "images"
        if not image_dir.exists():
            continue
        sizes, checked = check_split(image_dir, args.sample)
        print(f"{split_dir.name} ({checked} images checked):")
        for size, count in sizes.most_common():
            print(f"  {size[0]}x{size[1]}: {count} images")
        if len(sizes) > 1:
            print(f"  WARNING: inconsistent sizes found!")


if __name__ == "__main__":
    main()
