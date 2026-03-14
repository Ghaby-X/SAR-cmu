#!/usr/bin/env python3
"""
Verify and print statistics for a YOLO or COCO formatted dataset.
Auto-detects format based on directory structure.

Usage:
    python verify_dataset.py --data_dir <path>
"""

import argparse
import json
from pathlib import Path


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def detect_format(data_dir):
    for split in ("train", "val", "test"):
        if (data_dir / split / "annotations.json").exists():
            return "coco"
        if (data_dir / split / "labels").exists():
            return "yolo"
    return None


def split_image_stats(image_dir):
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    total_size = sum(p.stat().st_size for p in images)
    return len(images), total_size


def verify_yolo(data_dir):
    print(f"Format: YOLO")
    print(f"data.yaml: {'present' if (data_dir / 'data.yaml').exists() else 'MISSING'}\n")

    total_images, total_size = 0, 0
    splits = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name in ("train", "val", "test"))

    for split_dir in splits:
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        n_images, size = split_image_stats(image_dir) if image_dir.exists() else (0, 0)
        n_labels = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        total_images += n_images
        total_size += size

        issues = []
        if not image_dir.exists(): issues.append("missing images/")
        if not label_dir.exists(): issues.append("missing labels/")
        else:
            images_stems = set(p.stem for p in image_dir.glob("*.jpg")) | set(p.stem for p in image_dir.glob("*.png"))
            label_stems  = set(p.stem for p in label_dir.glob("*.txt"))
            if images_stems - label_stems: issues.append(f"{len(images_stems - label_stems)} images without labels")
            if label_stems - images_stems: issues.append(f"{len(label_stems - images_stems)} labels without images")

            bad = sum(
                1 for lbl in list(label_dir.glob("*.txt"))[:200]
                for line in lbl.read_text().splitlines()
                if len(line.split()) != 5 or not all(0 <= float(v) <= 1 for v in line.split()[1:])
            )
            if bad: issues.append(f"{bad} malformed label lines (sampled 200 files)")

        status = "OK" if not issues else "ISSUES"
        print(f"  [{status}] {split_dir.name}: {n_images} images, {n_labels} labels, {human_size(size)}")
        for issue in issues:
            print(f"           - {issue}")

    print(f"\n  Total images : {total_images}")
    print(f"  Total size   : {human_size(total_size)}")


def verify_coco(data_dir):
    print(f"Format: COCO\n")

    total_images, total_size = 0, 0
    splits = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name in ("train", "val", "test"))

    for split_dir in splits:
        image_dir = split_dir / "images"
        ann_file  = split_dir / "annotations.json"
        n_images, size = split_image_stats(image_dir) if image_dir.exists() else (0, 0)
        total_images += n_images
        total_size += size

        issues = []
        n_anns, n_cats = 0, 0
        if not ann_file.exists():
            issues.append("missing annotations.json")
        else:
            with open(ann_file) as f:
                coco = json.load(f)
            n_anns = len(coco.get("annotations", []))
            n_cats = len(coco.get("categories", []))
            n_ann_images = len(coco.get("images", []))
            if n_ann_images != n_images:
                issues.append(f"annotations.json has {n_ann_images} images but images/ has {n_images}")

        status = "OK" if not issues else "ISSUES"
        print(f"  [{status}] {split_dir.name}: {n_images} images, {n_anns} annotations, {n_cats} categories, {human_size(size)}")
        for issue in issues:
            print(f"           - {issue}")

    print(f"\n  Total images : {total_images}")
    print(f"  Total size   : {human_size(total_size)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fmt = detect_format(data_dir)
    if fmt is None:
        print("ERROR: could not detect dataset format (no YOLO labels/ or COCO annotations.json found)")
        return

    if fmt == "yolo":
        verify_yolo(data_dir)
    else:
        verify_coco(data_dir)


if __name__ == "__main__":
    main()
