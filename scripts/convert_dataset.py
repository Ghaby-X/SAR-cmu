"""
Convert Detectron2-format annotations to YOLO or COCO format.

Usage:
    python convert_dataset.py --input_dir <path/to/Actor001-ActorXXX>
                               --annotations <path/to/annotations.json>
                               --output_dir <output/path>
                               --format yolo|coco
"""

import argparse
import json
import shutil
from pathlib import Path


def get_actor(file_name):
    return file_name.split("_")[0]

def build_split_map(annotations):
    actors = sorted(set(get_actor(e["file_name"]) for e in annotations))
    n = len(actors)
    n_val  = max(1, round(n * 0.1))
    n_test = max(1, round(n * 0.1))
    val_actors  = set(actors[-(n_val + n_test):-n_test])
    test_actors = set(actors[-n_test:])
    print(f"Actors: {n} total | train: {n - n_val - n_test}, val: {n_val}, test: {n_test}")
    print(f"  Val actors:  {sorted(val_actors)}")
    print(f"  Test actors: {sorted(test_actors)}")
    def get_split(fname):
        a = get_actor(fname)
        if a in val_actors:  return "val"
        if a in test_actors: return "test"
        return "train"
    return get_split

def find_image(input_dir, file_name):
    actor = get_actor(file_name)
    angle = file_name.split("_")[1]
    return Path(input_dir) / actor / f"{actor}_{angle}" / file_name

def convert_yolo(annotations, input_dir, output_dir, get_split):
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    total = len(annotations)
    missing = 0
    for i, entry in enumerate(annotations, 1):
        if i % 500 == 0 or i == total:
            print(f"  [{i}/{total}] {i*100//total}%")

        fname  = entry["file_name"]
        split  = get_split(fname)
        W, H   = entry["width"], entry["height"]
        src    = find_image(input_dir, fname)
        dst    = output_dir / split / "images" / fname

        if not src.exists():
            missing += 1
            continue
        if not dst.exists():
            shutil.copy2(src, dst)

        with open(output_dir / split / "labels" / (Path(fname).stem + ".txt"), "w") as f:
            for ann in entry.get("annotations", []):
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}\n")

    with open(output_dir / "data.yaml", "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n")
        f.write("nc: 1\nnames: ['person']\n")

    print(f"Done. Missing images: {missing}")
    print(f"YOLO dataset written to {output_dir}")

def convert_coco(annotations, input_dir, output_dir, get_split):
    splits = {"train": [], "val": [], "test": []}
    for entry in annotations:
        splits[get_split(entry["file_name"])].append(entry)

    for split, entries in splits.items():
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        images, coco_anns = [], []
        ann_id = 1
        total = len(entries)
        missing = 0
        print(f"\nProcessing {split} ({total} images)...")
        for img_id, entry in enumerate(entries, 1):
            if img_id % 500 == 0 or img_id == total:
                print(f"  [{img_id}/{total}] {img_id*100//total}%")
            fname = entry["file_name"]
            src   = find_image(input_dir, fname)
            dst   = output_dir / split / "images" / fname
            if not src.exists():
                missing += 1
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
            images.append({"id": img_id, "file_name": fname,
                           "width": entry["width"], "height": entry["height"]})
            for ann in entry.get("annotations", []):
                x, y, w, h = ann["bbox"]
                coco_anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                  "bbox": [x, y, w, h], "area": w * h,
                                  "iscrowd": ann.get("iscrowd", 0)})
                ann_id += 1
        with open(output_dir / split / "annotations.json", "w") as f:
            json.dump({"images": images, "annotations": coco_anns,
                       "categories": [{"id": 1, "name": "person"}]}, f)
        print(f"  {split}: {len(images)} images, {len(coco_anns)} annotations. Missing: {missing}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--format",      required=True, choices=["yolo", "coco"])
    args = parser.parse_args()

    print(f"Loading annotations from {args.annotations}...")
    with open(args.annotations) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} entries.")

    # Filter to only entries whose image exists on disk
    annotations = [e for e in annotations if find_image(args.input_dir, e["file_name"]).exists()]
    print(f"After filtering to available images: {len(annotations)} entries.")

    get_split  = build_split_map(annotations)
    output_dir = Path(args.output_dir)

    if args.format == "yolo":
        convert_yolo(annotations, args.input_dir, output_dir, get_split)
    else:
        convert_coco(annotations, args.input_dir, output_dir, get_split)

if __name__ == "__main__":
    main()
