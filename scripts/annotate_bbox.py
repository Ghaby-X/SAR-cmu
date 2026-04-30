"""
Manual bbox annotation tool.
Usage: python scripts/annotate_bbox.py --images images/depth/drone_person_L18.5H0.jpg images/depth/drone_person_L24.5H0.jpg
Saves annotations to annotations.json  {filename: [x1, y1, x2, y2]}
"""
import cv2
import json
import argparse
from pathlib import Path

ANNOT_FILE = Path(__file__).resolve().parents[1] / "annotations.json"

drawing = False
ix, iy = -1, -1
bbox = None


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, bbox, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = param.copy()
        cv2.rectangle(tmp, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotate", tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        cv2.rectangle(param, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow("Annotate", param)


def annotate(image_paths: list[str]):
    annots = {}
    if ANNOT_FILE.exists():
        with open(ANNOT_FILE) as f:
            annots = json.load(f)

    for path in image_paths:
        global bbox, canvas
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read {path}")
            continue

        # scale down if too large to fit screen
        h, w = img.shape[:2]
        scale = min(1.0, 1200 / w, 800 / h)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        canvas = img.copy()
        bbox = None
        fname = Path(path).name

        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", mouse_cb, canvas)
        cv2.putText(canvas, f"{fname} — drag bbox, [s]=save, [q]=skip",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.imshow("Annotate", canvas)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s'):
                if bbox:
                    # scale bbox back to original image coords
                    x1, y1, x2, y2 = bbox
                    if scale < 1.0:
                        x1, y1, x2, y2 = (int(v / scale) for v in (x1, y1, x2, y2))
                    annots[fname] = [x1, y1, x2, y2]
                    print(f"  Saved {fname}: {annots[fname]}")
                else:
                    print("  No bbox drawn — draw one first")
                break
            elif key == ord('q'):
                print(f"  Skipped {fname}")
                break

        cv2.destroyAllWindows()

    with open(ANNOT_FILE, "w") as f:
        json.dump(annots, f, indent=2)
    print(f"\nAnnotations saved to {ANNOT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    args = parser.parse_args()
    annotate(args.images)
