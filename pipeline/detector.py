import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


def load_detector(weights: str) -> YOLO:
    return YOLO(weights)


def detect(model: YOLO, frame: np.ndarray, conf: float = 0.25, classes: list = None) -> list[dict]:
    """
    Run detection on a BGR frame.
    Returns list of {class_id, class_name, conf, bbox: (x1,y1,x2,y2)}.
    """
    kwargs = {"conf": conf, "verbose": False}
    if classes:
        kwargs["classes"] = classes
    results = model(frame, **kwargs)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return []
    out = []
    for cls_id, conf_val, xyxy in zip(
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy(),
        boxes.xyxy.cpu().numpy().astype(int),
    ):
        out.append({
            "class_id":   int(cls_id),
            "class_name": model.names[int(cls_id)],
            "conf":       float(conf_val),
            "bbox":       tuple(xyxy.tolist()),
        })
    return out
