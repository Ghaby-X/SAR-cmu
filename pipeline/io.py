import cv2
import numpy as np
from pathlib import Path
from typing import Iterator


class VideoReader:
    def __init__(self, path: str, frame_skip: int = 1):
        self.path = path
        self.frame_skip = frame_skip
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx % self.frame_skip == 0:
                yield idx, frame
            idx += 1

    def release(self):
        self._cap.release()


class VideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        self._writer.release()
