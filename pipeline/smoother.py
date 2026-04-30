from collections import defaultdict, deque
import numpy as np


class Smoother:
    """Per-track temporal depth smoother."""

    def __init__(self, method: str = "ema", alpha: float = 0.3, window: int = 5):
        assert method in ("none", "ema", "kalman")
        self.method = method
        self.alpha = alpha
        self.window = window
        self._ema: dict[int, float] = {}
        self._buf: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, depth: float) -> float:
        if self.method == "none":
            return depth

        if self.method == "ema":
            prev = self._ema.get(track_id, depth)
            smoothed = self.alpha * depth + (1 - self.alpha) * prev
            self._ema[track_id] = smoothed
            return smoothed

        # kalman (lightweight scalar version)
        buf = self._buf[track_id]
        buf.append(depth)
        if len(buf) < 2:
            return depth
        arr = np.array(buf)
        # process noise / measurement noise ratio drives how much we trust new obs
        weights = np.exp(np.linspace(-1, 0, len(arr)))
        weights /= weights.sum()
        return float(np.dot(weights, arr))

    def reset(self, track_id: int = None):
        if track_id is None:
            self._ema.clear()
            self._buf.clear()
        else:
            self._ema.pop(track_id, None)
            self._buf.pop(track_id, None)
