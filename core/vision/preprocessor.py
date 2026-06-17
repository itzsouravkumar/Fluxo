from __future__ import annotations

import cv2
import numpy as np


class FramePreprocessor:
    """CLAHE preprocessing for night/rain modes + resize."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8), target_size: tuple[int, int] = (640, 640)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def resize(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

    def preprocess(self, frame: np.ndarray, night_mode: bool = False) -> np.ndarray:
        if night_mode:
            frame = self.apply_clahe(frame)
        frame = self.resize(frame)
        return frame
