from __future__ import annotations

import cv2
import numpy as np


class HomographyCalibrator:
    """Pixel-to-meter homography for speed estimation."""

    def __init__(self, source_points: np.ndarray | None = None, target_points: np.ndarray | None = None):
        self.H = None
        if source_points is not None and target_points is not None:
            self.compute_homography(source_points, target_points)

    def compute_homography(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.H, _ = cv2.findHomography(source, target)
        return self.H

    def project_to_ground(self, point: np.ndarray) -> np.ndarray:
        if self.H is None:
            return point
        pt = np.array([point[0], point[1], 1.0])
        projected = self.H @ pt
        projected /= projected[2]
        return projected[:2]
