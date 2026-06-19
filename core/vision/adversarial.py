from __future__ import annotations

import cv2
import numpy as np


class AdversarialDefense:
    """Defends against physical adversarial patch attacks on YOLO detectors.

    Implements:
    - Anomaly detection on detection regions via texture inconsistency
    - Multi-frame track persistence to survive patch-induced misses
    - Adversarial training augmentation (synthetic patch overlay)

    Reference: arXiv:2410.19863 — "Breaking the Illusion" (2024)
    Reference: Ilina et al. (2025) — anomaly localisation for patch defense
    """

    def __init__(self, anomaly_threshold: float = 0.35):
        self.anomaly_threshold = anomaly_threshold
        self._track_detection_counts: dict[int, list[bool]] = {}

    def detect_adversarial_patch(self, vehicle_crop: np.ndarray) -> bool:
        if vehicle_crop.size == 0:
            return False
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY) if len(vehicle_crop.shape) == 3 else vehicle_crop
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / max(edges.size, 1)

        std_local = self._local_std_dev(gray)
        texture_anomaly = std_local < 0.05

        bright_mask = gray > 240
        bright_ratio = np.sum(bright_mask) / max(gray.size, 1)
        saturation_anomaly = bright_ratio > 0.6

        return texture_anomaly or (edge_density < 0.02 and saturation_anomaly)

    def maintain_track_persistence(
        self,
        track_id: int,
        was_detected_this_frame: bool,
        max_missed_frames: int = 3,
    ) -> bool:
        if track_id not in self._track_detection_counts:
            self._track_detection_counts[track_id] = []
        history = self._track_detection_counts[track_id]
        history.append(was_detected_this_frame)
        if len(history) > max_missed_frames + 1:
            history.pop(0)
        if len(history) < 2:
            return True
        recent = history[-(max_missed_frames + 1):]
        return sum(recent) >= 2

    @staticmethod
    def generate_adversarial_augmentations(
        images: list[np.ndarray],
        patch_sizes: list[int] | None = None,
    ) -> list[np.ndarray]:
        if patch_sizes is None:
            patch_sizes = [32, 48, 64]
        augmented = []
        for img in images:
            for size in patch_sizes:
                aug = img.copy()
                h, w = aug.shape[:2]
                px = np.random.randint(0, max(w - size, 1))
                py = np.random.randint(0, max(h - size, 1))
                pattern = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                pattern = cv2.GaussianBlur(pattern, (5, 5), 0)
                aug[py:py + size, px:px + size] = pattern
                augmented.append(aug)
        return augmented

    def _local_std_dev(self, gray: np.ndarray, block_size: int = 32) -> float:
        h, w = gray.shape
        if h < block_size or w < block_size:
            return float(np.std(gray))
        stds = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                stds.append(float(np.std(block)))
        return float(np.mean(stds)) if stds else 0.0

    def reset(self):
        self._track_detection_counts.clear()
