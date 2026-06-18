from __future__ import annotations

import numpy as np

from .types import ViolationType, ViolationEvent


class OverloadingDetector:
    """Detects overloaded goods vehicles.

    Examines the bounding box aspect ratio and visual density of
    detected heavy vehicles (bus, truck). An overloaded goods carrier
    typically has a wider, flatter appearance with cargo protruding
    above the roofline, or the vehicle sits lower due to weight.

    Reference: BTP ITeMS added overloaded goods vehicle detection as
    one of 13 violation types (Hindustan Times, Sep 2024).
    Reference: Karnataka Transport Dept AI cameras specifically target
    overloading on highways (The Hindu, Jun 2025).
    """

    VIOLATION_TYPE = ViolationType.OVERLOADING

    def __init__(self, aspect_ratio_threshold: float = 1.8, density_threshold: float = 0.6):
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.density_threshold = density_threshold

    def detect(
        self,
        detections,
        frame: np.ndarray,
        frame_idx: int,
        signal_state: str = "GREEN",
    ) -> list[ViolationEvent]:
        violations = []
        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        h, w = frame.shape[:2]
        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id not in (3, 4):
                continue

            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bw, bh = x2 - x1, y2 - y1
            if bw < 40 or bh < 30:
                continue

            aspect_ratio = bw / max(bh, 1)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            density_score = self._compute_visual_density(crop)

            is_overloaded = (
                aspect_ratio > self.aspect_ratio_threshold
                and density_score > self.density_threshold
            )

            if is_overloaded:
                violations.append(ViolationEvent(
                    type=self.VIOLATION_TYPE,
                    track_id=int(detections.tracker_id[i]) if detections.tracker_id is not None else -1,
                    frame=frame_idx,
                    confidence=min(0.5 + density_score * 0.3, 0.9),
                    bbox=tuple(bbox.astype(int).tolist()),
                ))

        return violations

    def _compute_visual_density(self, vehicle_crop: np.ndarray) -> float:
        import cv2
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        color_var = np.std(hsv[:, :, 0])

        density = (edge_density * 0.5 + (saturation / 255.0) * 0.3 + min(color_var / 50.0, 1.0) * 0.2)
        return density
