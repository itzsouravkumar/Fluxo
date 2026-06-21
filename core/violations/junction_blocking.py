"""Junction Blocking Detector.

Detects vehicles trapped in the central intersection box during
critical congestion (gridlock). Uses area-based density in a
dynamically computed central polygon.
"""

from __future__ import annotations

import numpy as np
import cv2
from .types import ViolationEvent, ViolationType


class JunctionBlockingDetector:
    """Detects vehicles blocking the central intersection during congestion.

    Computes an area-based density specifically for the center of the frame
    and flags vehicles trapped there when density exceeds a critical threshold
    for a sustained number of frames.
    """

    def __init__(self, density_threshold: float = 0.25, min_blocked_frames: int = 5):
        self.density_threshold = density_threshold
        self.min_blocked_frames = min_blocked_frames
        self._intersection_poly = None
        self._blocked_frames = 0
        self._frame_shape = None

    def _compute_intersection_polygon(self, frame_shape: tuple) -> np.ndarray:
        """Define a dynamic central box (approx 40% of the frame in the center)."""
        h, w = frame_shape[:2]
        margin_x = int(w * 0.3)
        margin_y = int(h * 0.3)
        return np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y]
        ], dtype=np.int32)

    def detect(
        self,
        detections,
        frame: np.ndarray,
        frame_idx: int,
        signal_state: str = "GREEN",
    ) -> list[ViolationEvent]:
        """Standard detect interface matching all other violation detectors."""
        violations = []
        if len(detections) == 0:
            self._blocked_frames = max(0, self._blocked_frames - 1)
            return violations

        # Recompute polygon if frame shape changes
        if self._intersection_poly is None or self._frame_shape != frame.shape[:2]:
            self._frame_shape = frame.shape[:2]
            self._intersection_poly = self._compute_intersection_polygon(frame.shape)

        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return violations

        # Count vehicles whose center is inside the intersection polygon
        vehicles_in_box = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            cx = float((bbox[0] + bbox[2]) / 2)
            cy = float((bbox[1] + bbox[3]) / 2)

            if cv2.pointPolygonTest(self._intersection_poly, (cx, cy), False) >= 0:
                vehicles_in_box.append(i)

        if not vehicles_in_box:
            self._blocked_frames = max(0, self._blocked_frames - 1)
            return violations

        # Compute density as ratio of occupied area to intersection area
        box_area = cv2.contourArea(self._intersection_poly)
        if box_area <= 0:
            return violations

        occupied_area = sum(
            float((detections.xyxy[i][2] - detections.xyxy[i][0]) *
                  (detections.xyxy[i][3] - detections.xyxy[i][1]))
            for i in vehicles_in_box
        )

        density = occupied_area / box_area

        if density > self.density_threshold:
            self._blocked_frames += 1
        else:
            self._blocked_frames = max(0, self._blocked_frames - 1)

        # Only flag after sustained blocking
        if self._blocked_frames >= self.min_blocked_frames:
            for i in vehicles_in_box:
                tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                if tid < 0:
                    continue
                bbox = detections.xyxy[i]
                violations.append(
                    ViolationEvent(
                        type=ViolationType.JUNCTION_BLOCKING,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=min(0.5 + density * 0.4, 0.95),
                        bbox=tuple(int(x) for x in bbox),
                    )
                )

        return violations
