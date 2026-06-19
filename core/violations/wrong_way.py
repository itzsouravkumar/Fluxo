from __future__ import annotations

from collections import defaultdict

import numpy as np
from .types import ViolationEvent, ViolationType


class WrongWayDetector:
    """Detects vehicles moving against the dominant traffic flow direction.

    Instead of relying on frame quadrants (which don't match real lanes),
    this detector computes the average movement direction of ALL tracked
    vehicles, then flags any vehicle moving against that dominant direction.

    Requires temporal confirmation to reduce false positives.
    """

    VEHICLE_CLASSES = {0, 1, 2, 3, 4}

    def __init__(self):
        self._track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._track_class: dict[int, int] = {}
        self._wrong_way_streak: dict[int, int] = defaultdict(int)
        self._max_history = 15
        self._min_confirm_frames = 5
        self._min_track_age = 12
        self._min_movement_px = 30
        self._dominant_direction: float | None = None
        self._direction_samples: list[float] = []

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return violations

        active_tids = set()
        current_directions = []

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if tid < 0:
                continue
            active_tids.add(tid)

            cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            self._track_class[tid] = cls_id

            bbox = detections.xyxy[i]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            self._track_history[tid].append((cx, cy))
            if len(self._track_history[tid]) > self._max_history:
                self._track_history[tid].pop(0)

            history = self._track_history[tid]
            if len(history) < self._min_track_age:
                continue

            dx = history[-1][0] - history[-8][0]
            dy = history[-1][1] - history[-8][1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < self._min_movement_px:
                continue

            angle = np.arctan2(dy, dx)
            current_directions.append(angle)

        self._direction_samples.extend(current_directions)
        if len(self._direction_samples) > 200:
            self._direction_samples = self._direction_samples[-200:]

        if len(self._direction_samples) < 10:
            return violations

        self._dominant_direction = self._circular_mean(self._direction_samples)

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if tid < 0:
                continue

            cls_id = self._track_class.get(tid, -1)
            if cls_id not in self.VEHICLE_CLASSES:
                continue

            history = self._track_history.get(tid, [])
            if len(history) < self._min_track_age:
                continue

            dx = history[-1][0] - history[-8][0]
            dy = history[-1][1] - history[-8][1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < self._min_movement_px:
                continue

            velocity_angle = np.arctan2(dy, dx)

            angle_diff = abs(velocity_angle - self._dominant_direction)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            if angle_diff > np.pi * 0.5:
                self._wrong_way_streak[tid] += 1
                if self._wrong_way_streak[tid] >= self._min_confirm_frames:
                    bbox = detections.xyxy[i]
                    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.7
                    violations.append(
                        ViolationEvent(
                            type=ViolationType.WRONG_WAY,
                            track_id=tid,
                            frame=frame_idx,
                            confidence=min(conf, 0.9),
                            bbox=tuple(bbox.astype(int)),
                        )
                    )
            else:
                self._wrong_way_streak[tid] = 0

        stale = [tid for tid in self._wrong_way_streak if tid not in active_tids]
        for tid in stale:
            del self._wrong_way_streak[tid]

        return violations

    @staticmethod
    def _circular_mean(angles: list[float]) -> float:
        sin_sum = sum(np.sin(a) for a in angles)
        cos_sum = sum(np.cos(a) for a in angles)
        return np.arctan2(sin_sum, cos_sum)
