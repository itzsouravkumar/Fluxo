from __future__ import annotations

from collections import defaultdict

import numpy as np
from .types import ViolationEvent, ViolationType


class WrongWayDetector:
    """Detects vehicles moving against expected lane direction.

    Uses a history of positions per track to compute velocity direction,
    then checks against expected direction per lane.
    """

    def __init__(self, expected_directions: dict[str, float] | None = None):
        self._track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._max_history = 15
        self.lane_directions = expected_directions or {
            "north": -np.pi / 2,
            "south": np.pi / 2,
            "east": 0.0,
            "west": np.pi,
        }

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return violations

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if tid < 0:
                continue

            bbox = detections.xyxy[i]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            self._track_history[tid].append((cx, cy))
            if len(self._track_history[tid]) > self._max_history:
                self._track_history[tid].pop(0)

            history = self._track_history[tid]
            if len(history) < 8:
                continue

            dx = history[-1][0] - history[-8][0]
            dy = history[-1][1] - history[-8][1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < 10:
                continue

            velocity_angle = np.arctan2(dy, dx)

            lane = self._get_lane(cx, cy, w, h)
            expected = self.lane_directions.get(lane, 0.0)

            angle_diff = abs(velocity_angle - expected)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            if angle_diff > np.pi * 0.6:
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.7
                violations.append(
                    ViolationEvent(
                        type=ViolationType.WRONG_WAY,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=min(conf, angle_diff / np.pi),
                        bbox=tuple(bbox.astype(int)),
                    )
                )

        return violations

    def _get_lane(self, cx: float, cy: float, w: int, h: int) -> str:
        if cy < h / 2:
            return "north" if cx < w / 2 else "east"
        return "south" if cx < w / 2 else "west"
