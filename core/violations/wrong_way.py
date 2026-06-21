from __future__ import annotations

from collections import defaultdict

import numpy as np
from .types import ViolationEvent, ViolationType


class WrongWayDetector:
    """Detects vehicles moving against the dominant traffic flow direction.

    Uses multi-flow clustering to handle TWO-WAY ROADS correctly:
    - Clusters vehicle trajectories into K dominant flow directions
    - A vehicle is flagged only if its direction doesn't match ANY
      of the learned flow clusters
    - This prevents flagging 50% of traffic on a two-way road

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
        self._direction_samples: list[float] = []
        # Store multiple flow clusters instead of a single dominant direction
        self._flow_clusters: list[float] = []
        # Minimum angle difference from ALL clusters to flag wrong-way
        self._wrong_way_angle_threshold = np.pi * 0.55  # ~100 degrees

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
        if len(self._direction_samples) > 300:
            self._direction_samples = self._direction_samples[-300:]

        # Need enough samples to establish flow patterns
        if len(self._direction_samples) < 20:
            return violations

        # Cluster directions into 1-2 dominant flows
        self._flow_clusters = self._cluster_directions(self._direction_samples)

        if not self._flow_clusters:
            return violations

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

            # Check against ALL flow clusters — wrong-way only if
            # the vehicle doesn't match ANY cluster
            is_wrong_way = True
            for cluster_angle in self._flow_clusters:
                angle_diff = abs(velocity_angle - cluster_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                if angle_diff < self._wrong_way_angle_threshold:
                    is_wrong_way = False
                    break

            if is_wrong_way:
                self._wrong_way_streak[tid] += 1
                if self._wrong_way_streak[tid] >= self._min_confirm_frames:
                    bbox = detections.xyxy[i]
                    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.7
                    violations.append(
                        ViolationEvent(
                            type=ViolationType.WRONG_WAY,
                            track_id=tid,
                            frame=frame_idx,
                            confidence=min(conf, 0.85),
                            bbox=tuple(bbox.astype(int)),
                        )
                    )
            else:
                self._wrong_way_streak[tid] = 0

        stale = [tid for tid in self._wrong_way_streak if tid not in active_tids]
        for tid in stale:
            del self._wrong_way_streak[tid]

        return violations

    def _cluster_directions(self, angles: list[float], max_clusters: int = 2) -> list[float]:
        """Cluster angles into 1-2 dominant flow directions.

        Uses a simple histogram-based approach on circular data:
        1. Bin angles into 12 sectors (30° each)
        2. Find the top 1-2 peaks
        3. Return the mean angle of each peak cluster

        This handles both one-way and two-way roads automatically.
        """
        n_bins = 12
        bin_size = 2 * np.pi / n_bins

        # Normalize angles to [0, 2π)
        norm_angles = [(a % (2 * np.pi)) for a in angles]

        # Count per bin
        counts = [0] * n_bins
        bin_angles: list[list[float]] = [[] for _ in range(n_bins)]
        for a in norm_angles:
            bin_idx = int(a / bin_size) % n_bins
            counts[bin_idx] += 1
            bin_angles[bin_idx].append(a)

        if max(counts) < 5:
            return []

        # Find peaks: bins with more than 15% of total samples
        threshold = len(angles) * 0.12
        peaks = []
        for i in range(n_bins):
            # Check bin and its neighbors (smoothed)
            total = counts[i] + counts[(i - 1) % n_bins] + counts[(i + 1) % n_bins]
            if total > threshold:
                # Compute mean angle for this cluster
                cluster_angles = (
                    bin_angles[(i - 1) % n_bins] +
                    bin_angles[i] +
                    bin_angles[(i + 1) % n_bins]
                )
                if cluster_angles:
                    mean_angle = self._circular_mean(cluster_angles)
                    # Avoid duplicate peaks (within 45° of existing)
                    is_duplicate = False
                    for existing in peaks:
                        diff = abs(mean_angle - existing)
                        if diff > np.pi:
                            diff = 2 * np.pi - diff
                        if diff < np.pi / 4:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        peaks.append(mean_angle)

        return peaks[:max_clusters]

    @staticmethod
    def _circular_mean(angles: list[float]) -> float:
        sin_sum = sum(np.sin(a) for a in angles)
        cos_sum = sum(np.cos(a) for a in angles)
        return np.arctan2(sin_sum, cos_sum)
