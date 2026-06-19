from __future__ import annotations

import cv2
import numpy as np


class HomographyCalibrator:
    """Pixel-to-meter homography for speed estimation with auto-re-calibration.

    Includes automatic drift detection using lane marking positions.
    If detected lane markings deviate > 5px from expected homography-projected
    positions over 100 consecutive frames, triggers re-calibration alert.

    Reference: Vol 9, EC-S2 — homography calibration drift.
    """

    def __init__(
        self,
        source_points: np.ndarray | None = None,
        target_points: np.ndarray | None = None,
        drift_threshold_px: float = 5.0,
        drift_window: int = 100,
        safety_margin_m: float = 0.15,
    ):
        self.H = None
        self.drift_threshold_px = drift_threshold_px
        self.drift_window = drift_window
        self.safety_margin_m = safety_margin_m
        self._expected_lane_positions: list[np.ndarray] = []
        self._drift_history: list[float] = []
        self._needs_recalibration = False

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

    def project_stop_line_with_margin(self, point: np.ndarray) -> np.ndarray:
        projected = self.project_to_ground(point)
        return projected + np.array([0, self.safety_margin_m])

    def detect_lane_drift(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return 0.0

        lane_lines = [line[0] for line in lines if abs(line[0][1] - line[0][3]) < 20]
        if not lane_lines:
            return 0.0

        avg_x = np.mean([(line[0] + line[2]) / 2 for line in lane_lines])

        if self._expected_lane_positions:
            expected_x = np.mean(self._expected_lane_positions)
            drift = abs(avg_x - expected_x)
            self._drift_history.append(drift)
            if len(self._drift_history) > self.drift_window:
                self._drift_history.pop(0)

            if len(self._drift_history) >= self.drift_window:
                mean_drift = np.mean(self._drift_history)
                if mean_drift > self.drift_threshold_px:
                    self._needs_recalibration = True
                else:
                    self._needs_recalibration = False

            return float(drift)

        self._expected_lane_positions.append(avg_x)
        if len(self._expected_lane_positions) > self.drift_window:
            self._expected_lane_positions.pop(0)
        return 0.0

    @property
    def needs_recalibration(self) -> bool:
        return self._needs_recalibration

    def get_drift_stats(self) -> dict:
        return {
            "current_drift_px": self._drift_history[-1] if self._drift_history else 0.0,
            "mean_drift_px": float(np.mean(self._drift_history)) if self._drift_history else 0.0,
            "needs_recalibration": self._needs_recalibration,
            "history_length": len(self._drift_history),
        }
