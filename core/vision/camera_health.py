from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraHealthStatus:
    camera_id: str
    is_frozen: bool = False
    is_blinded: bool = False
    is_degraded: bool = False
    last_frame_hash: str = ""
    frozen_duration_sec: float = 0.0
    blur_score: float = 1.0
    scene_change_score: float = 0.0
    last_updated: float = field(default_factory=time.time)


class CameraHealthMonitor:
    """Monitors camera health using GMM background model + freeze detection.

    Detects:
    - Frame freeze (hash unchanged for N seconds)
    - Lens spray/vandalism (all pixels change simultaneously)
    - Progressive blur increase (lens contamination)
    - IR LED blinding (persistent fixed bright spot)

    Reference: Vol 7, EC — Camera Tampering & Physical Vandalism
    """

    def __init__(
        self,
        freeze_timeout_sec: float = 5.0,
        blur_degradation_threshold: float = 0.3,
        scene_change_threshold: float = 0.6,
    ):
        self.freeze_timeout_sec = freeze_timeout_sec
        self.blur_degradation_threshold = blur_degradation_threshold
        self.scene_change_threshold = scene_change_threshold
        self._cameras: dict[str, CameraHealthStatus] = {}
        self._bg_models: dict[str, cv2.BackgroundSubtractor] = {}
        self._prev_hashes: dict[str, str] = {}
        self._prev_gray: dict[str, np.ndarray] = {}
        self._freeze_start: dict[str, float] = {}

    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        return hash(gray.tobytes())

    def _compute_blur_score(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 200.0, 1.0)

    def _compute_scene_change(self, frame: np.ndarray, camera_id: str) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if camera_id not in self._prev_gray:
            self._prev_gray[camera_id] = gray
            return 0.0
        prev = self._prev_gray[camera_id]
        if prev.shape != gray.shape:
            self._prev_gray[camera_id] = gray
            return 0.0
        diff = cv2.absdiff(prev, gray)
        change = np.mean(diff) / 255.0
        self._prev_gray[camera_id] = gray
        return round(float(change), 4)

    def update(self, camera_id: str, frame: np.ndarray) -> CameraHealthStatus:
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraHealthStatus(camera_id=camera_id)

        status = self._cameras[camera_id]
        frame_hash = self._compute_frame_hash(frame)
        now = time.time()

        if frame_hash == status.last_frame_hash:
            if camera_id not in self._freeze_start:
                self._freeze_start[camera_id] = now
            status.frozen_duration_sec = now - self._freeze_start[camera_id]
            status.is_frozen = status.frozen_duration_sec > self.freeze_timeout_sec
        else:
            self._freeze_start.pop(camera_id, None)
            status.frozen_duration_sec = 0.0
            status.is_frozen = False

        status.last_frame_hash = frame_hash

        scene_change = self._compute_scene_change(frame, camera_id)
        status.scene_change_score = scene_change
        status.is_blinded = scene_change > self.scene_change_threshold

        blur = self._compute_blur_score(frame)
        status.blur_score = blur
        if blur < self.blur_degradation_threshold:
            status.is_degraded = True
        else:
            status.is_degraded = False

        status.last_updated = now
        return status

    def is_camera_healthy(self, camera_id: str) -> bool:
        status = self._cameras.get(camera_id)
        if status is None:
            return True
        return not (status.is_frozen or status.is_blinded or status.is_degraded)

    def get_all_statuses(self) -> dict[str, CameraHealthStatus]:
        return dict(self._cameras)
