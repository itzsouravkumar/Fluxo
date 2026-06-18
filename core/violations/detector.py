from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .types import ViolationEvent
from .signal_jump import SignalJumpDetector
from .helmet import HelmetDetector
from .wrong_way import WrongWayDetector
from .triple_riding import TripleRidingDetector
from .fancy_plate import FancyPlateDetector
from .mirror import MirrorDetector
from .anpr import ANPRReader
from .clip_extractor import ClipExtractor

if TYPE_CHECKING:
    from .types import ViolationConfig


class ViolationDetector:
    """Orchestrates all violation detection sub-modules.

    Uses a single detection+tracking pass to drive all violation checks.
    Every classifier reads from the same per-vehicle track ID, so one
    detection+tracking pass drives all violation checks instead of
    separate inference calls per frame.

    Supports:
    - Signal jump, helmet, wrong-way, triple-riding (core)
    - Fancy plate, missing mirror (novel classes from research)
    - ANPR with SR preprocessing + plate validation
    - VLM evidence narration (post-confirmation, async)
    """

    def __init__(self, config: ViolationConfig | None = None):
        from .types import ViolationConfig
        self.config = config or ViolationConfig()
        self._detectors = []
        self._anpr = ANPRReader() if self.config.enable_anpr else None
        self._clip_extractor = ClipExtractor() if self.config.enable_clip_extract else None
        self._vlm = None
        self._ring_buffer: list[np.ndarray] = []
        self._ring_buffer_max = 300

        if self.config.enable_signal_jump:
            self._detectors.append(SignalJumpDetector(self.config.stop_line_y))
        if self.config.enable_helmet:
            self._detectors.append(HelmetDetector())
        if self.config.enable_wrong_way:
            self._detectors.append(WrongWayDetector())
        if self.config.enable_triple_riding:
            self._detectors.append(TripleRidingDetector())
        if self.config.enable_fancy_plate:
            self._detectors.append(FancyPlateDetector())
        if self.config.enable_missing_mirror:
            self._detectors.append(MirrorDetector())

        if self.config.enable_vlm_narration:
            self._init_vlm()

    def _init_vlm(self):
        try:
            from .vlm_evidence import VLMEvidenceLayer
            self._vlm = VLMEvidenceLayer()
        except Exception:
            self._vlm = None

    def feed_frame(self, frame: np.ndarray):
        if len(self._ring_buffer) >= self._ring_buffer_max:
            self._ring_buffer.pop(0)
        self._ring_buffer.append(frame.copy())

    def check(
        self,
        detections,
        frame: np.ndarray,
        frame_idx: int,
        signal_state: str = "GREEN",
    ) -> list[ViolationEvent]:
        violations = []
        for detector in self._detectors:
            violations.extend(detector.detect(detections, frame, frame_idx, signal_state))

        if self._anpr:
            for v in violations:
                plate = self._read_plate(detections, frame, v.track_id)
                if plate:
                    v.plate_number = plate

        if self._clip_extractor and violations and self._ring_buffer:
            for v in violations:
                clip_path = self._clip_extractor.extract(
                    self._ring_buffer,
                    len(self._ring_buffer) - 1,
                    f"outputs/clips/{v.type.value}_{v.track_id}_f{frame_idx}.mp4",
                    pre_seconds=self.config.clip_pre_seconds,
                    post_seconds=self.config.clip_post_seconds,
                )
                if clip_path:
                    v.clip_path = str(clip_path)

        if self._vlm and violations:
            for v in violations:
                clip_frames = self._ring_buffer[-90:] if self._ring_buffer else []
                narration = self._vlm.narrate(v, clip_frames, frame)
                if narration:
                    v.evidence_narration = narration

        return violations

    def _read_plate(self, detections, frame: np.ndarray, track_id: int) -> str | None:
        if self._anpr is None:
            return None
        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return None
        for i in range(len(detections)):
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if tid != track_id:
                continue
            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = bbox
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(x1, w_frame - 1))
            x2 = max(0, min(x2, w_frame))
            y1 = max(0, min(y1, h_frame - 1))
            y2 = max(0, min(y2, h_frame))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            plate = self._anpr.read_plate(crop)
            if plate:
                return plate
        return None
