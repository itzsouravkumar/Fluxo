from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import cv2

from .types import ViolationEvent
from .signal_jump import SignalJumpDetector
from .helmet import HelmetDetector
from .wrong_way import WrongWayDetector
from .triple_riding import TripleRidingDetector
from .fancy_plate import FancyPlateDetector
from .mirror import MirrorDetector
from .mobile_phone import MobilePhoneDetector
from .seatbelt import SeatbeltDetector
from .overloading import OverloadingDetector
from .anpr import ANPRReader
from .clip_extractor import ClipExtractor
from .clip_extractor import ClipExtractor
from .pedestrian import PedestrianRedLightDetector
from .junction_blocking import JunctionBlockingDetector

if TYPE_CHECKING:
    from .types import ViolationConfig


class ViolationDetector:
    """Orchestrates all violation detection sub-modules.

    Uses a single detection+tracking pass to drive all violation checks.
    Every classifier reads from the same per-vehicle track ID, so one
    detection+tracking pass drives all violation checks instead of
    separate inference calls per frame.

    Integrates:
    - Core: signal jump, helmet, wrong-way, triple-riding
    - Novel: fancy plate, missing mirror, pedestrian red-light
    - ANPR: SR preprocessing + per-character confidence + color-aware
    - Evidence: SHA-256 hash chain for legal defensibility
    - Track gating: per-violation-type consecutive-frame requirements

    Reference: Single-Pass Unified Pipeline (Vol 8, Research Brief)
    """

    def __init__(self, config: ViolationConfig | None = None):
        from .types import ViolationConfig
        self.config = config or ViolationConfig()
        self._detectors = []
        self._anpr = ANPRReader() if self.config.enable_anpr else None
        self._clip_extractor = ClipExtractor() if self.config.enable_clip_extract else None
        self._ring_buffer: list[np.ndarray] = []
        self._ring_buffer_max = 300
        self._track_lifetimes: dict[int, int] = {}
        self._reported_violations: set[tuple[int, str]] = set()

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
        if self.config.enable_mobile_phone:
            self._detectors.append(MobilePhoneDetector())
        if self.config.enable_seatbelt:
            self._detectors.append(SeatbeltDetector())
        if self.config.enable_overloading:
            self._detectors.append(OverloadingDetector())
        if self.config.enable_pedestrian_red_light:
            self._detectors.append(PedestrianRedLightDetector(self.config.stop_line_y))

    def reset(self):
        self._track_lifetimes.clear()
        self._reported_violations.clear()
        self._ring_buffer.clear()
        for d in self._detectors:
            if hasattr(d, '_track_history'):
                d._track_history.clear()
            if hasattr(d, '_wrong_way_streak'):
                d._wrong_way_streak.clear()
            if hasattr(d, '_direction_samples'):
                d._direction_samples.clear()
            if hasattr(d, '_dominant_direction'):
                d._dominant_direction = None

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
        tracker=None,
    ) -> list[ViolationEvent]:
        # Lazily initialize to fix Streamlit cached instances
        if not hasattr(self, "_track_lifetimes"):
            self._track_lifetimes = {}

        # Track lifetimes internally
        if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
            for tid in detections.tracker_id:
                tid = int(tid)
                self._track_lifetimes[tid] = self._track_lifetimes.get(tid, 0) + 1

        violations = []

        for detector in self._detectors:
            violations.extend(detector.detect(detections, frame, frame_idx, signal_state))

        violations = self._apply_track_gating(violations, tracker)
        violations = self._check_frame_completeness(violations, frame)
        violations = [v for v in violations if v.confidence >= self.config.min_violation_confidence]

        if self._anpr:
            for v in violations:
                if not self._is_plate_region_readable(detections, frame, v.track_id):
                    continue
                plate_result = self._read_plate_with_confidence(detections, frame, v.track_id)
                if plate_result and plate_result.get("plate"):
                    v.plate_number = plate_result["plate"]
                    v.partial_plate = plate_result.get("is_partial", False)
                    if v.partial_plate:
                        v.requires_human_review = True

                    if tracker and hasattr(tracker, 'verify_ocr_consistency') and self.config.enable_anpr:
                        is_consistent = tracker.verify_ocr_consistency(
                            v.track_id,
                            plate_result["plate"],
                            window=self.config.ocr_consistency_window,
                        )
                        if not is_consistent:
                            v.requires_human_review = True

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

        if self.config.evidence_hash_enabled:
            for v in violations:
                v.compute_evidence_hash()

        return violations

    def _apply_track_gating(self, violations: list[ViolationEvent], tracker=None) -> list[ViolationEvent]:
        gated = []
        for v in violations:
            min_frames = self._get_min_frames_for_violation(v.type)
            if min_frames <= 1:
                gated.append(v)
                continue

            # Lazily initialize
            if not hasattr(self, "_track_lifetimes"):
                self._track_lifetimes = {}

            # Use internal tracking instead of tracker object
            lifetime = self._track_lifetimes.get(v.track_id, 0)
            if lifetime >= min_frames:
                gated.append(v)

        return gated

    def _get_min_frames_for_violation(self, vtype) -> int:
        from .types import ViolationType
        mapping = {
            ViolationType.NO_HELMET: self.config.track_min_frames_helmet,
            ViolationType.WRONG_WAY: self.config.track_min_frames_wrong_way,
            ViolationType.SIGNAL_JUMP: self.config.track_min_frames_signal_jump,
            ViolationType.AMBER_VIOLATION: self.config.track_min_frames_yellow,
            ViolationType.TRIPLE_RIDING: self.config.track_min_frames_helmet,
            ViolationType.MOBILE_PHONE: self.config.track_min_frames_helmet,
        }
        return mapping.get(vtype, 3)

    def _check_frame_completeness(self, violations: list[ViolationEvent], frame: np.ndarray) -> list[ViolationEvent]:
        h, w = frame.shape[:2]
        complete = []
        for v in violations:
            if v.bbox == (0, 0, 0, 0):
                complete.append(v)
                continue
            x1, y1, x2, y2 = v.bbox
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            box_area = bw * bh
            if bw < 30 or bh < 30:
                continue
            partially_outside = (x1 < 0 or x2 > w or y1 < 0 or y2 > h)
            if partially_outside:
                clipped_area = max(0, min(x2, w) - max(x1, 0)) * max(0, min(y2, h) - max(y1, 0))
                completeness = clipped_area / max(box_area, 1)
                if completeness < 0.5:
                    continue
                if completeness < self.config.frame_completeness_threshold:
                    v.requires_human_review = True
            complete.append(v)
        return complete

    def _read_plate_with_confidence(self, detections, frame: np.ndarray, track_id: int) -> dict | None:
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
            return self._anpr.read_plate_with_confidence(crop)
        return None

    def _is_plate_region_readable(self, detections, frame: np.ndarray, track_id: int) -> bool:
        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return False
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
                return False
            ch, cw = crop.shape[:2]
            if cw < 60 or ch < 20:
                return False
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            if np.mean(gray) < 30:
                return False
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                return False
            return True
        return False
