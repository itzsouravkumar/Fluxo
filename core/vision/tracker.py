from __future__ import annotations

import numpy as np
from .detector import Detection


class FluxoTracker:
    """Multi-object tracking with BoT-SORT (primary) or ByteTrack (fallback).

    BoT-SORT adds camera motion compensation for junction cameras with
    pan/tilt/zoom, and better identity preservation through occlusion.
    Falls back to ByteTrack for speed on fixed cameras.

    Track lifetime gating: violations only flagged after N consecutive frames
    per violation type to reduce false positives.
    """

    def __init__(self, use_bot_sort: bool = True, track_buffer: int = 90):
        self._tracker = None
        self._use_bot_sort = use_bot_sort
        self._track_buffer = track_buffer
        self._track_states: dict[int, dict] = {}
        self._frame_counter = 0

    def reset(self):
        self._track_states.clear()
        self._frame_counter = 0
        self._tracker = None

    def _init_tracker(self):
        if self._tracker is not None:
            return self._tracker

        import supervision as sv

        if self._use_bot_sort:
            try:
                from supervision.tracker import BoTSort
                self._tracker = BoTSort(
                    track_thresh=0.25,
                    track_buffer=self._track_buffer,
                    match_thresh=0.8,
                    cam_motion_compensation=True,
                )
                return self._tracker
            except (ImportError, AttributeError, TypeError):
                pass

        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=self._track_buffer,
            minimum_matching_threshold=0.8,
        )
        return self._tracker

    def update(self, detections: list[Detection], frame_idx: int = 0):
        import supervision as sv

        tracker = self._init_tracker()

        if not detections:
            return []

        bboxes = np.array([d.bbox for d in detections])
        confidences = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids,
        )
        tracked = tracker.update_with_detections(sv_detections)

        self._update_track_states(tracked, frame_idx)
        return tracked

    def update_with_detections(self, sv_detections):
        tracker = self._init_tracker()
        self._frame_counter += 1
        if len(sv_detections) == 0:
            return sv_detections
        tracked = tracker.update_with_detections(sv_detections)
        self._update_track_states(tracked, self._frame_counter)
        return tracked

    def _update_track_states(self, tracked, frame_idx: int):
        if not hasattr(tracked, "tracker_id") or tracked.tracker_id is None:
            return

        active_ids = set()
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            active_ids.add(tid)

            if tid not in self._track_states:
                self._track_states[tid] = {
                    "track_id": tid,
                    "first_seen_frame": frame_idx,
                    "consecutive_detections": 1,
                    "last_seen_frame": frame_idx,
                    "plate_history": [],
                    "is_id_unstable": False,
                    "class_id": cls_id,
                }
            else:
                state = self._track_states[tid]
                if frame_idx - state["last_seen_frame"] <= 2:
                    state["consecutive_detections"] += 1
                else:
                    state["consecutive_detections"] = 1
                state["last_seen_frame"] = frame_idx
                state["class_id"] = cls_id

        stale = [tid for tid in self._track_states if tid not in active_ids]
        for tid in stale:
            if frame_idx - self._track_states[tid]["last_seen_frame"] > self._track_buffer:
                del self._track_states[tid]

    def get_track_state(self, track_id: int) -> dict | None:
        return self._track_states.get(track_id)

    def check_track_lifetime(self, track_id: int, min_frames: int) -> bool:
        state = self._track_states.get(track_id)
        if state is None:
            return False
        return state["consecutive_detections"] >= min_frames

    def verify_ocr_consistency(self, track_id: int, new_plate: str, window: int = 10) -> bool:
        state = self._track_states.get(track_id)
        if state is None:
            return True

        state["plate_history"].append(new_plate)
        if len(state["plate_history"]) > window:
            state["plate_history"] = state["plate_history"][-window:]

        if len(state["plate_history"]) < 3:
            return True

        recent = state["plate_history"][-5:]
        unique_plates = set(recent)
        if len(unique_plates) > 1:
            state["is_id_unstable"] = True
            return False
        return True

    @property
    def active_tracks(self) -> int:
        return len(self._track_states)
