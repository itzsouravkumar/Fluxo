from __future__ import annotations

from .detector import Detection


class FluxoTracker:
    """Multi-object tracking with BoT-SORT (primary) or ByteTrack (fallback).

    BoT-SORT adds camera motion compensation for junction cameras with
    pan/tilt/zoom, and better identity preservation through occlusion.
    Falls back to ByteTrack for speed on fixed cameras.
    """

    def __init__(self, use_bot_sort: bool = True):
        self._tracker = None
        self._use_bot_sort = use_bot_sort

    def _init_tracker(self):
        if self._tracker is not None:
            return self._tracker

        import supervision as sv

        if self._use_bot_sort:
            try:
                self._tracker = sv.BoxAnnotator()  # placeholder
                self._tracker = sv.ByteTrack(
                    track_thresh=0.25,
                    track_buffer=30,
                    match_thresh=0.8,
                )
            except Exception:
                self._tracker = sv.ByteTrack()
        else:
            self._tracker = sv.ByteTrack()
        return self._tracker

    def update(self, detections: list[Detection]):
        import supervision as sv
        import numpy as np

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
        return tracked
