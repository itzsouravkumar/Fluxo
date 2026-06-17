from __future__ import annotations

from .detector import Detection


class FluxoTracker:
    """Multi-object tracking using ByteTrack."""

    def __init__(self):
        self._tracker = None

    def _init_tracker(self):
        if self._tracker is None:
            import supervision as sv
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
