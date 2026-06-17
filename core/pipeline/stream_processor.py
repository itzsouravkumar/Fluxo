from __future__ import annotations

import cv2
import logging

logger = logging.getLogger(__name__)


class StreamProcessor:
    """RTSP ingestion + frame dispatch."""

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self._cap = None

    def connect(self) -> bool:
        self._cap = cv2.VideoCapture(self.rtsp_url)
        return self._cap.isOpened()

    def read_frame(self):
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def reconnect(self):
        if self._cap:
            self._cap.release()
        return self.connect()

    def release(self):
        if self._cap:
            self._cap.release()
