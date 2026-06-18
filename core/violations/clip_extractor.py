from __future__ import annotations

from pathlib import Path


class ClipExtractor:
    """Extracts evidence video clips from ring buffer using ffmpeg."""

    def __init__(self, fps: int = 30, buffer_size: int = 300):
        self.fps = fps
        self.buffer_size = buffer_size

    def extract(
        self,
        frames: list,
        event_frame: int,
        output_path: str | Path,
        pre_seconds: int = 2,
        post_seconds: int = 3,
    ) -> Path | None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start = max(0, event_frame - pre_seconds * self.fps)
        end = min(len(frames), event_frame + post_seconds * self.fps)

        if start >= end:
            return None

        clip_frames = frames[start:end]
        self._write_clip(clip_frames, output_path)
        return output_path

    def _write_clip(self, frames: list, output_path: Path):
        import cv2
        import numpy as np

        if not frames:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))

        for frame in frames:
            writer.write(frame)
        writer.release()
