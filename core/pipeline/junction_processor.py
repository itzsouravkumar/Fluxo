from __future__ import annotations

from collections import deque
import logging

logger = logging.getLogger(__name__)


class JunctionProcessor:
    """Per-junction processing loop: the main pipeline."""

    def __init__(self, junction_id: str, buffer_size: int = 300):
        self.junction_id = junction_id
        self.ring_buffer: deque = deque(maxlen=buffer_size)

    async def process_frame(self, frame, detector, density_scorer, violation_engine, rl_agent, redis_client):
        from core.vision.preprocessor import FramePreprocessor

        preprocessor = FramePreprocessor()
        processed = preprocessor.preprocess(frame)

        tracks = detector.detect(processed)
        density = density_scorer.compute_density(tracks)
        signal_state = "GREEN"
        violations = violation_engine.check(tracks, processed, signal_state)
        signal_rec = rl_agent.recommend([0.0] * 8)

        self.ring_buffer.append(processed)

        payload = {
            "junction_id": self.junction_id,
            "density_score": density,
            "rl_recommendation": signal_rec,
            "active_violations": len(violations),
        }

        if redis_client:
            import json
            await redis_client.publish("fluxo:junction:updates", json.dumps(payload))

        return payload
