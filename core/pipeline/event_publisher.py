from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class EventPublisher:
    """Redis pub/sub publisher for real-time updates."""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def publish_junction_update(self, payload: dict):
        await self.redis.publish("fluxo:junction:updates", json.dumps(payload))

    async def publish_violation(self, violation: dict):
        await self.redis.publish("fluxo:violations", json.dumps(violation))
