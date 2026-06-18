from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class EventBus:
    """In-memory event bus for real-time updates. No Redis dependency."""

    def __init__(self):
        self._subscribers: dict[str, list] = {}
        self._latest: dict[str, Any] = {}
        self._history: dict[str, list] = {}
        self._max_history = 200

    def publish(self, topic: str, data: Any):
        self._latest[topic] = data
        if topic not in self._history:
            self._history[topic] = []
        self._history[topic].append({"data": data, "ts": time.time()})
        if len(self._history[topic]) > self._max_history:
            self._history[topic] = self._history[topic][-self._max_history:]

        for callback in self._subscribers.get(topic, []):
            try:
                callback(data)
            except Exception:
                pass

    def subscribe(self, topic: str, callback):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def get_latest(self, topic: str) -> Any | None:
        return self._latest.get(topic)

    def get_history(self, topic: str, limit: int = 50) -> list:
        return self._history.get(topic, [])[-limit:]


event_bus = EventBus()


@dataclass
class JunctionState:
    junction_id: str
    name: str
    lat: float = 0.0
    lng: float = 0.0
    density_score: float = 0.0
    congestion_level: str = "CLEAR"
    vehicle_count: int = 0
    unique_vehicles: int = 0
    lane_states: dict = field(default_factory=dict)
    signal_state: str = "GREEN"
    signal_phase: str = "N-S"
    signal_remaining: int = 30
    rl_recommendation: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    prediction: dict = field(default_factory=dict)
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return {
            "junction_id": self.junction_id,
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "density_score": self.density_score,
            "congestion_level": self.congestion_level,
            "vehicle_count": self.vehicle_count,
            "unique_vehicles": self.unique_vehicles,
            "lane_states": self.lane_states,
            "signal_state": self.signal_state,
            "signal_phase": self.signal_phase,
            "signal_remaining": self.signal_remaining,
            "rl_recommendation": self.rl_recommendation,
            "violations": self.violations,
            "prediction": self.prediction,
            "last_update": self.last_update,
        }


junctions: dict[str, JunctionState] = {
    "j1": JunctionState("j1", "Veerannapalya", 12.9980, 77.6890),
    "j2": JunctionState("j2", "Gokaldas", 12.9950, 77.6830),
    "j3": JunctionState("j3", "Silk Board", 12.9180, 77.6210),
    "j4": JunctionState("j4", "Hebbal", 13.0350, 77.5970),
}


@dataclass
class ViolationRecord:
    id: str
    junction_id: str
    type: str
    track_id: int
    frame: int
    confidence: float
    plate_number: str | None = None
    clip_path: str | None = None
    bbox: tuple = (0, 0, 0, 0)
    timestamp: float = 0.0


violations_log: list[ViolationRecord] = []
MAX_VIOLATIONS = 500


def record_violation(v: ViolationRecord):
    violations_log.append(v)
    if len(violations_log) > MAX_VIOLATIONS:
        violations_log.pop(0)
    event_bus.publish("violation", {
        "id": v.id,
        "type": v.type,
        "plate": v.plate_number,
        "junction": v.junction_id,
        "confidence": v.confidence,
        "clip_path": v.clip_path,
        "ts": v.timestamp,
    })


def update_junction(junction_id: str, **kwargs):
    if junction_id in junctions:
        j = junctions[junction_id]
        for k, v in kwargs.items():
            if hasattr(j, k):
                setattr(j, k, v)
        j.last_update = time.time()
        event_bus.publish(f"junction:{junction_id}", j.to_dict())
        event_bus.publish("junctions", {jid: j.to_dict() for jid, j in junctions.items()})
