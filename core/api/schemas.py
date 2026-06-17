from pydantic import BaseModel


class JunctionUpdate(BaseModel):
    junction_id: str
    density_score: float
    congestion_level: str
    lane_states: list[dict] = []
    rl_recommendation: dict = {}
    prediction: dict = {}
    active_violations: int = 0


class ViolationEvent(BaseModel):
    id: str
    junction_id: str
    type: str
    timestamp: str
    plate_number: str | None = None
    clip_url: str | None = None
    confidence: float = 0.0
