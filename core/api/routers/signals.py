from fastapi import APIRouter

from ..state import junctions, update_junction
from core.rl.agent import RLAgent

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])

agent = RLAgent()


@router.get("/{junction_id}")
async def get_signal_state(junction_id: str):
    if junction_id not in junctions:
        return {"error": "junction not found"}
    j = junctions[junction_id]
    return {
        "junction_id": junction_id,
        "signal_state": j.signal_state,
        "signal_phase": j.signal_phase,
        "signal_remaining": j.signal_remaining,
        "rl_recommendation": j.rl_recommendation,
    }


@router.post("/{junction_id}/apply")
async def apply_signal(junction_id: str, duration_s: int = 30):
    if junction_id not in junctions:
        return {"error": "junction not found"}
    update_junction(junction_id, signal_remaining=duration_s)
    return {"junction_id": junction_id, "status": "applied", "duration_s": duration_s}


@router.post("/{junction_id}/override")
async def override_signal(junction_id: str, phase: str = "N-S", duration_s: int = 30):
    if junction_id not in junctions:
        return {"error": "junction not found"}
    update_junction(
        junction_id,
        signal_phase=phase,
        signal_remaining=duration_s,
        signal_state="GREEN",
    )
    return {"junction_id": junction_id, "status": "overridden", "phase": phase, "duration_s": duration_s}


@router.get("/{junction_id}/recommend")
async def get_rl_recommendation(junction_id: str):
    if junction_id not in junctions:
        return {"error": "junction not found"}
    j = junctions[junction_id]
    queues = [
        j.lane_states.get("north", {}).get("queue_length", 0),
        j.lane_states.get("south", {}).get("queue_length", 0),
        j.lane_states.get("east", {}).get("queue_length", 0),
        j.lane_states.get("west", {}).get("queue_length", 0),
    ]
    elapsed = j.signal_remaining
    import math
    hour_frac = (int(j.last_update) % 3600) / 3600.0
    state = queues + [elapsed / 120.0, math.sin(2 * math.pi * hour_frac), math.cos(2 * math.pi * hour_frac), j.density_score]
    rec = agent.recommend(state)
    update_junction(junction_id, rl_recommendation=rec)
    return rec
