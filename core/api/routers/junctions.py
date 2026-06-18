from fastapi import APIRouter

from ..state import junctions

router = APIRouter(prefix="/api/v1/junctions", tags=["junctions"])


@router.get("/")
async def list_junctions():
    return {
        "junctions": {jid: j.to_dict() for jid, j in junctions.items()},
        "total": len(junctions),
    }


@router.get("/{junction_id}")
async def get_junction(junction_id: str):
    if junction_id not in junctions:
        return {"error": "junction not found"}
    return junctions[junction_id].to_dict()


@router.get("/{junction_id}/lanes")
async def get_lane_states(junction_id: str):
    if junction_id not in junctions:
        return {"error": "junction not found"}
    j = junctions[junction_id]
    return {"junction_id": junction_id, "lanes": j.lane_states}


@router.get("/{junction_id}/density")
async def get_density(junction_id: str):
    if junction_id not in junctions:
        return {"error": "junction not found"}
    j = junctions[junction_id]
    return {
        "junction_id": junction_id,
        "density_score": j.density_score,
        "congestion_level": j.congestion_level,
        "vehicle_count": j.vehicle_count,
        "unique_vehicles": j.unique_vehicles,
    }
