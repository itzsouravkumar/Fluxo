from fastapi import APIRouter

from ..state import violations_log

router = APIRouter(prefix="/api/v1/violations", tags=["violations"])


@router.get("/")
async def list_violations(
    junction_id: str | None = None,
    type: str | None = None,
    limit: int = 50,
):
    results = violations_log
    if junction_id:
        results = [v for v in results if v.junction_id == junction_id]
    if type:
        results = [v for v in results if v.type == type]
    return {
        "violations": [
            {
                "id": v.id,
                "junction_id": v.junction_id,
                "type": v.type,
                "track_id": v.track_id,
                "confidence": v.confidence,
                "plate_number": v.plate_number,
                "clip_path": v.clip_path,
                "timestamp": v.timestamp,
            }
            for v in results[-limit:]
        ],
        "total": len(results),
    }


@router.get("/stats")
async def violation_stats():
    stats = {}
    for v in violations_log:
        stats[v.type] = stats.get(v.type, 0) + 1
    return {"stats": stats, "total": len(violations_log)}


@router.get("/{violation_id}")
async def get_violation(violation_id: str):
    for v in violations_log:
        if v.id == violation_id:
            return {
                "id": v.id,
                "junction_id": v.junction_id,
                "type": v.type,
                "track_id": v.track_id,
                "confidence": v.confidence,
                "plate_number": v.plate_number,
                "clip_path": v.clip_path,
                "timestamp": v.timestamp,
            }
    return {"error": "not found"}


@router.get("/{violation_id}/clip")
async def get_violation_clip(violation_id: str):
    for v in violations_log:
        if v.id == violation_id:
            return {"clip_url": v.clip_path or ""}
    return {"clip_url": ""}
