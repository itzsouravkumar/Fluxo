from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/violations", tags=["violations"])


@router.get("/")
async def list_violations(junction_id: str | None = None, type: str | None = None, limit: int = 50):
    return {"violations": []}


@router.get("/{violation_id}")
async def get_violation(violation_id: str):
    return {"violation_id": violation_id}


@router.get("/{violation_id}/clip")
async def get_violation_clip(violation_id: str):
    return {"clip_url": ""}
