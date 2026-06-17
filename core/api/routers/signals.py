from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


@router.post("/{junction_id}/apply")
async def apply_signal(junction_id: str):
    return {"junction_id": junction_id, "status": "applied"}


@router.post("/{junction_id}/override")
async def override_signal(junction_id: str):
    return {"junction_id": junction_id, "status": "overridden"}
