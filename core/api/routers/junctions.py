from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/junctions", tags=["junctions"])


@router.get("/")
async def list_junctions():
    return {"junctions": []}


@router.get("/{junction_id}")
async def get_junction(junction_id: str):
    return {"junction_id": junction_id, "density_score": 0.0, "congestion_level": "CLEAR"}


@router.get("/{junction_id}/density")
async def get_junction_density(junction_id: str, hours: int = 24):
    return {"junction_id": junction_id, "density_history": []}
