from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.get("/predict/{junction_id}")
async def predict_congestion(junction_id: str):
    return {
        "junction_id": junction_id,
        "forecast": [
            {"horizon_min": 15, "density_score": 0.5, "level": "MODERATE"},
            {"horizon_min": 30, "density_score": 0.5, "level": "MODERATE"},
        ],
    }
