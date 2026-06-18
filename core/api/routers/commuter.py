from fastapi import APIRouter
from pydantic import BaseModel

from ..state import junctions, violations_log

router = APIRouter(prefix="/api/v1/commuter", tags=["commuter"])


class CommuterQuery(BaseModel):
    origin_lat: float = 0.0
    origin_lng: float = 0.0
    dest_lat: float = 0.0
    dest_lng: float = 0.0


@router.get("/alerts")
async def get_alerts(junction_id: str | None = None, limit: int = 20):
    alerts = []
    for v in violations_log[-100:]:
        if junction_id and v.junction_id != junction_id:
            continue
        severity = "high" if v.type in ("signal_jump", "wrong_way") else "medium"
        alerts.append({
            "id": v.id,
            "type": v.type,
            "junction": v.junction_id,
            "severity": severity,
            "plate": v.plate_number,
            "timestamp": v.timestamp,
        })

    for jid, j in junctions.items():
        if j.density_score > 0.7:
            alerts.append({
                "id": f"congestion_{jid}",
                "type": "congestion",
                "junction": jid,
                "severity": "high",
                "density": j.density_score,
                "level": j.congestion_level,
            })

    return {"alerts": alerts[-limit:], "total": len(alerts)}


@router.get("/congestion")
async def get_congestion_map():
    result = []
    for jid, j in junctions.items():
        result.append({
            "junction_id": jid,
            "name": j.name,
            "density": j.density_score,
            "level": j.congestion_level,
            "lat": j.lat,
            "lng": j.lng,
        })
    return {"junctions": result}


@router.post("/route")
async def suggest_route(query: CommuterQuery):
    congestion_junctions = [
        jid for jid, j in junctions.items()
        if j.density_score > 0.5
    ]

    avoids = congestion_junctions[:3]

    return {
        "origin": {"lat": query.origin_lat, "lng": query.origin_lng},
        "destination": {"lat": query.dest_lat, "lng": query.dest_lng},
        "avoid_junctions": avoids,
        "congestion_level": "high" if len(congestion_junctions) > 2 else "moderate",
        "estimated_delay_min": len(congestion_junctions) * 5,
        "suggestion": "Consider alternate route via Ring Road" if len(congestion_junctions) > 2 else "Current route is clear",
    }


@router.get("/departure")
async def departure_planner(origin: str = "home"):
    now_hour = 9
    predictions = {
        "peak_morning": {"start": 8, "end": 10, "congestion": "high"},
        "peak_evening": {"start": 17, "end": 19, "congestion": "high"},
        "off_peak": {"start": 10, "end": 16, "congestion": "low"},
    }

    best_departures = [
        {"time": "07:30", "reason": "Before morning peak", "congestion": "low"},
        {"time": "10:30", "reason": "After morning peak", "congestion": "low"},
        {"time": "16:00", "reason": "Before evening peak", "congestion": "moderate"},
    ]

    return {
        "origin": origin,
        "current_hour": now_hour,
        "current_congestion": "high" if 8 <= now_hour <= 10 or 17 <= now_hour <= 19 else "low",
        "best_departures": best_departures,
        "predictions": predictions,
    }
