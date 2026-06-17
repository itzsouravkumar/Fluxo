# FLUXO API Reference

## REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/junctions/` | List all monitored junctions |
| GET | `/api/v1/junctions/{id}` | Junction detail + current state |
| GET | `/api/v1/junctions/{id}/density` | Historical density (?hours=24) |
| GET | `/api/v1/predict/{junction_id}` | 15/30 min forecast |
| POST | `/api/v1/signals/{id}/apply` | Apply RL recommendation |
| POST | `/api/v1/signals/{id}/override` | Manual signal override |
| GET | `/api/v1/violations/` | Violation feed |
| GET | `/api/v1/violations/{id}` | Violation detail + clip URL |
| GET | `/api/v1/violations/{id}/clip` | Stream evidence video |

## WebSocket

| Path | Description |
|------|-------------|
| `/ws/live` | Real-time junction updates |
| `/ws/violations` | Real-time violation alerts |

## WebSocket Message Format

```json
{
  "type": "junction_update",
  "junction_id": "veerannapalya_001",
  "data": {
    "density_score": 0.87,
    "congestion_level": "CRITICAL",
    "rl_recommendation": {
      "phase": "E-W",
      "duration_s": 45,
      "confidence": 0.91
    }
  }
}
```
