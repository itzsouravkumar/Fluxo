# FLUXO Architecture

## System Overview

FLUXO is a dual-layer urban intelligence platform:

**BTP Operator Layer (top-down):**
- Real-time CCTV vision AI
- RL-driven signal control
- Violation detection + evidence
- Congestion prediction

**Commuter Layer (bottom-up):**
- Smart departure planner
- Capacity-aware routing
- Signal green wave navigation
- Community incident reporting

## Data Flow

```
CCTV Feed → YOLOv11 Detection → ByteTrack → PCE Density Score
                                              ↓
                                    TFT Prediction (15/30 min)
                                              ↓
                                    PPO RL Signal Agent
                                              ↓
                                    FastAPI WebSocket → Dashboard
                                              ↓
                                    PostgreSQL (historical)
```

## Edge-Cloud Split

- **Edge**: YOLOv11 nano on Jetson Nano / RPi5 (~25ms inference)
- **Cloud**: RL coordination, TFT prediction, dashboard, database
- Raw video: ~10 MB/s per camera → Density JSON: ~1 KB/s

## Module Communication

All modules communicate through Redis PubSub for real-time updates and PostgreSQL for historical data.
