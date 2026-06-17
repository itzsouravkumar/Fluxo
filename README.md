# FLUXO

**Adaptive Urban Traffic Intelligence Platform**

Bengaluru loses 168 hours per person annually to traffic. ASTraM alerts every 15 minutes. By then, queues are already 4 km long. FLUXO closes the loop in real time: detect, predict, control, measure, under 200ms.

Built for the Gridlock Hackathon 2.0 (Flipkart x Bengaluru Traffic Police).

## Problem

- 1.23 crore registered vehicles, growing 2,000/day
- 823 vehicles per km of road
- Fixed-time signals hold green for empty lanes while others gridlock
- No system currently controls signals adaptively based on real-time density

## What FLUXO Does

1. **Vision AI**: YOLOv11 detects vehicles, classifies by type (two-wheeler, auto, bus, etc.), tracks across frames
2. **PCE Density Scoring**: Weighted per-lane density (not just vehicle count)
3. **Violation Detection**: Helmet, signal jump, wrong-way, triple riding with evidence clips
4. **Congestion Prediction**: TFT model forecasts 15/30 min ahead
5. **RL Signal Control**: PPO agent outputs optimal green phase timing
6. **Operator Dashboard**: Real-time map, CCTV grid, signal panel, violation feed

The feedback loop is the differentiator: detect > predict > control > measure > improve.

## Quick Start

```bash
# Clone
git clone https://github.com/itzsouravkumar/Fluxo.git
cd Fluxo

# Bootstrap environment
chmod +x scripts/bootstrap_env.sh
./scripts/bootstrap_env.sh

# Activate venv
source venv/bin/activate

# Start services (Redis + PostgreSQL)
docker-compose up -d redis postgres

# Start backend
uvicorn core.api.main:app --reload --port 8000

# Start dashboard (separate terminal)
cd dashboard && npm install && npm run dev
```

Backend: http://localhost:8000
Dashboard: http://localhost:5173
API Docs: http://localhost:8000/docs

## Project Structure

```
Fluxo/
├── core/                    # Python backend
│   ├── vision/              # YOLOv11 detection + tracking
│   ├── violations/          # Violation detection engine
│   ├── prediction/          # TFT congestion forecaster
│   ├── rl/                  # PPO signal controller
│   ├── api/                 # FastAPI endpoints
│   ├── pipeline/            # Stream processing orchestration
│   └── db/                  # PostgreSQL models
├── dashboard/               # React + Vite frontend
├── commuter/                # React Native commuter app
├── scripts/                 # Setup & training scripts
├── tests/                   # Test suite
├── models/                  # Trained model weights
└── docs/                    # Architecture documentation
```

## Make Commands

```
make help         Show all commands
make install      Install Python dependencies
make dev          Start development stack
make test         Run tests
make lint         Run linter
make demo         Start full demo stack
make train-rl     Train RL agent
make train-tft    Train TFT forecaster
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vision | YOLOv11, ByteTrack, OpenCV |
| Prediction | Temporal Fusion Transformer, PyTorch |
| Control | PPO (Stable Baselines3), SUMO |
| Backend | FastAPI, Redis, PostgreSQL |
| Frontend | React, Vite, Tailwind CSS |
| Mobile | React Native, Expo |
| Maps | MapmyIndia SDK |

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
DATABASE_URL=postgresql://fluxo:fluxo@localhost:5432/fluxo
REDIS_URL=redis://localhost:6379
YOLO_MODEL_PATH=models/yolo11n.pt
RL_MODEL_PATH=models/fluxo_rl_agent_v1.zip
MAP_API_KEY=your_mapmyindia_key
```

## Testing

```bash
make test
# or
pytest tests/ -v
```

## License

Hackathon project. Gridlock Hackathon 2.0

---

*FLUXO: Adaptive Urban Traffic Intelligence*
*Gridlock Hackathon 2.0. Flipkart x Bengaluru Traffic Police*
