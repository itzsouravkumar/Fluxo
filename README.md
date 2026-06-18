# FLUXO

**Automated Photo Identification and Classification for Traffic Violations Using Computer Vision**

Built for Gridlock Hackathon 2.0 (Flipkart x Bengaluru Traffic Police) — Theme 3.

BTP ran a 2-day manual enforcement drive and booked 573 violations by hand. FLUXO automates that entirely: detect, classify, identify, and generate evidence — at CCTV scale, in real time, with zero added manpower.

---

## Architecture Decisions

| Default approach | Documented failure | FLUXO's fix |
|---|---|---|
| YOLOv11 backbone | One generation behind; weaker INT8 quantization stability | **YOLO26** (NMS-free, 43% faster CPU inference) |
| Separate model per violation | Redundant inference, no shared tracking context | **Single tracked-ID drives all classifiers** |
| Person-count for triple-riding | Box merging/splitting on dense motorcycle clusters | **Trapezium rider boxes + amodal regression** |
| Generic helmet classifier | Confuses caps/turbans/hijabs with helmets; driver/passenger bleed-through | **Secondary headwear classifier + per-seat assignment** |
| Tesseract OCR | Weak on Indian fonts, low-res CCTV plates | **EasyOCR/PaddleOCR** |
| Continuous video storage | Unsustainable storage cost at scale | **Event-triggered clip extraction** |

## What FLUXO Does

### Core: Violation Detection & Classification (Theme 3)

1. **Vehicle & Rider Detection** — YOLO26, fine-tuned on Indian road conditions (auto-rickshaw, two-wheeler, bus, truck, LMV, pedestrian), with ByteTrack for multi-object tracking across frames
2. **Helmet Compliance** — Secondary headwear classifier (helmet vs. cap/turban/scarf) + per-seat-position helmet assignment using trapezium rider boxes
3. **Signal Violation** — Stop-line crossing detection during red phase
4. **Wrong-Way Driving** — Velocity vector vs. expected lane-flow direction, flagged after consecutive frames of mismatch
5. **Triple Riding** — Trapezium bounding box representation + amodal regression for occluded riders
6. **Automated Number Plate Recognition (ANPR)** — YOLO26 plate localization + EasyOCR (chosen for stronger performance on Indian fonts and low-resolution plates over Tesseract)
7. **Auto-Generated Evidence** — Event-triggered clip extraction (pre/post-violation buffer only, 80% storage reduction)

## Quick Start

```bash
# Clone
git clone https://github.com/itzsouravkumar/Fluxo.git
cd Fluxo

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app.py

# Or run vision pipeline demo
python scripts/demo_vision.py --source video.mp4 --show --violations
```

Dashboard: http://localhost:8501

## Project Structure

```
Fluxo/
├── core/                    # Python backend
│   ├── vision/              # YOLO26 detection + ByteTrack tracking
│   ├── violations/          # Helmet, signal-jump, wrong-way, triple-riding detectors
│   └── config.py            # Global settings
├── app.py                   # Streamlit dashboard
├── scripts/                 # Training & demo scripts
├── tests/                   # Test suite
├── models/                  # Trained model weights
└── data/                    # Datasets (gitignored)
```

## Make Commands

```
make help         Show all commands
make install      Install Python dependencies
make dev          Start Streamlit dashboard
make test         Run tests
make lint         Run linter
make demo         Run full demo pipeline
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection & Tracking | YOLO26 (NMS-free), ByteTrack, OpenCV |
| Plate Recognition | YOLO26 + EasyOCR (Indian fonts, low-res CCTV) |
| Low-Light Robustness | CLAHE preprocessing |
| Helmet Detection | YOLO classifier + headwear discriminator |
| Triple-Riding | Trapezium bounding boxes + amodal regression |
| Dashboard | Streamlit |

## Data Setup

Raw datasets are not committed to this repo (too large for git). See `data/README.md` for download sources and placement instructions:

- Vehicle/rider detection: India Driving Dataset (IDD Detection)
- Helmet compliance: Kaggle helmet detection dataset

## Testing

```bash
make test
# or
pytest tests/ -v
```

---

*FLUXO: Automated Photo Identification and Classification for Traffic Violations*
*Gridlock Hackathon 2.0. Flipkart x Bengaluru Traffic Police*
