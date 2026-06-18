# FLUXO

**Spots traffic violations on CCTV footage automatically — so officers don't have to watch every feed by hand.**

Built for Gridlock Hackathon 2.0 (Flipkart x Bengaluru Traffic Police) — Theme 3.

BTP ran a 2-day manual enforcement drive and booked 573 violations by hand. FLUXO automates that: detect, classify, read the plate, save the evidence — at CCTV scale, in real time.

---

## What it does

| Violation | How it works |
|-----------|-------------|
| No helmet | Looks at the rider's head. Knows the difference between a helmet and a turban, cap, or scarf — things that fool most other systems. |
| Triple riding | Counts people on a two-wheeler using a smarter shape than a simple rectangle, so it doesn't miscount when bikes are close together. |
| Red light jump | Watches for vehicles crossing the stop line while the signal is red. |
| Wrong-way driving | Spots vehicles moving against the expected lane direction. |
| Fancy / hidden plates | Catches modified or obscured number plates that try to dodge cameras. |
| Missing mirrors | Flags two-wheelers without rear-view mirrors (legally required). |
| Number plate reading | Reads Indian plates — including non-standard fonts and low-quality CCTV footage. |
| Evidence clips | Saves only the few seconds around each violation, not the whole video. |

## How it's different

Most traffic AI systems are built for Western roads. Indian traffic has auto-rickshaws, lane-splitting bikes, non-standard plates, and headwear that looks like helmets to a basic camera. FLUXO was built for these conditions from the start.

| What most teams do | What FLUXO does instead |
|---|---|
| Use YOLOv11 (from last year's tutorials) | Uses YOLO26 — faster, works better on cheaper hardware |
| Count heads in a box to detect triple riding | Uses a trapezium-shaped boundary that follows the actual bike shape |
| Treat every helmet-looking object as a helmet | Has a second check that tells helmets apart from turbans, caps, and scarves |
| Run a separate model for each violation | One detection pass, one vehicle ID, all checks happen together |
| Store the whole video clip | Only saves the seconds around the actual violation |
| Use Tesseract for plate reading | Uses EasyOCR, which works better on Indian plates |

## Quick start

```bash
# Clone the repo
git clone https://github.com/itzsouravkumar/Fluxo.git
cd Fluxo

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project structure

```
Fluxo/
├── core/
│   ├── vision/           # Vehicle detection + tracking
│   │   ├── detector.py   # YOLO26 wrapper
│   │   ├── tracker.py    # ByteTrack / BoT-SORT
│   │   ├── preprocessor.py  # Night mode (CLAHE)
│   │   ├── density.py    # Traffic density scoring
│   │   └── config.py     # Settings
│   └── violations/       # Each violation type
│       ├── helmet.py     # Helmet check (with headwear classifier)
│       ├── triple_riding.py  # Triple riding (trapezium boxes)
│       ├── signal_jump.py    # Red light jump
│       ├── wrong_way.py      # Wrong-way driving
│       ├── fancy_plate.py    # Non-standard plate detection
│       ├── mirror.py         # Missing mirror detection
│       ├── anpr.py           # Number plate reading + validation
│       ├── vlm_evidence.py   # Plain-English violation summaries
│       └── detector.py       # Orchestrates everything
├── app.py                # Streamlit dashboard
├── scripts/              # Training scripts
├── tests/                # 39 tests
└── requirements.txt
```

## Running tests

```bash
pytest tests/ -v
```

## How the pipeline works

1. **Detect** — YOLO26 finds vehicles, riders, and plates in each frame
2. **Track** — ByteTrack follows each vehicle across frames so we know it's the same one
3. **Check violations** — Each tracker ID gets checked for helmets, triple riding, red-light jumping, wrong-way driving, fancy plates, and missing mirrors — all from the same detection pass
4. **Read the plate** — EasyOCR reads the number plate (with super-resolution for blurry frames)
5. **Save evidence** — Only the few seconds around the violation are saved as a clip

## Research behind the decisions

Every design choice was made because we found a documented problem with the default approach:

- **Why YOLO26, not YOLOv11**: [YOLO26 eliminates NMS post-processing entirely](https://arxiv.org/abs/2601.12882), giving [43% faster CPU inference](https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai) and better INT8 quantization stability on edge hardware like Jetson Nano.

- **Why trapezium boxes for triple riding**: [Goyal et al. (CVPR 2022)](https://arxiv.org/abs/2204.08364) from IIIT Hyderabad showed that rectangular bounding boxes merge and split incorrectly in dense motorcycle traffic. Their trapezium representation follows the actual bike silhouette and reduces false positives. This work was [granted US Patent 12,315,264](https://patents.google.com/patent/US12315264B2) in May 2025.

- **Why a headwear classifier for helmets**: [Deshpande et al. (Frontiers in AI, 2025)](https://doi.org/10.3389/frai.2025.1582257) and [Deshpande et al. (Springer, 2025)](https://doi.org/10.1007/s44163-025-00263-3) documented that generic helmet classifiers confuse turbans, caps, and scarves with helmets in Indian traffic. A dedicated secondary classifier solves this.

- **Why EasyOCR over Tesseract**: [ResearchGate comparison study (2024)](https://www.researchgate.net/publication/378948224) showed EasyOCR outperforms Tesseract on Indian fonts and low-resolution CCTV captures — the exact conditions of real junction cameras.

- **Why fancy plate detection**: The [TR-TRVD paper (Nanotechnology Perceptions, 2024)](https://doi.org/10.5281/zenodo.13953874) noted that fancy number plate detection has "received little attention" — making this a genuine gap no competitor addresses.

- **Why missing mirror detection**: [arXiv:2511.12206 (IEEE, 2025)](https://arxiv.org/abs/2511.12206) introduced missing rear-view mirror detection as a novel violation class with mAP@50 = 0.843. It's legally enforceable under Indian motor vehicle rules.

- **Why event-triggered clips**: [IJARCCE (Jan 2026)](https://ijarcce.com/wp-content/uploads/2026/01/IJARCCE.2026.15133.pdf) reported that intelligent frame selection reduces storage by 80% while maintaining evidentiary standards — saving ~650GB per camera per year.

## Tech stack

| Part | Tool |
|------|------|
| Finding vehicles | [YOLO26](https://arxiv.org/abs/2601.12882) (NMS-free, end-to-end) |
| Following vehicles across frames | [ByteTrack](https://arxiv.org/abs/2110.06864) / BoT-SORT |
| Reading number plates | [EasyOCR](https://github.com/JaidedAI/EasyOCR) (Indian fonts optimized) |
| Night vision | CLAHE preprocessing on LAB color space |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Training | [Ultralytics](https://docs.ultralytics.com/) |

## Datasets

| Dataset | Used for | Source |
|---------|----------|--------|
| India Driving Dataset (IDD) | Vehicle detection fine-tuning | [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in/) |
| Helmet Detection (Kaggle) | Helmet classifier training | [kaggle.com](https://www.kaggle.com/datasets) |
| Indian License Plate | ANPR model training | [kaggle.com](https://www.kaggle.com/datasets) |

## License

Hackathon project. Gridlock Hackathon 2.0.

---

*FLUXO — Built for Bengaluru Traffic Police, June 2026*
