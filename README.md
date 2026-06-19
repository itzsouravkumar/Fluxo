---
title: FLUXO
emoji: ""
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: true
license: mit
---

# FLUXO

Automated traffic violation detection on CCTV footage. Built for Gridlock Hackathon 2.0 (Flipkart x Bengaluru Traffic Police), Theme 3.

BTP ran a 2-day manual enforcement drive and booked 573 violations. That's the upper bound of what manual review can do. I built FLUXO to automate it: detect vehicles, classify violations, read the plate, capture evidence with annotated frame snapshots, and generate a downloadable report.

---

## What it detects

| Violation | How |
|-----------|-----|
| No helmet | Secondary headwear classifier distinguishes helmets from turbans, caps, scarves |
| Triple riding | Trapezium bounding boxes instead of rectangles, handles dense bike clusters |
| Red light jump | Stop-line crossing while signal is red |
| Junction blocking | Flags vehicles trapped in the intersection during critical congestion (Gridlock) |
| Wrong-way driving | Tracks vehicle direction against expected lane flow |
| Mobile phone usage | Detects phone held to ear while riding/driving |
| No seat belt | Looks for diagonal seat belt stripe on car occupants |
| Overloading | Flags overloaded goods vehicles |
| Fancy/hidden plates | Novel class - no existing Indian system catches modified plates |
| Missing mirrors | Flags two-wheelers without rear-view mirrors |
| Number plate reading | EasyOCR with HSRP validation + state-code regex |
| Evidence capture | Annotated frame snapshots with red/green borders + HTML report |

## Technical differentiation

I studied what BTP's ASTraM/ITeMS and commercial systems (Safira, Recon, Vehant TrafficMon) actually do. These are AI cameras deployed at 75+ Bengaluru junctions that caught 87% of the city's 3M violations in 2025. They work. But they have gaps.

FLUXO addresses those gaps and introduces next-gen architecture for massive scale deployments:

### Automatic Lane Graph Builder (Architecture Concept)
Real-world traffic cameras require technicians to manually configure "North lane", "South lane", and wrong-way zones. FLUXO's architecture is designed to support **Unsupervised Trajectory Clustering** to automatically learn traffic flow directions and build a road topology graph. This means zero manual configuration per camera.

| Gap in deployed systems | FLUXO's approach |
|---|---|
| No fancy plate detection | Trained on modified/obscured plates as a novel violation class. [TR-TRVD, 2024](https://doi.org/10.5281/zenodo.13953874) documents this as an unaddressed gap. |
| Generic helmet classifiers produce false positives on turbans/scarves | Secondary classifier specifically trained for Indian headwear. Based on [Deshpande et al., Frontiers in AI, 2025](https://doi.org/10.3389/frai.2025.1582257). |
| Rectangular bboxes merge in dense bike traffic | Trapezium bounds that follow the motorcycle profile. [Goyal et al., CVPR 2022](https://arxiv.org/abs/2204.08364), [US Patent 12,315,264](https://patents.google.com/patent/US12315264B2). |
| Require clean, high-res camera feeds | Adaptive enhancement pipeline: auto-detect bad footage, apply super-resolution + sharpening + tile-based detection for far objects. |
| No evidence frame capture with visual highlighting | Annotated snapshots per violation: red border on violator, green on context vehicles, downloadable HTML report. |
| Raw data logs, no human-readable output | VLM generates plain-English evidence summaries for each violation. |
| Full video storage per violation | Event-triggered clips save only the seconds around the infraction. ~80% storage reduction ([IJARCCE, Jan 2026](https://ijarcce.com/wp-content/uploads/2026/01/IJARCCE.2026.15133.pdf)). |

### What BTP/ASTraM has that FLUXO doesn't

Honest gaps - FLUXO is a detection engine, not a full enforcement stack:

- E-challan generation (Parivahan/Vahan integration)
- Repeat offender tracking
- Digital twin / predictive analytics
- Citizen reporting app
- Drone camera feeds
- Radar-based speed detection (FLUXO uses tracking-based estimation)

## Evidence reports

When FLUXO detects a violation, it captures an annotated frame snapshot:

- **Red border** around the violating vehicle with violation type, plate number, and confidence
- **Green border** around all other tracked vehicles for spatial context
- Frame number and timestamp watermarked

After processing, you can:
- See each violation with its evidence frame inline in the dashboard
- Download a self-contained HTML evidence report with all annotated snapshots, plate numbers, timestamps, and VLM narrations (if enabled)
- Print the HTML report to PDF from any browser

## Quick start

```bash
git clone https://github.com/itzsouravkumar/Fluxo.git
cd Fluxo
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Project structure

```
Fluxo/
├── core/
│   ├── vision/
│   │   ├── detector.py      # YOLO26 wrapper + enhancement pipeline
│   │   ├── tracker.py       # ByteTrack / BoT-SORT
│   │   ├── preprocessor.py  # Night mode (CLAHE)
│   │   ├── enhancement.py   # Quality analysis, SR, tiling, adaptive conf, temporal boost
│   │   ├── density.py       # Traffic density scoring
│   │   └── config.py        # All tunable parameters
│   └── violations/
│       ├── helmet.py           # Helmet check + headwear classifier
│       ├── triple_riding.py    # Trapezium boxes
│       ├── signal_jump.py      # Red light jump
│       ├── wrong_way.py        # Wrong-way driving
│       ├── mobile_phone.py     # Phone usage
│       ├── seatbelt.py         # Seat belt
│       ├── overloading.py      # Goods vehicle overloading
│       ├── fancy_plate.py      # Non-standard plate detection
│       ├── mirror.py           # Missing mirror
│       ├── anpr.py             # Number plate reading + HSRP validation
│       ├── evidence_report.py  # Annotated frame capture + HTML report generation
│       ├── vlm_evidence.py     # Plain-English violation summaries
│       ├── clip_extractor.py   # Event-triggered clip saving
│       └── detector.py         # Orchestrates all violation checks
├── app.py                # Streamlit dashboard
├── scripts/              # Training scripts
├── tests/                # 75 tests
└── requirements.txt
```

## Pipeline

1. **Detect** - YOLO26 finds vehicles, riders, plates in each frame
2. **Enhance** - If footage is blurry/low-res: auto-sharpen + 2x upscale + tile-based detection
3. **Track** - ByteTrack assigns persistent IDs across frames
4. **Check violations** - All violation classifiers run from the same detection pass
5. **Read plate** - EasyOCR with super-resolution preprocessing for blurry frames
6. **Capture evidence** - Annotated frame snapshot per violation (red/green borders)
7. **Save clip** - Event-triggered clip extraction (seconds around the violation)

## Research basis

Every design choice addresses a documented problem:

- **YOLO26 over YOLOv11**: [Eliminates NMS entirely](https://arxiv.org/abs/2601.12882), [43% faster CPU inference](https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai), better INT8 quantization on edge hardware.

- **Trapezium boxes for triple riding**: Rectangular bboxes merge in dense motorcycle traffic. [Goyal et al. (CVPR 2022)](https://arxiv.org/abs/2204.08364), IIIT Hyderabad. [US Patent 12,315,264](https://patents.google.com/patent/US12315264B2), May 2025.

- **Headwear classifier**: Generic helmet classifiers confuse turbans/caps/scarves with helmets. [Deshpande et al. (Frontiers in AI, 2025)](https://doi.org/10.3389/frai.2025.1582257), [Deshpande et al. (Springer, 2025)](https://doi.org/10.1007/s44163-025-00263-3).

- **EasyOCR over Tesseract**: [ResearchGate comparison (2024)](https://www.researchgate.net/publication/378948224) - EasyOCR outperforms Tesseract on Indian fonts and low-res CCTV.

- **Fancy plate detection**: [TR-TRVD paper (2024)](https://doi.org/10.5281/zenodo.13953874) notes this has "received little attention" - genuine gap.

- **Missing mirror detection**: [arXiv:2511.12206 (IEEE, 2025)](https://arxiv.org/abs/2511.12206), mAP@50 = 0.843. Legally enforceable under Indian MV rules.

- **Mobile phone detection**: BTP ITeMS added this in 2024 ([Hindustan Times, Sep 2024](https://www.hindustantimes.com/cities/bengaluru-news/aipowered-cameras-to-detect-13-types-of-violations-in-bengaluru-traffic-report-101727404715450.html)). Karnataka highway cameras target it ([Indian Express, Dec 2024](https://indianexpress.com/article/cities/bangalore/ai-cameras-bengaluru-mysuru-highway-detect-violations-slap-fines-9723358/)).

- **Seat belt detection**: 16% of BTP's automated detections ([Times of India, Oct 2025](https://timesofindia.indiatimes.com/city/bengaluru/87-of-traffic-violation-detection-on-bengaluru-roads-now-contactless/articleshow/124535743.cms)).

- **Overloading detection**: Karnataka Transport Dept deploying AI cameras for this on highways ([The Hindu, Jun 2025](https://www.thehindu.com/news/national/karnataka/ai-powered-cameras-to-be-installed-on-karnataka-highways-to-curb-accidents-and-violations/article69745351.ece)).

- **Event-triggered clips**: [IJARCCE (Jan 2026)](https://ijarcce.com/wp-content/uploads/2026/01/IJARCCE.2026.15133.pdf) - 80% storage reduction, ~650GB saved per camera per year.

## Tech stack

| Component | Tool |
|-----------|------|
| Detection | [YOLO26](https://arxiv.org/abs/2601.12882) (NMS-free) |
| Tracking | [ByteTrack](https://arxiv.org/abs/2110.06864) / BoT-SORT |
| Plate reading | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Night vision | CLAHE on LAB color space |
| Quality enhancement | ESPCNN super-resolution + adaptive tiling + temporal boosting |
| VLM narration | [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Weather preprocessing | Dark-channel dehazing + rain streak removal |
| Evidence reports | OpenCV annotation + self-contained HTML generation |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Training | [Ultralytics](https://docs.ultralytics.com/) |

## Architecture Strategy: Why zero-shot?

I deliberately chose **not** to train a monolithic foundation model from scratch. Building an end-to-end model for 10+ Indian traffic violations requires millions of annotated frames and massive compute, which is unfeasible for a hackathon.

Instead, FLUXO uses a **Zero-Shot / Composable Pipeline** approach:
1. **Pre-trained Foundation Models**: I use state-of-the-art pre-trained weights (YOLO26 for raw object detection, EasyOCR for text) to handle the generic tasks of "find a vehicle" or "read text".
2. **Domain-Specific Logic**: The actual intelligence sits in the pipeline logic (the "Zero-Shot" part). For example, FLUXO doesn't need to be trained on "triple riding" — it uses mathematical heuristics (trapezium bounds) on top of the generic YOLO person detections.
3. **Temporal Tracking**: By accumulating evidence across 300+ frames using ByteTrack, FLUXO achieves high accuracy without needing a custom-trained video transformer.

### FLUXO vs Large Vision Models (like Claude/GPT-4V)
Why not just send the CCTV feed to Claude?
- **Speed & Cost**: VLMs take seconds per frame and cost thousands of dollars to run continuously. FLUXO runs at 30+ FPS locally on edge hardware.
- **Temporal consistency**: VLMs process frames in isolation and struggle to maintain the identity of a specific bike across a crowded junction. FLUXO's dedicated tracking layer (ByteTrack) assigns and holds a persistent ID.

## Tests

```bash
pytest tests/ -v
```

## License

Hackathon project. Gridlock Hackathon 2.0.

---

*FLUXO - Built for Bengaluru Traffic Police, June 2026*
