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
| Mobile phone usage | Catches riders holding a phone to their ear while riding or driving. |
| No seat belt | Looks for the diagonal stripe of a seat belt on car occupants. |
| Overloading | Flags goods vehicles carrying more cargo than they should. |
| Fancy / hidden plates | Catches modified or obscured number plates that try to dodge cameras. |
| Missing mirrors | Flags two-wheelers without rear-view mirrors (legally required). |
| Number plate reading | Reads Indian plates — including non-standard fonts and low-quality CCTV footage. |
| Evidence clips | Saves only the few seconds around each violation, not the whole video. |

## How FLUXO compares to existing systems

Bengaluru already has ASTraM (Actionable Intelligence for Sustainable Traffic Management) and ITeMS (Intelligent Traffic Management System) — AI cameras at 75+ junctions that caught 87% of the city's 3 million violations in 2025. Karnataka's Transport Department is deploying more on highways. Commercial systems like Safira (Auriga), Recon (Stellarview), and Vehant's TrafficMon are also operational.

Here's where FLUXO goes beyond what's already deployed:

| Feature | ASTraM / ITeMS (current BTP) | Safira / Recon / Vehant | FLUXO |
|---------|------|------|------|
| Helmet detection | Yes (36% of violations) | Yes | Yes + **secondary headwear classifier** (turban/cap/scarf vs helmet — reduces false positives on Sikh riders) |
| Triple riding | Yes | Yes | Yes + **trapezium bounding boxes** (handles dense bike clusters where rectangles merge) |
| Red light jump | Yes | Yes | Yes |
| Wrong-way driving | Yes | Yes | Yes |
| Mobile phone usage | Yes | Yes | Yes |
| Seat belt | Yes (16% of violations) | Yes | Yes |
| Overloading | Yes (planned) | Partial | Yes |
| Over-speeding | Yes (radar-based) | Yes (radar + camera) | Partial (tracking-based, no radar) |
| ANPR | Yes | Yes (99% accuracy, <40ms) | Yes + **HSRP/non-HSRP validation** + state-code regex + plate type classification |
| Fancy / hidden plates | No | No | **Yes — novel class, no competitor addresses this** |
| Missing mirrors | Added in 2024 (13 violation types) | Yes (Recon) | Yes |
| E-challan generation | Yes (Parivahan/Vahan integration) | Yes (600+/day) | No (detection only, no challan backend) |
| Repeat offender tracking | Yes | Yes | No |
| Digital twin / predictive analytics | Yes (ASTraM) | No | No |
| Citizen reporting app | Yes (BTP ASTraM app, 7220 reports in 6 months) | No | No |
| **Quality enhancement for far objects** | No | No | **Yes — auto-upscaling + sharpening + tile detection** |
| **Temporal confidence boosting** | No | No | **Yes — accumulates evidence across frames for live feeds** |
| **Smart ROI selection** | No | No | **Yes — focuses processing on motion regions** |
| **VLM evidence narration** | No | No | **Yes — plain-English summaries for tickets** |
| **Event-triggered clips** | No | No | **Yes — saves ~80% storage** |
| Drone camera integration | Yes (10 drones) | No | No |

### What FLUXO does that nobody else does

1. **Fancy plate detection** — no existing Indian traffic system catches modified or obscured plates. This is a documented gap ([TR-TRVD paper, 2024](https://doi.org/10.5281/zenodo.13953874)).

2. **Cultural headwear awareness** — BTP's own data shows helmet detection is the top violation type (36%), but generic classifiers produce false positives on turbans and scarves. FLUXO's secondary classifier specifically addresses this ([Deshpande et al., Frontiers in AI, 2025](https://doi.org/10.3389/frai.2025.1582257)).

3. **Far/low-quality vehicle detection** — existing systems need clean, high-res camera feeds. FLUXO auto-detects bad footage and kicks in super-resolution + sharpening + tile-based detection to catch vehicles that other systems miss.

4. **Trapezium bounding boxes** — standard rectangular boxes merge when bikes are packed tight. FLUXO's trapezium approach follows the actual bike shape ([Goyal et al., CVPR 2022](https://arxiv.org/abs/2204.08364), [US Patent 12,315,264](https://patents.google.com/patent/US12315264B2)).

5. **Plain-English evidence narration** — instead of raw data logs, FLUXO generates human-readable summaries for each violation using a vision-language model.

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
│   │   ├── enhancement.py   # Quality analysis, SR, tiling, adaptive conf, temporal boost
│   │   ├── density.py    # Traffic density scoring
│   │   └── config.py     # Settings
│   └── violations/       # Each violation type
│       ├── helmet.py     # Helmet check (with headwear classifier)
│       ├── triple_riding.py  # Triple riding (trapezium boxes)
│       ├── signal_jump.py    # Red light jump
│       ├── wrong_way.py      # Wrong-way driving
│       ├── mobile_phone.py   # Phone usage while riding
│       ├── seatbelt.py       # Seat belt violation
│       ├── overloading.py    # Goods vehicle overloading
│       ├── fancy_plate.py    # Non-standard plate detection
│       ├── mirror.py         # Missing mirror detection
│       ├── anpr.py           # Number plate reading + validation
│       ├── vlm_evidence.py   # Plain-English violation summaries
│       └── detector.py       # Orchestrates everything
├── app.py                # Streamlit dashboard
├── scripts/              # Training scripts
├── tests/                # 75 tests
└── requirements.txt
```

## Running tests

```bash
pytest tests/ -v
```

## How the pipeline works

1. **Detect** — YOLO26 finds vehicles, riders, and plates in each frame
2. **Enhance** — If footage is blurry or low-res, auto-sharpen + 2x upscale + tile-based detection catches far vehicles
3. **Track** — ByteTrack follows each vehicle across frames to maintain identity
4. **Check violations** — Each tracker ID gets checked for helmets, triple riding, red-light jumping, wrong-way driving, phone usage, seat belt, overloading, fancy plates, and missing mirrors — all from the same detection pass
5. **Read the plate** — EasyOCR reads the number plate (with super-resolution for blurry frames)
6. **Save evidence** — Only the few seconds around the violation are saved as a clip

## Research behind the decisions

Every design choice was made to address a documented problem with the default approach:

- **Why YOLO26, not YOLOv11**: [YOLO26 eliminates NMS post-processing entirely](https://arxiv.org/abs/2601.12882), giving [43% faster CPU inference](https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai) and better INT8 quantization stability on edge hardware like Jetson Nano.

- **Why trapezium boxes for triple riding**: [Goyal et al. (CVPR 2022)](https://arxiv.org/abs/2204.08364) from IIIT Hyderabad showed that rectangular bounding boxes merge and split incorrectly in dense motorcycle traffic. Their trapezium representation follows the actual bike silhouette and reduces false positives. This work was [granted US Patent 12,315,264](https://patents.google.com/patent/US12315264B2) in May 2025.

- **Why a headwear classifier for helmets**: [Deshpande et al. (Frontiers in AI, 2025)](https://doi.org/10.3389/frai.2025.1582257) and [Deshpande et al. (Springer, 2025)](https://doi.org/10.1007/s44163-025-00263-3) documented that generic helmet classifiers confuse turbans, caps, and scarves with helmets in Indian traffic. A dedicated secondary classifier solves this.

- **Why EasyOCR over Tesseract**: [ResearchGate comparison study (2024)](https://www.researchgate.net/publication/378948224) showed EasyOCR outperforms Tesseract on Indian fonts and low-resolution CCTV captures — the exact conditions of real junction cameras.

- **Why fancy plate detection**: The [TR-TRVD paper (Nanotechnology Perceptions, 2024)](https://doi.org/10.5281/zenodo.13953874) noted that fancy number plate detection has "received little attention" — making this a genuine gap no competitor addresses.

- **Why missing mirror detection**: [arXiv:2511.12206 (IEEE, 2025)](https://arxiv.org/abs/2511.12206) introduced missing rear-view mirror detection as a novel violation class with mAP@50 = 0.843. It's legally enforceable under Indian motor vehicle rules.

- **Why mobile phone detection**: BTP ITeMS added mobile phone usage as one of 13 violation types in 2024 ([Hindustan Times, Sep 2024](https://www.hindustantimes.com/cities/bengaluru-news/aipowered-cameras-to-detect-13-types-of-violations-in-bengaluru-traffic-report-101727404715450.html)). Karnataka highway cameras specifically target this ([Indian Express, Dec 2024](https://indianexpress.com/article/cities/bangalore/ai-cameras-bengaluru-mysuru-highway-detect-violations-slap-fines-9723358/)).

- **Why seat belt detection**: BTP data shows seat belt violations are 16% of all automated detections ([Times of India, Oct 2025](https://timesofindia.indiatimes.com/city/bengaluru/87-of-traffic-violation-detection-on-bengaluru-roads-now-contactless/articleshow/124535743.cms)). The Bengaluru-Mysuru highway cameras specifically target this ([Indian Express, Dec 2024](https://indianexpress.com/article/cities/bangalore/ai-cameras-bengaluru-mysuru-highway-detect-violations-slap-fines-9723358/)).

- **Why overloading detection**: Karnataka Transport Department's AI cameras will target overloaded goods vehicles on highways ([The Hindu, Jun 2025](https://www.thehindu.com/news/national/karnataka/ai-powered-cameras-to-be-installed-on-karnataka-highways-to-curb-accidents-and-violations/article69745351.ece)).

- **Why quality enhancement for far objects**: Existing systems like ASTraM need clean, high-res feeds. FLUXO's adaptive pipeline auto-detects bad footage and kicks in super-resolution + sharpening + tile-based detection to catch vehicles other systems miss.

- **Why event-triggered clips**: [IJARCCE (Jan 2026)](https://ijarcce.com/wp-content/uploads/2026/01/IJARCCE.2026.15133.pdf) reported that intelligent frame selection reduces storage by 80% while maintaining evidentiary standards — saving ~650GB per camera per year.

## Tech stack

| Part | Tool |
|------|------|
| Finding vehicles | [YOLO26](https://arxiv.org/abs/2601.12882) (NMS-free, end-to-end) |
| Following vehicles across frames | [ByteTrack](https://arxiv.org/abs/2110.06864) / BoT-SORT |
| Reading number plates | [EasyOCR](https://github.com/JaidedAI/EasyOCR) (Indian fonts optimized) |
| Night vision | CLAHE preprocessing on LAB color space |
| Quality enhancement | ESPCNN super-resolution + adaptive tiling + temporal boosting |
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
