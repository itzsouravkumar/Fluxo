## Data Setup

FLUXO requires external datasets that are not included in this repository due to size. Follow the instructions below to download and place them correctly.

### Directory Structure

```
data/
├── raw/
│   ├── idd_detection/        # Indian Driving Dataset (detection)
│   ├── helmet_kaggle/        # Helmet detection dataset
│   └── bengaluru_traffic.csv # BTP Round 1 traffic time series
├── processed/
│   ├── yolo_labels/          # Converted YOLO .txt labels
│   └── data.yaml             # YOLO training config
└── sumo/                     # SUMO network files (generated)
```

### 1. IDD (Indian Driving Dataset)

Source: https://idd.insaan.iiit.ac.in/

- Download the detection dataset
- Extract to `data/raw/idd_detection/`
- ~22 GB compressed

### 2. Helmet Detection Dataset

Source: Kaggle (search "helmet detection dataset")

- Download and extract to `data/raw/helmet_kaggle/`
- Used for binary helmet/no-helmet classification

### 3. BTP Round 1 Traffic Data

Source: Provided by hackathon organizers

- Place CSV at `data/raw/bengaluru_traffic.csv`
- 48-day Bengaluru traffic time series
- Used for TFT model training

### 4. SUMO Network (Optional, for RL training)

- Install SUMO: `apt install sumo sumo-tools`
- Run: `python /usr/share/sumo/tools/osmWebWizard.py`
- Select Veerannapalya area, download network
- Place files in `data/sumo/`

### 5. Process Data

After downloading raw datasets:

```bash
python scripts/prepare_data.py
```

This converts IDD annotations to YOLO format and generates `data/processed/data.yaml`.
