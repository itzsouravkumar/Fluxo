# FLUXO Demo Guide

## For Judges (8-10 minutes)

### 0:00-1:00: The Hook
Open dashboard live. Don't explain yet.

### 1:00-2:30: The Problem
Bengaluru loses ₹37,000 crore annually. ASTraM alerts every 15 minutes. FLUXO acts in under a second.

### 2:30-5:00: Live Demo
1. Show live density detection with PCE score
2. RL recommendation fires: extend green phase
3. Apply recommendation: show phase timer change
4. Show violation clip with bounding box and plate
5. Prediction strip shows CRITICAL incoming

### 5:00-6:30: Edge Cases
- Night mode (CLAHE preprocessing)
- Rain mode (augmented detection)
- Emergency vehicle override
- Starvation prevention

### 6:30-7:30: Numbers
- 27% reduction in average wait time
- < 1 second alert latency
- ₹2,400/junction/month at scale

### 7:30-8:30: Deployment
FLUXO runs on existing BTP CCTV. No new hardware. Propose 90-day shadow pilot.

## Running Demo Locally

```bash
# Generate demo data
python scripts/generate_demo_data.py

# Start demo stack
make demo
```

Dashboard loads with pre-populated junction data and violation feed.
