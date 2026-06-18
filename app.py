#!/usr/bin/env python3
"""FLUXO Streamlit Dashboard — Traffic Violation Detection Demo.

Run: streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from datetime import datetime

st.set_page_config(
    page_title="FLUXO — Traffic Violation Detection",
    page_icon="🚨",
    layout="wide",
)

st.markdown("""
<style>
    .stMetric { background: #1a1a2e; padding: 12px; border-radius: 8px; }
    .violation-card {
        background: #1a1a2e; padding: 12px; border-radius: 8px;
        border-left: 4px solid #ff4444; margin-bottom: 8px;
    }
    .safe-card {
        background: #1a1a2e; padding: 12px; border-radius: 8px;
        border-left: 4px solid #44ff44; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    from core.vision.detector import FluxoDetector
    from core.violations import ViolationDetector, ViolationConfig

    detector = FluxoDetector(model_path="yolo26n.pt")
    vconfig = ViolationConfig(
        enable_signal_jump=True,
        enable_helmet=True,
        enable_wrong_way=True,
        enable_triple_riding=True,
        enable_anpr=True,
        enable_clip_extract=False,
    )
    violation_engine = ViolationDetector(vconfig)
    return detector, violation_engine


def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    lightness = clahe.apply(lightness)
    lab = cv2.merge([lightness, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def annotate_frame(frame, tracked, violations, density_score, frame_idx, signal_state):
    from core.vision.config import VEHICLE_CLASSES

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for i in range(len(tracked)):
        bbox = tracked.xyxy[i].astype(int)
        cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
        conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
        tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1

        vclass = VEHICLE_CLASSES.get(cls_id)
        if vclass is None:
            continue

        color = vclass.color
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = f"#{tid} {vclass.name} {conf:.0%}"
        cv2.putText(annotated, label, (bbox[0], bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for v in violations:
        vb = v.bbox
        if len(vb) == 4:
            x1, y1, x2, y2 = int(vb[0]), int(vb[1]), int(vb[2]), int(vb[3])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            vlabel = v.type.value.upper()
            if v.plate_number:
                vlabel += f" [{v.plate_number}]"
            cv2.putText(annotated, vlabel, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    bar_w = int(density_score * 200)
    cv2.rectangle(annotated, (10, h - 60), (210, h - 40), (50, 50, 50), -1)
    bar_color = (0, int(255 * (1 - density_score)), int(255 * density_score))
    cv2.rectangle(annotated, (10, h - 60), (10 + bar_w, h - 40), bar_color, -1)
    cv2.putText(annotated, f"Density: {density_score:.3f}", (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def process_video(source_path, detector, violation_engine, signal_state, night_mode, max_frames=None):
    import supervision as sv
    from core.vision.config import VEHICLE_CLASSES, DEFAULT_CONFIG

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {source_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"Resolution: {width}x{height} | FPS: {fps:.0f} | Frames: {total_frames}")

    tracker = sv.ByteTrack()
    progress = st.progress(0)
    status_text = st.empty()
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    frame_placeholder = st.empty()
    violations_log = []

    frame_count = 0
    start_time = time.time()
    density_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if night_mode:
            frame = apply_clahe(frame)

        results = detector._load_model()(frame, conf=DEFAULT_CONFIG.detection.confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_mask = np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))
        vehicle_dets = detections[vehicle_mask]

        if len(vehicle_dets) > 0:
            tracked = tracker.update_with_detections(vehicle_dets)
        else:
            tracked = vehicle_dets

        violations = []
        violation_engine.feed_frame(frame)
        violations = violation_engine.check(tracked, frame, frame_count, signal_state)

        total_pce = sum(
            VEHICLE_CLASSES.get(int(tracked.class_id[i]), None).pce
            for i in range(len(tracked))
            if tracked.class_id is not None and int(tracked.class_id[i]) in VEHICLE_CLASSES
        )
        density_score = min(total_pce / ((width * height) / 1000.0), 1.0)
        density_scores.append(density_score)
        avg_density = np.mean(density_scores[-30:]) if density_scores else 0

        for v in violations:
            violations_log.append({
                "frame": frame_count,
                "type": v.type.value,
                "track_id": v.track_id,
                "confidence": round(v.confidence, 3),
                "plate": v.plate_number,
                "timestamp": datetime.now().isoformat(),
            })

        annotated = annotate_frame(frame, tracked, violations, density_score, frame_count, signal_state)
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        with metrics_col1:
            st.metric("Frame", frame_count)
        with metrics_col2:
            st.metric("Vehicles", len(tracked))
        with metrics_col3:
            st.metric("Violations", len(violations))
        with metrics_col4:
            level = DEFAULT_CONFIG.density_levels.get_level(avg_density)
            st.metric("Density", f"{avg_density:.3f}", delta=level)

        progress.progress(min(frame_count / max(total_frames, 1), 1.0))
        elapsed = time.time() - start_time
        status_text.text(f"Processing: {frame_count}/{total_frames} | {frame_count/max(elapsed, 0.01):.1f} FPS | Violations: {len(violations_log)}")

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    progress.progress(1.0)

    total_time = time.time() - start_time
    st.success(f"Processed {frame_count} frames in {total_time:.1f}s ({frame_count/max(total_time, 0.01):.1f} FPS)")

    return violations_log


def main():
    st.title("🚨 FLUXO — Traffic Violation Detection")
    st.caption("Automated Photo Identification and Classification for Traffic Violations Using Computer Vision")

    with st.sidebar:
        st.header("Configuration")
        signal_state = st.selectbox("Signal State", ["GREEN", "RED", "YELLOW"], index=0)
        night_mode = st.checkbox("Enable Night Mode (CLAHE)", value=False)
        max_frames = st.number_input("Max Frames (0 = all)", min_value=0, value=0, step=100)
        st.divider()
        st.subheader("Pipeline Info")
        st.markdown("""
        - **Detector**: YOLO26 (NMS-free)
        - **Tracker**: ByteTrack
        - **OCR**: EasyOCR
        - **Helmet**: YOLO cls + headwear classifier
        - **Triple-ride**: Trapezium boxes
        - **ANPR**: EasyOCR (Indian fonts)
        """)

    tab_video, tab_live, tab_results = st.tabs(["Video Analysis", "Live Camera", "Results"])

    with tab_video:
        uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
        sample_option = st.radio("Or use sample:", ["None", "Download sample"], horizontal=True)

        source_path = None
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.close()
            source_path = tmp.name
            st.video(source_path)
        elif sample_option == "Download sample":
            sample_url = "https://github.com/ultralytics/assets/raw/main/videos/demo.mp4"
            st.info(f"Download sample from: {sample_url}")

        if source_path is not None:
            if st.button(" Run Detection", type="primary", use_container_width=True):
                detector, violation_engine = load_models()
                violations_log = process_video(
                    source_path, detector, violation_engine,
                    signal_state, night_mode,
                    max_frames=max_frames if max_frames > 0 else None,
                )
                if violations_log:
                    st.divider()
                    st.subheader("Violations Detected")
                    for v in violations_log:
                        card_class = "violation-card" if v["type"] != "safe" else "safe-card"
                        plate_info = f" | Plate: `{v['plate']}`" if v.get("plate") else ""
                        st.markdown(
                            f'<div class="{card_class}">'
                            f'<b>{v["type"].upper()}</b> | Track #{v["track_id"]} | '
                            f'Conf: {v["confidence"]:.0%}{plate_info} | Frame: {v["frame"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    with tab_live:
        camera_source = st.number_input("Camera index", min_value=0, value=0, step=1)
        if st.button(" Start Live Detection", type="primary", use_container_width=True):
            detector, violation_engine = load_models()
            process_video(
                camera_source, detector, violation_engine,
                signal_state, night_mode,
                max_frames=300,
            )

    with tab_results:
        st.subheader("Architecture Overview")
        st.markdown("""
        | Component | Technology | Why |
        |-----------|-----------|-----|
        | Detector | **YOLO26** (NMS-free) | 43% faster CPU inference, INT8-quantization stable |
        | Tracker | **ByteTrack** | Stable per-vehicle IDs across frames |
        | Helmet | **YOLO cls + headwear classifier** | Avoids false-positives on caps/turbans/scarves |
        | Triple-riding | **Trapezium boxes** | Reduces false-positives from dense motorcycle clusters |
        | Signal Jump | **Stop-line crossing** | Homography-projected, red-phase detection |
        | Wrong Way | **Velocity vector** | 5+ consecutive frames against lane direction |
        | ANPR | **EasyOCR** | Stronger on Indian fonts + low-res CCTV plates |
        | Evidence | **Event-triggered clips** | 80% storage reduction vs continuous recording |
        """)
        st.subheader("Violation Summary")
        st.info("Run a video analysis to see results here.")


if __name__ == "__main__":
    main()
