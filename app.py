#!/usr/bin/env python3
"""FLUXO Streamlit Dashboard - Traffic Violation Detection Demo.

Run: streamlit run app.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import cv2
import tempfile
import time
from datetime import datetime

st.set_page_config(
    page_title="FLUXO",
    page_icon="🚨",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 300;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }

    .stMetric {
        background: #1a1a2e; padding: 12px; border-radius: 8px;
        font-family: 'Poppins', sans-serif !important;
    }
    .stMetric label { font-weight: 300 !important; }
    .stMetric [data-testid="stMetricValue"] { font-weight: 500 !important; }

    .violation-card {
        background: #1a1a2e; padding: 12px; border-radius: 8px;
        border-left: 4px solid #ff4444; margin-bottom: 8px;
        font-family: 'Poppins', sans-serif; font-weight: 300;
    }
    .narration-card {
        background: #0d1b2a; padding: 10px; border-radius: 6px;
        border-left: 4px solid #2e86c1; margin-top: 4px;
        font-family: 'Poppins', sans-serif; font-weight: 300;
        font-style: italic; color: #a0c4e8;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }
    .stSelectbox label, .stCheckbox label, .stNumberInput label, .stRadio label {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 300 !important;
    }
    .stButton button {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }
    .stMarkdown { font-family: 'Poppins', sans-serif; font-weight: 300; }
    .stCaption { font-family: 'Poppins', sans-serif; font-weight: 300; }
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
        enable_fancy_plate=True,
        enable_missing_mirror=True,
        enable_anpr=True,
        enable_clip_extract=False,
        enable_vlm_narration=False,
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
            vlabel = v.type.value.upper().replace("_", " ")
            if v.plate_number:
                vlabel += f" [{v.plate_number}]"
            cv2.putText(annotated, vlabel, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    bar_w = int(density_score * 200)
    cv2.rectangle(annotated, (10, h - 60), (210, h - 40), (50, 50, 50), -1)
    bar_color = (0, int(255 * (1 - density_score)), int(255 * density_score))
    cv2.rectangle(annotated, (10, h - 60), (10 + bar_w, h - 40), bar_color, -1)
    cv2.putText(annotated, f"Density: {density_score:.3f}", (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def process_video(source_path, detector, violation_engine, signal_state, night_mode, max_frames=None, enhance_quality=False):
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

    enhancer = None
    tile_det = None
    if enhance_quality:
        from core.vision.enhancement import FrameEnhancer, TileDetector
        enhancer = FrameEnhancer()
        tile_det = TileDetector(tile_size=640, overlap=0.2)

    tracker = sv.ByteTrack()
    progress = st.progress(0)
    status_text = st.empty()
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    if enhance_quality:
        metrics_col5 = st.columns(1)[0]
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

        if enhance_quality:
            detections_ultralytics, quality = detector.detect_with_enhancement(
                frame, enhancer=enhancer, tile_detector=tile_det,
            )
            dets_for_sv = sv.Detections(
                xyxy=np.array([d.bbox for d in detections_ultralytics]) if detections_ultralytics else np.empty((0, 4)),
                confidence=np.array([d.confidence for d in detections_ultralytics]) if detections_ultralytics else np.empty(0),
                class_id=np.array([d.class_id for d in detections_ultralytics]) if detections_ultralytics else np.empty(0, dtype=int),
            )
            with metrics_col5:
                st.metric("Quality", f"{quality['overall']:.2f}",
                          delta="Enhanced" if quality["needs_enhancement"] else "OK")
        else:
            results = detector._load_model()(frame, conf=DEFAULT_CONFIG.detection.confidence, verbose=False)[0]
            dets_for_sv = sv.Detections.from_ultralytics(results)

        vehicle_mask = np.isin(dets_for_sv.class_id, list(VEHICLE_CLASSES.keys()))
        vehicle_dets = dets_for_sv[vehicle_mask]

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
                "narration": v.evidence_narration,
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
    st.title("FLUXO")
    st.caption("Spots traffic violations on camera footage so officers don't have to watch every feed manually.")

    with st.sidebar:
        st.header("Settings")
        signal_state = st.selectbox("Current signal state", ["GREEN", "RED", "YELLOW"], index=0)
        night_mode = st.checkbox("Night mode (brightens dark footage)", value=False)
        enhance_quality = st.checkbox("Enhance low-quality footage (upscale + sharpen far objects)", value=False)
        max_frames = st.number_input("Stop after N frames (0 = process whole video)", min_value=0, value=0, step=100)
        enable_vlm = st.checkbox("Write a plain-English summary for each violation", value=False)
        st.divider()
        st.subheader("What's running under the hood")
        st.markdown("""
        - **Vehicle detection** - finds bikes, cars, buses, autos, trucks, pedestrians
        - **Tracking** - follows each vehicle across frames so we know it's the same one
        - **Helmet check** - looks at the rider's head, knows the difference between a helmet and a turban or cap
        - **Triple riding** - counts riders on a two-wheeler using a smarter shape than a simple box
        - **Red light jump** - watches for vehicles crossing the stop line while the light is red
        - **Wrong-way driving** - spots vehicles moving against the expected lane direction
        - **Number plate reading** - reads Indian plates even on low-quality CCTV footage
        - **Fancy plate detection** - catches modified or hidden plates that try to dodge cameras
        - **Missing mirror** - flags two-wheelers without rear-view mirrors
        - **Quality enhancement** - when footage is blurry or low-res, it sharpens and upscales before checking
        - **Evidence clips** - saves only the few seconds around each violation, not the whole video
        """)

    tab_video, tab_live, tab_about = st.tabs(["Analyze a video", "Live camera", "About"])

    with tab_video:
        uploaded = st.file_uploader("Drop a traffic video here", type=["mp4", "avi", "mov", "mkv"])

        source_path = None
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.close()
            source_path = tmp.name
            st.video(source_path)

        if source_path is not None:
            if st.button("Run detection", type="primary", use_container_width=True):
                detector, violation_engine = load_models()
                if enable_vlm:
                    from core.violations import ViolationConfig, ViolationDetector
                    vconfig = ViolationConfig(
                        enable_signal_jump=True,
                        enable_helmet=True,
                        enable_wrong_way=True,
                        enable_triple_riding=True,
                        enable_fancy_plate=True,
                        enable_missing_mirror=True,
                        enable_anpr=True,
                        enable_clip_extract=False,
                        enable_vlm_narration=True,
                    )
                    violation_engine = ViolationDetector(vconfig)
                violations_log = process_video(
                    source_path, detector, violation_engine,
                    signal_state, night_mode,
                    max_frames=max_frames if max_frames > 0 else None,
                    enhance_quality=enhance_quality,
                )
                if violations_log:
                    st.divider()
                    st.subheader("What we found")
                    for v in violations_log:
                        vtype = v["type"].replace("_", " ").title()
                        plate_info = f" - Plate: `{v['plate']}`" if v.get("plate") else ""
                        st.markdown(
                            f'<div class="violation-card">'
                            f'<b>{vtype}</b>{plate_info} - detected in frame {v["frame"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if v.get("narration"):
                            st.markdown(
                                f'<div class="narration-card">{v["narration"]}</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.success("No violations found in this video.")

    with tab_live:
        camera_source = st.number_input("Camera index (0 = your webcam)", min_value=0, value=0, step=1)
        if st.button("Start live detection", type="primary", use_container_width=True):
            detector, violation_engine = load_models()
            process_video(
                camera_source, detector, violation_engine,
                signal_state, night_mode,
                max_frames=300,
            )

    with tab_about:
        st.subheader("What is FLUXO?")
        st.markdown("""
        BTP had officers watching CCTV feeds one by one, writing down violations by hand.
        In a 2-day drive they caught 573 violations - but that's the ceiling of what humans can do.

        FLUXO watches the feeds for them. It spots vehicles, reads number plates,
        checks for helmets, catches red-light jumpers, and saves the evidence clip - all automatically.
        """)

        st.subheader("Why does it work well on Indian roads?")
        st.markdown("""
        Most traffic AI is trained on Western roads. Indian traffic is different:
        auto-rickshaws, lane-splitting two-wheelers, non-standard number plates, turbans and scarves
        that look like helmets to a basic camera. FLUXO was built specifically for these conditions.
        """)

        st.subheader("What violations can it spot?")
        st.markdown("""
        - **No helmet** - even when the rider is wearing a turban, cap, or scarf (things that fool other systems)
        - **Triple riding** - more than two people on a two-wheeler
        - **Red light jumping** - crossing the stop line while the signal is red
        - **Wrong-way driving** - moving against the expected direction of traffic
        - **Fancy or hidden number plates** - modified plates designed to avoid cameras
        - **Missing rear-view mirrors** - two-wheelers without the legally required mirrors
        - **Over-speeding** - when speed data is available from the tracking layer
        """)

        st.subheader("How does it handle bad lighting?")
        st.markdown("""
        Night-time CCTV footage is usually dark and grainy. FLUXO uses a technique called CLAHE
        that brightens the important parts of the image without washing everything out.
        This helps the system read number plates and spot helmets even in poor lighting.
        """)


if __name__ == "__main__":
    main()
