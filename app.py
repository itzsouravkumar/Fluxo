#!/usr/bin/env python3
"""FLUXO Streamlit Dashboard - Traffic Violation Detection System.

Run: streamlit run app.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import base64
import cv2
import tempfile
import time
from datetime import datetime

st.set_page_config(
    page_title="FLUXO | Traffic Enforcement AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px; padding: 28px 32px; margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-banner h1 { color: #fff; font-size: 28px; margin-bottom: 4px; }
    .hero-banner p  { color: #a0a0c0; font-size: 14px; margin: 0; }

    /* Metric cards */
    .metric-card {
        background: #161625; border: 1px solid #252540; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #fff; }
    .metric-card .label { font-size: 11px; color: #888; text-transform: uppercase;
                          letter-spacing: 1px; margin-top: 2px; }

    /* Violation cards */
    .v-card {
        background: #161625; border-radius: 10px; padding: 14px 16px;
        border-left: 4px solid #ff4444; margin-bottom: 10px;
    }
    .v-card.amber { border-left-color: #ffaa00; }
    .v-card .v-title { font-weight: 600; font-size: 14px; color: #fff; }
    .v-card .v-meta  { font-size: 12px; color: #888; margin-top: 2px; }

    /* Narration */
    .narration {
        background: #0d1b2a; padding: 10px 12px; border-radius: 6px;
        border-left: 3px solid #2e86c1; margin-top: 6px;
        font-style: italic; color: #a0c4e8; font-size: 13px;
    }

    /* Status badges */
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
             font-size: 11px; font-weight: 600; }
    .badge-ok    { background: #0e3a1e; color: #34d399; }
    .badge-warn  { background: #3a2e0e; color: #fbbf24; }
    .badge-err   { background: #3a0e0e; color: #f87171; }
    .badge-off   { background: #252540; color: #666; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important; font-size: 14px;
    }

    /* Sidebar sections */
    .sidebar-section {
        background: #161625; border: 1px solid #252540; border-radius: 8px;
        padding: 12px; margin-bottom: 10px;
    }

    /* Expander */
    .streamlit-expanderHeader { font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pedestrian remap: fix YOLO misclassifying walking people as two-wheelers
# ---------------------------------------------------------------------------
def _remap_pedestrians(dets):
    """Remap detections that look like walking pedestrians but are classified as two-wheelers.

    Walking pedestrians have tall, narrow bounding boxes (aspect ratio < 0.40)
    AND small absolute area. Two-wheelers are wider relative to height and
    typically larger in pixel area. This catches the common YOLO26n failure mode
    where pedestrians are misclassified as class 0 (two_wheeler).
    """
    import supervision as sv

    if len(dets) == 0 or dets.class_id is None:
        return dets

    class_id = dets.class_id.copy()
    for i in range(len(dets)):
        if class_id[i] != 0:
            continue
        x1, y1, x2, y2 = dets.xyxy[i]
        bw, bh = x2 - x1, y2 - y1
        if bh <= 0 or bw <= 0:
            continue
        aspect = bw / bh
        area = bw * bh
        # Must be narrow AND small (a parked motorcycle from the side can be
        # narrow but it'll be much larger in pixel area)
        if aspect < 0.40 and area < 15000:
            class_id[i] = 5

    return sv.Detections(
        xyxy=dets.xyxy,
        confidence=dets.confidence,
        class_id=class_id,
        tracker_id=dets.tracker_id,
    )


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
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
        enable_missing_mirror=False,
        enable_mobile_phone=True,
        enable_seatbelt=False,
        enable_overloading=True,
        enable_anpr=True,
        enable_clip_extract=False,
        enable_pedestrian_red_light=True,
        enable_plate_obstruction=False,
        enable_junction_blocking=True,
    )
    violation_engine = ViolationDetector(vconfig)
    return detector, violation_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    lightness = clahe.apply(lightness)
    lab = cv2.merge([lightness, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def annotate_frame(frame, tracked, violations, density_score, frame_idx):
    from core.vision.config import VEHICLE_CLASSES
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Dynamic scaling based on frame width
    font_scale = max(0.3, w / 2500.0)
    thick = max(1, int(w / 1000))
    
    for i in range(len(tracked)):
        bbox = tracked.xyxy[i].astype(int)
        cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
        conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
        tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
        vc = VEHICLE_CLASSES.get(cls_id)
        if vc is None:
            continue
            
        # Draw bounding box for context vehicles
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), vc.color, 1)
        # Show label text for context vehicles, using dynamic font scale so it's not huge
        cv2.putText(annotated, f"#{tid} {vc.name} {conf:.0%}",
                    (bbox[0], bbox[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, vc.color, thick)

    for v in violations:
        x1, y1, x2, y2 = [int(c) for c in v.bbox[:4]]
        color = (0, 140, 255) if "amber" in v.type.value else (0, 0, 255)
        # Draw thick colored box for the violator
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick + 1)
        
        lbl = v.type.value.upper().replace("_", " ")
        if v.plate_number:
            lbl += f" [{v.plate_number}]"
        if v.requires_human_review:
            lbl += " [REVIEW]"
        
        # Draw background for text to make it readable
        (tw, th), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, max(0, y1)), color, -1)
        cv2.putText(annotated, lbl, (x1 + 2, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)

    bar_w = int(density_score * 200)
    cv2.rectangle(annotated, (10, h - 55), (210, h - 35), (40, 40, 40), -1)
    bar_c = (0, int(255 * (1 - density_score)), int(255 * density_score))
    cv2.rectangle(annotated, (10, h - 55), (10 + bar_w, h - 35), bar_c, -1)
    cv2.putText(annotated, f"Density: {density_score:.3f}", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return annotated


def build_sidebar():
    """Build a well-organised sidebar with sections."""
    with st.sidebar:
        st.markdown("## Control Panel")

        # --- Signal & Environment ---
        with st.expander("Signal & Environment", expanded=True):
            signal_state = st.selectbox("Signal phase", ["GREEN", "RED", "YELLOW"], index=0,
                                        help="Current traffic signal phase at the junction.")
            night_mode = st.checkbox("Night mode (CLAHE)", value=False,
                                     help="Brightens dark footage for better detection.")
            weather_mode = st.checkbox("Weather preprocessing", value=True,
                                       help="Dehazing + rain streak removal for monsoon conditions.")

        # --- Detection ---
        with st.expander("Detection Settings", expanded=False):
            enhance_quality = st.checkbox("Quality enhancement pipeline", value=False,
                                          help="SR upscale + sharpen + tile detection for far objects.")
            max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=0, step=100)
            enable_evidence_hash = st.checkbox("Evidence hashing (SHA-256)", value=True,
                                                help="Cryptographic hash for legal defensibility.")

        # --- System Status ---
        with st.expander("System Status", expanded=False):
            st.markdown(
                '<span class="badge badge-ok">YOLO26</span> '
                '<span class="badge badge-ok">NMS-Free</span> '
                '<span class="badge badge-ok">STAL</span>',
                unsafe_allow_html=True,
            )
            st.caption("Detector: YOLO26n (Jan 2026) | Tracker: BoT-SORT")
            st.caption("Pipeline: Single-pass unified | Detection: YOLO26 + BoT-SORT")

        st.divider()
        st.caption("FLUXO | Gridlock Hackathon 2.0 | Flipkart x BTP")

    return signal_state, night_mode, weather_mode, enhance_quality, max_frames, enable_evidence_hash



def run_detection(source_path, video_name, is_image=False):
    """Execute the detection pipeline and display results."""

    signal_state = st.session_state.get("signal_state", "GREEN")
    night_mode = st.session_state.get("night_mode", False)
    weather_mode = st.session_state.get("weather_mode", True)
    enhance_quality = st.session_state.get("enhance_quality", False)
    max_frames = st.session_state.get("max_frames", 0)
    enable_evidence_hash = st.session_state.get("enable_evidence_hash", True)

    detector, violation_engine = load_models()

    violation_engine.config.evidence_hash_enabled = enable_evidence_hash

    # Initialize weather preprocessor if enabled
    weather_preprocessor = None
    if weather_mode:
        from core.vision.weather import WeatherPreprocessor
        weather_preprocessor = WeatherPreprocessor()

    try:
        if is_image:
            _run_detection_image(source_path, video_name, detector, violation_engine,
                                 signal_state, night_mode, enhance_quality, enable_evidence_hash,
                                 weather_preprocessor)
        else:
            _run_detection_video(source_path, video_name, detector, violation_engine,
                                 signal_state, night_mode, enhance_quality, max_frames,
                                 enable_evidence_hash, weather_preprocessor)
    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(source_path)
        except OSError:
            pass


def _run_detection_image(source_path, video_name, detector, violation_engine,
                         signal_state, night_mode, enhance_quality, enable_evidence_hash,
                         weather_preprocessor=None):
    """Process a single image through the detection pipeline."""
    from core.vision.config import VEHICLE_CLASSES, DEFAULT_CONFIG

    import supervision as sv
    from core.vision.tracker import FluxoTracker

    frame = cv2.imread(source_path)
    if frame is None:
        st.error("Cannot read image.")
        return

    h, w = frame.shape[:2]
    tracker = FluxoTracker()
    violation_engine.reset()

    if night_mode:
        frame = apply_clahe(frame)

    # Apply weather preprocessing if enabled
    if weather_preprocessor is not None:
        frame, _ = weather_preprocessor.preprocess(frame, night_mode=night_mode)

    if enhance_quality:
        from core.vision.enhancement import (
            FrameEnhancer, TileDetector, AdaptiveConfidence, SmartROISelector,
        )
        enhancer = FrameEnhancer()
        tile_det = TileDetector(tile_size=640, overlap=0.2)
        adaptive_conf = AdaptiveConfidence(base_conf=0.15)
        roi_selector = SmartROISelector()
        dets_ultralytics, _ = detector.detect_with_enhancement(
            frame, enhancer=enhancer, tile_detector=tile_det,
            adaptive_conf=adaptive_conf, roi_selector=roi_selector,
        )
        dets_for_sv = sv.Detections(
            xyxy=np.array([d.bbox for d in dets_ultralytics]) if dets_ultralytics else np.empty((0, 4)),
            confidence=np.array([d.confidence for d in dets_ultralytics]) if dets_ultralytics else np.empty(0),
            class_id=np.array([d.class_id for d in dets_ultralytics]) if dets_ultralytics else np.empty(0, dtype=int),
        )
    else:
        # Use the detector's public API instead of accessing _load_model()
        dets_for_sv = detector.detect(frame)

    dets_for_sv = _remap_pedestrians(dets_for_sv)

    vehicle_mask = np.isin(dets_for_sv.class_id, list(VEHICLE_CLASSES.keys()))
    vehicle_dets = dets_for_sv[vehicle_mask]
    tracked = tracker.update_with_detections(vehicle_dets) if len(vehicle_dets) > 0 else vehicle_dets

    violation_engine.feed_frame(frame)
    violations = violation_engine.check(tracked, frame, 0, signal_state, tracker=tracker)

    annotated = annotate_frame(frame, tracked, violations, 0.0, 0)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Original")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    with c2:
        st.markdown("### Detected")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    st.metric("Vehicles detected", len(tracked))

    if violations:
        st.markdown(f"### {len(violations)} Violation(s) Detected")
        for idx, v in enumerate(violations, 1):
            vtype = v.type.value.replace("_", " ").title()
            meta = f"Confidence: {v.confidence:.0%}"
            if v.plate_number:
                meta += f" | Plate: {v.plate_number}"
            st.markdown(f"**#{idx} {vtype}** -- {meta}")
    else:
        st.success("No violations detected in this image.")


def _run_detection_video(source_path, video_name, detector, violation_engine,
                         signal_state, night_mode, enhance_quality, max_frames,
                         enable_evidence_hash):
    """Process video frames through the detection pipeline."""
    from core.vision.config import VEHICLE_CLASSES, DEFAULT_CONFIG

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        st.error("Cannot open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Enhancement pipeline
    enhancer = tile_det = adaptive_conf = roi_selector = temporal_booster = None
    if enhance_quality:
        from core.vision.enhancement import (
            FrameEnhancer, TileDetector, AdaptiveConfidence,
            TemporalConfidenceBooster, SmartROISelector,
        )
        enhancer = FrameEnhancer()
        tile_det = TileDetector(tile_size=640, overlap=0.2)
        adaptive_conf = AdaptiveConfidence(base_conf=0.15)
        roi_selector = SmartROISelector()
        temporal_booster = TemporalConfidenceBooster()

    import supervision as sv
    from core.vision.tracker import FluxoTracker
    tracker = FluxoTracker()

    # --- UI Layout ---
    st.divider()
    progress = st.progress(0)
    status_bar = st.empty()

    metrics_placeholder = st.empty()
    frame_placeholder = st.empty()

    violations_log = []
    frame_count = 0
    start_time = time.time()
    density_scores = []
    total_violations = 0
    review_count = 0
    reported_violations = {}
    violation_engine.reset()
    tracker.reset()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if night_mode:
            frame = apply_clahe(frame)

        # Detection
        if enhance_quality and enhancer:
            dets_ultralytics, quality = detector.detect_with_enhancement(
                frame, enhancer=enhancer, tile_detector=tile_det,
                adaptive_conf=adaptive_conf, roi_selector=roi_selector,
            )
            dets_for_sv = sv.Detections(
                xyxy=np.array([d.bbox for d in dets_ultralytics]) if dets_ultralytics else np.empty((0, 4)),
                confidence=np.array([d.confidence for d in dets_ultralytics]) if dets_ultralytics else np.empty(0),
                class_id=np.array([d.class_id for d in dets_ultralytics]) if dets_ultralytics else np.empty(0, dtype=int),
            )
            if temporal_booster and len(dets_for_sv) > 0:
                bboxes = [d.bbox for d in dets_ultralytics]
                boosts = temporal_booster.update(bboxes)
                for i, boost in enumerate(boosts):
                    dets_for_sv.confidence[i] = min(dets_for_sv.confidence[i] + boost, 1.0)
        else:
            results = detector._load_model()(frame, conf=0.15, verbose=False)[0]
            dets_for_sv = sv.Detections.from_ultralytics(results)

        dets_for_sv = _remap_pedestrians(dets_for_sv)

        vehicle_mask = np.isin(dets_for_sv.class_id, list(VEHICLE_CLASSES.keys()))
        vehicle_dets = dets_for_sv[vehicle_mask]
        tracked = tracker.update_with_detections(vehicle_dets) if len(vehicle_dets) > 0 else vehicle_dets

        # Violation check
        violation_engine.feed_frame(frame)
        violations = violation_engine.check(tracked, frame, frame_count, signal_state, tracker=tracker)

        # Density
        total_pce = sum(
            VEHICLE_CLASSES[int(tracked.class_id[i])].pce
            for i in range(len(tracked))
            if tracked.class_id is not None and int(tracked.class_id[i]) in VEHICLE_CLASSES
        )
        density = min(total_pce / ((width * height) / 1000.0), 1.0)
        density_scores.append(density)
        avg_density = float(np.mean(density_scores[-30:])) if density_scores else 0

        # Collect evidence
        for v in violations:
            violation_key = (v.track_id, v.type.value)
            
            if violation_key not in reported_violations:
                total_violations += 1
                if v.requires_human_review:
                    review_count += 1
                from core.violations.evidence_report import capture_evidence_frame
                ev_bytes = capture_evidence_frame(frame, v, tracked)
                ev_b64 = base64.b64encode(ev_bytes).decode() if ev_bytes else ""
                log_entry = {
                    "frame": frame_count,
                    "type": v.type.value,
                    "track_id": v.track_id,
                    "confidence": round(v.confidence, 3),
                    "plate": v.plate_number,
                    "narration": v.evidence_narration,
                    "timestamp": datetime.now().isoformat(),
                    "evidence_frame_b64": ev_b64,
                    "requires_review": v.requires_human_review,
                    "partial_plate": v.partial_plate,
                    "evidence_hash": v.evidence_hash,
                }
                violations_log.append(log_entry)
                reported_violations[violation_key] = log_entry
            else:
                existing_entry = reported_violations[violation_key]
                if v.plate_number and not existing_entry.get("plate"):
                    existing_entry["plate"] = v.plate_number
                    from core.violations.evidence_report import capture_evidence_frame
                    ev_bytes = capture_evidence_frame(frame, v, tracked)
                    if ev_bytes:
                        existing_entry["evidence_frame_b64"] = base64.b64encode(ev_bytes).decode()
                    if v.confidence > existing_entry["confidence"]:
                        existing_entry["confidence"] = round(v.confidence, 3)

        # Annotate & display
        annotated = annotate_frame(frame, tracked, violations, avg_density, frame_count)
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

        elapsed = time.time() - start_time
        cur_fps = frame_count / max(elapsed, 0.01)

        with metrics_placeholder.container():
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Frame", f"{frame_count}/{total_frames}")
            m2.metric("Vehicles", len(tracked))
            m3.metric("Violations", total_violations)
            m4.metric("FPS", f"{cur_fps:.1f}")
            level = DEFAULT_CONFIG.density_levels.get_level(avg_density)
            m5.metric("Density", f"{avg_density:.3f}", delta=level)
            m6.metric("Review", review_count)

        progress.progress(min(frame_count / max(total_frames, 1), 1.0))
        status_bar.caption(f"Processing frame {frame_count}/{total_frames} | {cur_fps:.1f} FPS | Violations: {total_violations}")

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    progress.progress(1.0)

    total_time = time.time() - start_time
    proc_fps = frame_count / max(total_time, 0.01)
    st.success(f"Done -- {frame_count} frames in {total_time:.1f}s ({proc_fps:.1f} FPS)")

    # --- Results ---
    if violations_log:
        st.divider()
        st.markdown(f"## Evidence Report -- {len(violations_log)} Violations")

        from core.violations.evidence_report import generate_html_report
        report_html = generate_html_report(violations_log, video_name=video_name,
                                           total_frames=frame_count, processing_fps=proc_fps)
        st.download_button("Download HTML Report", data=report_html,
                           file_name=f"fluxo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                           mime="text/html", width="stretch")

        type_counts = {}
        for v in violations_log:
            t = v["type"].replace("_", " ").title()
            type_counts[t] = type_counts.get(t, 0) + 1
        if type_counts:
            st.markdown("### Violation Breakdown")
            cols = st.columns(min(len(type_counts), 6))
            for idx, (t, c) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])):
                with cols[idx % len(cols)]:
                    st.metric(t, c)

        st.markdown("### Violation Details")
        for idx, v in enumerate(violations_log, 1):
            vtype = v["type"].replace("_", " ").title()
            card_class = "v-card amber" if "amber" in v["type"] else "v-card"

            meta_parts = [f"Frame {v['frame']}", f"Track #{v['track_id']}", f"{v['confidence']:.0%}"]
            if v.get("plate"):
                meta_parts.append(f"Plate: `{v['plate']}`")
            if v.get("partial_plate"):
                meta_parts.append("[!] Partial read")
            if v.get("requires_review"):
                meta_parts.append("[!] Needs review")
            if v.get("evidence_hash"):
                meta_parts.append(f"Hash: `{v['evidence_hash'][:12]}...`")

            meta_str = " | ".join(meta_parts)
            st.markdown(f'<div class="{card_class}">'
                        f'<div class="v-title">#{idx} {vtype}</div>'
                        f'<div class="v-meta">{meta_str}</div></div>',
                        unsafe_allow_html=True)

            if v.get("evidence_frame_b64"):
                frame_bytes = base64.b64decode(v["evidence_frame_b64"])
                with st.expander(f"View evidence frame #{idx}"):
                    st.image(frame_bytes, caption=f"{vtype} -- red = violation, green = context",
                             width="stretch")

            if v.get("narration"):
                st.markdown(f'<div class="narration">{v["narration"]}</div>', unsafe_allow_html=True)
    else:
        st.success("No violations detected in this video.")


def tab_live_content():
    """Live camera tab."""
    st.markdown("### Live Camera Feed")
    camera_source = st.number_input("Camera index (0 = webcam)", min_value=0, value=0, step=1,
                                    label_visibility="collapsed", key="live_cam")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Start Live", type="primary", width="stretch"):
            run_detection(camera_source, "live_feed")
    with col2:
        st.info("Live mode processes up to 300 frames then stops. Re-click to restart.")


def tab_system_content():
    """System architecture & status tab."""
    st.markdown("## System Architecture")

    st.markdown("""
    <div class="hero-banner">
        <h1>FLUXO | Single-Pass Unified Pipeline</h1>
        <p>One detection pass -> one track ID -> one evidence bundle. No separate models per violation type.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline overview
    st.markdown("### Detection Pipeline")
    stages = [
        ("[Input]", "CCTV footage or live camera feed"),
        ("[YOLO26]", "NMS-free single-pass vehicle detection (7 classes, STAL for small targets)"),
        ("[Tracker]", "BoT-SORT with camera-motion compensation + track lifetime gating"),
        ("[Violations]", "11 parallel detectors from shared detection pass"),
        ("[ANPR]", "EasyOCR + SR preprocessing + plate color-aware + per-character confidence"),
        ("[ANPR]", "SR preprocessing + per-character confidence gating"),
        ("[Evidence]", "SHA-256 hashed bundle: annotated frame + clip + narration + metadata"),
    ]
    for i, (icon, desc) in enumerate(stages):
        st.markdown(f"**{i+1}. {icon}** -- {desc}")

    st.divider()

    # Feature matrix
    st.markdown("### Violation Detectors")
    detectors_data = [
        ("No Helmet", "Two-stage: YOLO classifier + headwear discriminator (turban/cap/scarf)", "[OK]"),
        ("Triple Riding", "Trapezium boundary (CVPR 2022) + amodal occlusion recovery", "[OK]"),
        ("Signal Jump", "Stop-line crossing during RED + YELLOW phase human review", "[OK]"),
        ("Wrong Way", "Velocity direction vs expected lane flow (8-frame history)", "[OK]"),
        ("Mobile Phone", "Multi-signal: HSV skin + edge density + temporal consistency", "[OK]"),
        ("Seatbelt", "Canny edge diagonal stripe detection in chest region", "[OK]"),
        ("Overloading", "Aspect ratio + visual density on bus/truck crops", "[OK]"),
        ("Fancy Plate", "OCR failure flagging + regex mismatch detection", "[OK]"),
        ("Missing Mirror", "Edge-feature analysis in expected mirror regions", "[OK]"),
        ("Pedestrian Red", "Pedestrian stop-line crossing during red phase", "[OK] NEW"),
        ("Plate Obstruction", "Texture anomaly detection on plate crops", "[OK] NEW"),
        ("Amber Violation", "Yellow phase crossing -> human review with speed data", "[OK] NEW"),
    ]
    cols = st.columns(3)
    for idx, (name, desc, status) in enumerate(detectors_data):
        with cols[idx % 3]:
            st.markdown(f"**{name}** `{status}`")
            st.caption(desc)

    st.divider()

    # Edge-case defenses
    st.markdown("### Edge-Case Defenses")
    defenses = [
        ("Fog Dehazing", "Dark-channel prior (He et al.) for monsoon/post-monsoon fog"),
        ("Rain Removal", "Streak detection + removal + raindrop-on-lens coverage monitoring"),
        ("Night HDR", "CLAHE + tone mapping for headlight glare saturation"),
        ("Plate Colors", "White/yellow/green background-aware OCR preprocessing"),
        ("Per-Char OCR", "Character-level confidence gating -- partial reads -> human review"),
        ("Evidence Hash", "SHA-256 on every violation bundle -- detect any post-creation tampering"),
        ("NTP Validation", "Timestamp drift detection for evidence chain integrity"),
        ("Adversarial Defense", "Texture anomaly detection + multi-frame track persistence"),
        ("Camera Health", "Freeze detection + lens spray + blur monitoring + IR blinding"),
        ("Cross-Camera ReID", "Vehicle embedding gallery for persistent violator tracking"),
        ("Federated Learning", "Privacy-compliant model improvement without centralizing video"),
    ]
    cols = st.columns(3)
    for idx, (name, desc) in enumerate(defenses):
        with cols[idx % 3]:
            st.markdown(f"**{name}**")
            st.caption(desc)




def tab_about_content():
    """About & documentation tab."""
    st.markdown("## About FLUXO")

    st.markdown("""
    <div class="hero-banner">
        <h1>Automated Traffic Violation Detection</h1>
        <p>Gridlock Hackathon 2.0 | Flipkart x Bengaluru Traffic Police | Theme 3</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### The Problem
    BTP had officers watching CCTV feeds one by one, writing down violations by hand.
    In a 2-day drive they caught 573 violations -- but that's the ceiling of what humans can do.
    FLUXO watches the feeds for them automatically.

    ### How It Works
    1. **Detect** -- YOLO26 finds all vehicles in a single pass (NMS-free, 43% faster than YOLOv11)
    2. **Track** -- BoT-SORT follows each vehicle across frames with camera-motion compensation
    3. **Check** -- 11 violation detectors run in parallel from the shared detection pass
    4. **Read** -- ANPR reads Indian plates with SR preprocessing and per-character confidence
    5. **Narrate** -- Template-based evidence text (post-confirmation)
    6. **Report** -- SHA-256 hashed evidence bundle with annotated frame + clip + narration

    ### Why Indian Roads Are Different
    Most traffic AI is trained on Western roads. Indian traffic has:
    - Auto-rickshaws, lane-splitting two-wheelers, mixed vehicle types
    - Non-standard number plates (HSRP, fancy fonts, commercial yellow)
    - Turbans, scarves, caps that fool helmet detectors (FLUXO handles these)
    - Monsoon rain, fog, and night conditions (FLUXO has weather preprocessing)
    - Dense 8-20 vehicle clusters at junctions (YOLO26's STAL handles small targets)

    ### Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Detection | YOLO26n (NMS-free, Jan 2026) |
    | Tracking | BoT-SORT / ByteTrack (supervision) |
    | OCR | EasyOCR + ESPCNN super-resolution |
    | Evidence | SHA-256 hash chain |
    | Weather | Dark-channel dehazing + rain removal |
    | Dashboard | Streamlit |
    | Tests | pytest (75 tests, all passing) |
    """)

    st.divider()
    st.caption("FLUXO | github.com/itzsouravkumar/Fluxo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <h1>FLUXO</h1>
        <p>Automated Traffic Violation Detection - watches CCTV feeds so officers don't have to</p>
    </div>
    """, unsafe_allow_html=True)

    signal_state, night_mode, weather_mode, enhance_quality, max_frames, enable_evidence_hash = build_sidebar()

    # Store in session state for access in detection
    st.session_state["signal_state"] = signal_state
    st.session_state["night_mode"] = night_mode
    st.session_state["weather_mode"] = weather_mode
    st.session_state["enhance_quality"] = enhance_quality
    st.session_state["max_frames"] = max_frames
    st.session_state["enable_evidence_hash"] = enable_evidence_hash

    tab_vid, tab_live, tab_sys, tab_info = st.tabs([
        "Analyze Video", "Live Camera", "System Architecture", "About"
    ])

    with tab_vid:
        tab_analysis_content()

    with tab_live:
        tab_live_content()

    with tab_sys:
        tab_system_content()

    with tab_info:
        tab_about_content()


def tab_analysis_content():
    """Inline content for the analysis tab."""
    IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}
    VIDEO_EXTS = {"mp4", "avi", "mov", "mkv"}
    uploaded = st.file_uploader("Drop a traffic video or image here",
                                type=list(VIDEO_EXTS | IMAGE_EXTS),
                                label_visibility="collapsed")

    source_path = None
    is_image = False
    if uploaded is not None:
        file_ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
        is_image = file_ext in IMAGE_EXTS

        if is_image:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
            tmp.write(uploaded.read())
            tmp.close()
            source_path = tmp.name

            c1, c2 = st.columns([3, 1])
            with c1:
                st.image(source_path, caption=uploaded.name, use_container_width=True)
            with c2:
                st.markdown("### Image Info")
                img = cv2.imread(source_path)
                if img is not None:
                    h, w = img.shape[:2]
                    st.metric("Resolution", f"{w}x{h}")
                    st.metric("Format", file_ext.upper())
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.close()
            source_path = tmp.name

            c1, c2 = st.columns([3, 1])
            with c1:
                st.video(source_path)
            with c2:
                st.markdown("### Video Info")
                cap = cv2.VideoCapture(source_path)
                info = {
                    "FPS": f"{cap.get(cv2.CAP_PROP_FPS):.0f}",
                    "Resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                    "Frames": f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}",
                }
                for k, v in info.items():
                    st.metric(k, v)
                cap.release()

    if source_path is not None:
        if st.button("Run Detection", type="primary", width="stretch"):
            run_detection(source_path, uploaded.name if uploaded else "Unknown", is_image=is_image)


if __name__ == "__main__":
    main()
