"""Evidence Report Generator for FLUXO.

Captures annotated violation frames with colored borders and generates
self-contained HTML evidence reports with embedded images.

Red border = violating vehicle / violation area
Green border = other tracked vehicles (context)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .types import ViolationEvent


def capture_evidence_frame(
    frame: np.ndarray,
    violation: ViolationEvent,
    tracked_detections,
    jpeg_quality: int = 85,
) -> bytes:
    """Render an annotated evidence snapshot for a single violation.

    Draws a thick red border around the violating vehicle and green
    borders around all other tracked vehicles for spatial context.
    Returns JPEG-compressed bytes to keep memory usage manageable
    on long videos.

    Args:
        frame: Raw BGR frame from the video.
        violation: The ViolationEvent being documented.
        tracked_detections: supervision.Detections with tracker_id.
        jpeg_quality: JPEG compression quality (0-100).

    Returns:
        JPEG bytes of the annotated frame.
    """
    from core.vision.config import VEHICLE_CLASSES

    evidence = frame.copy()
    h, w = evidence.shape[:2]

    # --- Removed Green context vehicles to reduce visual clutter ---
    # The user requested that only the red violation box should be visible
    # because the green boxes were blocking the view in dense traffic.

    # --- Red border on the violating vehicle ---
    vb = violation.bbox
    if len(vb) == 4 and any(v != 0 for v in vb):
        x1, y1, x2, y2 = int(vb[0]), int(vb[1]), int(vb[2]), int(vb[3])
        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 4)

        vlabel = violation.type.value.upper().replace("_", " ")
        if violation.plate_number:
            vlabel += f" [{violation.plate_number}]"
        vlabel += f" ({violation.confidence:.0%})"
        _draw_label_bg(evidence, vlabel, (x1, y1 - 12), (0, 0, 255))

    # --- Watermark: frame number + timestamp ---
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    watermark = f"Frame {violation.frame} | {ts}"
    _draw_label_bg(evidence, watermark, (10, h - 20), (60, 60, 60), font_scale=0.5)

    # Compress to JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, buf = cv2.imencode(".jpg", evidence, encode_params)
    if not success:
        return b""
    return buf.tobytes()


def _draw_label_bg(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.4,
    thickness: int = 1,
):
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # Background rectangle
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y + 4), color, -1)
    # Text in white
    cv2.putText(img, text, (x + 2, y), font, font_scale, (255, 255, 255), thickness)


def generate_html_report(
    violations_log: list[dict],
    video_name: str = "Unknown",
    total_frames: int = 0,
    processing_fps: float = 0.0,
) -> str:
    """Build a self-contained HTML evidence report.

    All violation frame images are embedded as base64 data URIs so the
    report can be opened in any browser without external dependencies.

    Args:
        violations_log: List of dicts from process_video(), each
            containing type, frame, confidence, plate, narration,
            timestamp, and optionally evidence_frame_b64.
        video_name: Original video filename.
        total_frames: Total frames processed.
        processing_fps: Average processing speed.

    Returns:
        Complete HTML string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    violation_count = len(violations_log)

    # Build violation cards
    cards_html = ""
    for idx, v in enumerate(violations_log, 1):
        vtype = v["type"].replace("_", " ").title()
        plate = v.get("plate") or "Not read"
        confidence = v.get("confidence", 0)
        frame_num = v.get("frame", 0)
        ts = v.get("timestamp", "")
        narration = v.get("narration", "")
        frame_b64 = v.get("evidence_frame_b64", "")

        img_html = ""
        if frame_b64:
            img_html = f"""
            <div class="evidence-img">
                <img src="data:image/jpeg;base64,{frame_b64}"
                     alt="Evidence frame for violation #{idx}" />
                <div class="img-caption">
                    <span class="legend-red">&#9632;</span> Violation area
                    &nbsp;&nbsp;
                    <span class="legend-green">&#9632;</span> Other vehicles (context)
                </div>
            </div>"""

        narration_html = ""
        if narration:
            narration_html = f"""
            <div class="narration">
                <strong>AI Summary:</strong> {narration}
            </div>"""

        cards_html += f"""
        <div class="violation-card">
            <div class="card-header">
                <span class="violation-number">#{idx}</span>
                <span class="violation-type">{vtype}</span>
                <span class="confidence">{confidence:.0%} confidence</span>
            </div>
            <div class="card-body">
                {img_html}
                <div class="card-details">
                    <table>
                        <tr><td class="detail-label">Plate</td><td>{plate}</td></tr>
                        <tr><td class="detail-label">Frame</td><td>{frame_num}</td></tr>
                        <tr><td class="detail-label">Timestamp</td><td>{ts}</td></tr>
                        <tr><td class="detail-label">Violation</td><td>{vtype}</td></tr>
                    </table>
                    {narration_html}
                </div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FLUXO Evidence Report - {now}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            padding: 24px;
            line-height: 1.6;
        }}
        .report-header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #2a2a4e;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
            text-align: center;
        }}
        .report-header h1 {{
            font-size: 28px;
            color: #fff;
            margin-bottom: 8px;
        }}
        .report-header .subtitle {{
            color: #888;
            font-size: 14px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 24px 0;
        }}
        .summary-card {{
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #fff;
        }}
        .summary-card .label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .violation-card {{
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-left: 4px solid #ff4444;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .card-header {{
            background: #16213e;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .violation-number {{
            background: #ff4444;
            color: #fff;
            padding: 2px 10px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 13px;
        }}
        .violation-type {{
            font-weight: 600;
            font-size: 16px;
            color: #fff;
            flex: 1;
        }}
        .confidence {{
            color: #ff8888;
            font-size: 13px;
            font-weight: 600;
        }}
        .card-body {{
            padding: 16px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .evidence-img {{
            flex: 1;
            min-width: 300px;
        }}
        .evidence-img img {{
            width: 100%;
            border-radius: 6px;
            border: 2px solid #2a2a4e;
        }}
        .img-caption {{
            font-size: 11px;
            color: #888;
            margin-top: 6px;
            text-align: center;
        }}
        .legend-red {{ color: #ff4444; font-size: 16px; }}
        .legend-green {{ color: #00cc00; font-size: 16px; }}
        .card-details {{
            flex: 1;
            min-width: 200px;
        }}
        .card-details table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .card-details td {{
            padding: 6px 8px;
            border-bottom: 1px solid #2a2a4e;
            font-size: 14px;
        }}
        .detail-label {{
            color: #888;
            font-weight: 600;
            width: 100px;
        }}
        .narration {{
            margin-top: 12px;
            padding: 10px;
            background: #0d1b2a;
            border-left: 3px solid #2e86c1;
            border-radius: 4px;
            font-style: italic;
            color: #a0c4e8;
            font-size: 13px;
        }}
        .footer {{
            text-align: center;
            color: #555;
            font-size: 12px;
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #2a2a4e;
        }}
        @media print {{
            body {{ background: #fff; color: #000; padding: 12px; }}
            .report-header {{ background: #f5f5f5; border-color: #ddd; }}
            .report-header h1 {{ color: #000; }}
            .summary-card {{ background: #f9f9f9; border-color: #ddd; }}
            .summary-card .value {{ color: #000; }}
            .violation-card {{ background: #fff; border-color: #ddd; }}
            .card-header {{ background: #f0f0f0; }}
            .violation-type {{ color: #000; }}
            .card-details td {{ border-color: #ddd; }}
            .narration {{ background: #f0f7ff; }}
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <h1>FLUXO Evidence Report</h1>
        <div class="subtitle">Automated Traffic Violation Detection</div>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{violation_count}</div>
            <div class="label">Violations Found</div>
        </div>
        <div class="summary-card">
            <div class="value">{total_frames}</div>
            <div class="label">Frames Processed</div>
        </div>
        <div class="summary-card">
            <div class="value">{processing_fps:.1f}</div>
            <div class="label">Avg FPS</div>
        </div>
        <div class="summary-card">
            <div class="value">{video_name[:20]}</div>
            <div class="label">Source</div>
        </div>
        <div class="summary-card">
            <div class="value">{now}</div>
            <div class="label">Generated</div>
        </div>
    </div>

    <h2 style="margin: 24px 0 16px; font-size: 20px; color: #fff;">Violations</h2>

    {cards_html}

    <div class="footer">
        Generated by FLUXO - Automated Traffic Violation Detection<br/>
        Report generated on {now}
    </div>
</body>
</html>"""

    return html
