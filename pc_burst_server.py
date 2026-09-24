"""
pc_burst_server.py - High-Speed Burst OCR Processing Server for Padlocks & Security Seals.
Listens for multi-image burst uploads from Raspberry Pi Zero 2W, evaluates frame sharpness,
runs RapidOCR ONNX character recognition across all burst shots, extracts padlock serials,
copies result to Windows clipboard, and returns recognized text and annotated images back to the Pi.
"""
import os
import sys
import time
import base64
import datetime
import subprocess
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from padlock_ai import PadlockDetector

# Paths & Directories
BASE_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = BASE_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# App & AI detector setup
app = FastAPI(title="StickCam Burst OCR Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = PadlockDetector(conf_threshold=0.40)
# Initialize detector engine synchronously for instant burst readiness
print("[Server] Initializing RapidOCR ONNX model for burst processing...")
dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
try:
    detector.engine(dummy_img)
    print("[Server] RapidOCR ONNX model pre-warmed and ready!")
except Exception as e:
    print(f"[Server] Pre-warm note: {e}")

# Global state for PC dashboard
latest_burst_result = {
    "timestamp": None,
    "serial_number": None,
    "confidence": 0.0,
    "ocr_time_ms": 0.0,
    "sharpness": 0.0,
    "frames_received": 0,
    "best_frame_index": 0,
    "candidates": [],
    "annotated_image": None,
    "burst_thumbnails": [],
    "saved_path": None,
}
result_lock = threading.Lock()


def copy_to_clipboard(text: str):
    """Copy detected text to Windows clipboard."""
    if not text:
        return
    try:
        cmd = f"Set-Clipboard -Value '{text}'"
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=True, timeout=2)
        print(f"[Server] Copied to Windows clipboard: {text}")
    except Exception as e:
        print(f"[Server] Clipboard copy warning: {e}")


def calculate_sharpness(image: np.ndarray) -> float:
    """Laplacian variance to gauge image focus/sharpness."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def annotate_frame(frame: np.ndarray, detections: list, best_id: str, best_conf: float, ocr_time_ms: float) -> np.ndarray:
    """Draw rich HUD bounding boxes, corner reticles, and status pill."""
    disp = frame.copy()
    h, w = disp.shape[:2]

    # Draw detection bounding boxes
    for det in detections:
        pts = det["pts"]
        text = det["text"]
        score = det["score"]
        is_serial = det["is_serial"]

        border_color = (0, 255, 100) if is_serial else (255, 190, 20)

        # Polygon box
        cv2.polylines(disp, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)
        for pt in pts:
            cv2.circle(disp, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Text pill
        label = f"{text} ({int(score * 100)}%)"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.55
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)

        min_y = int(np.min(pts[:, 1]))
        min_x = int(np.min(pts[:, 0]))
        lbl_y = max(th + 8, min_y - 6)
        lbl_x = max(10, min_x)

        x1 = max(0, lbl_x - 6)
        y1 = max(0, lbl_y - th - 3)
        x2 = min(w, lbl_x + tw + 6)
        y2 = min(h, lbl_y + baseline + 3)

        if y2 > y1 and x2 > x1:
            sub = disp[y1:y2, x1:x2]
            pill = np.zeros_like(sub)
            pill[:] = (20, 25, 32)
            disp[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.3, pill, 0.7, 0)
            cv2.rectangle(disp, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

        cv2.putText(disp, label, (lbl_x, lbl_y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Top HUD Banner
    hud_h = 44
    hud_sub = disp[0:hud_h, 0:w]
    bg = np.zeros_like(hud_sub)
    bg[:] = (12, 16, 24)
    disp[0:hud_h, 0:w] = cv2.addWeighted(hud_sub, 0.2, bg, 0.8, 0)
    cv2.line(disp, (0, hud_h), (w, hud_h), (45, 55, 70), 1)

    cv2.putText(disp, "STICKCAM BURST OCR", (14, 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 215, 255), 1, cv2.LINE_AA)

    if best_id:
        cv2.putText(disp, f"SEAL: {best_id} ({int(best_conf * 100)}%)", (220, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 100), 2, cv2.LINE_AA)
    else:
        cv2.putText(disp, "SEAL: NO SERIAL DETECTED", (220, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (130, 145, 160), 1, cv2.LINE_AA)

    stat_text = f"TIME: {int(ocr_time_ms)}ms"
    (sw, _), _ = cv2.getTextSize(stat_text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
    cv2.putText(disp, stat_text, (w - sw - 14, 28), cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 210, 225), 1, cv2.LINE_AA)

    return disp


def process_burst_images(frames: List[np.ndarray]) -> dict:
    """
    Evaluates each image in the burst, runs RapidOCR, extracts padlock serial codes,
    scores frames, and produces consensus results.
    """
    t0 = time.time()
    frame_evaluations = []

    for idx, frame in enumerate(frames):
        sharpness = calculate_sharpness(frame)

        # Run RapidOCR engine directly
        try:
            raw_results, _ = detector.engine(frame)
        except Exception as e:
            print(f"[Server] OCR error on frame {idx}: {e}")
            raw_results = None

        frame_detections = []
        best_frame_serial = None
        best_frame_conf = 0.0

        if raw_results:
            for item in raw_results:
                box, text, score_str = item
                score = float(score_str)
                clean_text = text.strip()

                if score < detector.conf_threshold or len(clean_text) < 2:
                    continue

                # Strict check for letters + numbers or serial pattern (e.g. C581819)
                has_digit = any(c.isdigit() for c in clean_text)
                is_serial = (has_digit and len(clean_text.replace(" ", "")) >= 4)

                pts = np.array(box, dtype=np.int32)
                frame_detections.append({
                    "text": clean_text,
                    "score": score,
                    "pts": pts,
                    "is_serial": is_serial,
                })

                if is_serial and score > best_frame_conf:
                    best_frame_serial = clean_text
                    best_frame_conf = score

        # Multi-angle fallback: If no serial found at 0°, check 180° flipped orientation (upside-down seals)
        if best_frame_serial is None:
            try:
                frame_180 = cv2.rotate(frame, cv2.ROTATE_180)
                raw_results_180, _ = detector.engine(frame_180)
                if raw_results_180:
                    fh, fw = frame.shape[:2]
                    for item in raw_results_180:
                        box, text, score_str = item
                        score = float(score_str)
                        clean_text = text.strip()
                        if score < detector.conf_threshold or len(clean_text) < 2:
                            continue
                        has_digit = any(c.isdigit() for c in clean_text)
                        is_serial = (has_digit and len(clean_text.replace(" ", "")) >= 4)
                        pts_rot = np.array([[fw - 1 - p[0], fh - 1 - p[1]] for p in box], dtype=np.int32)
                        frame_detections.append({
                            "text": clean_text,
                            "score": score,
                            "pts": pts_rot,
                            "is_serial": is_serial,
                        })
                        if is_serial and score > best_frame_conf:
                            best_frame_serial = clean_text
                            best_frame_conf = score
            except Exception as e:
                print(f"[Server] 180° rotation error on frame {idx}: {e}")

        frame_evaluations.append({
            "index": idx,
            "sharpness": sharpness,
            "detections": frame_detections,
            "best_serial": best_frame_serial,
            "best_conf": best_frame_conf,
        })

    t1 = time.time()
    total_ocr_ms = (t1 - t0) * 1000.0

    # Multi-shot consensus: Pick frame with highest confidence serial, breaking ties with sharpness
    valid_serial_frames = [f for f in frame_evaluations if f["best_serial"] is not None]
    if valid_serial_frames:
        best_eval = max(valid_serial_frames, key=lambda f: (f["best_conf"], f["sharpness"]))
    else:
        # Fallback to sharpest frame
        best_eval = max(frame_evaluations, key=lambda f: f["sharpness"])

    best_idx = best_eval["index"]
    winning_frame = frames[best_idx]
    best_serial = best_eval["best_serial"]
    best_conf = best_eval["best_conf"]
    sharpness = best_eval["sharpness"]
    detections = best_eval["detections"]

    # Collect all unique text candidates across burst
    candidates_set = {}
    for ev in frame_evaluations:
        for det in ev["detections"]:
            txt = det["text"]
            sc = det["score"]
            if txt not in candidates_set or sc > candidates_set[txt]:
                candidates_set[txt] = sc

    sorted_candidates = sorted(
        [{"text": k, "score": round(v * 100, 1)} for k, v in candidates_set.items()],
        key=lambda x: x["score"],
        reverse=True
    )

    # Annotate winning image
    annotated = annotate_frame(winning_frame, detections, best_serial, best_conf, total_ocr_ms)

    # Encode annotated image to JPEG base64
    ret, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ret else None

    # Generate thumbnails for all burst frames
    thumbs_b64 = []
    for f in frames:
        th_h, th_w = 120, int(120 * f.shape[1] / f.shape[0])
        thumb = cv2.resize(f, (th_w, th_h))
        r, tbuf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if r:
            thumbs_b64.append("data:image/jpeg;base64," + base64.b64encode(tbuf.tobytes()).decode("ascii"))

    # Save burst to disk in captures/
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    burst_dir = CAPTURES_DIR / f"burst_{ts_str}"
    burst_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(frames):
        cv2.imwrite(str(burst_dir / f"frame_{i}.jpg"), f)
    cv2.imwrite(str(burst_dir / "annotated.jpg"), annotated)

    # Copy recognized serial to Windows clipboard
    if best_serial:
        copy_to_clipboard(best_serial)

    result_payload = {
        "status": "ok",
        "serial_number": best_serial,
        "confidence": round(float(best_conf), 3),
        "ocr_time_ms": round(total_ocr_ms, 1),
        "sharpness": round(sharpness, 1),
        "frames_received": len(frames),
        "best_frame_index": best_idx,
        "candidates": sorted_candidates,
        "annotated_image": annotated_b64,
        "burst_thumbnails": thumbs_b64,
        "saved_path": str(burst_dir.relative_to(BASE_DIR)),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Update global state for PC dashboard
    global latest_burst_result
    with result_lock:
        latest_burst_result = result_payload

    return result_payload


@app.post("/api/process_burst")
async def handle_process_burst(files: List[UploadFile] = File(...)):
    """Receives multiple burst images from Pi Zero, processes OCR, and returns result."""
    if not files:
        return JSONResponse({"status": "error", "message": "No files uploaded"}, status_code=400)

    print(f"[Server] Received burst of {len(files)} image(s) from Pi Zero...")
    loaded_frames = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            loaded_frames.append(img)

    if not loaded_frames:
        return JSONResponse({"status": "error", "message": "Failed to decode any images"}, status_code=400)

    result = process_burst_images(loaded_frames)
    print(f"[Server] OCR Completed in {result['ocr_time_ms']}ms. Result: '{result['serial_number']}' (conf: {int(result['confidence']*100)}%)")
    return JSONResponse(result)


@app.get("/api/latest_result")
def get_latest_result():
    with result_lock:
        return JSONResponse(latest_burst_result)


@app.post("/api/test_sample")
def test_sample_seal():
    """Trigger test burst using local sample_seal.jpg for testing without Pi."""
    sample_path = BASE_DIR / "sample_seal.jpg"
    if not sample_path.exists():
        return JSONResponse({"status": "error", "message": "sample_seal.jpg not found"}, status_code=404)

    img = cv2.imread(str(sample_path))
    if img is None:
        return JSONResponse({"status": "error", "message": "Could not read sample_seal.jpg"}, status_code=500)

    # Synthesize a 3-frame burst (original, slightly cropped, slight brightness)
    h, w = img.shape[:2]
    f1 = img.copy()
    crop = img[int(h*0.02):int(h*0.98), int(w*0.02):int(w*0.98)]
    f2 = cv2.resize(crop, (w, h))
    f3 = cv2.convertScaleAbs(img, alpha=1.05, beta=5)

    result = process_burst_images([f1, f2, f3])
    return JSONResponse(result)


@app.get("/", response_class=HTMLResponse)
def pc_dashboard():
    """Interactive PC Web Dashboard for monitoring burst captures."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StickCam - Burst OCR PC Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card: #131b2e;
            --border: #223250;
            --accent: #38bdf8;
            --emerald: #10b981;
            --amber: #f59e0b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, sans-serif;
            min-height: 100vh;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            text-align: center;
            max-width: 1000px;
            width: 100%;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 1.8rem;
            font-weight: 800;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        .badge {
            background: var(--emerald);
            color: #fff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 999px;
            letter-spacing: 0.5px;
        }
        .container {
            max-width: 1000px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
        }
        .stat-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .stat-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        .stat-val {
            font-size: 1.5rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: #fff;
        }
        .stat-val.emerald { color: var(--emerald); }
        .stat-val.accent { color: var(--accent); }

        .preview-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .preview-header {
            padding: 12px 18px;
            background: #0d1424;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .preview-body {
            padding: 16px;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #05070d;
            min-height: 420px;
        }
        .preview-body img {
            max-width: 100%;
            max-height: 540px;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        }
        .thumbs-row {
            display: flex;
            gap: 10px;
            padding: 14px;
            background: #0d1424;
            border-top: 1px solid var(--border);
            overflow-x: auto;
        }
        .thumbs-row img {
            height: 75px;
            border-radius: 6px;
            border: 2px solid #334155;
            cursor: pointer;
            transition: all 0.2s;
        }
        .thumbs-row img.active {
            border-color: var(--emerald);
            transform: scale(1.05);
        }
        .controls-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }
        .btn {
            background: #1e293b;
            color: #fff;
            border: 1px solid #334155;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s;
        }
        .btn:hover {
            background: #334155;
            transform: translateY(-1px);
        }
        .btn-emerald {
            background: rgba(16, 185, 129, 0.2);
            border-color: var(--emerald);
            color: #34d399;
        }
        .btn-emerald:hover {
            background: var(--emerald);
            color: #fff;
        }
        .btn-accent {
            background: rgba(56, 189, 248, 0.2);
            border-color: var(--accent);
            color: #38bdf8;
        }
        .btn-accent:hover {
            background: var(--accent);
            color: #fff;
        }
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #1e293b;
            border: 1px solid var(--accent);
            color: #fff;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 0.9rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s;
            pointer-events: none;
            z-index: 1000;
        }
        .toast.show { opacity: 1; transform: translateY(0); }
    </style>
</head>
<body>
    <header>
        <h1>STICKCAM BURST OCR <span class="badge">ON-DEMAND MODE</span></h1>
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 6px;">
            Pi Zero 2W Burst Shot &bull; RapidOCR ONNX Processing &bull; Auto Clipboard Copy
        </p>
    </header>

    <div class="container">
        <div class="controls-bar">
            <button class="btn btn-emerald" onclick="copyResult()">📋 Copy Serial Code</button>
            <button class="btn btn-accent" onclick="testSample()">⚡ Test Local Sample Burst</button>
            <div style="margin-left: auto; color: var(--text-muted); font-size: 0.85rem;" id="timestampText">
                Listening for Pi Zero burst captures...
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-label">Detected Serial</span>
                <span class="stat-val emerald" id="valSerial">WAITING...</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Confidence</span>
                <span class="stat-val" id="valConf">-- %</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">OCR Speed</span>
                <span class="stat-val accent" id="valSpeed">-- ms</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Frames / Sharpness</span>
                <span class="stat-val" id="valFrames">--</span>
            </div>
        </div>

        <div class="preview-card">
            <div class="preview-header">
                <span style="font-weight: 700; font-size: 0.9rem;">Annotated AI Result & Bounding Boxes</span>
                <span id="savePath" style="font-size: 0.8rem; color: var(--text-muted); font-family: monospace;"></span>
            </div>
            <div class="preview-body" id="previewArea">
                <div style="color: var(--text-muted); font-size: 0.95rem; text-align: center;">
                    Press the button on Pi Zero (or click "Test Local Sample Burst" above)<br>to capture and read padlock serial codes.
                </div>
            </div>
            <div class="thumbs-row" id="thumbsRow" style="display: none;"></div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let currentSerial = "";

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2500);
        }

        function copyResult() {
            if (currentSerial) {
                navigator.clipboard.writeText(currentSerial);
                showToast("Copied: " + currentSerial);
            } else {
                showToast("No serial code available to copy.");
            }
        }

        async function testSample() {
            showToast("Running local sample burst test...");
            try {
                const res = await fetch('/api/test_sample', { method: 'POST' });
                const data = await res.json();
                renderResult(data);
                showToast("Detected: " + (data.serial_number || "None"));
            } catch (err) {
                showToast("Error testing sample: " + err);
            }
        }

        function renderResult(data) {
            if (!data || !data.timestamp) return;
            currentSerial = data.serial_number || "";
            document.getElementById('valSerial').innerText = data.serial_number || "NOT DETECTED";
            document.getElementById('valConf').innerText = Math.round((data.confidence || 0) * 100) + "%";
            document.getElementById('valSpeed').innerText = data.ocr_time_ms + " ms";
            document.getElementById('valFrames').innerText = data.frames_received + " shots (" + Math.round(data.sharpness) + ")";
            document.getElementById('timestampText').innerText = "Last Received: " + data.timestamp;
            document.getElementById('savePath').innerText = data.saved_path ? "Saved to: " + data.saved_path : "";

            if (data.annotated_image) {
                document.getElementById('previewArea').innerHTML = `<img src="${data.annotated_image}" alt="OCR Result">`;
            }

            const thumbsRow = document.getElementById('thumbsRow');
            if (data.burst_thumbnails && data.burst_thumbnails.length > 0) {
                thumbsRow.style.display = 'flex';
                thumbsRow.innerHTML = data.burst_thumbnails.map((t, idx) => `
                    <img src="${t}" class="${idx === data.best_frame_index ? 'active' : ''}" 
                         title="Shot #${idx + 1} ${idx === data.best_frame_index ? '(Best Selection)' : ''}" />
                `).join('');
            }
        }

        async function pollLatest() {
            try {
                const res = await fetch('/api/latest_result');
                const data = await res.json();
                if (data.timestamp && data.serial_number !== currentSerial) {
                    renderResult(data);
                }
            } catch (e) {}
        }

        setInterval(pollLatest, 1000);
    </script>
</body>
</html>"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="StickCam PC Burst OCR Processing Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    print("=" * 68)
    print("  STICKCAM PC BURST OCR SERVER")
    print(f"  Listening on http://{args.host}:{args.port}")
    print("  Ready to receive multi-image bursts from Raspberry Pi Zero 2W")
    print("=" * 68)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
