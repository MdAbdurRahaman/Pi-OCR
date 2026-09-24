#!/usr/bin/env python3
"""
pi_hdmi_scanner.py - Raspberry Pi 3.5" HDMI Display Live Scanner & Dual-Mode OCR Service.

Features:
1. Live Camera Ingestion: High-FPS streaming via Picamera2 (CSI Camera) or OpenCV V4L2 (USB Camera).
2. 3.5" HDMI Display Output: Direct GPU video output to /dev/fb0 at 30-60 FPS with zero SPI overhead.
3. Live Focus Telemetry: Real-time sharpness score & Green/Yellow/Red visual focus indicator.
4. Dual Physical Button Support (GPIO 17 & GPIO 27):
   - Button 1 (GPIO 17 / Pin 11 to GND): Burst Capture & Run OCR
   - Button 2 (GPIO 27 / Pin 13 to GND): Re-capture & Reset Viewfinder
5. Dual-Mode OCR:
   - Standalone Offline OCR: Local preprocessing + Tesseract / digits detection (autonomous operation).
   - High-Speed PC Server Offloading: Sends multi-image burst to PC server when available.
   - Automatic Fallback: If PC is unreachable, instantly falls back to on-device offline OCR.
6. Embedded Web Dashboard: Live MJPEG streaming on port 8000 with remote capture triggers.
"""
import os
import sys
import time
import json
import socket
import threading
import argparse
import subprocess
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from pi_hdmi_ui import StickCamHdmiUI

# Default Configuration
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_BURST_COUNT = 3
DEFAULT_PORT = 8000
DEFAULT_PC_URL = "http://192.168.68.123:5000"

# Global Hardware & State
camera_lock = threading.Lock()
picam2_obj = None
v4l2_cap = None
use_picam2 = False

hdmi_ui: Optional[StickCamHdmiUI] = None
running = True

# Shared state between threads & web interface
app_state = {
    "pc_url": None,
    "burst_count": DEFAULT_BURST_COUNT,
    "is_capturing": False,
    "last_result": {
        "status": "idle",
        "message": "Ready. Align seal inside target box.",
        "serial_number": None,
        "confidence": 0.0,
        "ocr_time_ms": 0.0,
        "annotated_image": None,
        "timestamp": None,
    },
    "roi": [0.25, 0.35, 0.50, 0.30],
    "sharpness": 0.0,
    "focus_quality": "BLURRY",
    "latest_camera_jpeg": None,
    "latest_hdmi_jpeg": None,
}
state_lock = threading.Lock()


def get_local_ip() -> str:
    """Detects local IP address on active network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Camera Ingestion Pipeline
# ---------------------------------------------------------------------------

def init_camera(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, device_id: int = 0):
    """Initializes CSI camera via Picamera2 or USB webcam via V4L2."""
    global picam2_obj, v4l2_cap, use_picam2
    print(f"[Camera] Initializing hardware camera ({width}x{height})...")

    # 1. Try Picamera2 (Raspberry Pi CSI Camera)
    try:
        from picamera2 import Picamera2
        picam2_obj = Picamera2(device_id)
        config = picam2_obj.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"}
        )
        picam2_obj.configure(config)
        picam2_obj.start()
        use_picam2 = True
        print("[Camera] Picamera2 CSI camera active (Native BGR888)!")
        return
    except Exception as e:
        print(f"[Camera] Picamera2 not available: {e}. Checking V4L2 USB camera...")

    # 2. Try OpenCV V4L2 (USB Webcam)
    try:
        v4l2_cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        v4l2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        v4l2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if v4l2_cap.isOpened():
            print(f"[Camera] OpenCV V4L2 camera active on /dev/video{device_id}!")
            return
        else:
            print("[Camera] Warning: No hardware camera device found. Running in simulated test mode.")
    except Exception as e:
        print(f"[Camera] V4L2 note: {e}")


def capture_single_frame() -> np.ndarray:
    """Acquires a single BGR frame from the active camera source."""
    global picam2_obj, v4l2_cap, use_picam2
    with camera_lock:
        if use_picam2 and picam2_obj is not None:
            try:
                return picam2_obj.capture_array("main")
            except Exception:
                return None
        elif v4l2_cap is not None and v4l2_cap.isOpened():
            ret, frame = v4l2_cap.read()
            return frame if ret and frame is not None else None
        else:
            # Fallback test pattern if running on desktop or without camera
            dummy = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
            dummy[:] = (20, 26, 36)
            cv2.putText(dummy, "STICK CAM HD", (DEFAULT_WIDTH // 2 - 130, DEFAULT_HEIGHT // 2 - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (56, 189, 248), 2)
            cv2.putText(dummy, "3.5\" HDMI Display Active", (DEFAULT_WIDTH // 2 - 120, DEFAULT_HEIGHT // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)
            return dummy


def capture_burst(count: int = DEFAULT_BURST_COUNT) -> List[np.ndarray]:
    """Captures N frames in rapid succession for peak-focus burst processing."""
    frames = []
    for _ in range(count):
        f = capture_single_frame()
        if f is not None:
            frames.append(f)
        time.sleep(0.04)
    return frames


# ---------------------------------------------------------------------------
# OCR Engines: Standalone Offline & PC Offloading
# ---------------------------------------------------------------------------

def calculate_sharpness(frame: np.ndarray, roi: List[float]) -> float:
    """Calculates Laplacian focus sharpness variance within the ROI bounding box."""
    if frame is None:
        return 0.0
    h, w = frame.shape[:2]
    rx, ry, rw, rh = roi
    x1, y1 = max(0, int(rx * w)), max(0, int(ry * h))
    x2, y2 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run_local_offline_ocr(frame: np.ndarray, roi: List[float]) -> Dict[str, Any]:
    """
    Standalone On-Device OCR Pipeline for Raspberry Pi.
    Uses bounded downscaling, CLAHE contrast equalization, adaptive Otsu binarization,
    and Tesseract OCR (with digit & seal whitelist) without needing internet or a PC.
    """
    t0 = time.time()
    h, w = frame.shape[:2]
    rx, ry, rw, rh = roi
    x1, y1 = max(0, int(rx * w)), max(0, int(ry * h))
    x2, y2 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return {"status": "error", "message": "Invalid crop", "serial_number": None}

    # Preprocessing
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 1000:
        scale = 1000.0 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Local contrast enhancement & unsharp mask
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 2.0)
    sharpened = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)

    # Otsu adaptive threshold
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)

    # Try running Tesseract
    candidates = {}
    temp_dir = Path("/tmp/stickcam_ocr")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for name, img in [("gray", sharpened), ("binary", binary)]:
        padded = cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        path = str(temp_dir / f"{name}.png")
        cv2.imwrite(path, padded)

        cmd = [
            "tesseract", path, "stdout",
            "-l", "eng",
            "--oem", "1",
            "--psm", "7",
            "-c", "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "tsv"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            for line in res.stdout.strip().split("\n"):
                cols = line.split("\t")
                if len(cols) >= 12:
                    text = cols[11].strip()
                    conf_str = cols[10].strip()
                    if text and any(c.isdigit() for c in text):
                        try:
                            conf = float(conf_str)
                            if conf > 0 and len(text) >= 3:
                                candidates[text] = max(conf, candidates.get(text, 0.0))
                        except ValueError:
                            pass
        except Exception:
            continue

    t1 = time.time()
    total_ms = (t1 - t0) * 1000.0

    if candidates:
        best_num = max(candidates.items(), key=lambda x: x[1])
        return {
            "status": "ok",
            "serial_number": best_num[0],
            "confidence": round(best_num[1] / 100.0, 3),
            "ocr_time_ms": round(total_ms, 1),
            "engine": "Standalone Tesseract (Local)",
        }

    return {
        "status": "not_found",
        "serial_number": None,
        "confidence": 0.0,
        "ocr_time_ms": round(total_ms, 1),
        "engine": "Standalone Tesseract (Local)",
    }


def send_burst_to_pc(frames: List[np.ndarray], pc_url: str) -> Optional[Dict[str, Any]]:
    """Uploads burst frames to PC neural inference server via multipart/form-data."""
    endpoint = pc_url.rstrip("/") + "/api/process_burst"
    boundary = "----WebKitFormBoundary" + hex(int(time.time() * 1000))[2:]
    body = bytearray()

    for idx, frame in enumerate(frames):
        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="files"; filename="shot_{idx}.jpg"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(buf.tobytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(endpoint, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            t1 = time.time()
            data = json.loads(response.read().decode("utf-8"))
            print(f"[Network] PC response in {(t1-t0)*1000:.0f}ms -> Serial: {data.get('serial_number')}")
            return data
    except Exception as e:
        print(f"[Network] PC server unreachable at {endpoint}: {e}")
        return None


def execute_capture_and_ocr():
    """Triggered by Button 1 (GPIO 17 / Pin 11 to GND) or UI Click."""
    global app_state, hdmi_ui

    with state_lock:
        if app_state["is_capturing"]:
            return
        app_state["is_capturing"] = True
        app_state["last_result"]["status"] = "processing"
        burst_n = app_state["burst_count"]
        pc_url = app_state["pc_url"]
        roi = app_state["roi"]

    if hdmi_ui:
        hdmi_ui.set_state(StickCamHdmiUI.STATE_CAPTURING, "Capturing burst & running OCR...")

    # 1. Capture Burst Frames
    frames = capture_burst(burst_n)
    if not frames:
        frames = [capture_single_frame()]

    # 2. Score Frames by Sharpness
    scored = [(calculate_sharpness(f, roi), f) for f in frames if f is not None]
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_sharpness, best_frame = scored[0]
    else:
        best_frame = capture_single_frame()
        best_sharpness = 0.0

    result = None

    # 3. Attempt PC Server Offload (if configured)
    if pc_url:
        result = send_burst_to_pc(frames, pc_url)
        if result and hdmi_ui:
            hdmi_ui.pc_connected = True

    # 4. Fallback to Local Offline OCR
    if not result or result.get("status") == "error":
        if pc_url and hdmi_ui:
            hdmi_ui.pc_connected = False
        print("[OCR] Running Standalone Local Offline OCR on Raspberry Pi...")
        result = run_local_offline_ocr(best_frame, roi)

    # 5. Update UI and Global State
    result["timestamp"] = time.strftime("%H:%M:%S")
    with state_lock:
        app_state["is_capturing"] = False
        app_state["last_result"] = result

    if hdmi_ui:
        hdmi_ui.set_result(result)


def execute_recapture_reset():
    """Triggered by Button 2 (GPIO 27 / Pin 13 to GND) or UI Click."""
    global app_state, hdmi_ui
    with state_lock:
        app_state["is_capturing"] = False
        app_state["last_result"] = {
            "status": "idle",
            "message": "Live Viewfinder Resumed. Ready.",
            "serial_number": None,
        }

    if hdmi_ui:
        hdmi_ui.trigger_recapture(source="BUTTON 2 (GPIO 27)")


# ---------------------------------------------------------------------------
# Background Ingestion & HDMI Refresh Threads
# ---------------------------------------------------------------------------

def camera_worker_loop():
    """Continuously ingests camera frames, computes focus telemetry, and feeds HDMI UI."""
    global running, app_state, hdmi_ui
    print("[Worker] Camera ingestion thread active.")

    while running:
        frame = capture_single_frame()
        if frame is not None:
            with state_lock:
                roi = app_state["roi"]

            sharpness = calculate_sharpness(frame, roi)

            # Classify focus
            if sharpness > 320:
                quality = "SHARP"
            elif sharpness > 160:
                quality = "FAIR"
            else:
                quality = "BLURRY"

            with state_lock:
                app_state["sharpness"] = sharpness
                app_state["focus_quality"] = quality
                # Encode preview JPEG for web
                ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if ret:
                    app_state["latest_camera_jpeg"] = buf.tobytes()

            if hdmi_ui:
                hdmi_ui.update_viewfinder_frame(frame, sharpness)

        time.sleep(0.033)  # ~30 FPS loop


def hdmi_display_loop():
    """Continuously renders the 3.5\" HDMI UI and pushes to /dev/fb0 at ~30 FPS."""
    global running, hdmi_ui, app_state
    print("[HDMI] Display refresh thread active.")

    while running:
        if hdmi_ui is not None:
            rendered = hdmi_ui.render()
            with state_lock:
                ret, buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    app_state["latest_hdmi_jpeg"] = buf.tobytes()

        time.sleep(0.033)


# ---------------------------------------------------------------------------
# Embedded Web Dashboard Server (Port 8000)
# ---------------------------------------------------------------------------

WEB_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stick Cam HD - 3.5" HDMI Seal Scanner</title>
<style>
  :root {
    --bg: #090d16;
    --card: #131b2e;
    --border: #223250;
    --accent: #38bdf8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #f8fafc;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
  }
  header {
    width: 100%;
    max-width: 900px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  h1 { font-size: 1.35rem; display: flex; align-items: center; gap: 8px; color: #fff; }
  .badge-hdmi {
    background: #0284c7;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
    width: 100%;
    max-width: 900px;
  }
  @media (min-width: 768px) {
    .grid { grid-template-columns: 1fr 1fr; }
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .card-header {
    padding: 10px 14px;
    background: #0d1424;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
  }
  .view-container {
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    min-height: 240px;
  }
  .view-container img { width: 100%; height: auto; display: block; }
  .controls-card {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .btn-row { display: flex; gap: 10px; }
  button {
    flex: 1;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 700;
    cursor: pointer;
    transition: filter 0.15s ease;
  }
  button:hover { filter: brightness(1.15); }
  button:active { filter: brightness(0.9); }
  .btn-cap { background: var(--success); color: #fff; }
  .btn-recap { background: #0284c7; color: #fff; }
  .result-box {
    background: #0d1424;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
  }
  .serial-display {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--success);
    letter-spacing: 2px;
    margin: 6px 0;
  }
  .telemetry-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--muted);
    padding: 4px 0;
  }
</style>
</head>
<body>
  <header>
    <h1>📸 STICK CAM <span class="badge-hdmi">3.5" HDMI DISPLAY</span></h1>
    <div style="font-size: 0.8rem; color: var(--muted);">Pi IP: <strong style="color: var(--accent);">{PI_IP}</strong></div>
  </header>

  <div class="grid">
    <!-- 3.5" HDMI Screen Live Mirror -->
    <div class="card">
      <div class="card-header">
        <span>3.5" HDMI Screen Mirror (Direct Display)</span>
        <span style="color: var(--accent);">Live 30 FPS</span>
      </div>
      <div class="view-container">
        <img src="/hdmi_mirror.mjpg" alt="3.5 HDMI Mirror">
      </div>
    </div>

    <!-- Controls & Results -->
    <div class="card controls-card">
      <div class="btn-row">
        <button class="btn-cap" onclick="triggerCapture()">📸 BTN 1: CAPTURE & OCR</button>
        <button class="btn-recap" onclick="triggerReset()">🔄 BTN 2: RESET</button>
      </div>

      <div class="result-box">
        <div style="font-size: 0.75rem; color: var(--muted); text-transform: uppercase;">Latest Recognized Seal Number</div>
        <div id="serialText" class="serial-display">---</div>
        <div class="telemetry-row">
          <span>Confidence: <strong id="confText">--</strong></span>
          <span>Time: <strong id="timeText">--</strong></span>
        </div>
      </div>

      <div style="background: #0d1424; padding: 12px; border-radius: 8px; border: 1px solid var(--border); font-size: 0.82rem;">
        <div style="font-weight: 600; margin-bottom: 6px; color: var(--accent);">Hardware Physical Buttons:</div>
        <div>• <strong>Button 1 (GPIO 17 / Pin 11 to GND)</strong>: Triggers Capture & OCR</div>
        <div>• <strong>Button 2 (GPIO 27 / Pin 13 to GND)</strong>: Resets Live Viewfinder</div>
      </div>
    </div>
  </div>

  <script>
    function triggerCapture() {
      fetch('/api/capture', {method: 'POST'});
    }
    function triggerReset() {
      fetch('/api/recapture', {method: 'POST'});
    }
    function pollStatus() {
      fetch('/api/status')
        .then(r => r.json())
        .then(data => {
          if (data.last_result && data.last_result.serial_number) {
            document.getElementById('serialText').innerText = data.last_result.serial_number;
            document.getElementById('confText').innerText = ((data.last_result.confidence || 0) * 100).toFixed(1) + '%';
            document.getElementById('timeText').innerText = (data.last_result.ocr_time_ms || 0) + 'ms';
          }
        });
    }
    setInterval(pollStatus, 800);
  </script>
</body>
</html>
"""


class WebStreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = WEB_PAGE_TEMPLATE.replace("{PI_IP}", get_local_ip())
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with state_lock:
                data = json.dumps(app_state, default=str)
            self.wfile.write(data.encode("utf-8"))

        elif self.path == "/hdmi_mirror.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while running:
                    with state_lock:
                        jpeg = app_state.get("latest_hdmi_jpeg")
                    if jpeg:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)
            except (ConnectionResetError, BrokenPipeError):
                pass

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while running:
                    with state_lock:
                        jpeg = app_state.get("latest_camera_jpeg")
                    if jpeg:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)
            except (ConnectionResetError, BrokenPipeError):
                pass

        elif self.path == "/snapshot":
            frame = capture_single_frame()
            if frame is not None:
                ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ret:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Disposition", 'attachment; filename="snapshot.jpg"')
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                    return
            self.send_error(503, "Camera frame unavailable")

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/capture":
            threading.Thread(target=execute_capture_and_ocr, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"capturing"}')

        elif self.path == "/api/recapture":
            execute_recapture_reset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"reset"}')
        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    global running, hdmi_ui, app_state

    parser = argparse.ArgumentParser(description="Stick Cam 3.5\" HDMI Display Scanner")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Web server port (default: {DEFAULT_PORT})")
    parser.add_argument("--pc", type=str, default=None, help="PC OCR server URL (e.g. http://192.168.68.123:5000)")
    parser.add_argument("--burst", type=int, default=DEFAULT_BURST_COUNT, help="Burst shot count (default: 3)")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Capture width (default: {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Capture height (default: {DEFAULT_HEIGHT})")
    parser.add_argument("--sim", action="store_true", help="Run in desktop simulator mode")
    parser.add_argument("--no-web", action="store_true", help="Disable embedded web server")
    args = parser.parse_args()

    app_state["pc_url"] = args.pc
    app_state["burst_count"] = args.burst

    # 1. Initialize 3.5" HDMI UI Engine
    print("\n" + "=" * 65)
    print("  STARTING STICK CAM 3.5\" HDMI DISPLAY SCANNER")
    print("=" * 65)
    hdmi_ui = StickCamHdmiUI(
        on_capture_cb=execute_capture_and_ocr,
        on_recapture_cb=execute_recapture_reset,
        sim_mode=args.sim,
        width=480,
        height=320
    )
    if args.pc:
        hdmi_ui.pc_connected = True
        hdmi_ui.server_mode = f"PC Server ({args.pc})"

    # 2. Initialize Camera
    init_camera(args.width, args.height, args.device)

    # 3. Start Background Ingestion & HDMI Refresh Threads
    t_cam = threading.Thread(target=camera_worker_loop, daemon=True)
    t_cam.start()

    t_disp = threading.Thread(target=hdmi_display_loop, daemon=True)
    t_disp.start()

    # 4. Start Embedded Web Server
    httpd = None
    if not args.no_web:
        server_addr = ("", args.port)
        httpd = HTTPServer(server_addr, WebStreamingHandler)
        my_ip = get_local_ip()
        print(f"[Web] Dashboard & stream active at http://{my_ip}:{args.port}")
        t_web = threading.Thread(target=httpd.serve_forever, daemon=True)
        t_web.start()

    # 5. Run Simulator or Service Loop
    try:
        if args.sim:
            hdmi_ui.start_gui_loop()
        else:
            print("[HDMI] Scanner service running! Press Ctrl+C to terminate.")
            while running:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Service] Terminating...")
    finally:
        running = False
        if httpd:
            httpd.server_close()
        print("[Service] Stopped cleanly.")


if __name__ == "__main__":
    main()
