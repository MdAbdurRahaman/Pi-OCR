import argparse
import csv
import io
import json
import os
import re
import subprocess
import threading
import time
from http import server
from pathlib import Path
import cv2
import numpy as np

# Global streaming state
latest_frame = None
frame_lock = threading.Lock()
running = True
active_camera_type = "Detecting..."

# ROI and camera config (normalized fractions)
roi_config = {
    "x": 0.25,
    "y": 0.35,
    "w": 0.50,
    "h": 0.30,
    "zoom": 1.0,
    "show_roi": True,
    "digits": 0,
}

# Live focus telemetry
live_telemetry = {
    "sharpness": 0.0,
    "focus_quality": "BLURRY",
    "focus_color": [0, 0, 255], # BGR
    "camera": "Unknown",
}

# Real-time scan state
scan_lock = threading.Lock()
scan_status = {
    "state": "idle",
    "message": "Ready. Align lock inside box. Watch the focus indicator turn green.",
    "candidates": [],
    "best_number": None,
    "best_score": 0.0,
    "frames_evaluated": 0,
    "best_sharpness": 0.0,
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pi Seal Scanner - CSI Camera Branch</title>
<style>
  :root {
    --bg: #0a0e17;
    --card: #131b2e;
    --card-border: #223250;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --success: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
  }
  header { margin-bottom: 16px; text-align: center; }
  h1 {
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: center;
  }
  .badge-live {
    background: var(--danger);
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 9999px;
    animation: pulse 2s infinite;
  }
  .badge-cam {
    background: var(--purple);
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 9999px;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .container {
    display: flex;
    flex-direction: column;
    max-width: 960px;
    width: 100%;
    gap: 16px;
  }
  .video-card {
    background: var(--card);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 16px 36px -8px rgba(0,0,0,0.6);
    border: 1px solid var(--card-border);
    position: relative;
  }
  .video-container {
    position: relative;
    width: 100%;
    background: #000;
    display: flex;
    justify-content: center;
  }
  .video-container img {
    max-width: 100%;
    height: auto;
    display: block;
  }
  .focus-bar-container {
    background: #0d1424;
    padding: 12px 18px;
    border-bottom: 1px solid var(--card-border);
    display: flex;
    align-items: center;
    gap: 15px;
  }
  .focus-meter-track {
    flex: 1;
    height: 12px;
    background: #1e293b;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
  }
  .focus-meter-fill {
    height: 100%;
    width: 0%;
    background: var(--danger);
    transition: width 0.15s ease, background-color 0.2s ease;
  }
  .focus-status-badge {
    font-size: 0.85rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 110px;
    text-align: center;
  }
  .controls-card {
    background: var(--card);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid var(--card-border);
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    padding: 11px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .btn-primary {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #fff;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
  }
  .btn-primary:hover:not(:disabled) {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    transform: translateY(-1px);
  }
  .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
  .btn-secondary {
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
  }
  .btn-secondary:hover { background: #334155; color: #fff; }
  .slider-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    background: #0d1424;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #1a253c;
  }
  .slider-item {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .slider-item label {
    font-size: 0.82rem;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
  }
  .slider-item span { color: var(--accent); font-family: monospace; font-weight: 600; }
  input[type="range"] { accent-color: var(--accent); cursor: pointer; }
  .result-banner {
    background: #090d16;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid var(--card-border);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .status-text { font-size: 0.95rem; color: #e2e8f0; }
  .best-match {
    display: flex;
    align-items: center;
    gap: 15px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 12px 16px;
    border-radius: 8px;
  }
  .best-digits {
    font-size: 2rem;
    font-weight: 800;
    font-family: "Courier New", Courier, monospace;
    color: #34d399;
    letter-spacing: 2px;
  }
  .candidates-list {
    font-family: monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.6;
    max-height: 120px;
    overflow-y: auto;
  }
  .tip-box {
    background: rgba(56, 189, 248, 0.08);
    border-left: 3px solid var(--accent);
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #bae6fd;
    border-radius: 0 8px 8px 0;
  }
</style>
</head>
<body>
  <header>
    <h1>Pi Seal Scanner <span class="badge-live">LIVE</span> <span class="badge-cam" id="camBadge">CSI Camera</span></h1>
    <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 4px;">
      Live CSI Camera Feed (Sony IMX219) with Real-Time Focus Assist & Auto-OCR
    </p>
  </header>

  <div class="container">
    <div class="video-card">
      <div class="focus-bar-container">
        <span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8; white-space: nowrap;">Focus Meter:</span>
        <div class="focus-meter-track">
          <div class="focus-meter-fill" id="focusMeterFill"></div>
        </div>
        <div class="focus-status-badge" id="focusStatusBadge" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">
          BLURRY
        </div>
      </div>
      <div class="video-container">
        <img src="/stream.mjpg" id="liveFeed" alt="Live Camera Feed">
      </div>
    </div>

    <div class="tip-box">
      📷 <strong>CSI Camera Tips:</strong> If using a lens with a twist-focus ring, gently rotate the lens ring while watching the Focus Meter turn Green. You can also use the <strong>Digital Zoom slider</strong> to enlarge the seal text!
    </div>

    <div class="controls-card">
      <div class="btn-row">
        <button class="btn btn-primary" id="btnStartScan" onclick="startScan()">
          ▶ Start Scan (Auto-Detect Best In-Focus Shots)
        </button>
        <button class="btn btn-secondary" onclick="saveSnapshot()">
          💾 Save Snapshot
        </button>
        <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 0.9rem; margin-left: auto;">
          <input type="checkbox" id="showRoi" checked onchange="updateRoi()"> Show ROI Box
        </label>
        <div style="display: flex; align-items: center; gap: 6px;">
          <label style="font-size: 0.85rem; color: var(--text-muted);">Exact Digits:</label>
          <input type="number" id="digitsFilter" value="0" min="0" max="20" style="width: 55px; padding: 5px 8px; background: #0d1424; color: #fff; border: 1px solid #334155; border-radius: 6px;" onchange="updateRoi()">
        </div>
      </div>

      <div class="slider-grid">
        <div class="slider-item">
          <label>Digital Zoom: <span id="valZoom">1.0x</span></label>
          <input type="range" id="zoom" min="1.0" max="3.0" step="0.1" value="1.0" oninput="updateRoi()">
        </div>
        <div class="slider-item">
          <label>ROI X (Left): <span id="valX">0.25</span></label>
          <input type="range" id="roiX" min="0" max="0.8" step="0.02" value="0.25" oninput="updateRoi()">
        </div>
        <div class="slider-item">
          <label>ROI Y (Top): <span id="valY">0.35</span></label>
          <input type="range" id="roiY" min="0" max="0.8" step="0.02" value="0.35" oninput="updateRoi()">
        </div>
        <div class="slider-item">
          <label>ROI Width: <span id="valW">0.50</span></label>
          <input type="range" id="roiW" min="0.1" max="0.9" step="0.02" value="0.50" oninput="updateRoi()">
        </div>
        <div class="slider-item">
          <label>ROI Height: <span id="valH">0.30</span></label>
          <input type="range" id="roiH" min="0.05" max="0.6" step="0.02" value="0.30" oninput="updateRoi()">
        </div>
      </div>

      <div class="result-banner">
        <div class="status-text" id="statusText">
          Ready. Align lock inside box. Watch the focus indicator turn green.
        </div>

        <div class="best-match" id="bestMatchBox" style="display: none;">
          <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #6ee7b7; font-weight: 600;">Detected Seal Number</div>
            <div class="best-digits" id="bestNumberDisplay">-------</div>
          </div>
          <div style="margin-left: auto; text-align: right;">
            <div style="font-size: 0.75rem; color: var(--text-muted);">Confidence Score</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;" id="bestScoreDisplay">0.0</div>
          </div>
        </div>

        <div class="candidates-list" id="candidatesList" style="display: none;"></div>
      </div>
    </div>
  </div>

  <script>
    let isScanning = false;
    let pollInterval = null;

    setInterval(() => {
      fetch('/telemetry')
        .then(r => r.json())
        .then(data => {
          if (data.camera) {
            document.getElementById('camBadge').innerText = data.camera;
          }
          const fill = document.getElementById('focusMeterFill');
          const badge = document.getElementById('focusStatusBadge');
          const pct = Math.min(100, Math.max(0, (data.sharpness / 600) * 100));
          fill.style.width = pct.toFixed(1) + '%';

          if (data.focus_quality === 'SHARP') {
            fill.style.backgroundColor = '#10b981';
            badge.style.background = 'rgba(16, 185, 129, 0.2)';
            badge.style.color = '#34d399';
            badge.innerText = 'SHARP (' + Math.round(data.sharpness) + ')';
          } else if (data.focus_quality === 'FAIR') {
            fill.style.backgroundColor = '#f59e0b';
            badge.style.background = 'rgba(245, 158, 11, 0.2)';
            badge.style.color = '#fbbf24';
            badge.innerText = 'FAIR (' + Math.round(data.sharpness) + ')';
          } else {
            fill.style.backgroundColor = '#ef4444';
            badge.style.background = 'rgba(239, 68, 68, 0.2)';
            badge.style.color = '#f87171';
            badge.innerText = 'BLURRY (' + Math.round(data.sharpness) + ')';
          }
        })
        .catch(() => {});
    }, 300);

    function updateRoi() {
      const x = parseFloat(document.getElementById('roiX').value);
      const y = parseFloat(document.getElementById('roiY').value);
      const w = parseFloat(document.getElementById('roiW').value);
      const h = parseFloat(document.getElementById('roiH').value);
      const zoom = parseFloat(document.getElementById('zoom').value);
      const show = document.getElementById('showRoi').checked;
      const digits = parseInt(document.getElementById('digitsFilter').value) || 0;

      document.getElementById('valX').innerText = x.toFixed(2);
      document.getElementById('valY').innerText = y.toFixed(2);
      document.getElementById('valW').innerText = w.toFixed(2);
      document.getElementById('valH').innerText = h.toFixed(2);
      document.getElementById('valZoom').innerText = zoom.toFixed(1) + 'x';

      fetch('/set_roi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({x, y, w, h, zoom, show_roi: show, digits})
      });
    }

    function startScan() {
      if (isScanning) return;
      isScanning = true;
      const btn = document.getElementById('btnStartScan');
      btn.disabled = true;
      btn.innerText = '🔍 Auto-Focus Tracking & Burst Scanning...';
      document.getElementById('statusText').innerText = 'Locking on... Evaluating frames for peak focus sharpness.';
      document.getElementById('bestMatchBox').style.display = 'none';
      document.getElementById('candidatesList').style.display = 'none';

      fetch('/start_scan', {method: 'POST'})
        .then(() => {
          pollInterval = setInterval(checkScanStatus, 500);
        })
        .catch(err => {
          endScan();
          document.getElementById('statusText').innerText = 'Failed to start scan: ' + err;
        });
    }

    function checkScanStatus() {
      fetch('/scan_status')
        .then(r => r.json())
        .then(data => {
          document.getElementById('statusText').innerText = data.message;
          if (data.state === 'complete' || data.state === 'error') {
            clearInterval(pollInterval);
            endScan();
            if (data.best_number) {
              document.getElementById('bestMatchBox').style.display = 'flex';
              document.getElementById('bestNumberDisplay').innerText = data.best_number;
              document.getElementById('bestScoreDisplay').innerText = data.best_score.toFixed(1) + '%';
            }
            if (data.candidates && data.candidates.length > 0) {
              const list = document.getElementById('candidatesList');
              list.style.display = 'block';
              list.innerHTML = '<strong>Detected Candidates:</strong><br>' + 
                data.candidates.map(c => `• ${c[0]} (score: ${c[1].toFixed(1)})`).join('<br>');
            }
          }
        });
    }

    function endScan() {
      isScanning = false;
      const btn = document.getElementById('btnStartScan');
      btn.disabled = false;
      btn.innerText = '▶ Start Scan (Auto-Detect Best In-Focus Shots)';
    }

    function saveSnapshot() {
      window.open('/snapshot', '_blank');
    }
  </script>
</body>
</html>
"""

def camera_thread_loop(backend="auto", device_id=0, width=1280, height=720):
    global latest_frame, running, active_camera_type

    # Try CSI (Picamera2) first if requested or auto
    use_picam2 = False
    picam2 = None

    if backend in ["auto", "csi"]:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2(0)
            config = picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
            picam2.configure(config)
            picam2.start()
            use_picam2 = True
            active_camera_type = "CSI Camera (Sony IMX219)"
            live_telemetry["camera"] = "CSI Camera (Sony IMX219)"
            print("[Camera] Picamera2 CSI camera successfully started!")
        except Exception as e:
            print(f"[Camera] Picamera2 init failed: {e}. Falling back to OpenCV V4L2...")

    cap = None
    if not use_picam2:
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        active_camera_type = "USB Camera (/dev/video0)"
        live_telemetry["camera"] = "USB Camera (/dev/video0)"
        if not cap.isOpened():
            print(f"[Error] Could not open camera {device_id}")
            return
        print(f"[Camera] V4L2 USB camera active on {device_id}")

    while running:
        if use_picam2:
            try:
                frame = picam2.capture_array()
            except Exception:
                time.sleep(0.04)
                continue
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

        # Handle digital zoom
        zoom = roi_config.get("zoom", 1.0)
        if zoom > 1.05:
            h, w = frame.shape[:2]
            zh, zw = int(h / zoom), int(w / zoom)
            y1 = (h - zh) // 2
            x1 = (w - zw) // 2
            frame = cv2.resize(frame[y1:y1+zh, x1:x1+zw], (w, h), interpolation=cv2.INTER_LINEAR)

        with frame_lock:
            latest_frame = frame.copy()

        # Update real-time focus telemetry
        h, w = frame.shape[:2]
        rx = int(roi_config["x"] * w)
        ry = int(roi_config["y"] * h)
        rw = int(roi_config["w"] * w)
        rh = int(roi_config["h"] * h)
        crop = frame[ry:ry+rh, rx:rx+rw]
        if crop.size > 0:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            s_val = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            live_telemetry["sharpness"] = s_val
            if s_val > 350:
                live_telemetry["focus_quality"] = "SHARP"
                live_telemetry["focus_color"] = [0, 255, 0] # Green
            elif s_val > 180:
                live_telemetry["focus_quality"] = "FAIR"
                live_telemetry["focus_color"] = [0, 215, 255] # Yellow
            else:
                live_telemetry["focus_quality"] = "BLURRY"
                live_telemetry["focus_color"] = [0, 0, 255] # Red

        time.sleep(0.03)

    if use_picam2 and picam2 is not None:
        picam2.stop()
    elif cap is not None:
        cap.release()

def enhance_text_focus(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 1200:
        scale = 1200 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 2.0)
    sharpened = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)

    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)

    return sharpened, binary

def perform_fast_ocr(image_bgr, digits_expected=0):
    sharpened, binary = enhance_text_focus(image_bgr)
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"
    candidates = {}

    for idx, processed in enumerate([sharpened, binary]):
        padded = cv2.copyMakeBorder(processed, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=255)
        temp_path = f"/tmp/crop_ocr_{idx}.png"
        cv2.imwrite(temp_path, padded)

        cmd = [
            "tesseract",
            temp_path,
            "stdout",
            "-l", "eng",
            "--oem", "1",
            "--psm", "7",
            "-c", "tessedit_char_whitelist=0123456789",
            "tsv",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, env=env)
            rows = csv.DictReader(io.StringIO(res.stdout), delimiter="\t")
            for row in rows:
                text = row.get("text", "").strip()
                if not re.fullmatch(r"[0-9]+", text):
                    continue
                if digits_expected > 0 and len(text) != digits_expected:
                    continue
                elif digits_expected == 0 and not (4 <= len(text) <= 14):
                    continue
                conf = float(row.get("conf", -1))
                if conf >= 0:
                    candidates[text] = max(conf, candidates.get(text, -1))
        except Exception:
            continue

    return candidates

def scan_worker():
    global scan_status
    with scan_lock:
        scan_status["state"] = "detecting"
        scan_status["message"] = "Auto-Focus Tracking: Capturing burst to select peak sharpest frames..."
        scan_status["candidates"] = []
        scan_status["best_number"] = None

    collected = []
    start_t = time.time()

    while time.time() - start_t < 2.5:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            h, w = frame.shape[:2]
            rx = int(roi_config["x"] * w)
            ry = int(roi_config["y"] * h)
            rw = int(roi_config["w"] * w)
            rh = int(roi_config["h"] * h)
            crop = frame[ry:ry+rh, rx:rx+rw]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                collected.append((sharpness, crop))
        time.sleep(0.07)

    if not collected:
        with scan_lock:
            scan_status["state"] = "error"
            scan_status["message"] = "No frames available. Check camera."
        return

    collected.sort(key=lambda x: x[0], reverse=True)
    best_shots = collected[:4]
    peak_sharpness = best_shots[0][0]

    with scan_lock:
        scan_status["state"] = "analyzing"
        scan_status["frames_evaluated"] = len(collected)
        scan_status["best_sharpness"] = peak_sharpness
        scan_status["message"] = f"Selected {len(best_shots)} peak frames (Sharpness: {peak_sharpness:.1f}). Running OCR..."

    all_scores = {}
    digits_expected = roi_config.get("digits", 0)

    for rank, (score, crop) in enumerate(best_shots):
        candidates = perform_fast_ocr(crop, digits_expected=digits_expected)
        for num, conf in candidates.items():
            if num not in all_scores:
                all_scores[num] = {"conf_max": conf, "count": 1, "conf_sum": conf}
            else:
                all_scores[num]["conf_max"] = max(all_scores[num]["conf_max"], conf)
                all_scores[num]["count"] += 1
                all_scores[num]["conf_sum"] += conf

    ranked = []
    for num, meta in all_scores.items():
        avg_conf = meta["conf_sum"] / meta["count"]
        final_score = avg_conf + (meta["count"] - 1) * 5.0
        ranked.append((num, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    with scan_lock:
        scan_status["state"] = "complete"
        scan_status["candidates"] = ranked
        if ranked:
            scan_status["best_number"] = ranked[0][0]
            scan_status["best_score"] = ranked[0][1]
            scan_status["message"] = f"Success! Number: {ranked[0][0]} (Score: {ranked[0][1]:.1f} across {len(best_shots)} shots)."
        else:
            scan_status["message"] = f"No digits recognized (Peak Sharpness: {peak_sharpness:.1f}). Try adjusting distance or increase Digital Zoom."

class StreamingHandler(server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(live_telemetry).encode("utf-8"))
        elif self.path == "/snapshot":
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                self.send_error(503, "Camera frame not available")
                return
            ret, jpeg = cv2.imencode(".jpg", frame)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Disposition", 'attachment; filename="snapshot.jpg"')
            self.end_headers()
            self.wfile.write(jpeg.tobytes())
        elif self.path == "/scan_status":
            with scan_lock:
                data = json.dumps(scan_status)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while running:
                    with frame_lock:
                        if latest_frame is None:
                            time.sleep(0.04)
                            continue
                        draw_frame = latest_frame.copy()

                    box_color = live_telemetry.get("focus_color", [0, 255, 255])
                    quality = live_telemetry.get("focus_quality", "BLURRY")
                    sharp_val = live_telemetry.get("sharpness", 0.0)

                    if roi_config.get("show_roi", True):
                        h, w = draw_frame.shape[:2]
                        rx = int(roi_config["x"] * w)
                        ry = int(roi_config["y"] * h)
                        rw = int(roi_config["w"] * w)
                        rh = int(roi_config["h"] * h)

                        cv2.rectangle(draw_frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
                        cx, cy = rx + rw // 2, ry + rh // 2
                        cv2.line(draw_frame, (cx - 15, cy), (cx + 15, cy), box_color, 1)
                        cv2.line(draw_frame, (cx, cy - 15), (cx, cy + 15), box_color, 1)

                        status_str = f"FOCUS: {quality} ({int(sharp_val)})"
                        cv2.putText(draw_frame, status_str, (rx, max(22, ry - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                    ret, jpeg = cv2.imencode(".jpg", draw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if not ret:
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    time.sleep(0.06)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/set_roi":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            roi_config.update(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/start_scan":
            t = threading.Thread(target=scan_worker, daemon=True)
            t.start()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"started"}')
        else:
            self.send_error(404)

def main():
    parser = argparse.ArgumentParser(description="Live camera stream server with CSI and USB support")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--camera", choices=["auto", "csi", "usb"], default="auto", help="Camera source")
    parser.add_argument("--device", type=int, default=0, help="Camera device index for USB (default: 0)")
    args = parser.parse_args()

    t = threading.Thread(target=camera_thread_loop, kwargs={"backend": args.camera, "device_id": args.device}, daemon=True)
    t.start()

    server_address = ("", args.port)
    httpd = server.ThreadingHTTPServer(server_address, StreamingHandler)
    print(f"[Server] Live video stream running on http://0.0.0.0:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        global running
        running = False
        httpd.server_close()

if __name__ == "__main__":
    main()
