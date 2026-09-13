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
    "x": 0.15,
    "y": 0.20,
    "w": 0.70,
    "h": 0.55,
    "zoom": 1.0,
    "show_roi": True,
    "digits": 0,
}

# Live focus telemetry
live_telemetry = {
    "sharpness": 0.0,
    "focus_quality": "BLURRY",
    "focus_color": [0, 0, 255],
    "camera": "Unknown",
}

# Real-time scan state
scan_lock = threading.Lock()
scan_status = {
    "state": "idle",
    "message": "Ready. Align seal inside box and click 'Start Fast Scan'.",
    "candidates": [],
    "all_words": [],
    "best_number": None,
    "best_score": 0.0,
    "ocr_time": 0.0,
    "frames_evaluated": 0,
    "best_sharpness": 0.0,
    "has_image": False,
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pi Seal Scanner - Fast Auto-OCR</title>
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
    font-size: 2.2rem;
    font-weight: 800;
    font-family: "Courier New", Courier, monospace;
    color: #34d399;
    letter-spacing: 2px;
  }
  .candidates-list {
    font-family: monospace;
    font-size: 0.88rem;
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
      Color-Aware Seal OCR with Fast Single-Shot Detection (< 2 sec)
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
      ⚡ <strong>Ultra-Fast OCR:</strong> The scanner now auto-isolates the yellow seal plastic and reads numbers with letters (e.g. <strong>C 581819</strong>) in under 2 seconds! Use Digital Zoom (e.g. 1.8x) to make the text larger and sharper.
    </div>

    <div class="controls-card">
      <div class="btn-row">
        <button class="btn btn-primary" id="btnStartScan" onclick="startScan()">
          ⚡ Start Fast Scan
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
          <label>ROI X (Left): <span id="valX">0.15</span></label>
          <input type="range" id="roiX" min="0" max="0.8" step="0.02" value="0.15" oninput="updateRoi()">
        </div>
        <div class="slider-item">
          <label>ROI Y (Top): <span id="valY">0.20</span></label>
          <input type="range" id="roiY" min="0" max="0.8" step="0.02" value="0.20" oninput="updateRoi()">
        </div>
        <div class="slider-item">
          <label>ROI Width: <span id="valW">0.70</span></label>
          <input type="range" id="roiW" min="0.1" max="0.95" step="0.02" value="0.70" oninput="updateRoi()">
        </div>
        <div class="slider-item">
          <label>ROI Height: <span id="valH">0.55</span></label>
          <input type="range" id="roiH" min="0.05" max="0.8" step="0.02" value="0.55" oninput="updateRoi()">
        </div>
      </div>

      <div class="result-banner">
        <div class="status-text" id="statusText">
          Ready. Align seal inside box and click "Start Fast Scan".
        </div>

        <div class="best-match" id="bestMatchBox" style="display: none;">
          <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #6ee7b7; font-weight: 600;">Detected Seal / Text</div>
            <div class="best-digits" id="bestNumberDisplay">-------</div>
          </div>
          <div style="margin-left: auto; text-align: right;">
            <div style="font-size: 0.75rem; color: var(--text-muted);">Speed / Confidence</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8;" id="bestScoreDisplay">0.0s</div>
          </div>
        </div>

        <div class="candidates-list" id="candidatesList" style="display: none;"></div>

        <div id="detectionPreviewBox" style="display: none; margin-top: 14px; text-align: center;">
          <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px; font-weight: 600;">EDATEC Localized Text Detection (Green Bounding Boxes):</div>
          <img id="detectedResultImg" style="max-width: 100%; border-radius: 8px; border: 1px solid #223250; box-shadow: 0 4px 12px rgba(0,0,0,0.5);" />
        </div>
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
      btn.innerText = '⚡ Reading Seal...';
      document.getElementById('statusText').innerText = 'Capturing sharpest frame & running color-isolated OCR...';
      document.getElementById('bestMatchBox').style.display = 'none';
      document.getElementById('candidatesList').style.display = 'none';

      fetch('/start_scan', {method: 'POST'})
        .then(() => {
          pollInterval = setInterval(checkScanStatus, 300);
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
              document.getElementById('bestScoreDisplay').innerText = data.ocr_time.toFixed(2) + 's (Score ' + data.best_score.toFixed(0) + '%)';
            }
            if (data.candidates && data.candidates.length > 0) {
              const list = document.getElementById('candidatesList');
              list.style.display = 'block';
              list.innerHTML = '<strong>Detected Seal Numbers:</strong><br>' + 
                data.candidates.map(c => `• ${c[0]} (confidence: ${c[1].toFixed(1)}%)`).join('<br>');
            } else if (data.all_words && data.all_words.length > 0) {
              const list = document.getElementById('candidatesList');
              list.style.display = 'block';
              list.innerHTML = '<strong>Recognized Words:</strong><br>' + 
                data.all_words.map(w => `• ${w[0]} (${w[1]}%)`).join(' ');
            }
            if (data.has_image) {
              const pbox = document.getElementById('detectionPreviewBox');
              const pimg = document.getElementById('detectedResultImg');
              pimg.src = '/detected_result.jpg?t=' + Date.now();
              pbox.style.display = 'block';
            }
          }
        });
    }

    function endScan() {
      isScanning = false;
      const btn = document.getElementById('btnStartScan');
      btn.disabled = false;
      btn.innerText = '⚡ Start Fast Scan';
    }

    function saveSnapshot() {
      window.open('/snapshot', '_blank');
    }
  </script>
</body>
</html>
"""

active_picam2 = None

def camera_thread_loop(backend="auto", device_id=0, width=1280, height=720):
    global latest_frame, running, active_camera_type, active_picam2

    use_picam2 = False
    picam2 = None

    if backend in ["auto", "csi"]:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2(0)
            # Hardware dual-stream: main (1280x720) for high-res OCR/snapshot, lores (640x360) for fast streaming
            config = picam2.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
                lores={"size": (640, 360), "format": "YUV420"}
            )
            picam2.configure(config)
            picam2.start()
            use_picam2 = True
            active_picam2 = picam2
            active_camera_type = "CSI Camera (Sony IMX219 Dual-Stream)"
            live_telemetry["camera"] = "CSI Camera (Sony IMX219)"
            print("[Camera] Picamera2 CSI hardware dual-stream active!")
        except Exception as e:
            print(f"[Camera] Picamera2 init: {e}. Falling back to USB...")

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

    frame_counter = 0
    while running:
        if use_picam2:
            try:
                # Fast lores hardware preview (640x360 YUV420)
                yuv = picam2.capture_array("lores")
                preview = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            except Exception:
                time.sleep(0.01)
                continue
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            preview = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

        # Digital zoom on preview if requested
        zoom = roi_config.get("zoom", 1.0)
        if zoom > 1.05:
            ph, pw = preview.shape[:2]
            zh, zw = int(ph / zoom), int(pw / zoom)
            y1 = (ph - zh) // 2
            x1 = (pw - zw) // 2
            preview = cv2.resize(preview[y1:y1+zh, x1:x1+zw], (pw, ph), interpolation=cv2.INTER_LINEAR)

        with frame_lock:
            latest_frame = preview

        # Throttle focus calculation to ~6-7 Hz (every 4th frame) to save CPU
        frame_counter += 1
        if frame_counter % 4 == 0:
            ph, pw = preview.shape[:2]
            rx = int(roi_config["x"] * pw)
            ry = int(roi_config["y"] * ph)
            rw = int(roi_config["w"] * pw)
            rh = int(roi_config["h"] * ph)
            crop = preview[ry:ry+rh, rx:rx+rw]
            if crop.size > 0:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                s_val = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
                live_telemetry["sharpness"] = s_val
                if s_val > 350:
                    live_telemetry["focus_quality"] = "SHARP"
                    live_telemetry["focus_color"] = [0, 255, 0]
                elif s_val > 160:
                    live_telemetry["focus_quality"] = "FAIR"
                    live_telemetry["focus_color"] = [0, 215, 255]
                else:
                    live_telemetry["focus_quality"] = "BLURRY"
                    live_telemetry["focus_color"] = [0, 0, 255]

        time.sleep(0.01)

    if use_picam2 and picam2 is not None:
        picam2.stop()
    elif cap is not None:
        cap.release()

def run_edatec_ocr_pipeline(crop_bgr, digits_expected=0):
    """
    EDATEC Multi-pass OCR Pipeline:
    Uses OpenCV Grayscale, Otsu Thresholding, Morphological Opening,
    Inverted Thresholding, and Color-Contrast Enhancement with Pytesseract
    bounding box extraction (https://edatec.cn/rpi-forum/hardware/1353.html).
    """
    annotated = crop_bgr.copy()
    ch, cw = crop_bgr.shape[:2]

    # 1. EDATEC Grayscale & Otsu
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. EDATEC Morphological Opening
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. EDATEC Inverted Thresholding
    thresh_inv = cv2.bitwise_not(thresh)

    # 4. Color-Contrast CLAHE difference (for colored/yellow security seals)
    r = crop_bgr[:, :, 2].astype(np.float32)
    b = crop_bgr[:, :, 0].astype(np.float32)
    diff = np.clip(r - b, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(diff)
    _, color_bin = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if color_bin.mean() < 127:
        color_bin = cv2.bitwise_not(color_bin)

    # Optimal passes: color contrast for colored seals, standard Otsu, and opening
    passes = [
        ("color_contrast", color_bin),
        ("edatec_thresh", thresh),
        ("edatec_opened", opened)
    ]

    candidates = {}
    all_words = []
    detected_boxes = []

    # Keep resolution ideal for fast Tesseract (height ~120-280px)
    if ch < 90:
        scale = 100.0 / ch
        scaled_base = cv2.resize(crop_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif ch > 320:
        scale = 280.0 / ch
    else:
        scale = 1.0

    for name, p_img in passes:
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            scaled = cv2.resize(p_img, (0, 0), fx=scale, fy=scale, interpolation=interp)
        else:
            scaled = p_img
        bordered = cv2.copyMakeBorder(scaled, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

        try:
            import pytesseract
            from pytesseract import Output
            d = pytesseract.image_to_data(
                bordered,
                output_type=Output.DICT,
                config="--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            )

            line_tokens = []
            for i in range(len(d['text'])):
                txt = d['text'][i].strip()
                conf = int(float(d['conf'][i]))
                if txt and conf > 30:
                    bx = int(max(0, (d['left'][i] - 12) / scale))
                    by = int(max(0, (d['top'][i] - 12) / scale))
                    bw = int(d['width'][i] / scale)
                    bh = int(d['height'][i] / scale)
                    line_tokens.append(txt)
                    all_words.append((txt, conf))
                    detected_boxes.append((txt, bx, by, bw, bh, conf))

                    # Draw EDATEC green bounding box and text annotation
                    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{txt} ({conf}%)", (bx, max(14, by - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            full_line = " ".join(line_tokens)
            # Match seal number patterns like C 581819, 581819
            matches = re.findall(r'[A-Z]?\s*\d{4,9}', full_line)
            for m in matches:
                clean_m = re.sub(r'\s+', ' ', m).strip()
                digits_only = re.sub(r'\D', '', clean_m)
                if digits_expected > 0 and len(digits_only) != digits_expected:
                    continue
                if 4 <= len(digits_only) <= 12:
                    candidates[clean_m] = max(candidates.get(clean_m, 0), 85.0)

            if candidates:
                break # Fast exit once seal number is recognized!
        except Exception as e:
            continue

    # Write annotated image for UI inspection
    cv2.imwrite("/tmp/detected_result.jpg", annotated)
    return candidates, all_words, annotated

def scan_worker():
    global scan_status
    t_start = time.time()
    with scan_lock:
        scan_status["state"] = "detecting"
        scan_status["message"] = "Capturing high-resolution frame & selecting peak sharpness..."
        scan_status["candidates"] = []
        scan_status["all_words"] = []
        scan_status["best_number"] = None
        scan_status["has_image"] = False

    # Capture high-resolution frame (from Picamera2 main stream or fallback)
    highres_frame = None
    if active_picam2 is not None:
        try:
            highres_frame = active_picam2.capture_array("main")
        except Exception:
            pass

    if highres_frame is None:
        with frame_lock:
            highres_frame = latest_frame.copy() if latest_frame is not None else None

    if highres_frame is None:
        with scan_lock:
            scan_status["state"] = "error"
            scan_status["message"] = "No camera frame available."
        return

    # Digital zoom crop if zoom > 1.05
    zoom = roi_config.get("zoom", 1.0)
    if zoom > 1.05:
        h, w = highres_frame.shape[:2]
        zh, zw = int(h / zoom), int(w / zoom)
        y1 = (h - zh) // 2
        x1 = (w - zw) // 2
        highres_frame = cv2.resize(highres_frame[y1:y1+zh, x1:x1+zw], (w, h), interpolation=cv2.INTER_LINEAR)

    # Crop the ROI
    h, w = highres_frame.shape[:2]
    rx = int(roi_config["x"] * w)
    ry = int(roi_config["y"] * h)
    rw = int(roi_config["w"] * w)
    rh = int(roi_config["h"] * h)
    crop = highres_frame[ry:ry+rh, rx:rx+rw]

    if crop.size == 0:
        with scan_lock:
            scan_status["state"] = "error"
            scan_status["message"] = "Invalid ROI crop size."
        return

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    s_val = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

    with scan_lock:
        scan_status["state"] = "analyzing"
        scan_status["best_sharpness"] = s_val
        scan_status["message"] = f"Sharpness: {s_val:.0f}. Running EDATEC multi-pass OCR..."

    # Run the EDATEC OCR pipeline
    digits_expected = roi_config.get("digits", 0)
    candidates, all_words, annotated = run_edatec_ocr_pipeline(crop, digits_expected=digits_expected)

    total_time = time.time() - t_start
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    with scan_lock:
        scan_status["state"] = "complete"
        scan_status["ocr_time"] = total_time
        scan_status["candidates"] = ranked
        scan_status["all_words"] = all_words
        scan_status["has_image"] = True
        if ranked:
            scan_status["best_number"] = ranked[0][0]
            scan_status["best_score"] = ranked[0][1]
            scan_status["message"] = f"Detected Seal: {ranked[0][0]} in {total_time:.2f} seconds!"
        elif all_words:
            first_word = all_words[0][0]
            scan_status["best_number"] = first_word
            scan_status["best_score"] = float(all_words[0][1])
            scan_status["message"] = f"Detected Text: {first_word} in {total_time:.2f}s."
        else:
            scan_status["message"] = f"No text recognized in {total_time:.2f}s. Align text inside box and adjust zoom."

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
        elif self.path.startswith("/detected_result.jpg"):
            if os.path.exists("/tmp/detected_result.jpg"):
                with open("/tmp/detected_result.jpg", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "No detection result image available")
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
                            time.sleep(0.02)
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
                        cv2.line(draw_frame, (cx - 12, cy), (cx + 12, cy), box_color, 1)
                        cv2.line(draw_frame, (cx, cy - 12), (cx, cy + 12), box_color, 1)

                        status_str = f"FOCUS: {quality} ({int(sharp_val)})"
                        cv2.putText(draw_frame, status_str, (rx, max(18, ry - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

                    ret, jpeg = cv2.imencode(".jpg", draw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
                    if not ret:
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03) # 30 FPS target
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
    parser = argparse.ArgumentParser(description="Live camera stream server with CSI and fast OCR")
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
