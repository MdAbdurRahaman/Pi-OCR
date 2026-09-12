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

# ROI default (normalized fractions)
roi_config = {
    "x": 0.25,
    "y": 0.35,
    "w": 0.50,
    "h": 0.30,
    "show_roi": True,
    "digits": 0,
}

# Real-time scan state
scan_lock = threading.Lock()
scan_status = {
    "state": "idle", # "idle", "detecting", "analyzing", "complete", "error"
    "message": "Ready. Place lock/seal in frame and click 'Start Scan'.",
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
<title>Pi Seal Scanner - Live & Auto-OCR</title>
<style>
  :root {
    --bg: #0b0f19;
    --card: #151d2e;
    --card-border: #223049;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --success: #10b981;
    --success-hover: #059669;
    --warn: #f59e0b;
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
  header {
    margin-bottom: 18px;
    text-align: center;
  }
  h1 {
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: center;
    letter-spacing: -0.02em;
  }
  .badge-live {
    background: #ef4444;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 9999px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .container {
    display: flex;
    flex-direction: column;
    max-width: 960px;
    width: 100%;
    gap: 18px;
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
    min-height: 360px;
  }
  .video-container img {
    max-width: 100%;
    height: auto;
    display: block;
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
    padding: 12px 22px;
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
  .btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .btn-secondary {
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
  }
  .btn-secondary:hover { background: #334155; color: #fff; }
  .slider-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
  .slider-item span {
    color: var(--accent);
    font-family: monospace;
  }
  input[type="range"] {
    accent-color: var(--accent);
    cursor: pointer;
  }
  .result-banner {
    background: #090d16;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid var(--card-border);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .status-text {
    font-size: 0.95rem;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: 8px;
  }
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
    font-size: 1.8rem;
    font-weight: 800;
    font-family: "Courier New", Courier, monospace;
    color: #34d399;
    letter-spacing: 2px;
  }
  .candidates-list {
    font-family: monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
    max-height: 120px;
    overflow-y: auto;
  }
</style>
</head>
<body>
  <header>
    <h1>Pi Seal Scanner <span class="badge-live">LIVE</span></h1>
    <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 5px;">
      Align seal in the yellow ROI box & click "Start Scan" to auto-detect best sharp frames
    </p>
  </header>

  <div class="container">
    <div class="video-card">
      <div class="video-container">
        <img src="/stream.mjpg" id="liveFeed" alt="Live Camera Feed">
      </div>
    </div>

    <div class="controls-card">
      <div class="btn-row">
        <button class="btn btn-primary" id="btnStartScan" onclick="startScan()">
          ▶ Start Scan (Auto-Detect Best Shots)
        </button>
        <button class="btn btn-secondary" onclick="saveSnapshot()">
          💾 Save Snapshot
        </button>
        <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 0.9rem; margin-left: auto;">
          <input type="checkbox" id="showRoi" checked onchange="updateRoi()"> Show ROI Guides
        </label>
        <div style="display: flex; align-items: center; gap: 6px;">
          <label style="font-size: 0.85rem; color: var(--text-muted);">Digits:</label>
          <input type="number" id="digitsFilter" value="0" min="0" max="20" style="width: 55px; padding: 5px 8px; background: #0d1424; color: #fff; border: 1px solid #334155; border-radius: 6px;" onchange="updateRoi()">
        </div>
      </div>

      <div class="slider-grid">
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
          Ready. Place lock/seal in frame and click "Start Scan".
        </div>

        <div class="best-match" id="bestMatchBox" style="display: none;">
          <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #6ee7b7; font-weight: 600;">Detected Seal Number</div>
            <div class="best-digits" id="bestNumberDisplay">-------</div>
          </div>
          <div style="margin-left: auto; text-align: right;">
            <div style="font-size: 0.75rem; color: var(--text-muted);">Confidence Score</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8;" id="bestScoreDisplay">0.0</div>
          </div>
        </div>

        <div class="candidates-list" id="candidatesList" style="display: none;"></div>
      </div>
    </div>
  </div>

  <script>
    let isScanning = false;
    let pollInterval = null;

    function updateRoi() {
      const x = parseFloat(document.getElementById('roiX').value);
      const y = parseFloat(document.getElementById('roiY').value);
      const w = parseFloat(document.getElementById('roiW').value);
      const h = parseFloat(document.getElementById('roiH').value);
      const show = document.getElementById('showRoi').checked;
      const digits = parseInt(document.getElementById('digitsFilter').value) || 0;

      document.getElementById('valX').innerText = x.toFixed(2);
      document.getElementById('valY').innerText = y.toFixed(2);
      document.getElementById('valW').innerText = w.toFixed(2);
      document.getElementById('valH').innerText = h.toFixed(2);

      fetch('/set_roi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({x, y, w, h, show_roi: show, digits})
      });
    }

    function startScan() {
      if (isScanning) return;
      isScanning = true;
      const btn = document.getElementById('btnStartScan');
      btn.disabled = true;
      btn.innerText = '🔍 Detecting & Analyzing Best Shots...';
      document.getElementById('statusText').innerText = 'Scanning lock... Gathering continuous frames to pick the sharpest shots.';
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
              list.innerHTML = '<strong>Candidate Numbers:</strong><br>' + 
                data.candidates.map(c => `• ${c[0]} (score: ${c[1].toFixed(1)})`).join('<br>');
            }
          }
        });
    }

    function endScan() {
      isScanning = false;
      const btn = document.getElementById('btnStartScan');
      btn.disabled = false;
      btn.innerText = '▶ Start Scan (Auto-Detect Best Shots)';
    }

    function saveSnapshot() {
      window.open('/snapshot', '_blank');
    }
  </script>
</body>
</html>
"""

def camera_thread_loop(device_id=0, width=1280, height=720):
    global latest_frame, running
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[Error] Could not open camera {device_id}")
        return

    print(f"[Camera] Stream thread active on {device_id} ({width}x{height})")
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.04)
            continue
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.03) # ~30 fps cap
    cap.release()

def calculate_sharpness(gray_crop):
    """Variance of Laplacian measures focus / high-frequency texture."""
    return cv2.Laplacian(gray_crop, cv2.CV_64F).var()

def perform_fast_ocr(image_bgr, digits_expected=0):
    """In-memory OCR using pre-allocated OpenCV operations and Tesseract."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 1200:
        scale = 1200 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)

    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"
    candidates = {}

    for idx, processed in enumerate([gray, binary]):
        padded = cv2.copyMakeBorder(processed, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
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
        scan_status["message"] = "Holding lock still... capturing 30 frames to select sharpest in-focus shots."
        scan_status["candidates"] = []
        scan_status["best_number"] = None

    # Step 1: Collect multiple frames over ~2.5 seconds (around 25-30 frames)
    collected = []
    start_t = time.time()
    last_sharpness = 0.0

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
                sharpness = calculate_sharpness(gray)
                collected.append((sharpness, crop))
                last_sharpness = sharpness
        time.sleep(0.08)

    with scan_lock:
        scan_status["state"] = "analyzing"
        scan_status["frames_evaluated"] = len(collected)
        scan_status["message"] = f"Analyzed {len(collected)} frames. Picking top 3 sharpest shots for multi-frame OCR..."

    if not collected:
        with scan_lock:
            scan_status["state"] = "error"
            scan_status["message"] = "No frames captured. Ensure camera is running."
        return

    # Step 2: Sort by sharpness descending and pick top 3 best frames
    collected.sort(key=lambda x: x[0], reverse=True)
    best_shots = collected[:3]
    top_sharpness = best_shots[0][0]

    with scan_lock:
        scan_status["best_sharpness"] = top_sharpness
        scan_status["message"] = f"Top sharpness: {top_sharpness:.1f}. Running OCR on best shots..."

    # Step 3: Run OCR on each best shot and aggregate candidates
    all_scores = {}
    digits_expected = roi_config.get("digits", 0)

    for rank, (score, crop) in enumerate(best_shots):
        candidates = perform_fast_ocr(crop, digits_expected=digits_expected)
        for num, conf in candidates.items():
            # Consensus boost: if seen in multiple shots, boost score
            if num not in all_scores:
                all_scores[num] = {"conf_max": conf, "count": 1, "conf_sum": conf}
            else:
                all_scores[num]["conf_max"] = max(all_scores[num]["conf_max"], conf)
                all_scores[num]["count"] += 1
                all_scores[num]["conf_sum"] += conf

    ranked = []
    for num, meta in all_scores.items():
        # Weighted score: average confidence + multi-shot consensus bonus (+5 per extra shot)
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
            scan_status["message"] = f"Scan complete! Best match: {ranked[0][0]} (Score: {ranked[0][1]:.1f} across {len(best_shots)} shots)."
        else:
            scan_status["message"] = "No digits found. Check focus, lighting, or crop alignment."

class StreamingHandler(server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
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

                    # Draw ROI overlay
                    if roi_config.get("show_roi", True):
                        h, w = draw_frame.shape[:2]
                        rx = int(roi_config["x"] * w)
                        ry = int(roi_config["y"] * h)
                        rw = int(roi_config["w"] * w)
                        rh = int(roi_config["h"] * h)
                        # Box
                        cv2.rectangle(draw_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                        # Crosshair
                        cx, cy = rx + rw // 2, ry + rh // 2
                        cv2.line(draw_frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
                        cv2.line(draw_frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
                        cv2.putText(draw_frame, f"ROI: {rw}x{rh}", (rx, max(22, ry - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
    parser = argparse.ArgumentParser(description="Live camera stream server with Auto-OCR")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    t = threading.Thread(target=camera_thread_loop, args=(args.device,), daemon=True)
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
