import argparse
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

# Global streaming state
latest_frame = None
frame_lock = threading.Lock()
running = True

# ROI default (normalized fractions)
roi_config = {"x": 0.25, "y": 0.40, "w": 0.50, "h": 0.20, "show_roi": True}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Seal Scanner - Live Feed</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f8fafc;
    --accent: #38bdf8;
    --accent-hover: #0ea5e9;
    --success: #22c55e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
  }
  header {
    margin-bottom: 20px;
    text-align: center;
  }
  h1 { font-size: 1.6rem; color: #fff; display: flex; align-items: center; gap: 8px; justify-content: center; }
  .badge {
    background: #ef4444;
    color: #fff;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 9999px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
  .container {
    display: flex;
    flex-direction: column;
    max-width: 900px;
    width: 100%;
    gap: 20px;
  }
  .video-card {
    background: var(--card);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
    border: 1px solid #334155;
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
  .controls-card {
    background: var(--card);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    align-items: center;
  }
  .btn {
    background: var(--accent);
    color: #0f172a;
    font-weight: 600;
    padding: 10px 18px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn:hover { background: var(--accent-hover); }
  .btn-success { background: var(--success); color: #fff; }
  .btn-success:hover { background: #16a34a; }
  .slider-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    min-width: 120px;
  }
  .slider-group label { font-size: 0.8rem; color: #94a3b8; }
  input[type="range"] { accent-color: var(--accent); }
  .results-box {
    background: #090d16;
    border-radius: 8px;
    padding: 15px;
    font-family: monospace;
    font-size: 0.95rem;
    border: 1px solid #1e293b;
    white-space: pre-wrap;
    min-height: 50px;
  }
</style>
</head>
<body>
  <header>
    <h1>Pi Seal Scanner <span class="badge">LIVE</span></h1>
    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Align the seal inside the yellow rectangle</p>
  </header>

  <div class="container">
    <div class="video-card">
      <div class="video-container">
        <img src="/stream.mjpg" id="liveFeed" alt="Live Camera Feed">
      </div>
    </div>

    <div class="controls-card">
      <div class="row">
        <button class="btn btn-success" id="btnOcr" onclick="triggerOcr()">📸 Capture & Run OCR</button>
        <button class="btn" onclick="saveSnapshot()">💾 Save Snapshot</button>
        <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 0.9rem;">
          <input type="checkbox" id="showRoi" checked onchange="updateRoi()"> Show ROI Guides
        </label>
      </div>

      <div class="row">
        <div class="slider-group">
          <label>ROI X (Left): <span id="valX">0.25</span></label>
          <input type="range" id="roiX" min="0" max="0.8" step="0.02" value="0.25" oninput="updateRoi()">
        </div>
        <div class="slider-group">
          <label>ROI Y (Top): <span id="valY">0.40</span></label>
          <input type="range" id="roiY" min="0" max="0.8" step="0.02" value="0.40" oninput="updateRoi()">
        </div>
        <div class="slider-group">
          <label>ROI Width: <span id="valW">0.50</span></label>
          <input type="range" id="roiW" min="0.1" max="0.9" step="0.02" value="0.50" oninput="updateRoi()">
        </div>
        <div class="slider-group">
          <label>ROI Height: <span id="valH">0.20</span></label>
          <input type="range" id="roiH" min="0.05" max="0.6" step="0.02" value="0.20" oninput="updateRoi()">
        </div>
      </div>

      <div>
        <label style="font-size: 0.85rem; color: #94a3b8;">OCR Results:</label>
        <div class="results-box" id="ocrOutput">Click "Capture & Run OCR" to read seal numbers.</div>
      </div>
    </div>
  </div>

  <script>
    function updateRoi() {
      const x = parseFloat(document.getElementById('roiX').value);
      const y = parseFloat(document.getElementById('roiY').value);
      const w = parseFloat(document.getElementById('roiW').value);
      const h = parseFloat(document.getElementById('roiH').value);
      const show = document.getElementById('showRoi').checked;

      document.getElementById('valX').innerText = x.toFixed(2);
      document.getElementById('valY').innerText = y.toFixed(2);
      document.getElementById('valW').innerText = w.toFixed(2);
      document.getElementById('valH').innerText = h.toFixed(2);

      fetch('/set_roi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({x, y, w, h, show_roi: show})
      });
    }

    function triggerOcr() {
      const btn = document.getElementById('btnOcr');
      const out = document.getElementById('ocrOutput');
      btn.disabled = true;
      btn.innerText = 'Processing OCR...';
      out.innerText = 'Capturing frame and running Tesseract OCR on Pi Zero...';

      fetch('/run_ocr', {method: 'POST'})
        .then(res => res.json())
        .then(data => {
          btn.disabled = false;
          btn.innerText = '📸 Capture & Run OCR';
          if (data.status === 'ok') {
            out.innerText = data.output;
          } else {
            out.innerText = 'Error: ' + data.error;
          }
        })
        .catch(err => {
          btn.disabled = false;
          btn.innerText = '📸 Capture & Run OCR';
          out.innerText = 'Request failed: ' + err;
        });
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

    print(f"[Camera] Stream thread started on device {device_id} ({width}x{height})")
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.03) # ~30 fps cap
    cap.release()
    print("[Camera] Camera released.")

class StreamingHandler(server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Suppress HTTP access logs for clean console

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
                            time.sleep(0.05)
                            continue
                        draw_frame = latest_frame.copy()
                    
                    # Draw ROI overlay if enabled
                    if roi_config.get("show_roi", True):
                        h, w = draw_frame.shape[:2]
                        rx = int(roi_config["x"] * w)
                        ry = int(roi_config["y"] * h)
                        rw = int(roi_config["w"] * w)
                        rh = int(roi_config["h"] * h)
                        # Bounding box
                        cv2.rectangle(draw_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                        # Center crosshair inside ROI
                        cx, cy = rx + rw // 2, ry + rh // 2
                        cv2.line(draw_frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
                        cv2.line(draw_frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
                        cv2.putText(draw_frame, f"ROI: {rw}x{rh}", (rx, max(20, ry - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Encode to JPEG for stream (quality 60 for low Pi Zero CPU load)
                    ret, jpeg = cv2.imencode(".jpg", draw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if not ret:
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    time.sleep(0.06) # ~15 fps streaming
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
        elif self.path == "/run_ocr":
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": "No frame captured"}).encode())
                return

            # Save snapshot to disk
            snapshot_path = "/home/stickcam/seal-scanner/live_snapshot.jpg"
            cv2.imwrite(snapshot_path, frame)

            # Call scan_seal.py with current ROI
            roi_args = [
                str(roi_config["x"]),
                str(roi_config["y"]),
                str(roi_config["w"]),
                str(roi_config["h"]),
            ]
            cmd = ["python3", "scan_seal.py", snapshot_path, "--roi", *roi_args]
            try:
                res = subprocess.run(cmd, cwd="/home/stickcam/seal-scanner", capture_output=True, text=True, timeout=20)
                out = res.stdout
                if not out.strip():
                    out = res.stderr
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "output": out}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode())
        else:
            self.send_error(404)

def main():
    parser = argparse.ArgumentParser(description="Live camera stream server")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--device", type=int, default=0, help="Camera V4L2 device index (default: 0)")
    args = parser.parse_args()

    t = threading.Thread(target=camera_thread_loop, args=(args.device,), daemon=True)
    t.start()

    server_address = ("", args.port)
    httpd = server.ThreadingHTTPServer(server_address, StreamingHandler)
    print(f"[Server] Live video stream running on http://0.0.0.0:{args.port}")
    print(f"[Server] Open in your browser: http://pizero2.local:{args.port} or http://192.168.68.145:{args.port}")
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
