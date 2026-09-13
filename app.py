"""
app.py - Main Live Video AI Padlock Character Recognition System.
Connects to Pi Zero 2W CSI camera, receives zero-latency feed, runs real-time
RapidOCR ONNX character recognition, and renders high-FPS interactive HUD.
"""
import os
import sys
import time
import datetime
import subprocess
import threading
import cv2
import numpy as np

from stream_receiver import StreamReceiver
from padlock_ai import PadlockDetector

# Optional FastAPI web preview
try:
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, StreamingResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


PI_STREAM_URL = "http://192.168.68.145:8000/stream.mjpg"
CAPTURES_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)


def copy_to_clipboard(text: str):
    """Copy text to Windows clipboard."""
    if not text:
        return
    try:
        cmd = f"Set-Clipboard -Value '{text}'"
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=True, timeout=2)
        print(f"[App] Copied to clipboard: {text}")
    except Exception as e:
        print(f"[App] Clipboard error: {e}")


class StickCamApp:
    def __init__(self, stream_url: str = PI_STREAM_URL):
        self.stream_url = stream_url
        self.receiver = StreamReceiver(stream_url)
        self.detector = PadlockDetector(conf_threshold=0.45)

        # Control states
        self.running = False
        self.paused = False
        self.show_boxes = True
        self.zoom_level = 1.0
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5

        self.latest_display_frame = None
        self.display_lock = threading.Lock()
        self.status_message = "Ready. Hold padlock in front of camera."
        self.status_msg_time = time.time()

    def start(self):
        """Start receiver, detector, and optional web server."""
        print("=" * 65)
        print("  STICKCAM PADLOCK AI RECOGNITION SYSTEM")
        print("  Pi Zero 2W Capture Stream -> PC ONNX Neural Inference")
        print("=" * 65)

        self.receiver.start()
        self.detector.start()
        self.running = True

        if HAS_FASTAPI:
            self._start_web_server()

        self._run_gui()

    def _start_web_server(self):
        """Runs background FastAPI server on port 5000 for browser viewing."""
        web_app = FastAPI(title="StickCam Padlock AI")

        @web_app.get("/", response_class=HTMLResponse)
        def index():
            return """<!DOCTYPE html>
<html>
<head>
    <title>StickCam Padlock AI - Live Feed</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { background: #0b0f19; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        h1 { margin-bottom: 4px; font-weight: 700; color: #38bdf8; }
        .subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 16px; }
        .container { max-width: 1040px; width: 100%; position: relative; }
        .video-box { position: relative; border-radius: 12px; overflow: hidden; border: 2px solid #1e293b; box-shadow: 0 20px 40px rgba(0,0,0,0.6); }
        .video-box img { width: 100%; display: block; }
        .badge { background: #10b981; color: #fff; padding: 3px 9px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; margin-left: 8px; }
    </style>
</head>
<body>
    <h1>StickCam Padlock AI <span class="badge">LIVE HUD</span></h1>
    <div class="subtitle">Zero-Latency Video from Pi Zero 2W &bull; Real-Time AI Character Detection</div>
    <div class="container">
        <div class="video-box">
            <img src="/stream.mjpg" alt="Live Stream">
        </div>
    </div>
</body>
</html>"""

        @web_app.get("/stream.mjpg")
        def web_stream():
            def frame_generator():
                while self.running:
                    with self.display_lock:
                        if self.latest_display_frame is None:
                            frame_data = None
                        else:
                            ret, buf = cv2.imencode(".jpg", self.latest_display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            frame_data = buf.tobytes() if ret else None

                    if frame_data is not None:
                        yield (
                            b"--FRAME\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_data)).encode("ascii") + b"\r\n\r\n" +
                            frame_data + b"\r\n"
                        )
                    time.sleep(0.033)

            return StreamingResponse(
                frame_generator(),
                media_type="multipart/x-mixed-replace; boundary=FRAME"
            )

        def run_uvicorn():
            uvicorn.run(web_app, host="0.0.0.0", port=5000, log_level="warning")

        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()
        print("[App] Web dashboard available at http://localhost:5000")

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Applies digital zoom centered at zoom_center."""
        if self.zoom_level <= 1.05:
            return frame

        h, w = frame.shape[:2]
        crop_w = int(w / self.zoom_level)
        crop_h = int(h / self.zoom_level)

        cx = int(self.zoom_center_x * w)
        cy = int(self.zoom_center_y * h)

        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _draw_bottom_toolbar(self, frame: np.ndarray) -> np.ndarray:
        """Draws clean bottom toolbar with hotkeys and status message."""
        h, w = frame.shape[:2]
        bar_h = 32
        y_start = h - bar_h

        # Semi-transparent dark bar
        roi = frame[y_start:h, 0:w]
        bg = np.zeros_like(roi)
        bg[:] = (12, 16, 24)
        frame[y_start:h, 0:w] = cv2.addWeighted(roi, 0.2, bg, 0.8, 0)
        cv2.line(frame, (0, y_start), (w, y_start), (45, 55, 70), 1)

        # Hotkey prompts
        hints = "[SPACE] Freeze  [S] Save  [C] Copy  [T] Boxes  [+/-] Zoom  [Q] Quit"
        cv2.putText(frame, hints, (14, h - 11), cv2.FONT_HERSHEY_DUPLEX, 0.40, (160, 175, 195), 1, cv2.LINE_AA)

        # Status alert message (if recent)
        if time.time() - self.status_msg_time < 3.0 and self.status_message:
            msg = self.status_message
            (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            cv2.putText(frame, msg, (w - mw - 14, h - 11), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        return frame

    def _run_gui(self):
        """Interactive high-FPS native OpenCV window loop."""
        window_name = "StickCam - Padlock AI Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1120, 630)

        frozen_frame = None
        last_submit_time = 0.0

        print("\n[App] Interactive Live Feed Running!")
        print("Controls:")
        print("  [SPACE]     Freeze / Resume live feed")
        print("  [S]         Save annotated snapshot to disk")
        print("  [C]         Copy detected padlock code to clipboard")
        print("  [T]         Toggle bounding box overlays")
        print("  [R]         Reset / clear detection state")
        print("  [+] / [-]   Digital Zoom In / Out")
        print("  [Q] / [ESC] Quit application\n")

        try:
            while self.running:
                if not self.paused:
                    raw_frame, fps, latency_ms, connected = self.receiver.get_frame()
                    if raw_frame is None:
                        # Waiting screen
                        wait_screen = np.zeros((540, 960, 3), dtype=np.uint8)
                        wait_screen[:] = (20, 24, 32)
                        cv2.putText(wait_screen, "CONNECTING TO PI ZERO 2W...", (260, 270),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 1, cv2.LINE_AA)
                        cv2.putText(wait_screen, f"Target: {self.stream_url}", (310, 310),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (120, 140, 160), 1, cv2.LINE_AA)
                        cv2.imshow(window_name, wait_screen)
                        key = cv2.waitKey(50) & 0xFF
                        if key in [ord('q'), ord('Q'), 27]:
                            break
                        continue

                    # Apply digital zoom if active
                    current_frame = self._apply_zoom(raw_frame)

                    # Periodically submit freshest frame to async AI worker (~5-10 Hz)
                    now = time.time()
                    if now - last_submit_time > 0.12:  # submit every 120ms
                        self.detector.submit_frame(current_frame)
                        last_submit_time = now

                    # Render live HUD
                    display = self.detector.draw_hud(
                        current_frame.copy(),
                        show_boxes=self.show_boxes,
                        fps=fps,
                        latency_ms=latency_ms
                    )
                else:
                    # Paused view
                    display = frozen_frame.copy() if frozen_frame is not None else np.zeros((540, 960, 3), dtype=np.uint8)
                    # Overlay "PAUSED" watermark
                    cv2.putText(display, "[PAUSED]", (400, 280), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 165, 255), 2, cv2.LINE_AA)

                # Draw bottom toolbar
                display = self._draw_bottom_toolbar(display)

                with self.display_lock:
                    self.latest_display_frame = display.copy()

                cv2.imshow(window_name, display)

                # Handle hotkeys (non-blocking 1ms poll)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("[App] Quitting...")
                    break

                elif key == 32:  # SPACE - Freeze/Unfreeze
                    self.paused = not self.paused
                    if self.paused:
                        frozen_frame = display.copy()
                        self.status_message = "Stream Paused."
                    else:
                        self.status_message = "Stream Resumed."
                    self.status_msg_time = time.time()

                elif key in [ord('s'), ord('S')]:  # Save snapshot
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(CAPTURES_DIR, f"padlock_{ts}.jpg")
                    cv2.imwrite(filename, display)
                    self.status_message = f"Saved to {os.path.basename(filename)}"
                    self.status_msg_time = time.time()
                    print(f"[App] Snapshot saved: {filename}")

                elif key in [ord('c'), ord('C')]:  # Copy padlock code
                    _, _, best_id, _ = self.detector.get_detections()
                    if best_id:
                        copy_to_clipboard(best_id)
                        self.status_message = f"Copied: {best_id}"
                    else:
                        self.status_message = "No padlock code detected to copy."
                    self.status_msg_time = time.time()

                elif key in [ord('t'), ord('T')]:  # Toggle boxes
                    self.show_boxes = not self.show_boxes
                    self.status_message = f"Bounding Boxes: {'ON' if self.show_boxes else 'OFF'}"
                    self.status_msg_time = time.time()

                elif key in [ord('r'), ord('R')]:  # Reset detections
                    self.detector.clear_detections()
                    self.status_message = "Detections Cleared."
                    self.status_msg_time = time.time()

                elif key in [ord('+'), ord('=')]:  # Zoom In
                    self.zoom_level = min(3.0, self.zoom_level + 0.25)
                    self.status_message = f"Zoom: {self.zoom_level:.1f}x"
                    self.status_msg_time = time.time()

                elif key in [ord('-'), ord('_')]:  # Zoom Out
                    self.zoom_level = max(1.0, self.zoom_level - 0.25)
                    self.status_message = f"Zoom: {self.zoom_level:.1f}x"
                    self.status_msg_time = time.time()

        finally:
            self.running = False
            self.detector.stop()
            self.receiver.stop()
            cv2.destroyAllWindows()
            print("[App] Stopped cleanly.")


if __name__ == "__main__":
    app = StickCamApp()
    app.start()
