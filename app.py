"""
app.py - Main StickCam Padlock AI Recognition System (Mobile Phone Powered).
Connects to smartphone camera (via low-latency Mobile Web App, IP Webcam, or USB),
runs real-time RapidOCR ONNX inference, and renders high-FPS interactive HUD.
"""
import os
import sys
import time
import datetime
import socket
import argparse
import subprocess
import threading
import base64
import cv2
import numpy as np

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from stream_receiver import StreamReceiver
from padlock_ai import PadlockDetector


CAPTURES_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def get_local_ip() -> str:
    """Find local LAN IP address for phone connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable, just triggers routing table lookup
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def get_free_port(preferred_port: int = 5000, host: str = "0.0.0.0") -> int:
    """Finds preferred_port if free, or the next available port without throwing WinError 10048."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, preferred_port))
            return preferred_port
        except OSError:
            print(f"\n[Warning] Port {preferred_port} is already in use by another process.")
            for p in range(preferred_port + 1, preferred_port + 50):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                        s2.bind((host, p))
                        print(f"[Info] Automatically bound to available port {p}\n")
                        return p
                except OSError:
                    continue
            return preferred_port


def print_terminal_qr(url: str):
    """Prints a clean ASCII QR code directly into the terminal."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        print("\n" + "=" * 68)
        print("  SCAN THIS QR CODE WITH YOUR PHONE CAMERA:")
        print("=" * 68)
        for row in qr.get_matrix():
            print("  " + "".join("##" if cell else "  " for cell in row))
        print("=" * 68 + "\n")
    except Exception:
        pass


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


def ensure_ssl_certificates(cert_path="cert.pem", key_path="key.pem", host_ip="127.0.0.1"):
    """
    Ensures SSL self-signed certificates exist for local HTTPS server.
    Modern mobile browsers (Chrome / Safari / Edge) strictly disable navigator.mediaDevices
    (the camera API) on non-localhost HTTP connections. Serving over HTTPS enables full camera access.
    """
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import ipaddress

        print(f"[SSL] Generating self-signed SSL certificate for {host_ip}...")
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        san_list = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
        try:
            san_list.append(x509.IPAddress(ipaddress.ip_address(host_ip)))
        except ValueError:
            san_list.append(x509.DNSName(host_ip))

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, f"StickCam Local ({host_ip})"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "StickCam Padlock AI"),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"[SSL] Self-signed certificate created: {cert_path}")
        return cert_path, key_path
    except Exception as e:
        print(f"[SSL] Warning: Could not generate SSL certificate: {e}. Falling back to HTTP.")
        return None, None


class StickCamApp:
    def __init__(self, port: int = 5000, stream_url: str = None, camera_index: int = None, no_gui: bool = False, conf: float = 0.45, ssl: bool = True):
        self.port = get_free_port(port)
        self.stream_url = stream_url
        self.camera_index = camera_index
        self.no_gui = no_gui
        self.local_ip = get_local_ip()

        # SSL Configuration for Mobile Camera Access
        self.use_ssl = ssl
        self.ssl_certfile = None
        self.ssl_keyfile = None
        if self.use_ssl:
            cert_file = os.path.join(os.path.dirname(__file__), "cert.pem")
            key_file = os.path.join(os.path.dirname(__file__), "key.pem")
            self.ssl_certfile, self.ssl_keyfile = ensure_ssl_certificates(cert_file, key_file, self.local_ip)
            if not self.ssl_certfile or not self.ssl_keyfile:
                self.use_ssl = False

        self.proto = "https" if self.use_ssl else "http"
        self.phone_url = f"{self.proto}://{self.local_ip}:{self.port}/phone"

        # Pre-render QR code and waiting screen
        self.qr_img = self._create_qr_image(self.phone_url, size=210)
        self.wait_screen = self._render_wait_screen()

        # Determine receiver mode
        if stream_url:
            mode = "mjpeg"
        elif camera_index is not None:
            mode = "capture"
        else:
            mode = "push"

        self.receiver = StreamReceiver(mode=mode, stream_url=stream_url, camera_index=camera_index)
        self.detector = PadlockDetector(conf_threshold=conf)

        # Control states
        self.running = False
        self.paused = False
        self.show_boxes = True
        self.zoom_level = 1.0
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5

        self.latest_display_frame = None
        self.latest_annotated_capture = None
        self.display_lock = threading.Lock()
        self.status_message = "Ready. Tap SCAN on phone to begin."
        self.status_msg_time = time.time()
        self.active_phone_ws = None

        # FastAPI app & templates
        self.web_app = FastAPI(title="StickCam Padlock AI")
        self.templates = Jinja2Templates(directory=TEMPLATES_DIR)
        self._setup_routes()

    def _setup_routes(self):
        app = self.web_app

        @app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "server_ip": self.local_ip,
                    "port": self.port,
                    "phone_url": self.phone_url,
                    "protocol": self.proto
                }
            )

        @app.get("/phone", response_class=HTMLResponse)
        async def phone_view(request: Request):
            return self.templates.TemplateResponse(
                request=request,
                name="phone.html",
                context={
                    "server_host": f"{self.local_ip}:{self.port}"
                }
            )

        @app.websocket("/ws/phone-stream")
        async def websocket_phone_stream(websocket: WebSocket):
            await websocket.accept()
            self.active_phone_ws = websocket
            self.status_message = "Phone Connected & Streaming!"
            self.status_msg_time = time.time()
            print(f"[WebSocket] Phone camera connected from {websocket.client.host}")

            last_sent_code = None
            try:
                while self.running:
                    # Receive binary packet from phone
                    data = await websocket.receive_bytes()
                    if len(data) > 8:
                        # Extract 8-byte client timestamp (float64)
                        client_ts = np.frombuffer(data[:8], dtype=np.float64)[0]
                        jpg_data = data[8:]
                        self.receiver.push_jpeg_bytes(jpg_data)

                        # Send back active telemetry to phone
                        _, _, best_id, best_conf = self.detector.get_detections()
                        repeated = (best_id == last_sent_code)
                        last_sent_code = best_id

                        await websocket.send_json({
                            "type": "telemetry",
                            "best_padlock_id": best_id,
                            "best_confidence": best_conf,
                            "repeated": repeated,
                            "rtt": 15.0
                        })

            except WebSocketDisconnect:
                print("[WebSocket] Phone camera disconnected.")
                self.receiver.mark_disconnected()
                self.active_phone_ws = None
            except Exception as e:
                print(f"[WebSocket] Error: {e}")
                self.receiver.mark_disconnected()
                self.active_phone_ws = None

        @app.get("/stream.mjpg")
        def web_stream():
            def frame_generator():
                while self.running:
                    with self.display_lock:
                        if self.latest_display_frame is None:
                            frame_data = None
                        else:
                            ret, buf = cv2.imencode(".jpg", self.latest_display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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

        @app.get("/api/detections")
        def api_detections():
            detections, ocr_time_ms, padlock_id, padlock_conf = self.detector.get_detections()
            _, fps, latency_ms, connected = self.receiver.get_frame()
            return {
                "connected": connected,
                "fps": fps,
                "latency_ms": latency_ms,
                "ocr_time_ms": ocr_time_ms,
                "best_padlock_id": padlock_id,
                "best_confidence": padlock_conf,
                "detections_count": len(detections)
            }

        @app.post("/api/action/{action}")
        def api_action(action: str):
            msg = "Action completed."
            if action == "freeze":
                self.paused = not self.paused
                msg = "Stream Paused" if self.paused else "Stream Resumed"
            elif action == "save":
                with self.display_lock:
                    if self.latest_display_frame is not None:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(CAPTURES_DIR, f"padlock_{ts}.jpg")
                        cv2.imwrite(filename, self.latest_display_frame)
                        msg = f"Saved {os.path.basename(filename)}"
            elif action == "toggle_boxes":
                self.show_boxes = not self.show_boxes
                msg = f"Boxes: {'ON' if self.show_boxes else 'OFF'}"
            elif action == "clear":
                self.detector.clear_detections()
                self.latest_annotated_capture = None
                msg = "Detections cleared"
            elif action == "zoom_in":
                self.zoom_level = min(3.0, self.zoom_level + 0.25)
                msg = f"Zoom {self.zoom_level:.1f}x"
            elif action == "zoom_out":
                self.zoom_level = max(1.0, self.zoom_level - 0.25)
                msg = f"Zoom {self.zoom_level:.1f}x"
            return {"status": "ok", "message": msg}

        @app.post("/api/scan-burst")
        async def api_scan_burst(request: Request):
            """
            Receives a burst of images captured on a single shutter click,
            runs RapidOCR ONNX character recognition, picks the best padlock ID,
            and updates the PC monitor and web dashboard.
            """
            try:
                body = await request.json()
                b64_list = body.get("images", [])
                if not b64_list:
                    return JSONResponse({"success": False, "error": "No images provided."}, status_code=400)

                frames = []
                for item in b64_list:
                    if "," in item:
                        item = item.split(",", 1)[1]
                    raw_bytes = base64.b64decode(item)
                    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        frames.append(img)

                if not frames:
                    return JSONResponse({"success": False, "error": "Could not decode image frames."}, status_code=400)

                result = self.detector.process_burst(frames)
                best_id = result.get("best_id")
                best_conf = result.get("confidence", 0.0)
                annotated_frame = result.get("annotated_frame")

                if annotated_frame is not None:
                    # Update receiver's latest frame and our persistent display capture
                    self.receiver.push_frame(frames[result.get("best_frame_idx", 0)])
                    self.latest_annotated_capture = annotated_frame.copy()
                    with self.display_lock:
                        self.latest_display_frame = annotated_frame.copy()

                    if best_id:
                        copy_to_clipboard(best_id)
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(CAPTURES_DIR, f"burst_padlock_{ts}.jpg")
                        cv2.imwrite(filename, annotated_frame)
                        result["saved_image"] = os.path.basename(filename)
                        self.status_message = f"Detected: {best_id} ({int(best_conf*100)}%)"
                        print(f"[Burst Scan] Successfully recognized: {best_id} ({int(best_conf*100)}%) -> Saved: {filename}")
                    else:
                        self.status_message = "Burst complete (No serial code detected)"
                        print("[Burst Scan] Processed burst: No padlock code identified above threshold.")

                    self.status_msg_time = time.time()

                return JSONResponse({
                    "success": result.get("success", False),
                    "best_id": best_id,
                    "confidence": best_conf,
                    "frames_count": result.get("frames_count", len(frames)),
                    "total_time_ms": result.get("total_time_ms", 0.0),
                    "all_texts": result.get("all_texts", [])
                })
            except Exception as e:
                print(f"[API] Error in /api/scan-burst: {e}")
                return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    def _create_qr_image(self, url: str, size: int = 210) -> np.ndarray:
        """Generates a high-contrast BGR QR code image for display in OpenCV."""
        try:
            import qrcode
            qr = qrcode.QRCode(box_size=5, border=1)
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            qr_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            return cv2.resize(qr_bgr, (size, size), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"[App] QR image fallback: {e}")
            try:
                encoder = cv2.QRCodeEncoder_create()
                mat = encoder.encode(url)
                qr_img = cv2.resize((mat * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
                return cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)
            except Exception:
                return None

    def _render_wait_screen(self) -> np.ndarray:
        """Draws a premium waiting screen with centered visual QR code, instructions, and hotkeys."""
        w, h = 960, 540
        screen = np.zeros((h, w, 3), dtype=np.uint8)
        screen[:] = (18, 22, 30)

        # Header Title
        title = "CONNECT YOUR SMARTPHONE CAMERA"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.85, 1)
        cv2.putText(screen, title, ((w - tw) // 2, 45), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 1, cv2.LINE_AA)

        subtitle = "Point your phone camera at the QR code below to connect"
        (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
        cv2.putText(screen, subtitle, ((w - sw) // 2, 75), cv2.FONT_HERSHEY_DUPLEX, 0.46, (160, 185, 205), 1, cv2.LINE_AA)

        # Draw QR Code if generated
        if self.qr_img is not None:
            qr_h, qr_w = self.qr_img.shape[:2]
            qr_x = (w - qr_w) // 2
            qr_y = 95

            # Glowing card backdrop
            pad = 8
            cv2.rectangle(screen, (qr_x - pad, qr_y - pad), (qr_x + qr_w + pad, qr_y + qr_h + pad), (255, 255, 255), -1)
            cv2.rectangle(screen, (qr_x - pad - 2, qr_y - pad - 2), (qr_x + qr_w + pad + 2, qr_y + qr_h + pad + 2), (0, 215, 255), 2, cv2.LINE_AA)
            screen[qr_y:qr_y+qr_h, qr_x:qr_x+qr_w] = self.qr_img

            text_y = qr_y + qr_h + 40
        else:
            text_y = 220

        # Direct link
        url_label = f"Open URL: {self.phone_url}"
        (uw, _), _ = cv2.getTextSize(url_label, cv2.FONT_HERSHEY_DUPLEX, 0.68, 1)
        cv2.putText(screen, url_label, ((w - uw) // 2, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 120), 1, cv2.LINE_AA)

        # Dashboard link
        dash_label = f"Web Dashboard: {self.proto}://localhost:{self.port}"
        (dw, _), _ = cv2.getTextSize(dash_label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(screen, dash_label, ((w - dw) // 2, text_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)

        # Bottom Hint
        hint = "[Q] / [ESC] Quit application"
        cv2.putText(screen, hint, (20, h - 15), cv2.FONT_HERSHEY_DUPLEX, 0.40, (100, 120, 140), 1, cv2.LINE_AA)

        return screen

    def start(self):
        """Start receiver, detector, web server, and OpenCV GUI."""
        print("=" * 68)
        print("  STICKCAM PADLOCK AI RECOGNITION SYSTEM (MOBILE PHONE POWERED)")
        print("=" * 68)
        print(f"  [+] Protocol:               {self.proto.upper()}")
        print(f"  [+] Local IP Address:       {self.local_ip}")
        print(f"  [+] Web Dashboard:          {self.proto}://localhost:{self.port}")
        print(f"  [+] Connect Phone at:       {self.phone_url}")
        if self.use_ssl:
            print("  [*] Mobile Browser Tip:     When prompted with 'Connection is not private',")
            print("                              tap 'Advanced' -> 'Proceed to IP (unsafe)'.")
            print("                              This grants instant camera access on Android / iOS!")
        if self.stream_url:
            print(f"  [+] External Stream URL:    {self.stream_url}")
        if self.camera_index is not None:
            print(f"  [+] Local Camera Index:     {self.camera_index}")
        print("=" * 68)

        # Print ASCII QR code in terminal
        print_terminal_qr(self.phone_url)

        self.running = True
        self.receiver.start()
        self.detector.start()

        # Start FastAPI in daemon thread
        def run_uvicorn():
            kwargs = {
                "host": "0.0.0.0",
                "port": self.port,
                "log_level": "warning"
            }
            if self.use_ssl and self.ssl_certfile and self.ssl_keyfile:
                kwargs["ssl_certfile"] = self.ssl_certfile
                kwargs["ssl_keyfile"] = self.ssl_keyfile
            uvicorn.run(self.web_app, **kwargs)

        web_thread = threading.Thread(target=run_uvicorn, daemon=True)
        web_thread.start()

        if not self.no_gui:
            self._run_gui()
        else:
            print("[App] Running in headless mode. Press Ctrl+C to stop.")
            self._run_headless()

    def _run_headless(self):
        last_submit_time = 0.0
        try:
            while self.running:
                raw_frame, fps, latency_ms, connected = self.receiver.get_frame()
                if raw_frame is not None and connected:
                    current_frame = self._apply_zoom(raw_frame)
                    now = time.time()
                    if now - last_submit_time > 0.12:
                        self.detector.submit_frame(current_frame)
                        last_submit_time = now

                    display = self.detector.draw_hud(
                        current_frame.copy(),
                        show_boxes=self.show_boxes,
                        fps=fps,
                        latency_ms=latency_ms,
                        source_label="PHONE"
                    )
                    with self.display_lock:
                        self.latest_display_frame = display
                time.sleep(0.03)
        except KeyboardInterrupt:
            print("\n[App] Stopping headless mode...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.detector.stop()
        self.receiver.stop()
        cv2.destroyAllWindows()
        print("[App] Stopped cleanly.")

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
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
        h, w = frame.shape[:2]
        bar_h = 32
        y_start = h - bar_h

        roi = frame[y_start:h, 0:w]
        bg = np.zeros_like(roi)
        bg[:] = (12, 16, 24)
        frame[y_start:h, 0:w] = cv2.addWeighted(roi, 0.2, bg, 0.8, 0)
        cv2.line(frame, (0, y_start), (w, y_start), (45, 55, 70), 1)

        hints = "[SPACE] Freeze  [S] Save  [C] Copy  [T] Boxes  [+/-] Zoom  [Q] Quit"
        cv2.putText(frame, hints, (14, h - 11), cv2.FONT_HERSHEY_DUPLEX, 0.40, (160, 175, 195), 1, cv2.LINE_AA)

        if time.time() - self.status_msg_time < 3.0 and self.status_message:
            msg = self.status_message
            (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            cv2.putText(frame, msg, (w - mw - 14, h - 11), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        return frame

    def _run_gui(self):
        window_name = "StickCam - Phone AI Padlock Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1120, 630)

        frozen_frame = None
        last_submit_time = 0.0

        print("\n[App] Interactive Live Desktop Feed Running!")
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
                    if raw_frame is not None and connected and self.receiver.mode != "push":
                        # Continuous streaming mode (MJPEG / RTSP / USB)
                        current_frame = self._apply_zoom(raw_frame)
                        now = time.time()
                        if now - last_submit_time > 0.12:
                            self.detector.submit_frame(current_frame)
                            last_submit_time = now

                        display = self.detector.draw_hud(
                            current_frame.copy(),
                            show_boxes=self.show_boxes,
                            fps=fps,
                            latency_ms=latency_ms,
                            source_label="PHONE"
                        )
                    else:
                        # Push / Burst Mode or Waiting Screen
                        if self.latest_annotated_capture is not None:
                            display = self.latest_annotated_capture.copy()
                        else:
                            display = self.wait_screen.copy()
                else:
                    display = frozen_frame.copy() if frozen_frame is not None else np.zeros((540, 960, 3), dtype=np.uint8)
                    cv2.putText(display, "[PAUSED]", (400, 280), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 165, 255), 2, cv2.LINE_AA)

                display = self._draw_bottom_toolbar(display)

                with self.display_lock:
                    self.latest_display_frame = display.copy()

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("[App] Quitting...")
                    break

                elif key == 32:  # SPACE
                    self.paused = not self.paused
                    if self.paused:
                        frozen_frame = display.copy()
                        self.status_message = "Stream Paused."
                    else:
                        self.status_message = "Stream Resumed."
                    self.status_msg_time = time.time()

                elif key in [ord('s'), ord('S')]:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(CAPTURES_DIR, f"padlock_{ts}.jpg")
                    cv2.imwrite(filename, display)
                    self.status_message = f"Saved to {os.path.basename(filename)}"
                    self.status_msg_time = time.time()
                    print(f"[App] Snapshot saved: {filename}")

                elif key in [ord('c'), ord('C')]:
                    _, _, best_id, _ = self.detector.get_detections()
                    if best_id:
                        copy_to_clipboard(best_id)
                        self.status_message = f"Copied: {best_id}"
                    else:
                        self.status_message = "No padlock code detected to copy."
                    self.status_msg_time = time.time()

                elif key in [ord('t'), ord('T')]:
                    self.show_boxes = not self.show_boxes
                    self.status_message = f"Bounding Boxes: {'ON' if self.show_boxes else 'OFF'}"
                    self.status_msg_time = time.time()

                elif key in [ord('r'), ord('R')]:
                    self.detector.clear_detections()
                    self.status_message = "Detections Cleared."
                    self.status_msg_time = time.time()

                elif key in [ord('+'), ord('=')]:
                    self.zoom_level = min(3.0, self.zoom_level + 0.25)
                    self.status_message = f"Zoom: {self.zoom_level:.1f}x"
                    self.status_msg_time = time.time()

                elif key in [ord('-'), ord('_')]:
                    self.zoom_level = max(1.0, self.zoom_level - 0.25)
                    self.status_message = f"Zoom: {self.zoom_level:.1f}x"
                    self.status_msg_time = time.time()

        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="StickCam Phone Padlock AI Scanner")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--stream", type=str, default=None, help="Optional external stream URL (e.g. http://192.168.68.x:8080/video)")
    parser.add_argument("--camera", type=int, default=None, help="Optional local OpenCV camera index (e.g. 0)")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold (default: 0.45)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless web server mode without OpenCV window")
    parser.add_argument("--no-ssl", action="store_true", help="Disable HTTPS and run in plain HTTP mode")
    args = parser.parse_args()

    app = StickCamApp(
        port=args.port,
        stream_url=args.stream,
        camera_index=args.camera,
        no_gui=args.no_gui,
        conf=args.conf,
        ssl=not args.no_ssl
    )
    app.start()


if __name__ == "__main__":
    main()
