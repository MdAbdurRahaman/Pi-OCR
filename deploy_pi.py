"""
deploy_pi.py - Deploys optimized zero-latency streaming capture service to Raspberry Pi Zero 2W.
"""
import paramiko
import sys
import time

PI_HOST = "192.168.68.145"
PI_USER = "stickcam"
PI_PASS = "Dubo2024"

OPTIMIZED_STREAM_SERVER = '''#!/usr/bin/env python3
"""
Ultra-Low Latency CSI Capture Streamer for Raspberry Pi Zero 2 W
Optimized for 30+ FPS Zero-Buffer Live Video Transmission to PC AI System.
"""
import os
import sys
import time
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import cv2
import numpy as np

# Camera settings
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
JPEG_QUALITY = 65

# Global frame buffer & synchronization
latest_jpeg = None
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)
camera_fps = 0.0
running = True

def camera_worker():
    global latest_jpeg, camera_fps, running
    print(f"[Camera] Initializing Picamera2 ({DEFAULT_WIDTH}x{DEFAULT_HEIGHT})...")
    
    picam2 = None
    use_picam2 = False
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2(0)
        config = picam2.create_video_configuration(
            main={"size": (DEFAULT_WIDTH, DEFAULT_HEIGHT), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        use_picam2 = True
        print("[Camera] Picamera2 hardware CSI pipeline active at 30 FPS!")
    except Exception as e:
        print(f"[Camera] Picamera2 init failed: {e}. Falling back to V4L2 OpenCV...")

    cap = None
    if not use_picam2:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # Frame timing
    fps_calc_interval = 20
    fps_counter = 0
    t_start = time.time()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    while running:
        if use_picam2:
            try:
                # Picamera2 RGB888 array
                frame = picam2.capture_array("main")
                # Fast encode directly
                ret, buf = cv2.imencode(".jpg", frame, encode_params)
            except Exception as e:
                time.sleep(0.01)
                continue
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            ret, buf = cv2.imencode(".jpg", frame, encode_params)

        if ret and buf is not None:
            raw_bytes = buf.tobytes()
            with frame_condition:
                latest_jpeg = raw_bytes
                frame_condition.notify_all()

        fps_counter += 1
        if fps_counter >= fps_calc_interval:
            now = time.time()
            elapsed = now - t_start
            camera_fps = fps_counter / elapsed if elapsed > 0 else 30.0
            fps_counter = 0
            t_start = now

    if picam2 is not None:
        picam2.stop()
    if cap is not None:
        cap.release()

class ZeroLagStreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Disable console logging for max performance

    def do_GET(self):
        # Enable TCP_NODELAY for immediate packet dispatch
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        if self.path in ["/stream.mjpg", "/raw_stream.mjpg", "/video_feed"]:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while running:
                    with frame_condition:
                        frame_condition.wait(timeout=0.1)
                        if latest_jpeg is None:
                            continue
                        jpeg_bytes = latest_jpeg

                    header = (
                        b"--FRAME\\r\\n"
                        b"Content-Type: image/jpeg\\r\\n"
                        + f"Content-Length: {len(jpeg_bytes)}\\r\\n\\r\\n".encode("ascii")
                    )
                    self.wfile.write(header)
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b"\\r\\n")
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

        elif self.path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            telemetry = {
                "fps": round(camera_fps, 2),
                "resolution": f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
                "status": "streaming"
            }
            self.wfile.write(json.dumps(telemetry).encode("utf-8"))

        elif self.path == "/snapshot":
            with frame_condition:
                data = latest_jpeg
            if data is None:
                self.send_error(503, "No frame available")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Disposition", "attachment; filename=snapshot.jpg")
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head><title>Pi Zero 2W Capture Streamer</title>
<style>
body {{ background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; text-align: center; margin: 0; padding: 20px; }}
h1 {{ color: #38bdf8; }}
img {{ width: 90%; max-width: 960px; border-radius: 12px; border: 2px solid #334155; }}
.badge {{ background: #10b981; color: #fff; padding: 4px 10px; border-radius: 9999px; font-weight: bold; }}
</style>
</head>
<body>
<h1>Pi Zero 2W Camera Capture <span class="badge">LIVE 30 FPS</span></h1>
<p>Endpoint: <code>/stream.mjpg</code> | Resolution: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}</p>
<img src="/stream.mjpg" alt="Live Feed">
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

class ThreadedHTTPServer(HTTPServer):
    """Serve requests in separate threads for non-blocking multi-client support."""
    def process_request(self, request, client_address):
        threading.Thread(target=self.finish_request, args=(request, client_address), daemon=True).start()

def main():
    cam_thread = threading.Thread(target=camera_worker, daemon=True)
    cam_thread.start()
    
    time.sleep(1)
    port = 8000
    print(f"[Server] Starting Zero-Lag HTTP Streamer on port {port}...")
    server = ThreadedHTTPServer(("0.0.0.0", port), ZeroLagStreamingHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
'''

def deploy():
    print(f"[*] Connecting to Raspberry Pi at {PI_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    print("[+] Connected to Pi successfully!")

    def run_sudo(cmd):
        stdin, stdout, stderr = ssh.exec_command(f"sudo -S {cmd}", get_pty=True)
        stdin.write(f"{PI_PASS}\\n")
        stdin.flush()
        return stdout.read().decode()

    # Backup existing stream_server.py
    print("[*] Backing up previous stream_server.py...")
    ssh.exec_command("cp /home/stickcam/seal-scanner/stream_server.py /home/stickcam/seal-scanner/stream_server.py.bak")

    # Upload new optimized server
    print("[*] Uploading ultra-low latency capture server to Pi...")
    sftp = ssh.open_sftp()
    with sftp.file("/home/stickcam/seal-scanner/stream_server.py", "w") as f:
        f.write(OPTIMIZED_STREAM_SERVER)
    sftp.chmod("/home/stickcam/seal-scanner/stream_server.py", 0o755)
    sftp.close()

    # Restart seal-stream.service
    print("[*] Restarting seal-stream.service...")
    res = run_sudo("systemctl restart seal-stream.service")
    print(res)

    time.sleep(2)

    # Check status
    print("[*] Checking service status...")
    stdin, stdout, stderr = ssh.exec_command("systemctl is-active seal-stream.service")
    status = stdout.read().decode().strip()
    print(f"[+] seal-stream.service is: {status}")

    # Check CPU usage of the streaming server
    time.sleep(2)
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep stream_server.py | grep -v grep")
    print("[+] Process running on Pi:\n" + stdout.read().decode().strip())

    ssh.close()
    print("[+] Deployment complete!")

if __name__ == "__main__":
    deploy()
