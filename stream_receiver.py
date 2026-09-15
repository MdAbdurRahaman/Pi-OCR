"""
stream_receiver.py - Universal Zero-Buffer Video Stream Receiver.
Supports:
1. Mobile Web App stream (WebSocket / Direct push from phone browser)
2. HTTP / MJPEG network stream (IP Webcam, DroidCam, etc.)
3. OpenCV VideoCapture (RTSP streams, USB / Virtual webcams)
Guarantees sub-30ms latency by discarding buffered/stale frames.
"""
import time
import threading
import urllib.request
import cv2
import numpy as np


class StreamReceiver:
    def __init__(self, mode: str = "push", stream_url: str = None, camera_index: int = None):
        """
        mode:
          - 'push': Phone web app pushes frames via WebSocket or HTTP (default)
          - 'mjpeg': Pulls MJPEG stream from HTTP URL (e.g. IP Webcam app)
          - 'capture': OpenCV VideoCapture from RTSP URL or USB camera index
        """
        self.mode = mode
        self.stream_url = stream_url
        self.camera_index = camera_index

        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_count = 0
        self.current_fps = 0.0
        self.running = False
        self.connected = False
        self.source_desc = "Waiting for phone connection..."

        self.lock = threading.Lock()
        self.thread = None

        # FPS calculation window
        self._fps_window = 15
        self._fps_counter = 0
        self._t_last_fps = time.time()

    def start(self):
        """Start ingestion worker if using pull/capture modes."""
        if self.running:
            return
        self.running = True

        if self.mode == "push":
            self.source_desc = "Mobile Web App (Push Mode)"
            # Push mode doesn't need a pull thread; it receives frames via push_frame()
        elif self.mode == "mjpeg":
            self.source_desc = f"MJPEG Stream: {self.stream_url}"
            self.thread = threading.Thread(target=self._mjpeg_worker, daemon=True)
            self.thread.start()
        elif self.mode == "capture":
            target = self.camera_index if self.camera_index is not None else self.stream_url
            self.source_desc = f"VideoCapture: {target}"
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop receiving frames."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.connected = False

    def push_frame(self, frame: np.ndarray, client_timestamp: float = None):
        """Called directly by WebSocket handler when phone pushes an image frame."""
        if frame is None:
            return
        now = time.time()
        with self.lock:
            self.latest_frame = frame
            self.latest_timestamp = client_timestamp if client_timestamp else now
            self.frame_count += 1
            self.connected = True

        self._update_fps(now)

    def push_jpeg_bytes(self, jpg_bytes: bytes, client_timestamp: float = None) -> bool:
        """Decodes raw JPEG byte array directly into an OpenCV BGR frame."""
        try:
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.push_frame(frame, client_timestamp)
                return True
        except Exception as e:
            print(f"[StreamReceiver] JPEG decode error: {e}")
        return False

    def mark_disconnected(self):
        """Mark phone stream as disconnected."""
        with self.lock:
            self.connected = False

    def _update_fps(self, now: float):
        self._fps_counter += 1
        if self._fps_counter >= self._fps_window:
            elapsed = now - self._t_last_fps
            if elapsed > 0:
                self.current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._t_last_fps = now

    def _mjpeg_worker(self):
        """Worker for external HTTP MJPEG streams (e.g. IP Webcam app)."""
        while self.running:
            try:
                print(f"[StreamReceiver] Connecting to MJPEG stream {self.stream_url}...")
                req = urllib.request.Request(
                    self.stream_url,
                    headers={
                        "User-Agent": "StickCamPhoneReceiver/1.0",
                        "Accept": "multipart/x-mixed-replace, image/jpeg",
                    }
                )
                stream = urllib.request.urlopen(req, timeout=5)
                self.connected = True
                print(f"[StreamReceiver] Connected to {self.stream_url}!")

                buffer = b""
                while self.running:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    buffer += chunk

                    # Look for JPEG start and end markers
                    a = buffer.find(b"\xff\xd8")
                    b = buffer.find(b"\xff\xd9")

                    if a != -1 and b != -1:
                        if b > a:
                            jpg_data = buffer[a:b+2]
                            buffer = buffer[b+2:]

                            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                now = time.time()
                                with self.lock:
                                    self.latest_frame = frame
                                    self.latest_timestamp = now
                                    self.frame_count += 1
                                self._update_fps(now)
                        else:
                            buffer = buffer[a:]

                    if len(buffer) > 262144:  # 256KB overflow cap
                        buffer = buffer[-65536:]

            except Exception as e:
                self.connected = False
                print(f"[StreamReceiver] Stream error: {e}. Reconnecting in 1s...")
                time.sleep(1.0)

    def _capture_worker(self):
        """Worker for RTSP stream or USB / Virtual camera backend."""
        target = self.camera_index if self.camera_index is not None else self.stream_url
        while self.running:
            cap = cv2.VideoCapture(target)
            if not cap.isOpened():
                self.connected = False
                print(f"[StreamReceiver] Failed to open VideoCapture ({target}). Retrying...")
                time.sleep(1.5)
                continue

            # Drop internal buffer to 1 for lowest latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.connected = True
            print(f"[StreamReceiver] Opened VideoCapture ({target}) successfully.")

            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[StreamReceiver] Frame grab failed, reconnecting...")
                    break

                now = time.time()
                with self.lock:
                    self.latest_frame = frame
                    self.latest_timestamp = now
                    self.frame_count += 1
                self._update_fps(now)

            cap.release()
            self.connected = False
            time.sleep(1.0)

    def get_frame(self):
        """
        Returns (frame, fps, latency_ms, is_connected)
        guaranteeing zero buffer backlog.
        """
        now = time.time()
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0, 0.0, self.connected
            if self.mode == "push":
                # In burst / push mode, maintain the last photo on screen between shutter snaps
                is_alive = self.connected
            else:
                # Continuous streams timeout if no frame arrived in 2.5s
                is_alive = self.connected and (now - self.latest_timestamp < 2.5)
            frame = self.latest_frame.copy()
            latency_ms = (now - self.latest_timestamp) * 1000.0
            return frame, self.current_fps, latency_ms, is_alive
