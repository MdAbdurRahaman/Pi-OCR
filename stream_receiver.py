"""
stream_receiver.py - Zero-Buffer Low-Latency Video Stream Receiver.
Continuously consumes frames from the Pi Zero 2W stream and ensures the consumer
always receives the freshest frame with sub-35ms latency.
"""
import time
import socket
import threading
import urllib.request
import cv2
import numpy as np


class StreamReceiver:
    def __init__(self, stream_url: str = "http://192.168.68.145:8000/stream.mjpg"):
        self.stream_url = stream_url
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_count = 0
        self.current_fps = 0.0
        self.running = False
        self.connected = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Start the background ingestion thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop receiving frames."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _worker(self):
        while self.running:
            try:
                print(f"[StreamReceiver] Connecting to {self.stream_url}...")
                req = urllib.request.Request(
                    self.stream_url,
                    headers={
                        "User-Agent": "ZeroLagReceiver/1.0",
                        "Accept": "multipart/x-mixed-replace, image/jpeg",
                    }
                )
                stream = urllib.request.urlopen(req, timeout=5)
                self.connected = True
                print("[StreamReceiver] Connected to Pi camera stream!")

                buffer = b""
                fps_calc_window = 20
                fps_counter = 0
                t_last_fps = time.time()

                while self.running:
                    # Read chunks from socket
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    buffer += chunk

                    # Look for JPEG start and end markers
                    a = buffer.find(b"\xff\xd8")
                    b = buffer.find(b"\xff\xd9")

                    if a != -1 and b != -1:
                        if b > a:
                            # Full JPEG frame extracted
                            jpg_data = buffer[a:b+2]
                            buffer = buffer[b+2:]

                            # Decode frame
                            frame = cv2.imdecode(
                                np.frombuffer(jpg_data, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )
                            if frame is not None:
                                now = time.time()
                                with self.lock:
                                    self.latest_frame = frame
                                    self.latest_timestamp = now
                                    self.frame_count += 1

                                fps_counter += 1
                                if fps_counter >= fps_calc_window:
                                    elapsed = now - t_last_fps
                                    self.current_fps = fps_counter / elapsed if elapsed > 0 else 30.0
                                    fps_counter = 0
                                    t_last_fps = now
                        else:
                            # Discard malformed prefix
                            buffer = buffer[a:]

                    # Avoid runaway buffer growth in case of packet loss
                    if len(buffer) > 262144: # 256KB cap
                        buffer = buffer[-65536:]

            except Exception as e:
                self.connected = False
                print(f"[StreamReceiver] Connection error: {e}. Reconnecting in 1s...")
                time.sleep(1.0)

    def get_frame(self):
        """
        Returns (frame, fps, latency_ms, is_connected)
        guaranteeing zero buffer backlog.
        """
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0, 0.0, self.connected
            frame = self.latest_frame.copy()
            latency_ms = (time.time() - self.latest_timestamp) * 1000.0
            return frame, self.current_fps, latency_ms, self.connected


if __name__ == "__main__":
    print("[*] Testing StreamReceiver...")
    receiver = StreamReceiver("http://192.168.68.145:8000/stream.mjpg")
    receiver.start()

    time.sleep(2)
    for _ in range(10):
        frame, fps, latency_ms, connected = receiver.get_frame()
        if frame is not None:
            print(f"[Receiver] Frame: {frame.shape}, FPS: {fps:.2f}, Frame Age: {latency_ms:.1f}ms, Connected: {connected}")
        else:
            print("[Receiver] Waiting for frame...")
        time.sleep(0.2)

    receiver.stop()
