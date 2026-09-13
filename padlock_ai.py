"""
padlock_ai.py - AI Padlock and Security Seal Character Recognition Engine.
Uses RapidOCR ONNX with asynchronous inference, temporal tracking,
and HUD overlay rendering.
"""
import time
import re
import threading
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class PadlockDetector:
    def __init__(self, conf_threshold: float = 0.50):
        self.conf_threshold = conf_threshold
        print("[AI] Initializing RapidOCR ONNX engine...")
        self.engine = RapidOCR()
        print("[AI] RapidOCR engine ready!")

        # Async inference control
        self.latest_frame = None
        self.frame_ready = threading.Event()
        self.lock = threading.Lock()
        self.running = False
        self.worker_thread = None

        # Tracking state
        self.tracked_detections = []
        self.last_inference_time_ms = 0.0
        self.best_padlock_id = None
        self.best_confidence = 0.0

    def start(self):
        """Start async AI background thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop async AI thread."""
        self.running = False
        self.frame_ready.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def submit_frame(self, frame: np.ndarray):
        """Submit a fresh frame for OCR without blocking."""
        with self.lock:
            self.latest_frame = frame
            self.frame_ready.set()

    def _ai_worker(self):
        """Dedicated background thread for AI character recognition."""
        while self.running:
            self.frame_ready.wait()
            self.frame_ready.clear()

            with self.lock:
                if self.latest_frame is None:
                    continue
                frame_to_process = self.latest_frame.copy()

            t0 = time.time()
            try:
                raw_results, elapse = self.engine(frame_to_process)
            except Exception as e:
                print(f"[AI] Inference error: {e}")
                continue
            t1 = time.time()

            self.last_inference_time_ms = (t1 - t0) * 1000.0

            new_detections = []
            best_id = None
            best_score = 0.0

            if raw_results:
                for item in raw_results:
                    box, text, score_str = item
                    score = float(score_str)
                    clean_text = text.strip()

                    if score < self.conf_threshold or len(clean_text) < 2:
                        continue

                    # Serial pattern (e.g. C 581819, 581819, C581819)
                    is_serial = bool(re.search(r'[A-Z0-9]{4,10}', clean_text.replace(" ", "")))

                    pts = np.array(box, dtype=np.int32)
                    new_detections.append({
                        "text": clean_text,
                        "score": score,
                        "pts": pts,
                        "is_serial": is_serial,
                        "timestamp": time.time()
                    })

                    if is_serial and score > best_score:
                        best_id = clean_text
                        best_score = score

            with self.lock:
                self._update_tracks(new_detections)
                if best_id:
                    self.best_padlock_id = best_id
                    self.best_confidence = best_score

    def _update_tracks(self, new_detections):
        """Merge new detections with temporal tracks to eliminate flicker."""
        now = time.time()
        ttl = 0.85  # Keep detections visible for 850ms across frames

        updated_tracks = []
        for det in new_detections:
            updated_tracks.append(det)

        for track in self.tracked_detections:
            if now - track["timestamp"] < ttl:
                matched = any(new_det["text"] == track["text"] for new_det in new_detections)
                if not matched:
                    updated_tracks.append(track)

        self.tracked_detections = updated_tracks

    def get_detections(self):
        """Get active smoothed detections and telemetry."""
        now = time.time()
        ttl = 0.85
        with self.lock:
            active = [d for d in self.tracked_detections if now - d["timestamp"] < ttl]
            return active, self.last_inference_time_ms, self.best_padlock_id, self.best_confidence

    def clear_detections(self):
        """Clear active detections."""
        with self.lock:
            self.tracked_detections = []
            self.best_padlock_id = None
            self.best_confidence = 0.0

    def draw_hud(self, frame: np.ndarray, show_boxes: bool = True, fps: float = 30.0, latency_ms: float = 10.0) -> np.ndarray:
        """
        Renders bounding boxes, text pills, corner brackets,
        and HUD status on top of the live video frame without darkening the video.
        """
        h, w = frame.shape[:2]
        detections, ocr_time_ms, padlock_id, padlock_conf = self.get_detections()

        if show_boxes:
            for det in detections:
                pts = det["pts"]
                text = det["text"]
                score = det["score"]
                is_serial = det["is_serial"]

                # Border colors
                border_color = (0, 255, 100) if is_serial else (255, 190, 20)

                # Draw polygon bounding box with antialiasing
                cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

                # Corner reticles
                for pt in pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)

                # Label text
                label = f"{text} ({int(score * 100)}%)"
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 0.52
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

                min_y = int(np.min(pts[:, 1]))
                min_x = int(np.min(pts[:, 0]))

                lbl_y = max(th + 8, min_y - 6)
                lbl_x = max(10, min_x)

                pad_x = 6
                pad_y = 3
                x1 = max(0, lbl_x - pad_x)
                y1 = max(0, lbl_y - th - pad_y)
                x2 = min(w, lbl_x + tw + pad_x)
                y2 = min(h, lbl_y + baseline + pad_y)

                # Alpha blend ONLY the small pill region
                if y2 > y1 and x2 > x1:
                    sub_roi = frame[y1:y2, x1:x2]
                    pill_bg = np.zeros_like(sub_roi)
                    pill_bg[:] = (20, 25, 32)
                    frame[y1:y2, x1:x2] = cv2.addWeighted(sub_roi, 0.35, pill_bg, 0.65, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

                cv2.putText(frame, label, (lbl_x, lbl_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Top HUD Banner (Draw over top 42px)
        hud_bar_h = 42
        banner_roi = frame[0:hud_bar_h, 0:w]
        banner_bg = np.zeros_like(banner_roi)
        banner_bg[:] = (12, 16, 24)
        frame[0:hud_bar_h, 0:w] = cv2.addWeighted(banner_roi, 0.15, banner_bg, 0.85, 0)
        cv2.line(frame, (0, hud_bar_h), (w, hud_bar_h), (45, 55, 70), 1)

        # STICKCAM AI logo
        cv2.putText(frame, "STICKCAM AI", (14, 27), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 215, 255), 1, cv2.LINE_AA)

        # Active Padlock text
        if padlock_id:
            cv2.putText(frame, f"PADLOCK: {padlock_id} ({int(padlock_conf*100)}%)", (170, 27), cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 100), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "PADLOCK: SEARCHING...", (170, 27), cv2.FONT_HERSHEY_DUPLEX, 0.58, (130, 145, 160), 1, cv2.LINE_AA)

        # Performance meters (FPS, Ingestion Lag, AI Speed)
        fps_color = (0, 255, 100) if fps >= 24 else (0, 200, 255)
        stat_text = f"CAM: {fps:.1f} FPS | LAG: {latency_ms:.0f}ms | AI: {int(ocr_time_ms)}ms"
        (sw, _), _ = cv2.getTextSize(stat_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(frame, stat_text, (w - sw - 14, 27), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 210, 225), 1, cv2.LINE_AA)

        return frame
