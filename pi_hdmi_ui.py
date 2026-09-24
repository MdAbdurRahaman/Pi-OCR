#!/usr/bin/env python3
"""
pi_hdmi_ui.py - 3.5" HDMI Display UI Engine & Dual Physical Button Controller.
Designed for Raspberry Pi Zero 2W / Pi 4 / Pi 5 with a 3.5" HDMI Display (e.g. 480x320 / 800x480).

Key Advantages over SPI Display:
1. Native HDMI Output: Direct GPU video output over HDMI with 30-60 FPS refresh rate.
   Zero CPU overhead from SPI bit-banging and eliminated white-screen / bus-collision glitches.
2. Dual Output Modes:
   - Direct Linux Framebuffer (/dev/fb0) for Raspberry Pi OS Lite (no X11 / desktop required).
   - Native OpenCV Window (cv2.imshow) for graphical desktop (X11 / Wayland) & PC simulator mode.
3. Physical Button Controller:
   - Button 1 (GPIO 17 / Physical Pin 11 to GND): Capture & Run OCR
   - Button 2 (GPIO 27 / Physical Pin 13 to GND): Re-capture & Reset Viewfinder
4. Live Focus Telemetry: Real-time sharpness meter with Green / Yellow / Red focus quality status.
5. On-Screen Touch / Mouse Support & High-Visibility Click Feedback Indicators.
"""
import os
import sys
import time
import mmap
import struct
import threading
from typing import Callable, Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

# Hardware GPIO Pins for Physical Push Buttons
GPIO_BTN_CAPTURE = 17    # Physical Pin 11 to GND (Button 1: Capture)
GPIO_BTN_RECAPTURE = 27  # Physical Pin 13 to GND (Button 2: Re-capture)


class HdmiFramebuffer:
    """High-speed direct Linux framebuffer (/dev/fb0) driver for HDMI displays."""

    def __init__(self, device: str = "/dev/fb0", target_w: int = 480, target_h: int = 320):
        self.device = device
        self.target_w = target_w
        self.target_h = target_h
        self.fb_w = target_w
        self.fb_h = target_h
        self.bpp = 32
        self.line_length = target_w * 4
        self.fb_file = None
        self.fb_mmap = None
        self.is_active = False

        self._init_framebuffer()

    def _init_framebuffer(self):
        if not os.path.exists(self.device):
            return

        try:
            # Query virtual size from sysfs if available
            size_path = "/sys/class/graphics/fb0/virtual_size"
            bpp_path = "/sys/class/graphics/fb0/bits_per_pixel"
            stride_path = "/sys/class/graphics/fb0/stride"

            if os.path.exists(size_path):
                with open(size_path, "r") as f:
                    parts = f.read().strip().split(",")
                    if len(parts) == 2:
                        self.fb_w = int(parts[0])
                        self.fb_h = int(parts[1])

            if os.path.exists(bpp_path):
                with open(bpp_path, "r") as f:
                    self.bpp = int(f.read().strip())

            if os.path.exists(stride_path):
                with open(stride_path, "r") as f:
                    self.line_length = int(f.read().strip())
            else:
                self.line_length = self.fb_w * (self.bpp // 8)

            fb_bytes = self.line_length * self.fb_h
            self.fb_file = open(self.device, "r+b")

            try:
                self.fb_mmap = mmap.mmap(
                    self.fb_file.fileno(),
                    fb_bytes,
                    mmap.MAP_SHARED,
                    mmap.PROT_WRITE | mmap.PROT_READ
                )
            except Exception:
                self.fb_mmap = None

            self.is_active = True
            print(f"[HDMI FB] /dev/fb0 active: {self.fb_w}x{self.fb_h} @ {self.bpp} bpp (stride: {self.line_length})")
        except Exception as e:
            print(f"[HDMI FB] Framebuffer init notice: {e}")
            self.is_active = False

    def write_frame(self, bgr_image: np.ndarray):
        """Pushes BGR image to HDMI framebuffer with resolution adaptation."""
        if not self.is_active:
            return

        h, w = bgr_image.shape[:2]
        if w != self.fb_w or h != self.fb_h:
            bgr_image = cv2.resize(bgr_image, (self.fb_w, self.fb_h), interpolation=cv2.INTER_LINEAR)

        try:
            if self.bpp == 32:
                # 32-bit: BGR to BGRA
                bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
                raw_bytes = bgra.tobytes()
            elif self.bpp == 16:
                # 16-bit: RGB565
                rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                r5 = (rgb[:, :, 0] >> 3).astype(np.uint16)
                g6 = (rgb[:, :, 1] >> 2).astype(np.uint16)
                b5 = (rgb[:, :, 2] >> 3).astype(np.uint16)
                rgb565 = (r5 << 11) | (g6 << 5) | b5
                raw_bytes = rgb565.tobytes()
            else:
                raw_bytes = bgr_image.tobytes()

            if self.fb_mmap is not None:
                self.fb_mmap.seek(0)
                self.fb_mmap.write(raw_bytes)
            elif self.fb_file is not None:
                self.fb_file.seek(0)
                self.fb_file.write(raw_bytes)
                self.fb_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            if self.fb_mmap is not None:
                self.fb_mmap.close()
            if self.fb_file is not None:
                self.fb_file.close()
        except Exception:
            pass
        self.is_active = False


class StickCamHdmiUI:
    """
    Manages the 3.5" HDMI display graphics, layout, focus telemetry,
    dual physical buttons, on-screen touch buttons, and inspection states.
    """

    STATE_VIEWFINDER = "viewfinder"
    STATE_CAPTURING = "capturing"
    STATE_RESULT = "result"

    def __init__(
        self,
        on_capture_cb: Optional[Callable[[], None]] = None,
        on_recapture_cb: Optional[Callable[[], None]] = None,
        sim_mode: bool = False,
        width: int = 480,
        height: int = 320,
        fb_device: str = "/dev/fb0",
    ):
        self.on_capture = on_capture_cb
        self.on_recapture = on_recapture_cb
        self.sim_mode = sim_mode
        self.fb_device = fb_device

        # Dimensions: 480x320 native 3.5" aspect ratio
        self.width = width
        self.height = height

        # Layout segments
        self.top_h = 32
        self.vp_y = 32
        self.vp_h = self.height - 32 - 68  # 220px viewport
        self.bottom_y = self.height - 68    # 252px
        self.bottom_h = 68

        # Interactive Button Hitboxes (Left: Capture, Right: Re-capture)
        margin = 10
        btn_w = (self.width - (margin * 3)) // 2
        self.btn_capture_rect = (margin, self.bottom_y + 8, margin + btn_w, self.height - 8)
        self.btn_recapture_rect = (margin * 2 + btn_w, self.bottom_y + 8, self.width - margin, self.height - 8)

        # State Machine & Data
        self.state = self.STATE_VIEWFINDER
        self.latest_frame: Optional[np.ndarray] = None
        self.result_data: Dict[str, Any] = {}
        self.status_msg = "Align seal inside box. Press Button 1 to Capture"
        self.pc_connected = False
        self.server_mode = "Standalone Offline"
        self.ip_address = "127.0.0.1"

        # Focus Telemetry (Updated dynamically in real-time)
        self.sharpness_score: float = 0.0
        self.focus_quality: str = "BLURRY"  # "BLURRY", "FAIR", "SHARP"
        self.focus_color: Tuple[int, int, int] = (0, 0, 255)  # BGR

        # Button highlight / click animation state
        self.btn_capture_active = False
        self.btn_recapture_active = False
        self.last_clicked_button: Optional[int] = None  # 1 or 2
        self.last_button_click_time: float = 0.0

        # Thread synchronization
        self.lock = threading.Lock()
        self.running = False

        # Rendering buffer
        self.buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Initialize Hardware
        self.hdmi_fb: Optional[HdmiFramebuffer] = None
        self._init_display_hardware()

        if not self.sim_mode:
            self._init_physical_gpio_buttons()

    def _init_display_hardware(self):
        """Initializes HDMI framebuffer if present."""
        if self.sim_mode:
            print("[HDMI UI] Initialized in simulator mode.")
            return

        if os.path.exists(self.fb_device):
            self.hdmi_fb = HdmiFramebuffer(device=self.fb_device, target_w=self.width, target_h=self.height)
            if self.hdmi_fb.is_active:
                print(f"[HDMI UI] Direct HDMI Framebuffer active on {self.fb_device}!")
                return

        print("[HDMI UI] Framebuffer not detected. Running with GUI / Mirror mode.")

    def _init_physical_gpio_buttons(self):
        """Registers physical tactile push buttons on GPIO 17 & GPIO 27 with debouncing."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Button 1: CAPTURE (GPIO 17 / Physical Pin 11 to GND)
            GPIO.setup(GPIO_BTN_CAPTURE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            # Button 2: RE-CAPTURE (GPIO 27 / Physical Pin 13 to GND)
            GPIO.setup(GPIO_BTN_RECAPTURE, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            def gpio_capture_cb(channel):
                print(f"[GPIO] Physical Button 1 (CAPTURE) pressed on GPIO {channel}!")
                self.trigger_capture(source="BUTTON 1 (GPIO 17)")

            def gpio_recapture_cb(channel):
                print(f"[GPIO] Physical Button 2 (RE-CAPTURE) pressed on GPIO {channel}!")
                self.trigger_recapture(source="BUTTON 2 (GPIO 27)")

            GPIO.add_event_detect(GPIO_BTN_CAPTURE, GPIO.FALLING, callback=gpio_capture_cb, bouncetime=300)
            GPIO.add_event_detect(GPIO_BTN_RECAPTURE, GPIO.FALLING, callback=gpio_recapture_cb, bouncetime=300)
            print(f"[GPIO] Dual physical buttons configured: [Btn 1: GPIO {GPIO_BTN_CAPTURE}], [Btn 2: GPIO {GPIO_BTN_RECAPTURE}]")
        except Exception as e:
            print(f"[GPIO] Note: Physical GPIO buttons disabled ({e})")

    def update_viewfinder_frame(self, frame: np.ndarray, sharpness: Optional[float] = None):
        """Feeds live camera frame and updates focus telemetry."""
        with self.lock:
            if frame is not None:
                self.latest_frame = frame

            if sharpness is not None:
                self.sharpness_score = sharpness
            elif frame is not None:
                # Calculate Laplacian variance on central crop
                h, w = frame.shape[:2]
                ch, cw = int(h * 0.4), int(w * 0.5)
                cy, cx = h // 2, w // 2
                crop = frame[cy - ch // 2: cy + ch // 2, cx - cw // 2: cx + cw // 2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    self.sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Focus classification
            if self.sharpness_score > 320:
                self.focus_quality = "SHARP"
                self.focus_color = (0, 255, 100)    # Green
            elif self.sharpness_score > 160:
                self.focus_quality = "FAIR"
                self.focus_color = (0, 215, 255)    # Yellow
            else:
                self.focus_quality = "BLURRY"
                self.focus_color = (0, 0, 255)      # Red

    def set_state(self, state: str, message: Optional[str] = None):
        with self.lock:
            self.state = state
            if message:
                self.status_msg = message

    def set_result(self, result_dict: Dict[str, Any]):
        """Sets OCR result payload and switches display to inspection view."""
        with self.lock:
            self.result_data = result_dict
            self.state = self.STATE_RESULT
            serial = result_dict.get("serial_number")
            if serial:
                self.status_msg = f"Detected: {serial}"
            else:
                self.status_msg = "No code recognized. Press Btn 2 to Re-capture"

    def trigger_capture(self, source: str = "BUTTON 1 (GPIO 17)"):
        """Triggers Capture & OCR workflow with animated button highlight."""
        with self.lock:
            self.last_clicked_button = 1
            self.last_button_click_time = time.time()
            self.btn_capture_active = True
            self.btn_recapture_active = False
            self.state = self.STATE_CAPTURING
            self.status_msg = "Capturing burst & analyzing seal..."

        def _reset_highlight():
            time.sleep(0.55)
            with self.lock:
                self.btn_capture_active = False

        threading.Thread(target=_reset_highlight, daemon=True).start()

        if self.on_capture:
            threading.Thread(target=self.on_capture, daemon=True).start()

    def trigger_recapture(self, source: str = "BUTTON 2 (GPIO 27)"):
        """Resets scanner back to live camera viewfinder."""
        with self.lock:
            self.last_clicked_button = 2
            self.last_button_click_time = time.time()
            self.btn_recapture_active = True
            self.btn_capture_active = False
            self.state = self.STATE_VIEWFINDER
            self.result_data = {}
            self.status_msg = "Live Viewfinder Resumed. Ready."

        def _reset_highlight():
            time.sleep(0.55)
            with self.lock:
                self.btn_recapture_active = False

        threading.Thread(target=_reset_highlight, daemon=True).start()

        if self.on_recapture:
            threading.Thread(target=self.on_recapture, daemon=True).start()

    def handle_touch_point(self, x: int, y: int) -> bool:
        """Processes on-screen touch taps / mouse clicks on buttons."""
        cx1, cy1, cx2, cy2 = self.btn_capture_rect
        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            self.trigger_capture(source="TOUCH BTN 1")
            return True

        rx1, ry1, rx2, ry2 = self.btn_recapture_rect
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            self.trigger_recapture(source="TOUCH BTN 2")
            return True

        return False

    def render(self) -> np.ndarray:
        """Renders complete 480x320 UI frame and pushes to HDMI framebuffer."""
        with self.lock:
            state = self.state
            status = self.status_msg
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            res = dict(self.result_data)
            cap_active = self.btn_capture_active
            recap_active = self.btn_recapture_active
            last_clicked = self.last_clicked_button
            click_age = time.time() - self.last_button_click_time
            recent_click = (click_age < 0.70)
            sharpness = self.sharpness_score
            focus_quality = self.focus_quality
            focus_color = self.focus_color

        w = self.width
        h = self.height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # -------------------------------------------------------------
        # 1. HEADER BAR (Top 32px)
        # -------------------------------------------------------------
        canvas[0:self.top_h, :] = (15, 23, 42)
        cv2.line(canvas, (0, self.top_h - 1), (w, self.top_h - 1), (34, 50, 80), 1)

        # App Title & HDMI Badge
        cv2.putText(canvas, "STICK CAM", (10, 21), cv2.FONT_HERSHEY_DUPLEX, 0.52, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (110, 6), (180, 24), (30, 41, 59), -1)
        cv2.rectangle(canvas, (110, 6), (180, 24), (56, 189, 248), 1)
        self._draw_centered_text(canvas, "HDMI 3.5\"", 145, 15, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (226, 232, 240), 1)

        # Center Status Banner
        if recent_click and last_clicked == 1:
            cv2.rectangle(canvas, (w // 2 - 80, 4), (w // 2 + 80, self.top_h - 5), (16, 185, 129), -1)
            self._draw_centered_text(canvas, "BTN 1 CLICKED", w // 2, 17, cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 0, 0), 1)
        elif recent_click and last_clicked == 2:
            cv2.rectangle(canvas, (w // 2 - 80, 4), (w // 2 + 80, self.top_h - 5), (11, 158, 245), -1)
            self._draw_centered_text(canvas, "BTN 2 CLICKED", w // 2, 17, cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 0, 0), 1)
        else:
            if state == self.STATE_CAPTURING:
                cv2.putText(canvas, "ANALYZING...", (w // 2 - 45, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
            elif state == self.STATE_RESULT:
                cv2.putText(canvas, "RESULT READY", (w // 2 - 45, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)
            else:
                status_color = (16, 185, 129) if focus_quality == "SHARP" else (148, 163, 184)
                cv2.putText(canvas, "LIVE VIEWFINDER", (w // 2 - 55, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1, cv2.LINE_AA)

        # Right Connection & Mode Dot
        dot_color = (16, 185, 129) if self.pc_connected else (56, 189, 248)
        cv2.circle(canvas, (w - 14, 16), 5, dot_color, -1)

        # -------------------------------------------------------------
        # 2. MAIN VIEWPORT (Middle 220px)
        # -------------------------------------------------------------
        vp = canvas[self.vp_y:self.bottom_y, :]
        vp_h, vp_w = vp.shape[:2]

        if state == self.STATE_VIEWFINDER:
            if frame is not None:
                vp[:] = cv2.resize(frame, (vp_w, vp_h), interpolation=cv2.INTER_LINEAR)
            else:
                vp[:] = (20, 24, 32)
                cv2.putText(vp, "INITIALIZING CAMERA...", (vp_w // 2 - 85, vp_h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.50, (148, 163, 184), 1, cv2.LINE_AA)

            # Central Target Reticle for Seal Positioning
            cx, cy = vp_w // 2, vp_h // 2
            rw = min(140, vp_w // 2 - 25)
            rh = min(60, vp_h // 2 - 25)
            x1, y1 = cx - rw, cy - rh
            x2, y2 = cx + rw, cy + rh

            reticle_col = focus_color
            arm = 18
            cv2.line(vp, (x1, y1), (x1 + arm, y1), reticle_col, 2)
            cv2.line(vp, (x1, y1), (x1, y1 + arm), reticle_col, 2)
            cv2.line(vp, (x2, y1), (x2 - arm, y1), reticle_col, 2)
            cv2.line(vp, (x2, y1), (x2, y1 + arm), reticle_col, 2)
            cv2.line(vp, (x1, y2), (x1 + arm, y2), reticle_col, 2)
            cv2.line(vp, (x1, y2), (x1, y2 - arm), reticle_col, 2)
            cv2.line(vp, (x2, y2), (x2 - arm, y2), reticle_col, 2)
            cv2.line(vp, (x2, y2), (x2 - arm, y2), reticle_col, 2)
            cv2.circle(vp, (cx, cy), 3, (0, 255, 128), -1)

            # Live Focus Sharpness Telemetry Bar (Top of Viewport)
            bar_w = 160
            bar_h = 10
            bar_x = vp_w - bar_w - 12
            bar_y = 10

            cv2.rectangle(vp, (bar_x - 4, bar_y - 4), (bar_x + bar_w + 4, bar_y + bar_h + 16), (15, 23, 42), -1)
            cv2.rectangle(vp, (bar_x - 4, bar_y - 4), (bar_x + bar_w + 4, bar_y + bar_h + 16), (51, 65, 85), 1)

            # Meter fill (scale 0-500 sharpness)
            fill_pct = min(1.0, max(0.0, sharpness / 500.0))
            fill_w = int(bar_w * fill_pct)
            cv2.rectangle(vp, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 41, 59), -1)
            if fill_w > 0:
                cv2.rectangle(vp, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), focus_color, -1)

            meter_lbl = f"FOCUS: {focus_quality} ({int(sharpness)})"
            cv2.putText(vp, meter_lbl, (bar_x, bar_y + bar_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, focus_color, 1, cv2.LINE_AA)

            # Bottom Hint Strip
            if recent_click and last_clicked == 1:
                cv2.rectangle(vp, (0, vp_h - 24), (vp_w, vp_h), (16, 150, 80), -1)
                self._draw_centered_text(vp, ">>> [BTN 1 PRESSED] CAPTURING BURST <<<",
                                         vp_w // 2, vp_h - 12, cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1)
            elif recent_click and last_clicked == 2:
                cv2.rectangle(vp, (0, vp_h - 24), (vp_w, vp_h), (11, 130, 220), -1)
                self._draw_centered_text(vp, ">>> [BTN 2 PRESSED] VIEWFINDER RESET <<<",
                                         vp_w // 2, vp_h - 12, cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1)
            else:
                cv2.rectangle(vp, (0, vp_h - 22), (vp_w, vp_h), (10, 15, 25), -1)
                self._draw_centered_text(vp, "Press Btn 1 to Capture  |  Btn 2 to Reset",
                                         vp_w // 2, vp_h - 11, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226, 232, 240), 1)

        elif state == self.STATE_CAPTURING:
            if frame is not None:
                vp[:] = cv2.resize(frame, (vp_w, vp_h), interpolation=cv2.INTER_LINEAR)
            else:
                vp[:] = (15, 20, 30)

            # Capture Feedback Card
            cv2.rectangle(vp, (25, vp_h // 2 - 45), (vp_w - 25, vp_h // 2 + 45), (15, 25, 40), -1)
            cv2.rectangle(vp, (25, vp_h // 2 - 45), (vp_w - 25, vp_h // 2 + 45), (16, 185, 129), 2)
            self._draw_centered_text(vp, "[BUTTON 1 TRIGGERED]", vp_w // 2, vp_h // 2 - 20,
                                     cv2.FONT_HERSHEY_DUPLEX, 0.52, (16, 255, 180), 1)
            self._draw_centered_text(vp, "CAPTURING & ANALYZING BURST...", vp_w // 2, vp_h // 2 + 6,
                                     cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 2)
            self._draw_centered_text(vp, "Evaluating peak focus sharpness...", vp_w // 2, vp_h // 2 + 28,
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)

        elif state == self.STATE_RESULT:
            vp[:] = (11, 15, 25)

            # Extract winning image / crop
            annotated_frame = None
            annotated_b64 = res.get("annotated_image")
            if annotated_b64 and annotated_b64.startswith("data:image"):
                try:
                    import base64
                    comma = annotated_b64.find(",")
                    raw_data = base64.b64decode(annotated_b64[comma + 1:])
                    arr = np.frombuffer(raw_data, dtype=np.uint8)
                    annotated_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception:
                    pass

            if annotated_frame is None and frame is not None:
                annotated_frame = frame

            img_w = int(vp_w * 0.52)
            if annotated_frame is not None:
                vp[:, 0:img_w] = cv2.resize(annotated_frame, (img_w, vp_h))
                cv2.line(vp, (img_w, 0), (img_w, vp_h), (34, 50, 80), 2)

            # Right details pane
            rx = img_w + 12
            cv2.putText(vp, "DETECTED SEAL:", (rx, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

            serial = res.get("serial_number")
            if serial:
                cv2.putText(vp, str(serial), (rx, 65),
                            cv2.FONT_HERSHEY_DUPLEX, 0.88, (52, 211, 153), 2, cv2.LINE_AA)

                conf = res.get("confidence", 0.0)
                conf_pct = conf * 100.0 if conf <= 1.0 else conf
                cv2.putText(vp, f"Conf: {conf_pct:.1f}%", (rx, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

                ocr_time = res.get("ocr_time_ms", 0.0)
                cv2.putText(vp, f"Time: {ocr_time:.0f}ms", (rx, 122),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)

                # Reset Prompt
                cv2.putText(vp, "Press Btn 2 to Reset", (rx, vp_h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)
            else:
                cv2.putText(vp, "NO CODE", (rx, 65),
                            cv2.FONT_HERSHEY_DUPLEX, 0.82, (239, 68, 68), 2, cv2.LINE_AA)
                cv2.putText(vp, "Check focus / distance", (rx, 96),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 158, 11), 1, cv2.LINE_AA)
                cv2.putText(vp, "Press Button 2 to Reset", (rx, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (56, 189, 248), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. BOTTOM TOOLBAR: DUAL PHYSICAL BUTTONS
        # -------------------------------------------------------------
        bottom_bar = canvas[self.bottom_y:h, :]
        bottom_bar[:] = (9, 13, 22)
        cv2.line(canvas, (0, self.bottom_y), (w, self.bottom_y), (34, 50, 80), 2)

        # Button 1: [ BTN 1: CAPTURE ]
        cx1, cy1, cx2, cy2 = self.btn_capture_rect
        btn1_cx = (cx1 + cx2) // 2
        if cap_active:
            btn_cap_col = (255, 255, 255)
            btn_cap_text = (10, 150, 80)
            btn_cap_sub = (20, 20, 20)
            txt1 = ">>> BTN 1 CLICKED <<<"
            txt2 = "[ CAPTURING BURST ]"
        else:
            btn_cap_col = (16, 185, 129)
            btn_cap_text = (255, 255, 255)
            btn_cap_sub = (220, 255, 240)
            txt1 = "BTN 1: CAPTURE"
            txt2 = "[ GPIO 17 ]"

        self._draw_rounded_button(canvas, cx1, cy1, cx2, cy2, btn_cap_col, radius=8)
        self._draw_centered_text(canvas, txt1, btn1_cx, cy1 + 18, cv2.FONT_HERSHEY_DUPLEX, 0.48, btn_cap_text, 1)
        self._draw_centered_text(canvas, txt2, btn1_cx, cy1 + 36, cv2.FONT_HERSHEY_SIMPLEX, 0.34, btn_cap_sub, 1)

        # Button 2: [ BTN 2: RE-CAPTURE ]
        rx1, ry1, rx2, ry2 = self.btn_recapture_rect
        btn2_cx = (rx1 + rx2) // 2
        if recap_active:
            btn_recap_col = (255, 255, 255)
            btn_recap_text = (11, 120, 220)
            btn_recap_sub = (20, 20, 20)
            txt1 = ">>> BTN 2 CLICKED <<<"
            txt2 = "[ RESET VIEWFINDER ]"
        else:
            btn_recap_col = (11, 158, 245)
            btn_recap_text = (255, 255, 255)
            btn_recap_sub = (220, 240, 255)
            txt1 = "BTN 2: RE-CAPTURE"
            txt2 = "[ GPIO 27 ]"

        self._draw_rounded_button(canvas, rx1, ry1, rx2, ry2, btn_recap_col, radius=8)
        self._draw_centered_text(canvas, txt1, btn2_cx, ry1 + 18, cv2.FONT_HERSHEY_DUPLEX, 0.48, btn_recap_text, 1)
        self._draw_centered_text(canvas, txt2, btn2_cx, ry1 + 36, cv2.FONT_HERSHEY_SIMPLEX, 0.34, btn_recap_sub, 1)

        with self.lock:
            self.buffer[:] = canvas

        # Push to HDMI Framebuffer
        if self.hdmi_fb is not None and self.hdmi_fb.is_active:
            self.hdmi_fb.write_frame(canvas)

        return canvas

    def _draw_centered_text(self, img: np.ndarray, text: str, cx: int, cy: int,
                            font=cv2.FONT_HERSHEY_DUPLEX, scale: float = 0.5,
                            color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(img, text, (cx - tw // 2, cy + th // 2), font, scale, color, thickness, cv2.LINE_AA)

    def _draw_rounded_button(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                             color: Tuple[int, int, int], radius: int = 8):
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), (20, 20, 20), 1)

    def get_latest_jpeg(self) -> bytes:
        """Returns JPEG mirror of the current 3.5" HDMI screen."""
        with self.lock:
            ret, buf = cv2.imencode(".jpg", self.buffer, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes() if ret else b""

    def start_gui_loop(self, window_name: str = "3.5 inch HDMI Display Simulator", scale: float = 1.5):
        """Interactive desktop simulator window with mouse touch and keyboard triggers."""
        self.running = True
        disp_w = int(self.width * scale)
        disp_h = int(self.height * scale)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_w, disp_h)

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                touch_x = int(x / scale)
                touch_y = int(y / scale)
                self.handle_touch_point(touch_x, touch_y)

        cv2.setMouseCallback(window_name, mouse_cb)

        print("\n" + "=" * 65)
        print(f"  3.5\" HDMI DISPLAY SIMULATOR ({self.width}x{self.height})")
        print("  - Button 1 (Capture): Press [1] or [SPACE] or Click [BTN 1]")
        print("  - Button 2 (Re-capture): Press [2] or [R] or Click [BTN 2]")
        print("  - Physical Buttons on Pi: Btn 1=GPIO 17, Btn 2=GPIO 27")
        print("  - Exit: Press [Q] or [ESC]")
        print("=" * 65 + "\n")

        try:
            while self.running:
                canvas = self.render()
                preview = cv2.resize(canvas, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(30) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                elif key in [32, ord('1')]:
                    self.trigger_capture(source="SIMULATOR BTN 1")
                elif key in [ord('r'), ord('R'), ord('2')]:
                    self.trigger_recapture(source="SIMULATOR BTN 2")
        finally:
            self.running = False
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ui = StickCamHdmiUI(sim_mode=True)
    sample_path = os.path.join(os.path.dirname(__file__), "sample_seal.jpg")
    if os.path.exists(sample_path):
        sample = cv2.imread(sample_path)
        ui.update_viewfinder_frame(sample)
    ui.start_gui_loop()
