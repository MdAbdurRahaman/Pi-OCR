# Pi-OCR: Raspberry Pi Seal Scanner with 3.5" HDMI Display

An embedded, high-performance optical character recognition (OCR) scanner and live viewfinder designed for the **UVSS Stick Cam** using a **Raspberry Pi Zero 2 W** equipped with a **3.5" HDMI Display**, **Dual Physical Push Buttons**, and either a **CSI Camera** (IMX219 / OV5647) or **USB Webcam**.

---

## 🌟 Key Architecture & Upgrades: HDMI vs SPI

| Feature | Old 3.5" SPI Display (ILI9488) | New 3.5" HDMI Display |
| :--- | :--- | :--- |
| **Video Interface** | 8 manual jumper wires over SPI bus | Standard Mini-HDMI / HDMI digital port |
| **Refresh Rate** | 12 - 20 FPS (heavy CPU bottleneck) | **30 - 60 FPS (Hardware GPU VideoCore)** |
| **CPU Overhead** | ~35 - 50% CPU used by SPI bit-banging | **0% CPU for display output** |
| **Reliability** | Bus collisions, white-screen glitch risk | **Rock-solid stable digital signal** |
| **Pin Utilization** | Occupied pins 18, 19, 21, 23, 24, 25, 26 | **Frees up all SPI GPIO pins completely** |
| **Direct Framebuffer** | Required custom ILI9488 register writes | **Direct native `/dev/fb0` (Pi OS Lite)** |

---

## 🛠 Hardware Architecture & Pinout

- **SBC**: Raspberry Pi Zero 2 W (Debian Bookworm 64-bit)
- **Display**: 3.5" HDMI LCD (480x320 native or 800x480) connected via Mini-HDMI
- **Camera Options**:
  - Sony IMX219 / OV5647 CSI Camera (Picamera2 native BGR888 capture)
  - Standard UVC USB Webcam (OpenCV V4L2)
- **Dual Tactile Physical Push Buttons (Stick Handle)**:

| Physical Button | Action Triggered | Pi Signal Pin | Pi Ground Pin | Raspberry Pi GPIO |
| :---: | :--- | :---: | :---: | :--- |
| **Button 1** | **📸 CAPTURE & RUN OCR** | **Pin 11** | **Pin 9 (GND)** | `GPIO 17` (Internal Pull-Up) |
| **Button 2** | **🔄 RE-CAPTURE & RESET** | **Pin 13** | **Pin 14 (GND)** | `GPIO 27` (Internal Pull-Up) |

*(Both buttons connect directly between the GPIO pin and GND. Internal pull-ups are enabled automatically in software).*

---

## 🔍 How It Works

1. **Live HDMI Viewfinder & Focus Telemetry**:
   - The 3.5" HDMI screen continuously renders a 30+ FPS live camera feed.
   - A reactive **Live Focus Sharpness Meter** evaluates image sharpness in real-time:
     - 🟢 **SHARP** (Score > 320): Crystal clear, optimal for OCR.
     - 🟡 **FAIR** (Score 160-320): Acceptable focus.
     - 🔴 **BLURRY** (Score < 160): Warning to adjust distance/focus before capturing.
2. **Button 1 (Capture & OCR)**:
   - Pressing **Button 1 (GPIO 17)** triggers an instantaneous burst capture.
   - On-screen Button 1 flashes in glowing emerald green (`>>> BTN 1 CLICKED <<<`).
   - Evaluates peak sharpest frames and executes OCR.
3. **Dual-Mode OCR Engine**:
   - **Standalone Offline Mode**: Runs on-device OCR (CLAHE contrast equalization + Otsu adaptive binarization + Tesseract engine) completely offline—no PC or internet needed.
   - **PC Burst Offload Mode**: If a high-power PC server is present (`--pc http://<PC_IP>:5000`), offloads burst frames to RapidOCR ONNX neural engine on the PC.
   - **Auto-Fallback**: If the PC is unreachable, seamlessly executes standalone local OCR without interruption.
4. **Inspection Result Screen**:
   - Displays the detected padlock serial number (e.g. `C 581819`), confidence score, processing time, and the captured image with bounding box highlights.
5. **Button 2 (Re-capture & Reset)**:
   - Pressing **Button 2 (GPIO 27)** resets the result and immediately returns to the live camera viewfinder.

---

## 📂 Project Structure

```
Pi-OCR/
├── pi_hdmi_ui.py          # 3.5" HDMI Display UI engine, /dev/fb0 framebuffer driver & button handlers
├── pi_hdmi_scanner.py     # Main Pi service: live capture, focus telemetry, offline/PC OCR & web stream
├── HARDWARE_HDMI_SETUP.md # Complete hardware wiring diagrams, GPIO pinout & config.txt guide
├── test_hdmi_ui.py        # Automated test suite for HDMI UI, states, buttons & rendering
├── deploy_hdmi_pi.py      # Automated 1-command remote deployment to Raspberry Pi Zero 2W
├── run_hdmi_simulator.bat # 1-click Windows batch launcher for desktop simulator
├── pc_burst_server.py     # High-speed RapidOCR ONNX burst server for PC offloading
├── run_burst_server.bat   # Windows launcher for PC burst server
├── app.py                 # PC interactive HUD and live video receiver
├── stream_receiver.py     # Zero-buffer socket video ingestion thread
├── padlock_ai.py          # RapidOCR ONNX inference and HUD overlay
├── scan_seal.py           # Core standalone OCR preprocessing script
├── capture_seal.py        # Picamera2 / rpicam capture script
├── requirements.txt       # Dependencies for PC and Pi
└── README.md
```

---

## 🚀 Quick Start

### 1. Test Locally on PC (Simulator Mode)
You can test the exact 3.5" HDMI display interface locally on Windows/Linux:

```bash
python pi_hdmi_scanner.py --sim
```
*(Or double-click `run_hdmi_simulator.bat`)*

**Controls in Simulator:**
- **`[SPACE]`** or **`[1]`**: Trigger Button 1 (Capture & OCR)
- **`[R]`** or **`[2]`**: Trigger Button 2 (Re-capture & Reset)
- **`Mouse Click`**: Tap on-screen Button 1 or Button 2
- **`[Q]`** or **`[ESC]`**: Exit

Run the automated test suite:
```bash
python test_hdmi_ui.py
```

---

### 2. Deploy to Raspberry Pi Zero 2W
Deploy the scanner service over WiFi with one command:

```bash
python deploy_hdmi_pi.py --host 192.168.68.129
```
This automatically configures and starts `seal-hdmi.service` at boot!

---

### 3. Run Manually on Raspberry Pi
```bash
# Standalone Offline Mode (Autonomous on Stick Cam):
python3 pi_hdmi_scanner.py --port 8000

# Connected to PC AI Server:
python3 pi_hdmi_scanner.py --port 8000 --pc http://192.168.68.123:5000
```

---

### 4. Embedded Web UI (Dual Mirror)
Open your browser at **`http://<PI_IP>:8000`** (e.g. `http://pizero2.local:8000`):
- View a live 30 FPS mirror of the 3.5" HDMI screen.
- Trigger **Capture** and **Reset** from any smartphone, tablet, or PC on the network.
- Monitor live focus telemetry and recognized seal serial numbers.

---

## 📄 License
MIT License
