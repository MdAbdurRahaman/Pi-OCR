# Pi-OCR: Raspberry Pi Offline Seal Scanner

An offline, embedded optical character recognition (OCR) scanner designed for reading printed seal numbers using a Raspberry Pi Zero 2 W with either a CSI camera module or a USB webcam.

---

## 🛠 Hardware Architecture

- **SBC**: Raspberry Pi Zero 2 W (Debian Bookworm / 64-bit)
- **Camera Options**:
  - 5 MP CSI Camera (e.g., OV5647 with wide-angle / fisheye lens and manual focus)
  - Standard UVC USB Webcam (e.g., Logitech C270)
- **Diffused Lighting**: Uniform white illumination to prevent specular glare on glossy seal surfaces.

---

## 📦 Software Stack

- **OS**: Raspberry Pi OS Lite (64-bit Bookworm)
- **Image Processing**: OpenCV (`python3-opencv`)
- **Camera Stack**: `picamera2` / `rpicam-apps` / `V4L2`
- **OCR Engine**: Tesseract OCR (`tesseract-ocr`, `tesseract-ocr-eng`)

---

## 🚀 Setup & Installation on the Pi

1. **Install Dependencies**:
   ```bash
   sudo apt update
   sudo apt install -y python3-picamera2 python3-opencv tesseract-ocr tesseract-ocr-eng v4l-utils
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/MdAbdurRahaman/Pi-OCR.git ~/seal-scanner
   cd ~/seal-scanner
   ```

3. **Verify Camera Detection**:
   - For CSI Camera:
     ```bash
     rpicam-hello --list-cameras
     ```
   - For USB Webcam:
     ```bash
     v4l2-ctl --list-devices
     ```

*(Note: If a background daemon such as `motion` is running, ensure it is disabled: `sudo systemctl disable --now motion`)*

---

## 🔍 How It Works

1. **Positioning**: Align the stationary seal in front of the lens so that the digits are sharp, horizontal, and well-lit.
2. **Capture**: 
   - `capture_seal.py`: Captures via `Picamera2` in RGB888 format (which maps directly to OpenCV BGR byte ordering without needing color conversions) or `rpicam-still`.
   - `capture_usb.py`: Captures via OpenCV V4L2 backend for USB cameras.
3. **Region of Interest (ROI) & Rotation**:
   - The image is cropped using normalized coordinates `[X, Y, WIDTH, HEIGHT]` (values between 0.0 and 1.0).
   - Optional 90°, 180°, or 270° clockwise rotation can be applied for vertical text.
4. **Preprocessing**:
   - Bounded downscaling (max width 1200 px) to optimize memory and processing speed on the Pi Zero 2 W.
   - Grayscale conversion + Otsu adaptive binarization with polarity correction (dark text on light background).
   - Border padding to improve edge-character recognition in Tesseract.
5. **OCR Recognition**:
   - Single-line PSM (`--psm 7`) with a digits-only whitelist (`0123456789`).
   - Confidence scoring and length filtering (e.g. `--digits 7`).

---

## 💻 Usage

### Direct on Raspberry Pi

```bash
cd ~/seal-scanner

# 1. Capture photo
python3 capture_usb.py
# Or with CSI camera:
python3 capture_seal.py test.jpg --delay 3

# 2. Run OCR (e.g., for a 7-digit seal)
python3 scan_seal.py test.jpg --digits 7

# Custom crop: 35% from left, 45% from top, 30% width, 10% height
python3 scan_seal.py test.jpg --roi 0.35 0.45 0.30 0.10 --digits 7
```

### Remote Controller from PC (`pi_client.py`)

Run commands on your PC to trigger captures and inspect the results locally:

```powershell
# Set credentials (or you will be prompted securely)
$env:PI_HOST = "pizero2.local"
$env:PI_USER = "stickcam"
$env:PI_PASS = "YourPassword"

# Capture and download to test.jpg
python pi_client.py capture --usb test.jpg

# Run OCR and download debug crops
python pi_client.py scan test.jpg --digits 7
```

---

## 📂 Project Structure

```
Pi-OCR/
├── capture_seal.py        # Picamera2 / rpicam-still capture script
├── scan_seal.py           # Core OCR and preprocessing engine
├── run_offline_scanner.py # All-in-one capture & OCR runner
├── pi_client.py           # PC client to trigger and sync with the Pi
├── .gitignore
└── README.md
```
