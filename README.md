# StickCam Padlock AI: Mobile Phone Scanner

An ultra-low latency, embedded optical character recognition (OCR) scanner designed for reading printed security seal and padlock numbers using a **smartphone camera** connected to PC-side RapidOCR ONNX neural inference.

---

## 🌟 Key Features

- **Phone Camera Ingestion (Zero-Install)**:
  - Simply open `http://<YOUR-PC-IP>:5000/phone` on your smartphone browser (Safari / Chrome).
  - Automatically activates the rear camera, enables torch/flash controls, and streams video frames over a low-latency binary WebSocket directly to the PC.
- **Third-Party IP Camera Support**:
  - Also works seamlessly with apps like *IP Webcam* (Android), *DroidCam*, *Iriun*, or RTSP network streams (`--stream http://...`).
- **Real-Time ONNX Inference**:
  - Employs **RapidOCR ONNX Runtime** asynchronously so video rendering never drops frames.
  - Recognizes padlocks, security seals, and serial numbers with high accuracy.
- **Temporal Tracking & Smoothing**:
  - Multi-frame temporal tracking eliminates bounding box flicker and jitter.
- **Interactive Native HUD & Web Dashboard**:
  - Native high-FPS OpenCV window with hotkeys (`[SPACE]` freeze, `[S]` save, `[C]` copy serial to clipboard, `[T]` toggle boxes, `[+]`/`[-]` digital zoom).
  - Web dashboard accessible at `http://localhost:5000` with live stream, QR code generator for phone pairing, and remote controls.
  - Mobile transmitter page also receives live detection feedback and provides haptic vibration when a padlock code is verified!

---

## 🛠 Software Requirements

- **Operating System**: Windows 10/11
- **Python**: Python 3.9 – 3.14
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 How to Run

### Method 1: Using Your Phone Browser (Recommended - No App Needed!)

1. **Start the System**:
   ```powershell
   python app.py
   ```
   *Or double-click `run_stickcam.bat`.*

2. **Connect Your Phone**:
   - The terminal and the Desktop Web Dashboard (`https://localhost:5000`) will display your phone link:
     ```
     https://192.168.68.128:5000/phone
     ```
   - Open that URL on your phone or scan the QR code displayed on the screen.
   - **Important SSL Note**: Because modern mobile browsers (Chrome / Safari) strictly require HTTPS for camera access, the server generates a local self-signed certificate. When Chrome displays *"Your connection is not private"*, simply tap **"Advanced"** ➔ **"Proceed to 192.168.68.128 (unsafe)"**.
   - Tap **"Allow"** when the browser asks for Camera permissions.
   - Tap **"📹 Start Stream"** to begin transmitting.
   - Use the **"💡 Torch"** button for dark environments.

   > **Note on Chrome Flag alternative (Plain HTTP)**: If you run with `python app.py --no-ssl`, mobile Chrome requires enabling `chrome://flags/#unsafely-treat-insecure-origin-as-secure` with `http://192.168.68.128:5000`.

3. **Scan Padlocks**:
   - Aim your phone at the padlock or security seal.
   - The PC and phone screen will highlight the detected characters with glowing bounding boxes.
   - Press **`[C]`** in the PC window or click **"📋 Copy Serial Number"** on the dashboard to copy the recognized code to your Windows clipboard!

---

### Method 2: Using an IP Camera App (DroidCam / IP Webcam / RTSP)

If you already use an Android/iOS streaming app:

```powershell
# Example with IP Webcam Android app
python app.py --stream http://192.168.68.150:8080/video

# Example with DroidCam MJPEG stream
python app.py --stream http://192.168.68.150:4747/video

# Example with RTSP camera feed
python app.py --stream rtsp://192.168.68.150:554/live/ch0
```

---

### Method 3: Using a USB-Connected Phone (DroidCam / Iriun Webcam)

If your phone is plugged in via USB and appears as a standard camera:

```powershell
# Using camera index 0 (or 1)
python app.py --camera 0
```

---

## ⌨️ Desktop Controls & Hotkeys

| Hotkey | Action |
| :--- | :--- |
| **`[SPACE]`** | Freeze / Resume live stream |
| **`[S]`** | Save timestamped snapshot with HUD overlay to `captures/` |
| **`[C]`** | Copy recognized padlock serial to Windows clipboard |
| **`[T]`** | Toggle bounding box overlays on / off |
| **`[R]`** | Reset / clear detection cache |
| **`[+]` / `[-]`** | Digital zoom In / Out (1.0x – 3.0x) |
| **`[Q]` / `[ESC]`** | Quit application |

---

## 🧪 Standalone Static Image OCR Testing

To test the character recognition model against a static photo:

```powershell
python scan_seal.py sample_seal.jpg
```

This runs RapidOCR ONNX, prints confidence scores and detected serial numbers, and outputs an annotated verification image (`sample_seal_annotated.jpg`).

---

## 📂 Project Structure

```
V6 BY MOBILE APP/
├── app.py               # Main launcher integrating FastAPI server & OpenCV HUD
├── stream_receiver.py   # Universal zero-buffer frame receiver (WebSocket, MJPEG, OpenCV)
├── padlock_ai.py        # Asynchronous RapidOCR ONNX character recognition engine
├── scan_seal.py         # Standalone CLI image OCR tool
├── run_stickcam.bat     # Windows double-click launcher
├── requirements.txt     # Python package requirements
├── sample_seal.jpg      # Sample test image of a security seal
├── templates/
│   ├── phone.html       # Mobile camera streaming web app (with torch & haptics)
│   └── index.html       # Desktop Web Dashboard with live HUD & phone QR code
├── captures/            # Saved snapshots and captures
└── README.md            # System documentation
```
