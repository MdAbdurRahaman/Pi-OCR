# Hardware & Wiring Guide: 3.5" HDMI Display + Dual Physical Buttons

Complete hardware setup, wiring diagrams, and Raspberry Pi OS configuration for the **UVSS Stick Cam with 3.5" HDMI Display** and **Dual Physical Push Buttons**.

---

## 1. Why 3.5" HDMI Display is Superior to SPI TFT

| Feature | Old SPI Display (ILI9488) | New 3.5" HDMI Display |
| :--- | :--- | :--- |
| **Interface** | 8 SPI jumper wires + manual bridging | Standard HDMI cable / adapter |
| **Refresh Rate** | 12 - 20 FPS (high latency) | **30 - 60 FPS (Zero Latency GPU)** |
| **CPU Usage on Pi Zero 2W** | 35% - 50% CPU used by SPI bit-banging | **0% CPU (Hardware VideoCore GPU)** |
| **Bus Collisions / Glitches** | Prone to white-screen & clock noise | **Rock-solid stable digital signal** |
| **GPIO Pin Utilization** | Consumes 8 GPIO pins (MOSI, SCLK, CS, etc.) | **Leaves all SPI GPIO pins 100% free** |
| **OS Compatibility** | Requires custom register initialization | **Standard Linux `/dev/fb0` or DRM** |

---

## 2. Wiring Summary

### A. 3.5" HDMI Display Connections
1. **Video Signal**: Connect the Raspberry Pi Zero 2W's **Mini-HDMI port** to the 3.5" HDMI Display's HDMI input using a Mini-HDMI to HDMI cable or adapter board.
2. **Display Power**: Power the display via its Micro-USB power port (5V 1A), or connect to Pi 5V (Pin 2 or 4) and GND (Pin 6).

---

### B. Dual Physical Push Buttons (Stick Handle)

Connect two tactile push buttons on the stick handle to the Raspberry Pi 40-pin GPIO header:

| Physical Button | Function | Pi Signal Pin | Pi Ground Pin | Raspberry Pi GPIO |
| :---: | :--- | :---: | :---: | :--- |
| **Button 1** | **📸 CAPTURE & RUN OCR** | **Pin 11** | **Pin 9 (GND)** | `GPIO 17` (Internal Pull-Up) |
| **Button 2** | **🔄 RE-CAPTURE & RESET** | **Pin 13** | **Pin 14 (GND)** | `GPIO 27` (Internal Pull-Up) |

> [!TIP]
> Both buttons use internal pull-up resistors configured in software. No external resistors or breadboards are needed—just wire each button directly between its GPIO pin and GND.

---

### 40-Pin Raspberry Pi Header Pinout Reference

```
 3.3V DC (Pin 1)  [ .  . ] (Pin 2)  5V Power
   GPIO 2 (SDA)   [ .  . ] (Pin 4)  5V Power
   GPIO 3 (SCL)   [ .  . ] (Pin 6)  GND (Display Power GND)
   GPIO 4         [ .  . ] (Pin 8)  GPIO 14 (TXD)
      GND (Pin 9) [ .  . ] (Pin 10) GPIO 15 (RXD)
  GPIO 17 (Pin 11)[ .  . ] (Pin 12) GPIO 18
  GPIO 27 (Pin 13)[ .  . ] (Pin 14) GND (Btn 2 GND)
  GPIO 22 (Pin 15)[ .  . ] (Pin 16) GPIO 23
```
- **Button 1**: Pin 11 (`GPIO 17`) & Pin 9 (`GND`)
- **Button 2**: Pin 13 (`GPIO 27`) & Pin 14 (`GND`)

---

## 3. Raspberry Pi HDMI Display Configuration

### A. Raspberry Pi OS Legacy / Bullseye (`config.txt`)
Edit `/boot/config.txt` to set native 480x320 timings for the 3.5" HDMI LCD:

```ini
# Force HDMI output even without hotplug detection
hdmi_force_hotplug=1

# Custom CVT Timings for 3.5" HDMI Display (480x320 @ 60Hz)
hdmi_group=2
hdmi_mode=87
hdmi_cvt 480 320 60 6 0 0 0
hdmi_drive=2

# (Optional) For screens with 800x480 resolution:
# hdmi_cvt 800 480 60 6 0 0 0

# Orientation Rotation: 0 = Normal, 1 = 90° CW, 2 = 180°, 3 = 270° (90° CCW)
display_rotate=0
```

### B. Raspberry Pi OS Bookworm (64-bit KMS / DRM)
On modern Bookworm systems using the `vc4-kms-v3d` driver, custom resolution is configured in `/boot/firmware/cmdline.txt`. Append to the end of the single line:

```
video=HDMI-A-1:480x320M@60
```
*(Or `video=HDMI-A-1:800x480M@60` if your 3.5" display uses an 800x480 controller chip).*

---

## 4. Camera Setup

1. **CSI Camera (Recommended: Sony IMX219 or OV5647)**:
   - Insert the camera ribbon cable into the CAM port of the Pi Zero 2W (blue tape facing outwards towards the edge).
   - Test detection:
     ```bash
     rpicam-hello --list-cameras
     ```

2. **USB Webcam**:
   - Plug into the micro-USB OTG port using a USB OTG adapter.
   - Test detection:
     ```bash
     v4l2-ctl --list-devices
     ```

---

## 5. Software Execution & Autostart

### Run Manually on Raspberry Pi:
```bash
python3 pi_hdmi_scanner.py
```

### Test on PC (Simulator Mode):
```bash
python pi_hdmi_scanner.py --sim
```
Or run `run_hdmi_simulator.bat`.

### Automatic Startup at Boot (systemd service):
The scanner can run completely headless on Raspberry Pi OS Lite, outputting directly to the HDMI screen `/dev/fb0`.

1. Create `/etc/systemd/system/seal-hdmi.service`:
   ```ini
   [Unit]
   Description=StickCam 3.5" HDMI Display Scanner & OCR Service
   After=network.target

   [Service]
   Type=simple
   User=stickcam
   WorkingDirectory=/home/stickcam/seal-scanner
   ExecStart=/usr/bin/python3 -u /home/stickcam/seal-scanner/pi_hdmi_scanner.py --port 8000
   Restart=always
   RestartSec=3

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable seal-hdmi.service
   sudo systemctl start seal-hdmi.service
   ```

---

## 6. How to Operate the Scanner on the Stick

1. **Power On**: Pi boots in ~15 seconds and automatically displays the live viewfinder on the 3.5" HDMI screen.
2. **Align Seal**: Position the camera lens ~5-15 cm from the padlock seal. Keep the seal digits centered inside the viewfinder target reticle.
3. **Check Focus Meter**: Observe the live focus meter on the top right of the 3.5" HDMI screen:
   - 🟢 **SHARP**: Optimal focus! High OCR accuracy.
   - 🟡 **FAIR**: Readable, but minor adjustment may help.
   - 🔴 **BLURRY**: Adjust distance before capturing.
4. **Capture (Button 1)**: Press the top tactile button (**Button 1 / GPIO 17**). The on-screen button lights up green, captures a peak-focus burst, and performs OCR.
5. **View Result**: The detected serial number (e.g. `C 581819`) and confidence percentage are displayed in large high-contrast text alongside the captured image.
6. **Reset (Button 2)**: Press the bottom tactile button (**Button 2 / GPIO 27**). The on-screen button lights up blue and immediately returns to the live viewfinder for the next inspection.
