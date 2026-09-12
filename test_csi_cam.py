import argparse
import time
from pathlib import Path
import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

def capture_csi(output_path="csi_photo.jpg", width=1920, height=1080, delay=2.0):
    if Picamera2 is None:
        raise RuntimeError("Picamera2 is not installed. Install via: sudo apt install python3-picamera2")

    print(f"[CSI] Initializing Sony IMX219 camera ({width}x{height})...")
    picam2 = Picamera2(0)
    config = picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print(f"[CSI] Settling exposure for {delay}s...")
    time.sleep(delay)
    frame = picam2.capture_array()
    picam2.stop()

    cv2.imwrite(output_path, frame)
    print(f"[CSI] Successfully saved photo to {output_path} (Resolution: {frame.shape[1]}x{frame.shape[0]})")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CSI camera (Sony IMX219) on Raspberry Pi")
    parser.add_argument("output", default="csi_test.jpg", nargs="?", help="Output image filename")
    parser.add_argument("--width", type=int, default=1920, help="Image width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Image height (default: 1080)")
    parser.add_argument("--delay", type=float, default=2.0, help="Exposure settling delay (default: 2.0)")
    args = parser.parse_args()

    capture_csi(args.output, args.width, args.height, args.delay)
