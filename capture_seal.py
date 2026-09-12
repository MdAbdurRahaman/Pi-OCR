import argparse
import sys
import time
import subprocess
from pathlib import Path

def capture_with_picam2(output_path: str, delay: float = 3.0, width: int = 1920, height: int = 1080):
    try:
        from picamera2 import Picamera2
        import cv2
    except ImportError as e:
        print(f"Picamera2 or OpenCV import failed: {e}. Falling back to rpicam-still...")
        return capture_with_rpicam_still(output_path, delay, width, height)

    picam2 = Picamera2()
    # RGB888 format produces BGR byte order for OpenCV (no channel swapping needed)
    config = picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print(f"Camera started. Waiting {delay}s for auto-exposure...")
    time.sleep(delay)
    frame = picam2.capture_array()
    picam2.stop()

    cv2.imwrite(output_path, frame)
    print(f"Captured photo saved to {output_path}")

def capture_with_rpicam_still(output_path: str, delay: float = 3.0, width: int = 1920, height: int = 1080):
    timeout_ms = int(delay * 1000)
    cmd = [
        "rpicam-still",
        "--nopreview",
        "--timeout", str(timeout_ms),
        "--width", str(width),
        "--height", str(height),
        "--output", output_path
    ]
    print("Executing:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"rpicam-still failed with code {res.returncode}")
    print(f"Captured photo saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture image on Raspberry Pi")
    parser.add_argument("output", default="test.jpg", nargs="?", help="Output image file path (default: test.jpg)")
    parser.add_argument("--delay", type=float, default=3.0, help="Wait time before capture in seconds (default: 3.0)")
    parser.add_argument("--width", type=int, default=1920, help="Capture width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Capture height (default: 1080)")
    parser.add_argument("--mode", choices=["picam2", "rpicam-still"], default="picam2", help="Capture backend")
    args = parser.parse_args()

    if args.mode == "picam2":
        capture_with_picam2(args.output, args.delay, args.width, args.height)
    else:
        capture_with_rpicam_still(args.output, args.delay, args.width, args.height)
