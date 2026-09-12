import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Full workflow: capture seal photo -> OCR number")
    parser.add_argument("--output", help="Output filename (default: seal_YYYYMMDD_HHMMSS.jpg)")
    parser.add_argument("--digits", type=int, default=0, help="Expected digit count (e.g. 7 or 5)")
    parser.add_argument("--roi", nargs=4, type=float, default=[0.25, 0.40, 0.50, 0.20],
                        metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Crop fractions (default: 0.25 0.40 0.50 0.20)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotation angle")
    parser.add_argument("--delay", type=float, default=3.0, help="Settling delay in seconds (default: 3.0)")
    args = parser.parse_args()

    filename = args.output
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"seal_{timestamp}.jpg"

    print(f"\n[Step 1/2] Capturing image to {filename}...")
    capture_cmd = [
        sys.executable, "capture_seal.py", filename,
        "--delay", str(args.delay)
    ]
    res = subprocess.run(capture_cmd)
    if res.returncode != 0:
        print("[Error] Image capture failed.")
        sys.exit(res.returncode)

    print(f"\n[Step 2/2] Running OCR analysis on {filename}...")
    scan_cmd = [
        sys.executable, "scan_seal.py", filename,
        "--roi", *[str(v) for v in args.roi],
        "--rotate", str(args.rotate),
    ]
    if args.digits > 0:
        scan_cmd.extend(["--digits", str(args.digits)])

    res_ocr = subprocess.run(scan_cmd)
    sys.exit(res_ocr.returncode)

if __name__ == "__main__":
    main()
