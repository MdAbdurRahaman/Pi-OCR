import argparse
import csv
import io
import os
import re
import subprocess
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description="Read seal number from image using OCR")
parser.add_argument("image", help="Captured image filename")
parser.add_argument(
    "--roi",
    nargs=4,
    type=float,
    default=[0.25, 0.40, 0.50, 0.20],
    metavar=("X", "Y", "WIDTH", "HEIGHT"),
    help="Crop fractions from 0 to 1; default is a central strip [X, Y, WIDTH, HEIGHT]",
)
parser.add_argument(
    "--digits",
    type=int,
    default=0,
    help="Exact number of digits, if known",
)
parser.add_argument(
    "--rotate",
    type=int,
    choices=[0, 90, 180, 270],
    default=0,
    help="Clockwise rotation applied after cropping",
)
args = parser.parse_args()

frame = cv2.imread(args.image)
if frame is None:
    raise SystemExit("Cannot read image: " + args.image)

x, y, rw, rh = args.roi
if (min(x, y) < 0 or rw <= 0 or rh <= 0 or x + rw > 1 or y + rh > 1):
    raise SystemExit("Invalid ROI: crop must fit inside the image.")

height, width = frame.shape[:2]
x1, y1 = int(x * width), int(y * height)
x2, y2 = int((x + rw) * width), int((y + rh) * height)
crop = frame[y1:y2, x1:x2]
if crop.size == 0:
    raise SystemExit("Crop is empty.")

rotations = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
if args.rotate:
    crop = cv2.rotate(crop, rotations[args.rotate])

# Separate debug folder for each input image
debug = Path(args.image).with_suffix("")
debug = debug.parent / (debug.name + "_ocr")
debug.mkdir(exist_ok=True, parents=True)
cv2.imwrite(str(debug / "crop.jpg"), crop)

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# Keep processing bounded on the Pi Zero
if gray.shape[1] > 1200:
    scale = 1200 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Expect dark digits on a lighter background
if binary.mean() < 127:
    binary = cv2.bitwise_not(binary)

env = os.environ.copy()
env["OMP_THREAD_LIMIT"] = "1"
candidates = {}

for name, processed in [("gray", gray), ("binary", binary)]:
    processed = cv2.copyMakeBorder(
        processed, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255
    )
    image_path = debug / (name + ".png")
    cv2.imwrite(str(image_path), processed)

    command = [
        "tesseract",
        str(image_path),
        "stdout",
        "-l",
        "eng",
        "--oem",
        "1",
        "--psm",
        "7",
        "-c",
        "tessedit_char_whitelist=0123456789",
        "tsv",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(name + ": OCR timed out.")
        continue
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr)

    rows = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
    for row in rows:
        text = row.get("text", "").strip()
        # Do not remove characters or join unrelated text.
        if not re.fullmatch(r"[0-9]+", text):
            continue
        if args.digits:
            if len(text) != args.digits:
                continue
        elif not 5 <= len(text) <= 12:
            continue

        confidence = float(row["conf"])
        if confidence < 0:
            continue
        candidates[text] = max(confidence, candidates.get(text, -1))

print("Debug images:", debug)
if not candidates:
    print("No candidate found. Check focus, crop, and orientation.")
else:
    print("\nOCR candidates — visually verify before using:")
    for number, score in sorted(
        candidates.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {number}  (OCR score: {score:.1f})")
    print("\nThe OCR score is NOT a probability of correctness.")
