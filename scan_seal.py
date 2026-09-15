"""
scan_seal.py - Standalone Padlock & Seal Character Recognition Scanner.
Can be run on static images or captures directly from the command line.
"""
import sys
import os
import re
import argparse
import time
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def scan_image(image_path: str, conf_threshold: float = 0.40, save_annotated: bool = True):
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[-] Error: Failed to load image: {image_path}")
        return None

    h, w = img.shape[:2]
    print(f"[*] Processing image: {image_path} ({w}x{h})")

    engine = RapidOCR()
    t0 = time.time()
    results, elapse = engine(img)
    t1 = time.time()
    elapsed_ms = (t1 - t0) * 1000.0

    print(f"[*] OCR Completed in {elapsed_ms:.1f}ms")

    detections = []
    best_id = None
    best_score = 0.0

    if results:
        for box, text, score_str in results:
            score = float(score_str)
            clean_text = text.strip()
            if score < conf_threshold or len(clean_text) < 2:
                continue

            is_serial = bool(re.search(r'[A-Z0-9]{4,10}', clean_text.replace(" ", "")))
            pts = np.array(box, dtype=np.int32)
            detections.append({
                "text": clean_text,
                "score": score,
                "pts": pts,
                "is_serial": is_serial
            })

            if is_serial and score > best_score:
                best_id = clean_text
                best_score = score

            border_color = (0, 255, 100) if is_serial else (255, 190, 20)
            cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=2)
            lbl = f"{clean_text} ({int(score*100)}%)"
            min_y = int(np.min(pts[:, 1]))
            min_x = int(np.min(pts[:, 0]))
            cv2.putText(img, lbl, (min_x, max(20, min_y - 6)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    print("\n" + "=" * 50)
    print(f"  FOUND {len(detections)} DETECTIONS:")
    print("=" * 50)
    for d in detections:
        marker = "[SERIAL]" if d["is_serial"] else "  [TEXT]"
        print(f"  {marker} | '{d['text']}' | Confidence: {d['score']*100:.1f}%")
    print("=" * 50)
    if best_id:
        print(f"  --> BEST PADLOCK ID: {best_id} ({best_score*100:.1f}%)")
    else:
        print("  --> No valid padlock serial number detected.")
    print("=" * 50 + "\n")

    if save_annotated:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_annotated{ext}"
        cv2.imwrite(out_path, img)
        print(f"[+] Saved annotated visualization to: {out_path}")

    return best_id


def main():
    parser = argparse.ArgumentParser(description="Scan seal/padlock characters from image")
    parser.add_argument("image", type=str, help="Path to input image file")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated output image")
    args = parser.parse_args()

    scan_image(args.image, conf_threshold=args.conf, save_annotated=not args.no_save)


if __name__ == "__main__":
    main()
