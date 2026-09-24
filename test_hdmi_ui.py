#!/usr/bin/env python3
"""
test_hdmi_ui.py - Automated test suite for 3.5" HDMI Display UI,
Focus Telemetry, State Machine, and Dual Physical Button Indicators.
"""
import os
import sys
import time
import cv2
import numpy as np

from pi_hdmi_ui import StickCamHdmiUI


def run_tests():
    print("=" * 65)
    print("  RUNNING 3.5\" HDMI DISPLAY SUITE TESTS")
    print("=" * 65)

    capture_events = []
    recapture_events = []

    def mock_capture():
        capture_events.append(time.time())
        print("  -> on_capture callback triggered successfully!")

    def mock_recapture():
        recapture_events.append(time.time())
        print("  -> on_recapture callback triggered successfully!")

    # 1. Initialize UI
    print("\n[Test 1] Initializing StickCamHdmiUI in simulator mode...")
    ui = StickCamHdmiUI(
        on_capture_cb=mock_capture,
        on_recapture_cb=mock_recapture,
        sim_mode=True,
        width=480,
        height=320
    )
    assert ui.width == 480 and ui.height == 320, "Incorrect screen dimensions"
    assert ui.state == StickCamHdmiUI.STATE_VIEWFINDER, "Default state should be VIEWFINDER"
    print("  [PASS] Initialized 480x320 HDMI UI instance.")

    # 2. Viewfinder Rendering & Focus Telemetry
    print("\n[Test 2] Testing Viewfinder rendering with focus telemetry...")
    sample_path = os.path.join(os.path.dirname(__file__), "sample_seal.jpg")
    if os.path.exists(sample_path):
        sample = cv2.imread(sample_path)
        ui.update_viewfinder_frame(sample)
    else:
        # Create synthetic test pattern with text
        sample = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(sample, "SEAL C581819", (150, 180), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        ui.update_viewfinder_frame(sample)

    frame1 = ui.render()
    assert frame1.shape == (320, 480, 3), f"Rendered frame shape mismatch: {frame1.shape}"
    assert ui.sharpness_score >= 0.0, "Sharpness score should be >= 0"
    out_vf = os.path.join(os.path.dirname(__file__), "test_hdmi_viewfinder.jpg")
    cv2.imwrite(out_vf, frame1)
    print(f"  [PASS] Viewfinder rendered with Focus: {ui.focus_quality} (score: {ui.sharpness_score:.1f}) -> saved to {out_vf}")

    # 3. Button 1 (Capture) Click & High-Visibility Feedback
    print("\n[Test 3] Testing Button 1 (Capture) click trigger...")
    ui.trigger_capture(source="PHYSICAL BTN 1 (GPIO 17)")
    time.sleep(0.05)
    assert len(capture_events) == 1, "Capture callback was not called"
    assert ui.state == StickCamHdmiUI.STATE_CAPTURING, f"State should be CAPTURING, got {ui.state}"
    assert ui.btn_capture_active is True, "Button 1 highlight should be active"
    assert ui.last_clicked_button == 1, "last_clicked_button should be 1"

    frame2 = ui.render()
    assert frame2.shape == (320, 480, 3)
    out_btn1 = os.path.join(os.path.dirname(__file__), "test_hdmi_btn1_clicked.jpg")
    cv2.imwrite(out_btn1, frame2)
    print(f"  [PASS] Button 1 active state rendered -> saved to {out_btn1}")

    # 4. Inspection Result Display
    print("\n[Test 4] Testing OCR result display rendering...")
    mock_result = {
        "status": "ok",
        "serial_number": "C581819",
        "confidence": 0.982,
        "ocr_time_ms": 135.6,
        "annotated_image": None,
    }
    ui.set_result(mock_result)
    assert ui.state == StickCamHdmiUI.STATE_RESULT, f"State should be RESULT, got {ui.state}"
    assert ui.result_data["serial_number"] == "C581819"

    frame3 = ui.render()
    assert frame3.shape == (320, 480, 3)
    out_res = os.path.join(os.path.dirname(__file__), "test_hdmi_result.jpg")
    cv2.imwrite(out_res, frame3)
    print(f"  [PASS] Inspection result 'C581819' rendered -> saved to {out_res}")

    # 5. Button 2 (Re-capture / Reset) Click
    print("\n[Test 5] Testing Button 2 (Re-capture) click trigger...")
    ui.trigger_recapture(source="PHYSICAL BTN 2 (GPIO 27)")
    time.sleep(0.05)
    assert len(recapture_events) == 1, "Recapture callback was not called"
    assert ui.state == StickCamHdmiUI.STATE_VIEWFINDER, f"State should reset to VIEWFINDER, got {ui.state}"
    assert ui.result_data == {}, "Result data should be cleared on re-capture"
    assert ui.btn_recapture_active is True, "Button 2 highlight should be active"
    assert ui.last_clicked_button == 2, "last_clicked_button should be 2"

    frame4 = ui.render()
    assert frame4.shape == (320, 480, 3)
    out_btn2 = os.path.join(os.path.dirname(__file__), "test_hdmi_btn2_clicked.jpg")
    cv2.imwrite(out_btn2, frame4)
    print(f"  [PASS] Button 2 active state and reset viewfinder rendered -> saved to {out_btn2}")

    # 6. Touch Hit-Testing on Screen
    print("\n[Test 6] Testing on-screen touch button hit-testing...")
    # Touch inside Button 1 hitbox
    c_x1, c_y1, c_x2, c_y2 = ui.btn_capture_rect
    hit_btn1 = ui.handle_touch_point((c_x1 + c_x2) // 2, (c_y1 + c_y2) // 2)
    assert hit_btn1 is True, "Touch inside Button 1 rect should return True"
    assert len(capture_events) == 2, "Touch on Button 1 should trigger capture event"

    # Touch inside Button 2 hitbox
    r_x1, r_y1, r_x2, r_y2 = ui.btn_recapture_rect
    hit_btn2 = ui.handle_touch_point((r_x1 + r_x2) // 2, (r_y1 + r_y2) // 2)
    assert hit_btn2 is True, "Touch inside Button 2 rect should return True"
    assert len(recapture_events) == 2, "Touch on Button 2 should trigger recapture event"

    # Touch outside buttons (e.g. top header)
    hit_miss = ui.handle_touch_point(100, 15)
    assert hit_miss is False, "Touch outside buttons should return False"
    print("  [PASS] Touch coordinate hit-testing verified for Button 1 & Button 2.")

    # 7. JPEG Screen Mirror Export
    print("\n[Test 7] Testing JPEG screen mirror generation...")
    jpeg = ui.get_latest_jpeg()
    assert len(jpeg) > 1000, f"JPEG stream too small ({len(jpeg)} bytes)"
    print(f"  [PASS] Screen mirror JPEG generated ({len(jpeg)} bytes).")

    print("\n" + "=" * 65)
    print("  ALL 7 TESTS PASSED SUCCESSFULLY!")
    print("  3.5\" HDMI Display UI Engine is verified and ready!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
