@echo off
title StickCam 3.5" HDMI Display Simulator
echo ===================================================================
echo   STICKCAM 3.5" HDMI DISPLAY SIMULATOR
echo ===================================================================
echo.
echo Controls:
echo   - Button 1 (Capture): Press [1] or [SPACE] or Click [BTN 1]
echo   - Button 2 (Re-capture): Press [2] or [R] or Click [BTN 2]
echo   - Exit: Press [Q] or [ESC]
echo.
python pi_hdmi_scanner.py --sim
pause
