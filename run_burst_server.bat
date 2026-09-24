@echo off
title StickCam PC Burst OCR Processing Server
echo =======================================================
echo   Launching StickCam PC Burst OCR Server...
echo   Listening for Pi Zero multi-image burst captures
echo =======================================================
python pc_burst_server.py --port 5000
pause
