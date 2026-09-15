@echo off
title StickCam Mobile Phone AI Padlock Scanner
echo ====================================================================
echo   StickCam Padlock AI Recognition System (Mobile Phone Powered)
echo ====================================================================
echo.

:: Automatically clean up any orphaned process occupying port 5000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do (
    echo [Info] Freeing occupied port 5000 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

echo Starting StickCam Padlock AI...
echo.
python app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
