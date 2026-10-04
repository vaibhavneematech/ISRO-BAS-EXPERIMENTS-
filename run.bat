@echo off
title VYOM - Mission Control Dashboard
cd /d "%~dp0"
python scripts\run_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
