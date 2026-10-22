@echo off
rem Double-click to set up (first time) and start the TSE demo. See setup_and_run.ps1.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1" %*
echo.
pause
