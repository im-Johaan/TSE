# Start the real-time TSE server (scripts/realtime_server.py) and open it in the browser.
# Close this window or press Ctrl+C to stop it. Models come from models.yaml.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_realtime.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)
$Host.UI.RawUI.WindowTitle = "TSE real-time server (http://127.0.0.1:7870)"
& ".\.venv\Scripts\python.exe" scripts/realtime_server.py @args
