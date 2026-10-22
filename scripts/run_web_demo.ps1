# Start the interactive TSE web demo (scripts/web_demo.py) and open it in the browser.
# Close this window or press Ctrl+C to stop the server.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_web_demo.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)
$Host.UI.RawUI.WindowTitle = "TSE web demo (http://127.0.0.1:7860)"
& ".\.venv\Scripts\python.exe" scripts/web_demo.py @args
