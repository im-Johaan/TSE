# Train + evaluate the proposed model, then the TD-SpeakerBeam baseline (sequentially, one GPU).
# Re-running this script resumes an interrupted run and skips runs that are already evaluated.
#
# Usage (opens nothing by itself; run it in any PowerShell window):
#   powershell -ExecutionPolicy Bypass -File scripts\run_main.ps1

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "TSE training (proposed -> baseline)"

$runs = @(
    @{ cfg = "configs/proposed.yaml"; name = "proposed" },
    @{ cfg = "configs/baseline_tdspeakerbeam.yaml"; name = "baseline_tdspeakerbeam" }
)
foreach ($run in $runs) {
    $dir = "runs/$($run.name)"
    if (Test-Path "$dir/test_metrics.json") {
        Write-Host ">>> $($run.name): already trained and evaluated, skipping" -ForegroundColor Yellow
        continue
    }
    $resume = @()
    if (Test-Path "$dir/last.pt") { $resume = @("--resume") }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] training $($run.name) $resume" -ForegroundColor Cyan
    & $py scripts/train.py --config $run.cfg @resume
    if ($LASTEXITCODE -ne 0) { Write-Host ">>> training $($run.name) failed (exit $LASTEXITCODE)" -ForegroundColor Red; continue }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] evaluating $($run.name) on the full test set" -ForegroundColor Cyan
    & $py scripts/evaluate.py --ckpt "$dir/best.pt"
}
Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] all done" -ForegroundColor Green
