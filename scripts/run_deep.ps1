# Train the deeper (R=6) U-eConv-TSE from scratch on 2-4 speaker dynamic mixtures, then
# evaluate it on the 2-, 3- and 4-speaker test sets. Waits until the R=3 fine-tune
# (runs/finetune_multispk) has finished training so the two runs don't share the GPU.
# Re-running resumes an interrupted run and skips finished evaluations.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_deep.ps1

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "TSE deep model R=6 (2-4 speakers, from scratch)"
$run = "runs/multispk_deep_r6"

$ftlog = "runs/finetune_multispk/train.log"
if (-not (Test-Path "$run/train_done.txt")) {
    while ((Test-Path $ftlog) -and -not (Select-String -Path $ftlog -Pattern "Done\. Best" -Quiet)) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] waiting for the R=3 fine-tune to finish training..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 60
    }
    $resume = @()
    if (Test-Path "$run/last.pt") { $resume = @("--resume") }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] training R=6 model $resume" -ForegroundColor Cyan
    & $py scripts/train.py --config configs/multispk_deep.yaml @resume
    if ($LASTEXITCODE -ne 0) { Write-Host ">>> training failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
    Set-Content -Path "$run/train_done.txt" -Value (Get-Date -Format s)
}

$evals = @(
    @{ out = "test_metrics.json"; args = @() },
    @{ out = "Libri3Mix_test_metrics.json"; args = @("--eval-root", "generated/Libri3Mix") },
    @{ out = "Libri4Mix_test_metrics.json"; args = @("--eval-root", "generated/Libri4Mix") }
)
foreach ($e in $evals) {
    if (Test-Path "$run/$($e.out)") { continue }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] evaluating -> $($e.out)" -ForegroundColor Cyan
    & $py scripts/evaluate.py --ckpt "$run/best.pt" @($e.args)
}
Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] R=6 training and evaluation done" -ForegroundColor Green
