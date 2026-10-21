# Fine-tune the saved 2-speaker model on dynamically mixed 2-4 speaker mixtures (25 epochs),
# then evaluate the best checkpoint on the 2-, 3- and 4-speaker test sets.
# Re-running resumes an interrupted fine-tune and skips evaluations that already exist.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_finetune.ps1

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "TSE fine-tuning (2-4 speakers)"
$run = "runs/finetune_multispk"

if (-not (Test-Path "$run/best.pt") -or (Test-Path "$run/last.pt")) {
    $resume = @()
    if (Test-Path "$run/last.pt") { $resume = @("--resume") }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] fine-tuning $resume" -ForegroundColor Cyan
    & $py scripts/train.py --config configs/finetune_multispk.yaml @resume
    if ($LASTEXITCODE -ne 0) { Write-Host ">>> fine-tuning failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
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
Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] fine-tuning and evaluation done" -ForegroundColor Green
