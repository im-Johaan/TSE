# Train the R=6 + frozen-ECAPA model from scratch on 2-4 speaker dynamic mixtures, then
# evaluate it on the 2-, 3- and 4-speaker test sets. Everything is written to
# experiments/ecapa_frozen/r6_ecapa/. Re-running resumes an interrupted run and skips
# finished evaluations.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments\ecapa_frozen\run.ps1

$ErrorActionPreference = "Continue"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$py = ".\.venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "TSE R=6 + frozen ECAPA (2-4 speakers, from scratch)"
$exp = "experiments/ecapa_frozen"
$run = "$exp/r6_ecapa"

if (-not (Test-Path "$run/train_done.txt")) {
    $resume = @()
    if (Test-Path "$run/last.pt") { $resume = @("--resume") }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] training R=6 + ECAPA $resume" -ForegroundColor Cyan
    & $py scripts/train.py --config "$exp/config.yaml" --runs-dir $exp @resume
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
Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] R=6 + ECAPA training and evaluation done" -ForegroundColor Green
