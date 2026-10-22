# Fine-tune one arm of the ECAPA-judge experiment, then evaluate it on the 2-, 3- and
# 4-speaker test sets. Outputs go to experiments/ecapa_judge/<arm>/. Resumable.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments\ecapa_judge\run.ps1 judge
#         powershell -ExecutionPolicy Bypass -File experiments\ecapa_judge\run.ps1 control
param([Parameter(Mandatory = $true)][ValidateSet("judge", "control")][string]$Arm)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$py = ".\.venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "TSE ECAPA-judge experiment: $Arm"
$exp = "experiments/ecapa_judge"
$run = "$exp/$Arm"

if (-not (Test-Path "$run/train_done.txt")) {
    $resume = @()
    if (Test-Path "$run/last.pt") { $resume = @("--resume") }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] fine-tuning ($Arm) $resume" -ForegroundColor Cyan
    & $py scripts/train.py --config "$exp/$Arm.yaml" --runs-dir $exp @resume
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
Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] $Arm done" -ForegroundColor Green
