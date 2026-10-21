# Baseline pipeline (runs alongside the proposed-model fine-tune):
#   1. train TD-SpeakerBeam on 2-speaker Libri2Mix for 80 epochs (same budget as the proposed model)
#   2. evaluate it on the 2-, 3- and 4-speaker test sets
#   3. fine-tune it on 2-4 speaker dynamic mixtures (configs/finetune_multispk_baseline.yaml)
#   4. evaluate the fine-tuned baseline on the 2-, 3- and 4-speaker test sets
# Re-running resumes interrupted training and skips finished steps.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_baseline_overnight.ps1

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "TSE baseline (TD-SpeakerBeam): train -> eval -> 2-4 spk fine-tune -> eval"

function Train-Run($cfg, $name, $extra) {
    $dir = "runs/$name"
    if ((Test-Path "$dir/train_done.txt")) { Write-Host ">>> $name already trained" -ForegroundColor Yellow; return $true }
    $resume = @()
    if (Test-Path "$dir/last.pt") { $resume = @("--resume") }
    Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] training $name $resume" -ForegroundColor Cyan
    & $py scripts/train.py --config $cfg @resume --set @extra
    if ($LASTEXITCODE -ne 0) { Write-Host ">>> training $name failed (exit $LASTEXITCODE)" -ForegroundColor Red; return $false }
    Set-Content -Path "$dir/train_done.txt" -Value (Get-Date -Format s)
    return $true
}

function Eval-Run($name) {
    $dir = "runs/$name"
    $evals = @(
        @{ out = "test_metrics.json"; args = @() },
        @{ out = "Libri3Mix_test_metrics.json"; args = @("--eval-root", "generated/Libri3Mix") },
        @{ out = "Libri4Mix_test_metrics.json"; args = @("--eval-root", "generated/Libri4Mix") }
    )
    foreach ($e in $evals) {
        if (Test-Path "$dir/$($e.out)") { continue }
        Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] evaluating $name -> $($e.out)" -ForegroundColor Cyan
        & $py scripts/evaluate.py --ckpt "$dir/best.pt" @($e.args)
    }
}

if (Train-Run "configs/baseline_tdspeakerbeam.yaml" "baseline_tdspeakerbeam" @("train.max_epochs=80")) {
    Eval-Run "baseline_tdspeakerbeam"
    if (Train-Run "configs/finetune_multispk_baseline.yaml" "finetune_multispk_baseline" @("train.max_epochs=25")) {
        Eval-Run "finetune_multispk_baseline"
    }
}
Write-Host "`n>>> [$(Get-Date -Format 'HH:mm:ss')] baseline pipeline finished" -ForegroundColor Green
