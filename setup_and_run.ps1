# One-step setup and launch of the TSE demo on Windows (e.g. a laptop with an NVIDIA GPU).
#
#   1. update the code (git pull)
#   2. find Python 3.10-3.13
#   3. create the virtual environment (.venv)
#   4. install PyTorch (CUDA build if an NVIDIA GPU is present, else CPU)
#   5. install the demo packages (requirements-demo.txt)
#   6. self-test: GPU, models, extraction quality, live speed
#   7. start the demo server and open http://127.0.0.1:8000
#
# Every step is skipped when it is already done, so run it again any time (it is also how to
# start the demo later). Double-click START_DEMO.bat, or run:
#   powershell -ExecutionPolicy Bypass -File setup_and_run.ps1
# Options: -NoPull  -SkipTest  -Reinstall (fresh .venv)  -Cuda cu128|cu126|cpu
#          -Port 8000  -Passcode CODE  -NoBrowser
param(
    [switch]$NoPull,
    [switch]$SkipTest,
    [switch]$Reinstall,
    [ValidateSet("auto", "cu128", "cu126", "cpu")][string]$Cuda = "auto",
    [int]$Port = 8000,
    [string]$Passcode = "",
    [switch]$NoBrowser
)
$ErrorActionPreference = "Continue"   # native tools write to stderr; exit codes are checked instead
Set-Location $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "TSE demo setup"
$total = 7
$t0 = Get-Date

function Step([int]$n, [string]$msg) { Write-Host "`n[$n/$total] $msg" -ForegroundColor Cyan }
function Ok([string]$msg) { Write-Host "      $msg" -ForegroundColor Green }
function Info([string]$msg) { Write-Host "      $msg" -ForegroundColor Gray }
function Fail([string]$msg) {
    Write-Host "`n  ERROR: $msg" -ForegroundColor Red
    Write-Host "  Fix the problem above and run this script again (finished steps are skipped)." -ForegroundColor Yellow
    exit 1
}
function Run([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { Fail "'$exe $($argv -join ' ')' failed (exit $LASTEXITCODE)" }
}

Write-Host "Target Speaker Extraction - demo setup" -ForegroundColor White
Write-Host "Folder: $PSScriptRoot" -ForegroundColor DarkGray

# PyTorch contains deep paths; without Windows long-path support the total must stay < 260
$longPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -ErrorAction SilentlyContinue).LongPathsEnabled
if ($longPaths -ne 1 -and $PSScriptRoot.Length -gt 70) {
    Fail ("this folder path is $($PSScriptRoot.Length) characters long and Windows long paths are off, so installing " +
          "PyTorch would fail. Move/clone the project to a short folder such as C:\TSE and run again " +
          "(or enable long paths: https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation).")
}

# 1. update ---------------------------------------------------------------------------
Step 1 "Updating the code"
if ($NoPull) { Info "skipped (-NoPull)" }
elseif (-not (Get-Command git -ErrorAction SilentlyContinue)) { Info "git not found; using the files as they are" }
elseif (-not (Test-Path ".git")) { Info "not a git checkout; using the files as they are" }
else {
    & git pull --ff-only 2>&1 | ForEach-Object { Info $_ }
    if ($LASTEXITCODE -ne 0) { Info "git pull failed (offline or local changes?); continuing with the current files" }
}

# 2. python ---------------------------------------------------------------------------
Step 2 "Finding Python 3.10-3.13"
$py = $null; $pyArgs = @()
$candidates = @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("python"), @("python3"))
foreach ($c in $candidates) {
    if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
    $extra = @($c | Select-Object -Skip 1)
    $ver = & $c[0] @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.(1[0-3])$') { $py = $c[0]; $pyArgs = $extra; Ok "Python $ver ($($c -join ' '))"; break }
}
if (-not $py) { Fail "Python 3.10-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and run again." }

# 3. venv -----------------------------------------------------------------------------
Step 3 "Virtual environment (.venv)"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if ($Reinstall -and (Test-Path ".venv")) { Info "removing the old .venv (-Reinstall)"; Remove-Item -Recurse -Force ".venv" }
if (Test-Path $venvPy) { Ok "exists" }
else {
    Run $py ($pyArgs + @("-m", "venv", ".venv"))
    Ok "created"
}
Run $venvPy @("-m", "pip", "install", "--quiet", "--upgrade", "pip")

# 4. pytorch --------------------------------------------------------------------------
Step 4 "PyTorch"
$hasTorch = $false
& $venvPy -c "import torch, torchaudio" 2>$null
if ($LASTEXITCODE -eq 0) { $hasTorch = $true }
$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
}
if ($gpu) { Info "NVIDIA GPU: $gpu" } else { Info "no NVIDIA GPU found (nvidia-smi missing)" }
$want = $Cuda
if ($want -eq "auto") { $want = if ($gpu) { "cu128" } else { "cpu" } }
if ($hasTorch) {
    $cudaOk = & $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null
    if ($want -ne "cpu" -and $cudaOk -ne "1") {
        Info "installed PyTorch cannot use the GPU; reinstalling the CUDA build"
        $hasTorch = $false
    }
    else { Ok ("already installed: " + (& $venvPy -W ignore -c "import torch; print(torch.__version__)")) }
}
if (-not $hasTorch) {
    $index = "https://download.pytorch.org/whl/$want"
    Info "installing torch + torchaudio ($want) from $index"
    if ($want -ne "cpu") { Info "this is a ~3 GB download; it can take a while" }
    Run $venvPy @("-m", "pip", "install", "--upgrade", "torch", "torchaudio", "--index-url", $index)
    Ok ("installed: " + (& $venvPy -W ignore -c "import torch; print(torch.__version__)"))
}

# 5. packages -------------------------------------------------------------------------
Step 5 "Demo packages (requirements-demo.txt)"
Run $venvPy @("-m", "pip", "install", "--quiet", "-r", "requirements-demo.txt")
Ok "installed"

# 6. self-test ------------------------------------------------------------------------
Step 6 "Self-test (GPU, models, quality, live speed)"
if ($SkipTest) { Info "skipped (-SkipTest)" }
else {
    & $venvPy scripts\self_test.py
    if ($LASTEXITCODE -ne 0) { Fail "self-test failed (see above)" }
}

# 7. run ------------------------------------------------------------------------------
Step 7 "Starting the demo server"
$elapsed = [int]((Get-Date) - $t0).TotalSeconds
Info "setup took $elapsed s"
$url = "http://127.0.0.1:$Port"
if ($Passcode) { $env:TSE_PASSCODE = $Passcode; Info "passcode: $Passcode" } else { Remove-Item Env:TSE_PASSCODE -ErrorAction SilentlyContinue }
Write-Host ""
Write-Host "  Demo: $url   (overview, Mixer and Live pages)" -ForegroundColor Green
Write-Host "  Use Chrome or Edge. Live microphone: wear headphones." -ForegroundColor Green
Write-Host "  Stop the server with Ctrl+C or by closing this window." -ForegroundColor DarkGray
Write-Host ""
if (-not $NoBrowser) {
    # open the browser once the server answers
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 120; $i++) {
            try { Invoke-WebRequest "$u/api/info" -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; return } catch { Start-Sleep -Seconds 1 }
        }
    } -ArgumentList $url | Out-Null
}
$Host.UI.RawUI.WindowTitle = "TSE demo server ($url)"
& $venvPy scripts\serve.py --port $Port
