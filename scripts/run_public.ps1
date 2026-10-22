# Make the TSE demo public: start scripts/serve.py on this GPU machine and expose it with a
# Cloudflare quick tunnel (no account needed). Prints the public URL and a ready-to-share link
# with the server address and passcode built in, saves them to public_url.txt and copies the
# link to the clipboard. Closing this window stops the tunnel; the server runs in its own window.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_public.ps1
#         ... -Passcode mycode     (default: a random 6-character passcode)
#         ... -NoPasscode          (open to anyone with the link)
#         ... -SiteUrl https://tse.vercel.app   (static frontend; default: the tunnel itself)
param(
    [string]$Passcode = "",
    [switch]$NoPasscode,
    [string]$SiteUrl = "",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$Host.UI.RawUI.WindowTitle = "TSE public tunnel"

$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) { $cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe" }
if (-not (Test-Path $cloudflared)) { throw "cloudflared not found: winget install --id Cloudflare.cloudflared" }

if (-not $NoPasscode -and -not $Passcode) {
    $chars = "abcdefghjkmnpqrstuvwxyz23456789".ToCharArray()
    $Passcode = -join (1..6 | ForEach-Object { $chars | Get-Random })
}

# 1. the inference server (own window, so its log stays visible)
$serverArgs = @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command",
    "`$Host.UI.RawUI.WindowTitle='TSE server :$Port'; Set-Location '$root'; " +
    $(if ($Passcode) { "`$env:TSE_PASSCODE='$Passcode'; " } else { "" }) +
    ".\.venv\Scripts\python.exe scripts\serve.py --port $Port")
Start-Process powershell -ArgumentList $serverArgs -WorkingDirectory $root
Write-Host "Starting the server on port $Port ..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    try { Invoke-RestMethod "http://127.0.0.1:$Port/api/info" -TimeoutSec 2 | Out-Null; $ready = $true; break } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw "the server did not start; check its window" }

# 2. the tunnel
$log = Join-Path $env:TEMP "tse_cloudflared.log"
Remove-Item $log -ErrorAction SilentlyContinue
$tunnel = Start-Process $cloudflared -ArgumentList @("tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$Port", "--logfile", $log) -PassThru -WindowStyle Minimized
Write-Host "Opening the Cloudflare tunnel ..." -ForegroundColor Cyan
$url = $null
for ($i = 0; $i -lt 60 -and -not $url; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $log) {
        $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value }
    }
}
if (-not $url) { Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue; throw "no tunnel URL after 60 s; see $log" }

$site = if ($SiteUrl) { $SiteUrl.TrimEnd("/") } else { $url }
$share = "$site/?api=$url" + $(if ($Passcode) { "&code=$Passcode" } else { "" })
@("tunnel: $url", "passcode: $(if ($Passcode) { $Passcode } else { '(none)' })", "share: $share") |
    Set-Content -Path (Join-Path $root "public_url.txt") -Encoding utf8
try { Set-Clipboard -Value $share } catch { }

Write-Host ""
Write-Host "  Public server : $url" -ForegroundColor Green
Write-Host "  Passcode      : $(if ($Passcode) { $Passcode } else { '(none)' })" -ForegroundColor Green
Write-Host "  Share this    : $share" -ForegroundColor Yellow
Write-Host "  (copied to the clipboard and saved in public_url.txt)"
Write-Host ""
Write-Host "  Keep this window open. The URL changes every time the tunnel restarts." -ForegroundColor DarkGray
Write-Host "  Waiting for the tunnel process; Ctrl+C or closing the window stops it." -ForegroundColor DarkGray
Wait-Process -Id $tunnel.Id
