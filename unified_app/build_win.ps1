
# ── build_win.ps1 ────────────────────────────────────────────────────────────
# Workaround for the winCodeSign symlink privilege issue on non-admin Windows.
# Strategy: start electron-builder, detect which hash it wants, copy a good
# cache entry there BEFORE it tries to extract the .7z, then let it continue.
# ─────────────────────────────────────────────────────────────────────────────

$goodCache = "C:\Users\user\AppData\Local\electron-builder\Cache\winCodeSign\331205323"
$cacheDir  = "C:\Users\user\AppData\Local\electron-builder\Cache\winCodeSign"

Write-Host "Building frontend..." -ForegroundColor Cyan
npm run build
if ($LASTEXITCODE -ne 0) { Write-Host "Frontend build failed" -ForegroundColor Red; exit 1 }

Write-Host "Starting electron-builder (background)..." -ForegroundColor Cyan
$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
$env:WIN_CSC_LINK = ""

# Run electron-builder, watch for the new hash folder, and immediately copy the good cache into it
$job = Start-Job -ScriptBlock {
    param($wd)
    Set-Location $wd
    $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
    $env:WIN_CSC_LINK = ""
    & node_modules\.bin\electron-builder --win --config 2>&1
} -ArgumentList (Get-Location).Path

Write-Host "Watching for new winCodeSign hash folder..." -ForegroundColor Yellow
$good = $goodCache
$attempts = 0
$filled = @()

while ($job.State -eq "Running" -and $attempts -lt 120) {
    Start-Sleep -Milliseconds 500
    $attempts++

    # Find any new folder created today that isn't yet fully populated
    $dirs = Get-ChildItem $cacheDir -Directory |
            Where-Object { $_.Name -notin $filled -and -not (Test-Path "$($_.FullName)\windows-10") }

    foreach ($d in $dirs) {
        Write-Host "  → Detected new hash: $($d.Name) — copying good cache..." -ForegroundColor Green
        Copy-Item -Path "$good\*" -Destination $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $filled += $d.Name
        Write-Host "  ✓ Done." -ForegroundColor Green
    }
}

Write-Host "Waiting for build to finish..." -ForegroundColor Cyan
Receive-Job $job -Wait | Write-Host
Remove-Job $job

Write-Host ""
$exe = Get-ChildItem "dist" -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($exe) {
    Write-Host "✅ EXE ready: dist\$($exe.Name) ($([math]::Round($exe.Length/1MB,1)) MB)" -ForegroundColor Green
} else {
    Write-Host "⚠ No EXE found in dist\. Check output above." -ForegroundColor Yellow
}
