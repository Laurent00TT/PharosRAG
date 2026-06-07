# Download MILCO via curl.exe through the Clash proxy, with per-file retries to
# ride out intermittent schannel/TLS handshake failures (TUN flakiness).
$proxy = 'http://127.0.0.1:7897'
$repo  = 'omai-research/milco-650m'
$base  = 'https://hf-mirror.com'
$dest  = 'C:\Users\11541\Desktop\projects\navikb\deploy\local-48g\milco-650m-dl'
New-Item -ItemType Directory -Force $dest | Out-Null

function Fetch($url, $out) {
  for ($i = 1; $i -le 30; $i++) {
    # -C - resumes from the partial file so drops don't restart the 1.3G blob.
    curl.exe -C - -x $proxy -L --retry 3 --retry-delay 2 -m 1800 -sS -o $out $url 2>$null
    if ($LASTEXITCODE -eq 0) { return $true }
    $mb = if (Test-Path $out) { [math]::Round((Get-Item $out).Length/1MB,1) } else { 0 }
    Write-Host ("    retry $i (exit $LASTEXITCODE, now ${mb}MB) " + (Split-Path $out -Leaf))
    Start-Sleep -Seconds 3
  }
  return $false
}

$apiOut = "$env:TEMP\milco_api.json"
Fetch "$base/api/models/$repo" $apiOut | Out-Null
$files = (Get-Content $apiOut -Raw | ConvertFrom-Json).siblings.rfilename
Write-Host ("files (" + $files.Count + "): " + ($files -join ', '))

$ok = $true
foreach ($f in $files) {
  $out = Join-Path $dest $f
  New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 0) -and ($f -ne 'pytorch_model.bin')) { Write-Host ("  skip (exists) " + $f); continue }
  Write-Host ("downloading " + $f + " ...")
  if (Fetch "$base/$repo/resolve/main/$f" $out) {
    Write-Host ("  ok " + $f + " = " + (Get-Item $out).Length + " bytes")
  } else {
    Write-Host ("  FAILED " + $f); $ok = $false
  }
}
if ($ok) { Write-Host "MILCO_CURL_DONE" } else { Write-Host "MILCO_CURL_INCOMPLETE" }
