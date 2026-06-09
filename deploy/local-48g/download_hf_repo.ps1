# Generic HF repo downloader via curl.exe through the Clash proxy (CONNECT tunnel,
# -L follows 308, -C - resumes, per-file retries). Usage:
#   download_hf_repo.ps1 -repo <org/name> -dest <windows-dir>
param([Parameter(Mandatory=$true)][string]$repo, [Parameter(Mandatory=$true)][string]$dest)
$proxy = 'http://127.0.0.1:7897'
$base  = 'https://hf-mirror.com'
New-Item -ItemType Directory -Force $dest | Out-Null

function Fetch($url, $out) {
  for ($i = 1; $i -le 20; $i++) {
    curl.exe -C - -x $proxy -L --retry 3 --retry-delay 2 -m 1200 -sS -o $out $url 2>$null
    if ($LASTEXITCODE -eq 0) { return $true }
    Start-Sleep -Seconds 2
  }
  return $false
}

$apiOut = Join-Path $env:TEMP ("hfapi_" + ($repo -replace '[/\\]','_') + ".json")
Fetch "$base/api/models/$repo" $apiOut | Out-Null
$files = (Get-Content $apiOut -Raw | ConvertFrom-Json).siblings.rfilename
Write-Host ("files (" + $files.Count + "): " + ($files -join ', '))
$ok = $true
foreach ($f in $files) {
  $out = Join-Path $dest $f
  New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 0) -and ($f -notmatch '\.(bin|safetensors)$')) { Write-Host "  skip $f"; continue }
  Write-Host "  get $f ..."
  if (Fetch "$base/$repo/resolve/main/$f" $out) { Write-Host ("    ok " + (Get-Item $out).Length + "B") } else { Write-Host "    FAILED $f"; $ok=$false }
}
if ($ok) { Write-Host "HF_REPO_DONE" } else { Write-Host "HF_REPO_INCOMPLETE" }
