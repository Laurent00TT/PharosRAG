# NaviKB 本地 4090-48G 控制台 / control console
# Monitoring is pure-PowerShell (Windows reaches the WSL services via localhost
# forwarding; GPU via nvidia-smi). Start/Stop shell out to WSL.
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
# Health checks hit localhost (WSL services) — must NOT go through the system
# Clash proxy, or every check wrongly reads DOWN. Force direct connection.
[System.Net.WebRequest]::DefaultWebProxy = $null
# Warm up the .NET HTTP stack once — the FIRST Invoke-WebRequest in a fresh
# process has multi-second cold-start overhead that would falsely time out.
try { Invoke-WebRequest 'http://localhost:8000/health' -TimeoutSec 8 -UseBasicParsing | Out-Null } catch {}

$here   = $PSScriptRoot
$wslDir = '/mnt/' + $here.Substring(0,1).ToLower() + ($here.Substring(2) -replace '\\','/')
$distro = 'Ubuntu'
$gpuIdx = 1   # RTX 4090 (index 0 is the 5070)

$services = @(
  @{n='embed'; port=8003; path='health'},
  @{n='sparse';port=8004; path='health'},
  @{n='rerank';port=8005; path='health'},
  @{n='gen';   port=8006; path='health'},
  @{n='mineru';port=8101; path='health'},
  @{n='serve'; port=8000; path='health'},
  @{n='qdrant';port=6333; path='healthz'}
)

function Test-Svc($port,$path){
  try { return ((Invoke-WebRequest "http://localhost:$port/$path" -TimeoutSec 4 -UseBasicParsing).StatusCode -eq 200) }
  catch { return $false }
}
function Get-Gpu {
  $l = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits -i $gpuIdx 2>$null
  if(-not $l){ return $null }
  $p = ($l -split ',') | ForEach-Object { $_.Trim() }
  [pscustomobject]@{ name=$p[0]; util=[int]$p[1]; used=[int]$p[2]; total=[int]$p[3]; temp=[int]$p[4] }
}
function Get-Worker {
  $r = wsl -d $distro -- bash -lc "pgrep -f scripts/worker.py >/dev/null 2>&1 && echo UP || echo DOWN" 2>$null
  return ($r -match 'UP')
}
function Bar($pct,$w=24){ $f=[math]::Max(0,[math]::Min($w,[int][math]::Round($pct/100*$w))); ('#'*$f)+('-'*($w-$f)) }
function Col($pct,$hi,$mid){ if($pct -ge $hi){'Red'}elseif($pct -ge $mid){'Yellow'}else{'Green'} }

function Show-Dash($withWorker){
  $g = Get-Gpu
  Write-Host ""
  if($g){
    $mp = [int][math]::Round($g.used/$g.total*100)
    Write-Host ("  GPU  {0}        {1}C" -f $g.name,$g.temp) -ForegroundColor Cyan
    Write-Host ("  Load [{0}] {1,3}%" -f (Bar $g.util),$g.util) -ForegroundColor (Col $g.util 90 60)
    Write-Host ("  VRAM [{0}] {1,3}%   {2} / {3} MiB" -f (Bar $mp),$mp,$g.used,$g.total) -ForegroundColor (Col $mp 92 80)
  } else { Write-Host "  GPU: nvidia-smi unavailable" -ForegroundColor Red }
  Write-Host ("  " + ('-'*54)) -ForegroundColor DarkGray
  $upN = 0
  foreach($s in $services){
    if(Test-Svc $s.port $s.path){ $upN++; Write-Host ("   {0,-8} :{1}    " -f $s.n,$s.port) -NoNewline; Write-Host "UP" -ForegroundColor Green }
    else { Write-Host ("   {0,-8} :{1}    " -f $s.n,$s.port) -NoNewline; Write-Host "DOWN" -ForegroundColor Red }
  }
  if($withWorker){
    if(Get-Worker){ Write-Host ("   {0,-8} {1}   " -f 'worker','(queue)') -NoNewline; Write-Host "UP" -ForegroundColor Green }
    else { Write-Host ("   {0,-8} {1}   " -f 'worker','(queue)') -NoNewline; Write-Host "DOWN" -ForegroundColor Red }
  }
  Write-Host ("  " + ('-'*54)) -ForegroundColor DarkGray
  Write-Host ("  {0}/{1} model services up" -f $upN,$services.Count) -ForegroundColor White
}

function Monitor {
  while($true){
    Clear-Host
    Write-Host ("  NaviKB 48G  实时监控 / live   {0}" -f (Get-Date -Format 'HH:mm:ss')) -ForegroundColor White
    Show-Dash $true
    Write-Host ""
    Write-Host "  每 3 秒刷新  —  按 Q 返回菜单 / press Q to return" -ForegroundColor Yellow
    for($i=0;$i -lt 30;$i++){
      Start-Sleep -Milliseconds 100
      if([Console]::KeyAvailable){ if([Console]::ReadKey($true).Key -eq 'Q'){ return } }
    }
  }
}
function Start-Stack {
  Write-Host "在新窗口拉起全栈(加载所有模型约需几分钟)..." -ForegroundColor Yellow
  Start-Process cmd -ArgumentList '/k',"title NaviKB Startup && wsl -d $distro -- bash $wslDir/start_everything.sh"
}
function Stop-Stack {
  Write-Host "停止全部服务..." -ForegroundColor Yellow
  wsl -d $distro -- bash "$wslDir/stop_all.sh"
}

while($true){
  Clear-Host
  Write-Host "==========================================================" -ForegroundColor Cyan
  Write-Host "        NaviKB 本地 4090-48G 控制台 / Control Console"        -ForegroundColor Cyan
  Write-Host "==========================================================" -ForegroundColor Cyan
  Show-Dash $false
  Write-Host ""
  Write-Host "   [1] 一键拉起全部模型+服务      Start all"
  Write-Host "   [2] 实时监控 GPU/显存/各服务   Live monitor"
  Write-Host "   [3] 刷新一次                   Refresh"
  Write-Host "   [4] 停止全部                   Stop all"
  Write-Host "   [5] 打开日志目录(资源管理器)  Open logs"
  Write-Host "   [0] 退出                       Exit"
  Write-Host ""
  switch(Read-Host "  选择 / choice"){
    '1'{ Start-Stack; Read-Host "  已在新窗口启动,回车继续" }
    '2'{ Monitor }
    '3'{ }
    '4'{ Stop-Stack; Read-Host "  已停止,回车继续" }
    '5'{ Start-Process "\\wsl.localhost\$distro\home\tiantian\navikb-serving\logs" }
    '0'{ return }
    default { }
  }
}
