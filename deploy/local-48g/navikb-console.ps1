# NaviKB 本地 4090-48G 控制台 / control console
#
# Lifetime model: THIS console OWNS the stack. A hidden watchdog watches this
# console's PID; the moment the console goes away — X button, Ctrl+C, force-kill,
# or crash — the watchdog runs stop_all.sh, so the 4090 VRAM is always freed.
# To keep the stack running WITHOUT the console, use menu [9] (detaches the
# watchdog) or run start_everything.sh directly in a WSL terminal.
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$here   = $PSScriptRoot
$wslDir = '/mnt/' + $here.Substring(0,1).ToLower() + ($here.Substring(2) -replace '\\','/')
$distro = 'Ubuntu'
$gpuIdx = 1   # RTX 4090 (index 0 is the 5070)

# Health checks hit localhost (WSL services) — bypass the system Clash proxy or
# every check wrongly reads DOWN; warm the .NET HTTP stack to avoid a cold-start
# first-call timeout.
[System.Net.WebRequest]::DefaultWebProxy = $null
try { Invoke-WebRequest 'http://localhost:8000/health' -TimeoutSec 8 -UseBasicParsing | Out-Null } catch {}

# --- watchdog: close console (any way) => stop_all ----------------------------
$wdScript = "while(Get-Process -Id $PID -ErrorAction SilentlyContinue){Start-Sleep -Seconds 2}; & wsl.exe -d $distro -- bash '$wslDir/stop_all.sh' | Out-Null"
$script:wd = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command',$wdScript
function Stop-Watchdog { if($script:wd){ Stop-Process -Id $script:wd.Id -Force -ErrorAction SilentlyContinue; $script:wd=$null } }
function Stop-Stack    { wsl -d $distro -- bash "$wslDir/stop_all.sh" 2>$null | Out-Null }

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

while($true){
  Clear-Host
  Write-Host "==========================================================" -ForegroundColor Cyan
  Write-Host "        NaviKB 本地 4090-48G 控制台 / Control Console"        -ForegroundColor Cyan
  Write-Host "==========================================================" -ForegroundColor Cyan
  Show-Dash $false
  Write-Host ""
  Write-Host "  ⚠ 关闭本窗口(X)/ Ctrl+C / 选 [0] → 自动停止所有服务、释放显存" -ForegroundColor DarkYellow
  Write-Host "    想让服务脱离本窗口长期运行,选 [9],或直接跑 start_everything.sh" -ForegroundColor DarkGray
  Write-Host ""
  Write-Host "   [1] 一键拉起全部模型+服务      Start all"
  Write-Host "   [2] 实时监控 GPU/显存/各服务   Live monitor"
  Write-Host "   [3] 刷新一次                   Refresh"
  Write-Host "   [4] 停止全部(不退出)          Stop all"
  Write-Host "   [5] 打开日志目录(资源管理器)  Open logs"
  Write-Host "   [9] 退出但保留服务运行         Exit, keep running"
  Write-Host "   [0] 退出并停止全部             Exit and stop"
  Write-Host ""
  switch(Read-Host "  选择 / choice"){
    '1'{ Start-Stack; Read-Host "  已在新窗口启动,回车继续" }
    '2'{ Monitor }
    '3'{ }
    '4'{ Write-Host "  停止全部服务..." -ForegroundColor Yellow; Stop-Stack; Read-Host "  已停止,回车继续" }
    '5'{ Start-Process "\\wsl.localhost\$distro\home\tiantian\navikb-serving\logs" }
    '9'{ Stop-Watchdog; Write-Host "  退出,服务保持后台运行" -ForegroundColor Yellow; exit 0 }
    '0'{ Write-Host "  停止全部并退出..." -ForegroundColor Yellow; Stop-Watchdog; Stop-Stack; exit 0 }
    default { }
  }
}
