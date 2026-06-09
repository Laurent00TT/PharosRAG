<#
.SYNOPSIS
    Knowledge Base CLI — 把 12 个 HTTP 端点包成顺手的 PowerShell 命令。

.DESCRIPTION
    所有请求走 Invoke-RestMethod + hashtable,完全绕开 PowerShell 对 curl.exe
    参数和 JSON 引号的双层解析(避免 '{"x":1}' 里的双引号被吞)。

    Base URL 来自 $env:KB_URL,默认 http://localhost:8000。

.EXAMPLE
    pwsh scripts/kb.ps1 ask "by 的常见用法"
    pwsh scripts/kb.ps1 search "BE 动词否定句" -k 5
    pwsh scripts/kb.ps1 add C:\docs\manual.pdf
    pwsh scripts/kb.ps1 list
    pwsh scripts/kb.ps1 status <job_id>

    # 推荐:在 PS profile 里加个 alias,直接打 'kb'(见末尾 hint)
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
$BaseUrl = if ($env:KB_URL) { $env:KB_URL } else { "http://localhost:8000" }

# Force UTF-8 console output so Chinese / emoji from the API don't get mangled
# by the default GBK (cp936) codepage on Chinese Windows.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# ── HTTP 助手 ─────────────────────────────────────────────────────────────────
function Invoke-Kb {
    param(
        [string]$Method,
        [string]$Path,
        $Body = $null,
        [int]$TimeoutSec = 600
    )
    $uri = $BaseUrl + $Path
    try {
        # Use Invoke-WebRequest so we can read RawContentStream and decode UTF-8
        # ourselves. Invoke-RestMethod on PS 5.1 ignores charset=utf-8 and decodes
        # as ISO-8859-1, corrupting Chinese.
        $iwrArgs = @{
            Method     = $Method
            Uri        = $uri
            TimeoutSec = $TimeoutSec
            UseBasicParsing = $true
        }
        if ($Body) {
            $json = $Body | ConvertTo-Json -Depth 10 -Compress
            $iwrArgs['ContentType'] = 'application/json; charset=utf-8'
            $iwrArgs['Body'] = [System.Text.Encoding]::UTF8.GetBytes($json)
        }
        $resp = Invoke-WebRequest @iwrArgs
        $bytes = $resp.RawContentStream.ToArray()
        $text = [System.Text.Encoding]::UTF8.GetString($bytes)
        if ([string]::IsNullOrWhiteSpace($text)) { return $null }
        return $text | ConvertFrom-Json
    } catch {
        Write-Host ""
        Write-Host "✗ 请求失败: $Method $uri" -ForegroundColor Red
        # Try to extract the response body for 4xx/5xx
        $respBody = $null
        try {
            if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
                $respBody = $_.ErrorDetails.Message
            } elseif ($_.Exception.Response) {
                $stream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($stream)
                $respBody = $reader.ReadToEnd()
            }
        } catch { }
        if ($respBody) { Write-Host $respBody -ForegroundColor DarkRed }
        Write-Host $_.Exception.Message -ForegroundColor DarkRed
        if ($_.Exception.Message -match "actively refused|unable to connect") {
            Write-Host "  → tool_server 没起,跑 'pwsh scripts/start_all.ps1'" -ForegroundColor Yellow
        }
        exit 1
    }
}

# ── 子命令实现 ────────────────────────────────────────────────────────────────

function Cmd-Help {
    Write-Host @"

Knowledge Base CLI    (base: $BaseUrl)

  ── 查询 ──────────────────────────────────────────────────────
  ask    <question> [-iters N]      Agent 多轮综合答案(慢,60-90s)
  search <query>    [-k N]          原始 chunk 检索(快,~5s)

  ── 文档管理 ──────────────────────────────────────────────────
  add    <pdf_path> [<pdf_path> …]  排队入库 (worker 异步处理)
  list                              列出所有文档
  show   <doc_id>                   单个文档详情
  delete <doc_id>                   删除文档(向量+元数据)

  ── 入库任务 ──────────────────────────────────────────────────
  status <job_id>                   看 job 当前状态 + items
  events <job_id>                   看 job 事件流(parse/embed/...)
  retry  <job_id>                   重跑失败 items
  cancel <job_id>                   取消

  ── 其他 ─────────────────────────────────────────────────────
  health                            tool_server 健康检查
  help                              显示本帮助

环境变量:
  KB_URL    覆盖默认 base URL (e.g. http://192.168.1.10:8000)

"@ -ForegroundColor Cyan
}

function Cmd-Health {
    $r = Invoke-Kb -Method GET -Path "/health"
    Write-Host "tool_server: $($r.status)" -ForegroundColor Green
}

function Cmd-Ask {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb ask <question> [-iters N]" -ForegroundColor Yellow
        return
    }
    # 解析 -iters
    $iters = 3
    $qparts = @()
    for ($i = 0; $i -lt $script:Rest.Count; $i++) {
        if ($script:Rest[$i] -eq '-iters' -and $i + 1 -lt $script:Rest.Count) {
            $iters = [int]$script:Rest[$i + 1]
            $i++
        } else {
            $qparts += $script:Rest[$i]
        }
    }
    $question = $qparts -join ' '
    Write-Host ""
    Write-Host "❓ $question" -ForegroundColor Cyan
    Write-Host "  (max_iterations=$iters, 期望耗时 60-120s)" -ForegroundColor DarkGray
    Write-Host ""

    $t0 = Get-Date
    $r = Invoke-Kb -Method POST -Path "/ask" -Body @{
        question       = $question
        max_iterations = $iters
    }
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds

    Write-Host "── 答案 ───────────────────────────────────────" -ForegroundColor Green
    Write-Host $r.answer
    Write-Host ""

    if ($r.citations -and $r.citations.Count -gt 0) {
        Write-Host "── 引用 ($($r.citations.Count)) ─────────────────────────────────" -ForegroundColor Green
        $r.citations | ForEach-Object {
            Write-Host ("  • {0}  page {1}" -f $_.doc_id.Substring(0, 12), $_.page_num) -ForegroundColor DarkGray
        }
        Write-Host ""
    }

    if ($r.citations_hallucinated -and $r.citations_hallucinated.Count -gt 0) {
        Write-Host "⚠ 幻觉引用 (LLM 自造,KB 里没有):" -ForegroundColor Yellow
        $r.citations_hallucinated | ForEach-Object {
            Write-Host ("  • {0} p{1}" -f $_.doc_id.Substring(0, 12), $_.page_num) -ForegroundColor DarkYellow
        }
        Write-Host ""
    }

    $conf = $r.confidence
    $confColor = if ($conf.status -eq 'grounded') { 'Green' } else { 'Yellow' }
    Write-Host ("confidence: {0}  retrieved={1}  elapsed={2}s" -f $conf.status, $r.retrieved.Count, $elapsed) -ForegroundColor $confColor
    if ($conf.reasons) {
        Write-Host ("  reasons: {0}" -f ($conf.reasons -join ', ')) -ForegroundColor DarkGray
    }
}

function Cmd-Search {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb search <query> [-k N]" -ForegroundColor Yellow
        return
    }
    $topK = 5
    $qparts = @()
    for ($i = 0; $i -lt $script:Rest.Count; $i++) {
        if ($script:Rest[$i] -eq '-k' -and $i + 1 -lt $script:Rest.Count) {
            $topK = [int]$script:Rest[$i + 1]
            $i++
        } else {
            $qparts += $script:Rest[$i]
        }
    }
    $query = $qparts -join ' '
    Write-Host ""
    Write-Host "🔍 $query  (top_k=$topK)" -ForegroundColor Cyan
    Write-Host ""

    $r = Invoke-Kb -Method POST -Path "/search_docs" -Body @{
        query = $query
        top_k = $topK
    }

    if (-not $r.results -or $r.results.Count -eq 0) {
        Write-Host "(无结果)" -ForegroundColor Yellow
        return
    }

    $i = 1
    foreach ($hit in $r.results) {
        $chs = $hit.match_channels -join '+'
        Write-Host ("#{0}  {1}  p{2}  [{3}]  score={4:F4}" -f $i, $hit.doc_id.Substring(0, 12), $hit.page_num, $chs, $hit.score) -ForegroundColor Green
        if ($hit.heading_path) {
            Write-Host ("    § {0}" -f $hit.heading_path) -ForegroundColor DarkGray
        }
        $text = if ($hit.text_chunk) { $hit.text_chunk } else { '' }
        if ($text.Length -gt 200) { $text = $text.Substring(0, 200) + '…' }
        Write-Host ("    {0}" -f ($text -replace "`n", ' ')) -ForegroundColor Gray
        if ($hit.image_url) {
            Write-Host ("    🖼  {0}" -f $hit.image_url) -ForegroundColor DarkCyan
        }
        Write-Host ""
        $i++
    }
}

function Cmd-Add {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb add <pdf_path> [<pdf_path> …]" -ForegroundColor Yellow
        return
    }
    # Resolve each path to absolute & verify exists
    $paths = @()
    foreach ($p in $script:Rest) {
        $resolved = Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue
        if (-not $resolved) {
            Write-Host "✗ 找不到文件: $p" -ForegroundColor Red
            exit 1
        }
        $paths += $resolved.Path.Replace('\', '/')
    }

    Write-Host "📥 排队 $($paths.Count) 个文档..." -ForegroundColor Cyan
    $r = Invoke-Kb -Method POST -Path "/ingestion/jobs" -Body @{
        paths        = $paths
        max_attempts = 3
    }
    Write-Host "✓ job_id: $($r.job_id)" -ForegroundColor Green
    Write-Host "  status: $($r.status)  total_items: $($r.total_items)"
    Write-Host ""
    Write-Host "查进度:  kb status $($r.job_id)" -ForegroundColor DarkGray
    Write-Host "看事件:  kb events $($r.job_id)" -ForegroundColor DarkGray
}

function Cmd-List {
    $r = Invoke-Kb -Method GET -Path "/documents"
    $docs = if ($r -is [array]) { $r } else { $r.documents }
    if (-not $docs -or $docs.Count -eq 0) {
        Write-Host "(知识库为空)" -ForegroundColor Yellow
        return
    }
    Write-Host ""
    Write-Host ("Total: {0} documents" -f $docs.Count) -ForegroundColor Cyan
    Write-Host ""
    $docs | ForEach-Object {
        $id = $_.doc_id.Substring(0, 12)
        $name = $_.doc_name
        if ($name.Length -gt 60) { $name = $name.Substring(0, 60) + '…' }
        $status = $_.status
        $statusColor = if ($status -eq 'active') { 'Green' } else { 'DarkGray' }
        Write-Host ("  {0}  " -f $id) -NoNewline -ForegroundColor DarkGray
        Write-Host ("[{0,-8}]  " -f $status) -NoNewline -ForegroundColor $statusColor
        Write-Host $name
    }
    Write-Host ""
}

function Cmd-Show {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb show <doc_id>" -ForegroundColor Yellow
        return
    }
    $r = Invoke-Kb -Method GET -Path "/documents/$($script:Rest[0])"
    $r | ConvertTo-Json -Depth 10
}

function Cmd-Delete {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb delete <doc_id>" -ForegroundColor Yellow
        return
    }
    $confirm = Read-Host "确认删除 $($script:Rest[0])? (y/N)"
    if ($confirm -notmatch '^[Yy]') { Write-Host "取消"; return }
    Invoke-Kb -Method DELETE -Path "/documents/$($script:Rest[0])" | Out-Null
    Write-Host "✓ 已删除 $($script:Rest[0])" -ForegroundColor Green
}

function Cmd-Status {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb status <job_id>" -ForegroundColor Yellow
        return
    }
    $r = Invoke-Kb -Method GET -Path "/ingestion/jobs/$($script:Rest[0])"
    $j = $r.job
    $statusColor = switch ($j.status) {
        'succeeded' { 'Green' }
        'completed' { 'Green' }
        'failed'    { 'Red' }
        'running'   { 'Cyan' }
        default     { 'Yellow' }
    }
    Write-Host ""
    Write-Host ("Job $($j.job_id)") -ForegroundColor Cyan
    Write-Host ("  status: ") -NoNewline; Write-Host $j.status -ForegroundColor $statusColor
    Write-Host ("  progress: {0}/{1} ok, {2} failed, {3} skipped" -f $j.succeeded_items, $j.total_items, $j.failed_items, $j.skipped_items)
    Write-Host ""
    if ($r.items) {
        Write-Host "Items:"
        $r.items | ForEach-Object {
            $itemColor = switch ($_.status) {
                'succeeded' { 'Green' }
                'failed'    { 'Red' }
                'running'   { 'Cyan' }
                default     { 'DarkGray' }
            }
            $name = Split-Path $_.path -Leaf
            Write-Host ("  ") -NoNewline
            Write-Host ("[{0,-10}]" -f $_.status) -NoNewline -ForegroundColor $itemColor
            Write-Host ("  attempt {0}/{1}  {2}" -f $_.attempt, $_.max_attempts, $name)
        }
        Write-Host ""
    }
}

function Cmd-Events {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb events <job_id>" -ForegroundColor Yellow
        return
    }
    $r = Invoke-Kb -Method GET -Path "/ingestion/jobs/$($script:Rest[0])/events"
    if (-not $r.events -or $r.events.Count -eq 0) {
        Write-Host "(无事件)" -ForegroundColor Yellow
        return
    }
    Write-Host ""
    $r.events | ForEach-Object {
        $ts = if ($_.ts) { $_.ts } else { '?' }
        Write-Host ("  {0}  " -f $ts) -NoNewline -ForegroundColor DarkGray
        Write-Host $_.event_type -ForegroundColor Cyan
        if ($_.payload) {
            $payloadJson = $_.payload | ConvertTo-Json -Depth 5 -Compress
            if ($payloadJson.Length -gt 200) { $payloadJson = $payloadJson.Substring(0, 200) + '…' }
            Write-Host ("    {0}" -f $payloadJson) -ForegroundColor Gray
        }
    }
    Write-Host ""
}

function Cmd-Retry {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb retry <job_id>" -ForegroundColor Yellow
        return
    }
    $r = Invoke-Kb -Method POST -Path "/ingestion/jobs/$($script:Rest[0])/retry"
    Write-Host "✓ 已重试 $($r.retried_items) 个 items" -ForegroundColor Green
}

function Cmd-Cancel {
    if (-not $script:Rest -or $script:Rest.Count -lt 1) {
        Write-Host "用法: kb cancel <job_id>" -ForegroundColor Yellow
        return
    }
    $r = Invoke-Kb -Method POST -Path "/ingestion/jobs/$($script:Rest[0])/cancel"
    Write-Host "✓ 已取消 (status: $($r.job.status))" -ForegroundColor Green
}

# ── 分发 ──────────────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    'ask'     { Cmd-Ask }
    'search'  { Cmd-Search }
    'add'     { Cmd-Add }
    'list'    { Cmd-List }
    'ls'      { Cmd-List }
    'show'    { Cmd-Show }
    'delete'  { Cmd-Delete }
    'rm'      { Cmd-Delete }
    'status'  { Cmd-Status }
    'events'  { Cmd-Events }
    'log'     { Cmd-Events }
    'retry'   { Cmd-Retry }
    'cancel'  { Cmd-Cancel }
    'health'  { Cmd-Health }
    'help'    { Cmd-Help }
    '-h'      { Cmd-Help }
    '--help'  { Cmd-Help }
    default {
        Write-Host "未知命令: $Command" -ForegroundColor Red
        Cmd-Help
        exit 1
    }
}
