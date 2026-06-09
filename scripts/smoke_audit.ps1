# T2 smoke — exercises audit + brute-force against a running server.
#
# Uses an ISOLATED temp directory (NOT ./qdrant_data) so this script
# cannot wipe real production data. Earlier versions did
# `Remove-Item -Recurse -Force .\qdrant_data, .\logs` which would
# silently delete everything an operator has indexed — that was a
# footgun and has been removed. If you want to reset YOUR real data,
# do it explicitly elsewhere, never as a side effect of smoke.
#
# Parameters:
#   -Port    127.0.0.1 port to bind (default 8765 — avoids dev :8000)
#   -SmokeDir  isolated workspace (default $env:TEMP\kb-smoke-audit)
#   -Keep    keep $SmokeDir on exit instead of deleting it
#
# Backends:
#   - Qdrant: this script forces local-file mode in $SmokeDir, so a
#     running Qdrant server is NOT required for /audit smoke.
#   - mineru / embedding / LLM are lazy and not touched by /audit.

param(
    [int]$Port = 8765,
    [string]$SmokeDir = (Join-Path $env:TEMP "kb-smoke-audit"),
    [switch]$Keep
)

$ErrorActionPreference = "Stop"

# 1. Fresh isolated tmp dir — explicit, no surprise-deletion of repo paths
if (Test-Path $SmokeDir) { Remove-Item -Recurse -Force $SmokeDir }
New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null
Write-Host "Smoke workspace: $SmokeDir"

# 2. Temporarily move .env aside so production config (QDRANT_URL pointing
#    at the Docker container, MinerU tokens, etc.) doesn't bleed into the
#    isolated smoke run. Pydantic Settings on Windows treats $env:VAR=""
#    as "unset" and falls back to .env, so setting an empty env var does
#    NOT override a non-empty .env value. Renaming is the only reliable
#    isolation.
$envBackup = $null
if (Test-Path .env) {
    $envBackup = ".env.smoke-bak-$(Get-Date -Format 'yyyyMMddHHmmss')"
    Move-Item .env $envBackup
    Write-Host "Moved .env -> $envBackup (will restore on exit)"
}

# 3. Isolate all data paths via env vars (now uncontested by .env).
$env:QDRANT_PATH         = $SmokeDir
$env:QDRANT_URL          = ""
$env:IMAGE_STORAGE_PATH  = Join-Path $SmokeDir "images"
$env:LOG_DIR             = Join-Path $SmokeDir "logs"
$env:NAV_DB_PATH         = Join-Path $SmokeDir "nav_index.db"
$env:AUDIT_FALLBACK_PATH = Join-Path $SmokeDir "audit_fallback.jsonl"
$env:KB_USERS_DB_URL     = "sqlite+aiosqlite:///$($SmokeDir.Replace('\','/'))/kb_metadata.db"

# 3. Bootstrap users via CLI (creates audit_log table + user.create audit rows)
$adminOut = & .\.venv\Scripts\python.exe scripts\manage_users.py create smoke_admin --role admin
$aliceOut = & .\.venv\Scripts\python.exe scripts\manage_users.py create smoke_alice --role member

$adminKey = ([regex]'kb_smoke_admin_\S+').Match($adminOut).Value
$aliceKey = ([regex]'kb_smoke_alice_\S+').Match($aliceOut).Value
if (-not $adminKey) { throw "failed to parse admin key from manage_users.py output" }
Write-Host "Admin key prefix: $($adminKey.Substring(0, [Math]::Min($adminKey.Length, 24)))..."

# 4. Start server in background
$server = Start-Process -PassThru -NoNewWindow `
    .\.venv\Scripts\python.exe `
    -ArgumentList "scripts\serve.py","--host","127.0.0.1","--port","$Port"

# Poll /health until ready (bge-m3 model load can take 30+ seconds on
# cold start; a fixed Start-Sleep is fragile). Timeout 60s total.
$base = "http://127.0.0.1:$Port"
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-RestMethod -Uri "$base/health" -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ready) { throw "server did not become ready within 60s" }
Write-Host "Server ready after ~$i seconds"

try {

    # 5. Admin sees user.create audit rows from CLI bootstrap (expect 2)
    $resp = Invoke-RestMethod -Uri "$base/audit?action=user.create" `
                              -Headers @{ "X-API-Key" = $adminKey }
    Write-Host "[1/3] user.create rows: $($resp.rows.Count) (expect 2)"
    if ($resp.rows.Count -ne 2) { throw "expected 2 user.create rows, got $($resp.rows.Count)" }

    # 6. Brute-force trigger: 10 garbage-key requests
    1..10 | ForEach-Object {
        try {
            Invoke-RestMethod -Uri "$base/audit" `
                              -Headers @{ "X-API-Key" = "garbage-$_" } | Out-Null
        } catch { } # 401s expected
    }
    Start-Sleep -Seconds 1
    $bf = Invoke-RestMethod -Uri "$base/audit?action=auth.brute_force" `
                            -Headers @{ "X-API-Key" = $adminKey }
    Write-Host "[2/3] auth.brute_force rows: $($bf.rows.Count) (expect 1)"
    if ($bf.rows.Count -ne 1) { throw "expected exactly 1 brute-force row, got $($bf.rows.Count)" }
    Write-Host "       source_ip: $($bf.rows[0].payload.source_ip)"
    Write-Host "       count:     $($bf.rows[0].payload.count)"

    # 7. Member scope — alice sees only her own
    $aliceView = Invoke-RestMethod -Uri "$base/audit" `
                                   -Headers @{ "X-API-Key" = $aliceKey }
    Write-Host "[3/3] Alice sees $($aliceView.rows.Count) rows scoped to herself"

    # 8. T3 permission matrix — direct-seed two docs (bypass MinerU/embedding/LLM)
    #    so smoke verifies ownership routing without external services up.
    $aliceUserId = & .\.venv\Scripts\python.exe -c @"
import asyncio
from kb.auth.users import UsersStore
async def main():
    s = UsersStore(db_url='sqlite+aiosqlite:///$($SmokeDir.Replace('\','/'))/kb_metadata.db')
    await s.init()
    u = await s.get_by_username('smoke_alice')
    print(u.user_id)
    await s.aclose()
asyncio.run(main())
"@
    $aliceUserId = $aliceUserId.Trim()
    if (-not $aliceUserId) { throw "failed to resolve smoke_alice user_id" }

    & .\.venv\Scripts\python.exe -c @"
import asyncio
from kb.storage.metadata_db import MetadataDB
async def main():
    db = MetadataDB(db_url='sqlite+aiosqlite:///$($SmokeDir.Replace('\','/'))/kb_metadata.db')
    await db.init()
    await db.upsert_document(
        doc_id='smoke-alice', doc_name='alice.pdf',
        version=None, effective_date=None,
        expiry_date=None, supersedes=None,
        owner_id='$aliceUserId',
    )
    await db.upsert_document(
        doc_id='smoke-legacy', doc_name='legacy.pdf',
        version=None, effective_date=None,
        expiry_date=None, supersedes=None,
        owner_id=None,
    )
    await db.aclose()
asyncio.run(main())
"@

    # [4/4] permission matrix
    Write-Host "[4/4] T3 permission matrix"

    # bob (member) deletes alice's doc → 403
    $bobOut = & .\.venv\Scripts\python.exe scripts\manage_users.py create smoke_bob --role member
    $bobKey = ([regex]'kb_smoke_bob_\S+').Match($bobOut).Value
    if (-not $bobKey) { throw "failed to parse bob key from manage_users.py output" }
    try {
        Invoke-RestMethod -Method Delete -Uri "$base/documents/smoke-alice" `
                          -Headers @{ "X-API-Key" = $bobKey } | Out-Null
        throw "bob delete should have failed"
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code -ne 403) {
            throw "expected 403 from bob, got $code"
        }
        Write-Host "       bob -> alice's doc: 403 OK"
    }

    # alice deletes own → 200
    Invoke-RestMethod -Method Delete -Uri "$base/documents/smoke-alice" `
                      -Headers @{ "X-API-Key" = $aliceKey } | Out-Null
    Write-Host "       alice -> own doc: 200 OK"

    # admin deletes NULL-owner legacy doc → 200
    Invoke-RestMethod -Method Delete -Uri "$base/documents/smoke-legacy" `
                      -Headers @{ "X-API-Key" = $adminKey } | Out-Null
    Write-Host "       admin -> null-owner doc: 200 OK"

    Write-Host "`nSmoke PASS"
} finally {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    # Restore .env regardless of smoke outcome — running again must not
    # find the .env still moved aside.
    if ($envBackup -and (Test-Path $envBackup)) {
        Move-Item $envBackup .env -Force
        Write-Host "Restored .env from $envBackup"
    }
    if (-not $Keep) {
        Remove-Item -Recurse -Force $SmokeDir -ErrorAction SilentlyContinue
        Write-Host "Cleaned up $SmokeDir"
    } else {
        Write-Host "Kept $SmokeDir for inspection (re-run without -Keep to auto-clean)"
    }
}
