#!/usr/bin/env bash
# T2 smoke — exercises audit + brute-force against a running server.
#
# Uses an ISOLATED temp dir (NOT ./qdrant_data) so this script cannot
# wipe real production data. Earlier versions did `rm -rf qdrant_data
# logs` which silently deleted everything an operator had indexed —
# that was a footgun and has been removed. If you want to reset YOUR
# real data, do it explicitly elsewhere, never as a smoke side effect.
#
# Flags:
#   --port N        bind port (default 8765, avoids dev 8000)
#   --smoke-dir DIR isolated workspace (default mktemp -d)
#   --keep          keep workspace on exit (default: cleaned)
set -euo pipefail

PORT=8765
SMOKE_DIR=""
KEEP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)       PORT="$2"; shift 2 ;;
    --smoke-dir)  SMOKE_DIR="$2"; shift 2 ;;
    --keep)       KEEP=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SMOKE_DIR" ]]; then
  SMOKE_DIR=$(mktemp -d -t kb-smoke-audit.XXXXXX)
fi
echo "Smoke workspace: $SMOKE_DIR"

# 1. Isolate all data paths via env vars so the running server cannot
#    touch real data. QDRANT_URL="" forces local-file mode.
export QDRANT_PATH="$SMOKE_DIR"
export QDRANT_URL=""
export IMAGE_STORAGE_PATH="$SMOKE_DIR/images"
export LOG_DIR="$SMOKE_DIR/logs"
export NAV_DB_PATH="$SMOKE_DIR/nav_index.db"
export AUDIT_FALLBACK_PATH="$SMOKE_DIR/audit_fallback.jsonl"
export KB_USERS_DB_URL="sqlite+aiosqlite:///$SMOKE_DIR/kb_metadata.db"

# 2. Bootstrap users via CLI
ADMIN_KEY=$(python scripts/manage_users.py create smoke_admin --role admin \
            | grep -oE 'kb_smoke_admin_[^[:space:]]+' | head -1)
ALICE_KEY=$(python scripts/manage_users.py create smoke_alice --role member \
            | grep -oE 'kb_smoke_alice_[^[:space:]]+' | head -1)
if [[ -z "$ADMIN_KEY" ]]; then
  echo "failed to parse admin key from manage_users.py output" >&2
  exit 1
fi

# 3. Start server in background
python scripts/serve.py --host 127.0.0.1 --port "$PORT" &
SERVER=$!

cleanup() {
  kill "$SERVER" 2>/dev/null || true
  if [[ "$KEEP" -eq 0 ]]; then
    rm -rf "$SMOKE_DIR"
    echo "Cleaned up $SMOKE_DIR"
  else
    echo "Kept $SMOKE_DIR for inspection"
  fi
}
trap cleanup EXIT

sleep 6
BASE="http://127.0.0.1:$PORT"

# 4. Admin sees user.create rows from CLI bootstrap (expect 2)
COUNT=$(curl -s "$BASE/audit?action=user.create" \
             -H "X-API-Key: $ADMIN_KEY" | jq '.rows | length')
echo "[1/3] user.create rows: $COUNT (expect 2)"
if [[ "$COUNT" != "2" ]]; then
  echo "expected 2 user.create rows, got $COUNT" >&2
  exit 1
fi

# 5. Brute-force trigger: 10 garbage-key requests
echo "[..] triggering 10 garbage-key requests"
for i in $(seq 1 10); do
  curl -s "$BASE/audit" -H "X-API-Key: garbage-$i" > /dev/null || true
done
sleep 1
BF_COUNT=$(curl -s "$BASE/audit?action=auth.brute_force" \
                -H "X-API-Key: $ADMIN_KEY" | jq '.rows | length')
echo "[2/3] auth.brute_force rows: $BF_COUNT (expect 1)"
if [[ "$BF_COUNT" != "1" ]]; then
  echo "expected exactly 1 brute-force row, got $BF_COUNT" >&2
  exit 1
fi
curl -s "$BASE/audit?action=auth.brute_force" \
     -H "X-API-Key: $ADMIN_KEY" | jq '.rows[0].payload'

# 6. Member scope — alice sees only her own
ALICE_ROWS=$(curl -s "$BASE/audit" \
                  -H "X-API-Key: $ALICE_KEY" | jq '.rows | length')
echo "[3/3] Alice sees $ALICE_ROWS rows scoped to herself"

# 7. T3 permission matrix — direct-seed two docs (bypass MinerU/embedding/LLM)
#    so smoke verifies ownership routing without external services up.
ALICE_USER_ID=$(python -c "
import asyncio
from kb.auth.users import UsersStore
async def main():
    s = UsersStore(db_url='sqlite+aiosqlite:///$SMOKE_DIR/kb_metadata.db')
    await s.init()
    u = await s.get_by_username('smoke_alice')
    print(u.user_id)
    await s.aclose()
asyncio.run(main())
")
ALICE_USER_ID=$(echo "$ALICE_USER_ID" | tr -d '[:space:]')
if [[ -z "$ALICE_USER_ID" ]]; then
  echo "failed to resolve smoke_alice user_id" >&2
  exit 1
fi

python -c "
import asyncio
from kb.storage.metadata_db import MetadataDB
async def main():
    db = MetadataDB(db_url='sqlite+aiosqlite:///$SMOKE_DIR/kb_metadata.db')
    await db.init()
    await db.upsert_document(
        doc_id='smoke-alice', doc_name='alice.pdf',
        version=None, effective_date=None,
        expiry_date=None, supersedes=None,
        owner_id='$ALICE_USER_ID',
    )
    await db.upsert_document(
        doc_id='smoke-legacy', doc_name='legacy.pdf',
        version=None, effective_date=None,
        expiry_date=None, supersedes=None,
        owner_id=None,
    )
    await db.aclose()
asyncio.run(main())
"

echo "[4/4] T3 permission matrix"

# Bootstrap a third smoke_bob user (member)
BOB_OUT=$(python scripts/manage_users.py create smoke_bob --role member)
BOB_KEY=$(echo "$BOB_OUT" | grep -oE 'kb_smoke_bob_[^[:space:]]+' | head -1)
if [[ -z "$BOB_KEY" ]]; then
  echo "failed to parse bob key from manage_users.py output" >&2
  exit 1
fi

# bob (member) DELETE alice's doc → 403
status=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
              "$BASE/documents/smoke-alice" -H "X-API-Key: $BOB_KEY")
if [[ "$status" != "403" ]]; then
  echo "expected 403 from bob, got $status" >&2
  exit 1
fi
echo "       bob -> alice's doc: 403 OK"

# alice DELETE own → 200
status=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
              "$BASE/documents/smoke-alice" -H "X-API-Key: $ALICE_KEY")
if [[ "$status" != "200" ]]; then
  echo "expected 200 from alice, got $status" >&2
  exit 1
fi
echo "       alice -> own doc: 200 OK"

# admin DELETE null-owner legacy doc → 200
status=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
              "$BASE/documents/smoke-legacy" -H "X-API-Key: $ADMIN_KEY")
if [[ "$status" != "200" ]]; then
  echo "expected 200 from admin, got $status" >&2
  exit 1
fi
echo "       admin -> null-owner doc: 200 OK"

echo
echo "Smoke PASS"
