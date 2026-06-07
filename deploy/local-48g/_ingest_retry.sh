#!/usr/bin/env bash
# Retry ingest: first attempt loads the mineru model (may client-timeout or 503
# while busy); the model stays resident in mineru_server, so a later attempt is
# fast and succeeds. Pipeline parse mode (set in .env) avoids the hybrid spawn deadlock.
for i in $(seq 1 15); do
  echo "=== ingest attempt $i ($(date +%H:%M:%S)) ==="
  bash /mnt/c/Users/11541/Desktop/projects/navikb/deploy/local-48g/_ingest.sh > /tmp/ing_$i.log 2>&1
  if grep -qE "chunks stored: [1-9]" /tmp/ing_$i.log; then
    echo "INGEST_OK on attempt $i"
    grep -E "chunks stored|Stored|nav|pages" /tmp/ing_$i.log | tail -6
    exit 0
  fi
  if grep -q "503" /tmp/ing_$i.log; then echo "  mineru busy (503) — model still loading/parsing, wait 25s"; sleep 25; continue; fi
  if grep -qiE "timeout|ReadError|ConnectError|RemoteProtocol" /tmp/ing_$i.log; then echo "  client timeout (model loading) — wait 30s, model stays resident"; sleep 30; continue; fi
  echo "  other error:"; tail -5 /tmp/ing_$i.log; sleep 15
done
echo "INGEST_GAVE_UP"; tail -10 /tmp/ing_15.log 2>/dev/null
