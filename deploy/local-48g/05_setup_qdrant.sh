#!/usr/bin/env bash
# NaviKB local-48G — Qdrant server binary (no docker, no sudo).
# all-online REQUIRES server mode (local-file Qdrant is single-writer -> worker
# and tool_server cannot both run). Binary -> ~/navikb-serving/qdrant/.
set -euo pipefail
QV="${QDRANT_VERSION:-v1.12.0}"
DEST="$HOME/navikb-serving/qdrant"; mkdir -p "$DEST"
if [ -x "$DEST/qdrant" ]; then echo "qdrant already present: $("$DEST/qdrant" --version 2>&1)"; exit 0; fi
URL="https://github.com/qdrant/qdrant/releases/download/${QV}/qdrant-x86_64-unknown-linux-gnu.tar.gz"
echo "downloading qdrant ${QV} (curl + retries) ..."
curl -fSL --retry 12 --retry-delay 3 --connect-timeout 15 -o /tmp/qdrant.tar.gz "$URL"
tar -xzf /tmp/qdrant.tar.gz -C "$DEST"
"$DEST/qdrant" --version 2>&1 || true
echo "SETUP_QDRANT_DONE"
