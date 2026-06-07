#!/usr/bin/env bash
# Qdrant server (REST 6333 / gRPC 6334), storage on WSL ext4.
set -euo pipefail
export QDRANT__STORAGE__STORAGE_PATH="$HOME/navikb-serving/qdrant_storage"
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334
mkdir -p "$QDRANT__STORAGE__STORAGE_PATH"
exec "$HOME/navikb-serving/qdrant/qdrant"
