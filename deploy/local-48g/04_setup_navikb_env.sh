#!/usr/bin/env bash
# NaviKB local-48G — control-plane conda env (serve.py + worker.py).
# Separate env: navikb pins transformers<5.0 / torch<2.9 (conflicts with vllm env).
# Runs CPU-only (it calls the model servers over HTTP; no local GPU model).
set -euo pipefail
MC="$HOME/miniconda3"; CONDA="$MC/bin/conda"; REPO="/mnt/c/Users/11541/Desktop/projects/navikb"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
export PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=120
CF="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

if ! "$CONDA" env list | grep -q "/envs/navikb$"; then
  "$CONDA" create -y -n navikb --override-channels -c "$CF" python=3.12
fi
"$CONDA" run -n navikb python -m pip install --upgrade pip
# editable install with the FastAPI/MCP server extras (NOT [mineru] — mineru has
# its own env). torch resolves to the tuna-mirrored CUDA wheel; control plane
# never moves tensors to GPU so it stays at 0 VRAM.
"$CONDA" run -n navikb python -m pip install -e "${REPO}[server]"
"$CONDA" run -n navikb python -c "import kb, fastapi, qdrant_client; print('navikb control-plane import OK')"
echo "SETUP_NAVIKB_DONE"
