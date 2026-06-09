#!/usr/bin/env bash
# NaviKB local-48G bring-up — Phase 0: Miniconda + vLLM conda env
# Run inside WSL2 Ubuntu. NO sudo required. Idempotent (safe to re-run).
set -euo pipefail

MC="$HOME/miniconda3"
CONDA="$MC/bin/conda"

# China mirrors for pip (inherited by all `conda run ... pip` subprocesses).
# tuna confirmed reachable (200); aliyun timed out via the host TUN proxy.
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
export PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=120   # ride out the flaky TUN proxy

echo "[1/5] Miniconda"
if [ ! -x "$CONDA" ]; then
  echo "  downloading miniconda (tsinghua, curl + retries) ..."
  curl -fSL --retry 8 --retry-delay 3 --connect-timeout 15 \
    -o /tmp/miniconda.sh \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$MC"
fi
"$CONDA" --version

# tuna conda-forge FULL URL — used directly in -c so it never resolves to the
# default conda.anaconda.org (which times out via the TUN proxy + has a ToS gate).
CF="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

echo "[2/5] conda config -> tuna mirror, disable notices"
"$CONDA" config --set show_channel_urls yes || true
"$CONDA" config --set number_channel_notices 0 || true
"$CONDA" config --set channel_alias https://mirrors.tuna.tsinghua.edu.cn/anaconda || true
"$CONDA" config --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud || true
"$CONDA" config --remove channels defaults 2>/dev/null || true

echo "[3/5] create env 'vllm' (python 3.12 from tuna conda-forge)"
if ! "$CONDA" env list | grep -q "/envs/vllm$"; then
  "$CONDA" create -y -n vllm --override-channels -c "$CF" python=3.12
fi

echo "[4/5] pip install vllm (big: pulls torch + cuda runtime)"
"$CONDA" run -n vllm python -m pip install --upgrade pip
"$CONDA" run -n vllm python -m pip install vllm

echo "[5/5] verify torch sees the GPUs + vllm imports"
"$CONDA" run -n vllm python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(), "| n_devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {round(p.total_memory/1024**3,1)}GB sm{p.major}{p.minor}")
print("vllm", vllm.__version__)
import transformers
print("transformers", transformers.__version__)
PY
echo "SETUP_VLLM_DONE"
