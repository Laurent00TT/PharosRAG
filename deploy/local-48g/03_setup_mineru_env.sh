#!/usr/bin/env bash
# NaviKB local-48G — Phase: create the isolated `mineru` conda env.
# mineru pins transformers~=4.57 / torch~=2.8 which conflict with the vllm env,
# so it MUST live in its own env (deployment-guide §3.2).
set -euo pipefail
MC="$HOME/miniconda3"; CONDA="$MC/bin/conda"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
export PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=120

CF="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
if ! "$CONDA" env list | grep -q "/envs/mineru$"; then
  "$CONDA" create -y -n mineru --override-channels -c "$CF" python=3.12
fi
"$CONDA" run -n mineru python -m pip install --upgrade pip
# hybrid backend needs BOTH pipeline + vlm extras (pyproject note)
"$CONDA" run -n mineru python -m pip install "mineru[pipeline,vlm]>=3.1.0" fastapi uvicorn python-multipart pypdfium2
"$CONDA" run -n mineru python -c "import mineru; print('mineru ok')"
echo "SETUP_MINERU_DONE"
