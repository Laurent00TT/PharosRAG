#!/usr/bin/env bash
# NaviKB local-48G — ISOLATED quantization env (llmcompressor).
# MUST be separate from the vllm env: llmcompressor pins transformers<5 +
# compressed-tensors 0.16, which conflict with vllm 0.22.1 (transformers 5.10 /
# compressed-tensors 0.15.0.1). Used only to produce FP8 checkpoints.
set -euo pipefail
MC="$HOME/miniconda3"; CONDA="$MC/bin/conda"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
export PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=120
CF="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
if ! "$CONDA" env list | grep -q "/envs/quant$"; then
  "$CONDA" create -y -n quant --override-channels -c "$CF" python=3.12
fi
"$CONDA" run -n quant python -m pip install --upgrade pip
"$CONDA" run -n quant python -m pip install llmcompressor
"$CONDA" run -n quant python -c "import llmcompressor, transformers; print('llmcompressor', llmcompressor.__version__, 'transformers', transformers.__version__)"
echo "SETUP_QUANT_DONE"
