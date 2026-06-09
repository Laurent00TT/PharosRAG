#!/usr/bin/env bash
# conda-forge gcc installs only the prefixed `x86_64-conda-linux-gnu-gcc`; Triton
# JIT looks for plain `gcc`/`cc` on PATH. Symlink them into the env bin.
set -e
B=/home/tiantian/miniconda3/envs/vllm/bin
GCC=$(ls "$B"/*-linux-gnu-gcc 2>/dev/null | head -1)
GXX=$(ls "$B"/*-linux-gnu-g++ 2>/dev/null | head -1)
echo "GCC=$GCC"
echo "GXX=$GXX"
[ -n "$GCC" ] && ln -sf "$GCC" "$B/gcc" && ln -sf "$GCC" "$B/cc"
[ -n "$GXX" ] && ln -sf "$GXX" "$B/g++" && ln -sf "$GXX" "$B/c++"
echo "--- verify ---"
"$B/gcc" --version | head -1
"$B/cc" --version | head -1
echo "SETUP_CC_DONE"
