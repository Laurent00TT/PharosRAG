#!/usr/bin/env python
"""Quantize Qwen3-VL-Reranker-8B (BF16) -> FP8_DYNAMIC for vLLM (gating #1).

FP8_DYNAMIC = weights static per-channel + activations dynamic per-token; data-free
(no calibration set). Run inside the `vllm` conda env on the 4090.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python quantize_rerank_fp8.py

Notes / risk: the reranker is a pooling model with a custom
Qwen3VLForSequenceClassification head — whether llmcompressor loads it and whether
the FP8 result serves under vLLM is the highest-risk step of the whole plan. We
print(model) first so the ignore patterns can be verified against actual module names.
"""
import os, shutil, sys
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
SRC = HOME / "navikb-serving/models/qwen3-vl-rerank-8b"
DST = HOME / "navikb-serving/models/qwen3-vl-rerank-8b-fp8"

# Keep the visual tower + LM head + any score/classifier head in BF16 (quantizing
# them hurts recall / breaks the head). Verified against print(model) below.
IGNORE = ["lm_head", r"re:model\.visual\..*", r"re:.*\.score.*", r"re:.*classifier.*"]

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if not SRC.exists():
        sys.exit(f"source model not found: {SRC} (download with 02_download_models.sh rerank)")

    print(f"loading {SRC} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(SRC), torch_dtype="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(str(SRC), trust_remote_code=True)

    # Dump top-level module names so IGNORE can be confirmed / corrected.
    print("=== top-level modules (verify IGNORE patterns) ===", flush=True)
    for name, _ in model.named_modules():
        if name.count(".") <= 1:
            print("  ", name)

    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    print("running oneshot FP8_DYNAMIC ...", flush=True)
    oneshot(model=model, recipe=recipe)

    DST.mkdir(parents=True, exist_ok=True)
    print(f"saving -> {DST}", flush=True)
    model.save_pretrained(str(DST))
    tok.save_pretrained(str(DST))

    # Copy aux files vLLM/trust_remote_code needs but save_pretrained may skip
    # (chat template, custom modeling .py, preprocessor config, jinja).
    for f in SRC.iterdir():
        if f.suffix in {".safetensors", ".bin"}:
            continue
        target = DST / f.name
        if f.is_file() and not target.exists():
            shutil.copy2(f, target)
            print("  copied", f.name)

    print("QUANTIZE_RERANK_DONE")

if __name__ == "__main__":
    main()
