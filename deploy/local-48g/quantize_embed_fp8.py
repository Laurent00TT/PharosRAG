#!/usr/bin/env python
"""Quantize Qwen3-VL-Embedding-8B (BF16) -> FP8_DYNAMIC for vLLM (gating #2).

We download the official BF16 from modelscope and quantize locally because the
community FP8 repo is HF-only and hf-mirror's file CDN is unreachable via the TUN.
Ignore the LM head + the entire visual tower (keep them BF16) — the community
embed-FP8 recipe (RamManavalan) does exactly this to preserve recall.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python quantize_embed_fp8.py
"""
import os, shutil, sys
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
SRC = HOME / "navikb-serving/models/qwen3-vl-embed-8b"
DST = HOME / "navikb-serving/models/qwen3-vl-embed-8b-fp8"
IGNORE = ["lm_head", r"re:model\.visual\..*"]

def main():
    from transformers import AutoModel, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if not SRC.exists():
        sys.exit(f"source model not found: {SRC} (download with 02_download_models.sh embed)")

    print(f"loading {SRC} ...", flush=True)
    model = AutoModel.from_pretrained(str(SRC), torch_dtype="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(str(SRC), trust_remote_code=True)

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

    # chat_template.jinja is REQUIRED at serve time (vision channel) — copy all
    # non-weight aux files the FP8 dir is missing.
    for f in SRC.iterdir():
        if f.suffix in {".safetensors", ".bin"}:
            continue
        target = DST / f.name
        if f.is_file() and not target.exists():
            shutil.copy2(f, target)
            print("  copied", f.name)

    print("QUANTIZE_EMBED_DONE")

if __name__ == "__main__":
    main()
