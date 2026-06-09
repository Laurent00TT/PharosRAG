#!/usr/bin/env bash
set -e
SRC=/mnt/c/Users/11541/Desktop/projects/navikb/deploy/local-48g/splade-v3-dl
export SPL=/home/tiantian/navikb-serving/models/splade-v3
export MILCO=/home/tiantian/navikb-serving/models/milco-650m
rm -rf "$SPL"; mkdir -p "$SPL"
for f in config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$SPL/$f"
done
echo "=== splade-v3 tokenizer files (WSL) ==="; ls "$SPL"
# point MILCO's two sub-tokenizer checkpoints at local paths (offline load)
python3 - <<'PY'
import json, os
p = os.path.join(os.environ["MILCO"], "config.json")
c = json.load(open(p))
c["lsr_encoder_checkpoint"] = os.environ["SPL"]      # splade-v3 (en, BERT vocab)
c["multilingual_encoder_checkpoint"] = os.environ["MILCO"]  # bge-m3 tokenizer ships inside MILCO
json.dump(c, open(p, "w"), indent=2)
print("lsr_encoder_checkpoint   ->", c["lsr_encoder_checkpoint"])
print("multilingual_encoder...  ->", c["multilingual_encoder_checkpoint"])
PY
echo "SPARSE_FIX_DONE"
