#!/usr/bin/env bash
SRC=/mnt/c/Users/11541/Desktop/projects/navikb/deploy/local-48g/milco-650m-dl
DST=/home/tiantian/navikb-serving/models/milco-650m
echo "=== source (Windows dl dir) ==="
ls -la "$SRC" 2>&1 | grep -vE '^total|\.cache'
rm -rf "$DST" /home/tiantian/navikb-serving/models/milco-650m-ms
mkdir -p "$DST"
for f in config.json milco.py pytorch_model.bin sentencepiece.bpe.model special_tokens_map.json tokenizer.json tokenizer_config.json; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"
done
echo "=== installed (WSL) ==="
ls -la "$DST" 2>&1 | grep -vE '^total'
echo "INSTALL_MILCO_DONE"
