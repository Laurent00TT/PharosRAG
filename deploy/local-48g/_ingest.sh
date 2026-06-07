#!/usr/bin/env bash
MC="$HOME/miniconda3"; REPO="/mnt/c/Users/11541/Desktop/projects/navikb"
cd "$HOME/navikb-serving/runtime"
export PYTHONPATH="$REPO/src"
PDF="${1:-/mnt/c/Users/11541/Desktop/projects/knowledge-base/datasets/mmdocir/extracted/doc_pdfs/a4f3ced0696009fec3179f493e4f28c4.pdf}"
echo "ingesting: $PDF"
exec "$MC/bin/conda" run -n navikb --no-capture-output python "$REPO/scripts/ingest.py" --path "$PDF"
