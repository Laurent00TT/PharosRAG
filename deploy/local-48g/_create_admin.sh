#!/usr/bin/env bash
MC="$HOME/miniconda3"; REPO="/mnt/c/Users/11541/Desktop/projects/navikb"
cd "$HOME/navikb-serving/runtime"
export PYTHONPATH="$REPO/src"
"$MC/bin/conda" run -n navikb --no-capture-output \
  python "$REPO/scripts/manage_users.py" create admin --role admin
