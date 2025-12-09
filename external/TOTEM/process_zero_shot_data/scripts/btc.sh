#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON:-python}"
TSF="/Users/jan/Documents/working papers/project 1/data/totem_zero_shot/btc/btc_tsf/saugeen.tsf"
SAVE="/Users/jan/Documents/working papers/project 1/data/totem_zero_shot/btc/data"
echo "[PROCESS] $TSF -> $SAVE"
"$PY" "/Users/jan/Documents/working papers/project 1/external/TOTEM/process_zero_shot_data/process_saugeen_sun_births.py" --base_path "$TSF" --save_path "$SAVE"
echo "[PROCESS] Done."
