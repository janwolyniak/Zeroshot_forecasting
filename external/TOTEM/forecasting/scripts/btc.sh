#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON:-python}"
echo "[FORECAST] Running with autodetected flags:"
echo ""$PY" "/Users/jan/Documents/working papers/project 1/external/TOTEM/process_zero_shot_data/forecasting_saugeen_sun_births.py" --base_path "/Users/jan/Documents/working papers/project 1/data/totem_zero_shot/btc/btc_tsf/saugeen.tsf" --save_path "/Users/jan/Documents/working papers/project 1/outputs/totem/btc""
"$PY" "/Users/jan/Documents/working papers/project 1/external/TOTEM/process_zero_shot_data/forecasting_saugeen_sun_births.py" --base_path "/Users/jan/Documents/working papers/project 1/data/totem_zero_shot/btc/btc_tsf/saugeen.tsf" --save_path "/Users/jan/Documents/working papers/project 1/outputs/totem/btc"
echo "[FORECAST] Done."
