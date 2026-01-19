# Zeroshot_forecasting

Directional trading experiments on BTC price data using zero-shot forecasters
 (TimesFM, Chronos/Chronos2, Lag-Llama, Moirai, TTM, MOMENT, TOTO). Forecasts
 are converted to long/short/flat signals and evaluated with a shared backtest
 and metrics toolset.

## Repo Layout
- `models/` – runners and sweep scripts for each model family plus notebooks
  (`timesfm_runner.py`, `chronos_runner.py`, `chronos2_runner.py`,
  `lagllama_runner.py`, `moirai_runner.py`, `ttm_runner.py`, `moment_runner.py`,
  `toto_runner.py`, `sweep_*.py`).
- `utils/` – shared helpers for metrics, plotting, validation, and ad-hoc
  evaluation (`adhoc_eval*.py`, `metrics.py`, `plotting.py`).
- `data/`, `preprocessing/` – BTC datasets and prep notebooks/scripts.
- `timesfm-results/`, `chronos-results/`, `chronos2-results/`, `ttm-results/`,
  `lagllama-results/`, `moirai-results/`, `moment-results/`, `toto-results/` –
  saved runs and per-run artifacts (`*_step1.csv`, `*_wide.csv`, `metrics.json`,
  `summary_row.csv`, plots).
- `models_comparison/` – aggregation scripts, leaderboards, and plots.
- `external/chronos-forecasting/`, `timesfm/`, `lag-llama/`, `moirai/` – upstream
  model code used by the runners.
- `TOTO/` – upstream Toto model code used by the runner.

## Running Experiments
- Most runner/sweep scripts contain user-specific defaults (absolute paths) near
  the top; edit those or pass CLI args where available.
- Example sweeps:
  ```bash
  python3 models/sweep_timesfm.py
  python3 models/sweep_chronos.py
  python3 models/sweep_chronos2.py
  python3 models/sweep_ttm.py
  python3 models/sweep_lagllama.py
  python3 models/sweep_moirai.py
  python3 models/sweep_moment.py
  python3 models/sweep_toto.py
  ```
- Outputs are written to the matching `*-results/` folder with a per-run
  `summary_row.csv` and a `summary_ctx*.csv` aggregate.

## Ad-hoc Evaluation
- TimesFM: `python3 utils/adhoc_eval.py`
- Chronos: `python3 utils/adhoc_eval_chronos.py`
- Lag-Llama: `python3 utils/adhoc_eval_lagllama.py`
- TTM: `python3 utils/adhoc_eval_ttm.py`

## Comparing Runs
Aggregate Chronos/Chronos2/TimesFM/TTM/Lag-Llama/Moirai/MOMENT/TOTO runs into leaderboards and plots:
```bash
python3 models_comparison/compare_results.py --metric final_equity_tc --top-k 3
```
Outputs include `models_comparison/combined_results.csv`,
`models_comparison/leaderboard_*.csv`, and plots under `models_comparison/plots/`.

## Notes
- Backtest logic lives in `utils/metrics.py` (execution lag, transaction costs,
  ARC/ASD/MDD/IR calculations).
