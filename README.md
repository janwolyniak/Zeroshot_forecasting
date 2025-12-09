# Zeroshot_forecasting

Directional trading experiments on BTC price data using TimesFM and several baselines (TTM, TOTEM, MOMENT, Moirai, Lag-Llama). Forecasts are converted to long/short/flat signals and evaluated with a consistent backtest/metrics toolset.

## Repo Layout
- `models/` – notebooks and scripts to generate forecasts for each model family (TimesFM, TTM, TOTEM, MOMENT, Moirai, Lag-Llama).
- `utils/` – shared helpers for metrics, plotting, validation, and ad-hoc evaluation (`adhoc_eval.py`, `metrics.py`, etc.).
- `timesfm-results/` – saved TimesFM runs (per-context/horizon folders) plus summary CSVs; `_adhoc_eval/` holds custom threshold experiments.
- `data/`, `preprocessing/`, `outputs/` – local data, prep steps, and generated artifacts (not all checked in).
- `TODO_5.md` – current research plan and milestone checklist.

## Latest Work: `_adhoc_eval` (TimesFM Step-1 trades)
Ad-hoc backtests on TimesFM step-1 predictions with different thresholds (starting capital 100k, fee_rate 0.001, no TC applied to the equity curves below):
- `timesfm-results/_adhoc_eval/ctx512_h1_thr0.00000_notc` – zero threshold; very frequent trading (~4.1k trades). Final equity (no TC): **433,770**; with TC: **114** due to ~155k fees and near-50/50 long/short time.
- `timesfm-results/_adhoc_eval/ctx512_h1_thr0.00500_notc` – 0.5% threshold; trades drop to ~2.6k with ~77% flat time. Final equity (no TC): **191,717**; with TC: **10,394**; fees ~125k. Risk metrics: ARC 1.05, ASD 0.46, MDD 0.18, IR* 2.30.
- `timesfm-results/_adhoc_eval/ctx2048_h1_thr0.00500_notc` – larger context, same threshold. Final equity (no TC): **186,832**; with TC: **9,294**; trades ~2.6k, ~76% flat time; fees ~123k. Risk metrics: ARC 0.99, ASD 0.46, MDD 0.22, IR* 2.15.
Artifacts per run: `adhoc_metrics.json`, `adhoc_equity.csv`, and `adhoc_equity_notc.png`.

## Running Ad-hoc Evaluation
1. Ensure the target TimesFM run exists under `timesfm-results/ctx{CTX}_h{H}_norm...` with `timesfm_step1.csv` and `timesfm_wide.csv`.
2. Set the parameters near the top of `utils/adhoc_eval.py` (`CTX`, `H`, `THRESHOLD`, `USE_TC`).
3. Run from repo root:
   ```bash
   python3 utils/adhoc_eval.py
   ```
4. Results are written to `timesfm-results/_adhoc_eval/ctx{CTX}_h{H}_thr{THRESHOLD}_{tc|notc}/`.

## Notes
- Metrics/backtest logic lives in `utils/metrics.py` (handles execution lag, transaction costs, ARC/ASD/MDD/IR calculations).
- Full TimesFM sweep summaries are in `timesfm-results/summary_ctx*.csv` if you need the broader context/horizon grid.
