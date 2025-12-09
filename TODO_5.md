# Research plan: 17.09 → 29.09

Legend:  
- **[M]** = Must-have (core deliverable — absolutely needed to finish cleanly).  
- **[S]** = Should-have (important, but could be simplified if pressed).  
- **[N]** = Nice-to-have (stretch goal if time allows).

---

## 17.09 — Metrics & plotting foundation
- [M] Finalize `utils/metrics.py` with:  
  `carry_forward_positions`, `equity_line_prevbar`, `equity_line_samebar`, `equity_line_tc_samebar`, `directional_accuracy_trading`, `directional_accuracy_model`.
- [M] Add `utils/plotting.py` with `plot_equity`.
- [M] Unit tests in `tests/test_metrics.py`.
- [M] Retrofit **TTM** with utils.  
- [S] Save “golden outputs” for regression testing.

## 18.09 — ASD module
- [M] Implement `compute_asd` in `utils/asd.py` + test.
- [M] Integrate into `btc_preprocessing.ipynb`.
- [S] Add `plot_asd` to plotting.

## 19.09 — Retrofit existing pipelines
- [M] Replace inline code in **TTM** and **TimesFM** with utils.
- [M] Add equity (with & without TC), logy plots.
- [S] Confirm outputs match and document differences.

## 20.09 — Look-ahead bias guardrails I
- [M] Add `utils/validation.py` with `assert_no_lookahead`.
- [M] Write failing + passing test.
- [S] Add leakage sentinel checks into notebooks.

## 21.09 — New model pipeline 1 (MOMENT)
- [M] Scaffold full pipeline with utils.
- [M] Confirm guardrails pass.

## 22.09 — New model pipeline 2 (Moirai)
- [M] Scaffold pipeline with configs.

## 23.09 — New model pipeline 3 (Lag-Llama)
- [M] Scaffold pipeline (stub if install fails).

## 24.09 — Look-ahead bias guardrails II
- [M] Add explicit cutoff logic in all pipelines.
- [S] Add smoke test: shift prices → DA collapses.
- [S] Document timing contract per pipeline.

## 25.09 — Hyperparameter menus
- [M] Draft `docs/hparam_menus.md` with ranges.
- [S] YAML sweep configs.

## 26.09 — Unified comparison runner
- [M] `scripts/run_all_models.py` (BTC slice → all models → metrics/plots).
- [S] `scripts/metrics_runner.py` (external CSV runner).

## 27.09 — Robustness passes & ablations
- [S] Re-run on 1h + 15m slices.  
- [N] Ablations: cost levels, thresholds.  
- [N] Write `docs/robustness.md`.

## 28.09 — Docs & reproducibility polish
- [M] Update `README.md` (usage + layout).  
- [M] Parameter cells in notebooks.  
- [S] Pin environments.  
- [N] Fresh-clone reproducibility test.

## 29.09 — Final run & summary
- [M] Run all pipelines full BTC.  
- [M] Export key plots/tables.  
- [M] Write `docs/summary_2909.md`.  
- [S] Package `artifacts_2909/`.

---

### If time is short
- **Always do [M]** → you’ll end up with working pipelines, guardrails, and final results.  
- **Do [S]** if pace allows → improves robustness and usability.  
- **[N]** can be dropped without hurting core outcomes.  
