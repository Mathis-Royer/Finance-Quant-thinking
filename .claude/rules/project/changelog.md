# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 1 | 2026-02-01 | **Refactored benchmarks for factor allocation relevance**: Removed Market (Mkt-RF) benchmark (not relevant for factor timing). Added 3 new benchmarks: (1) **Risk Parity** (inverse volatility weighting), (2) **Factor Momentum** (allocate to positive trailing 12M factors), (3) **Best Single Factor** (100% in best in-sample Sharpe factor). Kept Equal-Weight 6F and 50/50 Cyc/Def. Updated `holdout_plots.py` color scheme. |
| 2 | 2026-02-01 | **Improved holdout visualizations with benchmarks**: (1) Bar chart split into 2×2 grid (Sup, E2E, Multi, Binary) sorted by max(Sharpe), (2) Added 3rd heatmap showing Delta (Ensemble - Final), (3) New cumulative returns chart with Top 3 models vs benchmarks (solid lines) and benchmarks (dashed), (4) Combined summary table with Top 3 models + benchmarks. |
| 3 | 2026-02-01 | **Simplified notebook using externalized modules**: Reduced notebook from ~1400 lines to ~200 lines. Created `src/visualization/` (walk_forward_plots.py, holdout_plots.py, colormaps.py) and `src/pipelines/three_step_pipeline.py` (ThreeStepEvaluation class). Notebook now uses `plot_all_walk_forward()`, `plot_all_holdout()`, and `ThreeStepEvaluation` for clean orchestration. |
| 4 | 2026-02-01 | **Added composite score to all summary tables**: Score = 0.4×Sharpe + 0.3×IC + 0.3×MaxDD (normalized). Added `compute_composite_score()` function. Updated walk-forward summary, holdout summary, and returns summary to display Sharpe, IC, MaxDD, Score, and Rank columns. |
| 5 | 2026-02-01 | **Redesigned walk-forward visualizations**: (1) 4 heatmaps (2×2) showing returns by year × horizon per combination, (2) Cumulative returns split into 2×2 grid instead of 16 overlapping lines, (3) Bar chart of total returns by combination, (4) Bar chart of maximum drawdown by combination. Summary table now includes TOTAL and MAX DD columns, plus Top 3 rankings. |
| 6 | 2026-01-31 | **Fixed Binary mode horizon differentiation**: Binary backtest now uses prediction probability as weights (`pred * cyclical + (1-pred) * defensive`) instead of hard threshold (`>0.5`). This allows different horizons to produce different results because prediction magnitude matters, not just direction. Fixed in `main_strategy.py` and `comparison_runner.py`. |
| 7 | 2026-01-31 | **Fixed horizon seed bug**: Seed was `42 + window_id`, causing all horizons (1M, 3M, 6M, 12M) to have identical initialization → identical Binary predictions. Fixed: seed now includes horizon (`42 + i*100 + horizon`). Also added walk-forward visualizations (heatmaps, cumulative returns by year). |
| 8 | 2026-01-31 | **All 16 combinations through 3 steps**: Steps 2-3 now loop through all 16 combos (not just best). Holdout period based on actual data range (fixes 0-sample bug for longer horizons). `RUN_TUNING=False` and `USE_FEATURE_SELECTION=False` by default. |
| 9 | 2026-01-31 | **Rigorous 3-step methodology**: (1) Walk-forward 2017-2022 with holdout, (2) Final model on 2000-holdout, (3) Holdout evaluation. Added `train_final_model()`, `evaluate_on_holdout()`, `ensemble_predict()`, `compare_final_vs_ensemble()`. |
| 10 | 2026-01-31 | **Walk-forward HP tuning**: Added `hyperparameter_tuning.py` with `WalkForwardTuner`. Bayesian optimization (Optuna) or random search fallback. Tunes lr, dropout, weight_decay, epochs per window. |

---

## Current State

Factor allocation strategy using **Point-in-Time FRED-MD** macroeconomic data with **MICRO Transformer** model.

- **Status**: Development/Research
- **Model**: MICRO Transformer only (12k params, d_model=32, 1 layer, 1 head)
- **Evaluation Methodology** (rigorous 3-step):
  1. **Walk-forward (2017-2021)**: N models, non-overlapping test periods, holdout reserved
  2. **Final model**: Trained on 2000-2021 (all data except holdout) for production
  3. **Holdout comparison (2022+)**: All 16 combos evaluated (Final vs Ensemble), **fixed start date** for all horizons
- **Defaults**: `RUN_TUNING=False`, `USE_FEATURE_SELECTION=False`
- **Main features**:
  - Point-in-time data loading exclusively (305 FRED-MD vintages)
  - 6-factor allocation (cyclical, defensive, value, growth, quality, momentum)
  - 3-phase training (Binary → Regression → Sharpe optimization)
  - **Multi-horizon support**: 4 horizons (1M, 3M, 6M, 12M)
  - **Feature selection & PCA**: Reduce 112 indicators via mutual info or PCA
  - **Walk-forward HP tuning**: Auto-tune hyperparameters per window (Optuna/random search)
  - **Model persistence**: Walk-forward models saved for ensemble
- **Key finding**: Multi-factor (6F) outperforms Binary (2F)
- **Notebook**: Simplified ~200 lines using externalized modules (was ~1400 lines)
- **New modules**: `src/visualization/` (plots), `src/pipelines/` (ThreeStepEvaluation)
- **Visualizations**: 4 heatmaps (return × year × horizon), 2×2 cumulative returns grid, total returns bar chart, max drawdown bar chart, holdout comparison with Delta heatmap
- **Benchmarks**: Equal-Weight 6F, 50/50 Cyc/Def, Risk Parity, Factor Momentum, Best Single Factor
- **Composite scoring**: All summary tables include Score = 0.4×Sharpe + 0.3×IC + 0.3×MaxDD (normalized) and Rank columns

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
