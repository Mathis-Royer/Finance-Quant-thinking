# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 1 | 2026-02-01 | **Enhanced training with regularization and score improvements**: (1) **New EarlyStopping + ModelCheckpoint** in `src/utils/training_utils.py`, (2) **Improved composite score**: asymmetric IC penalty (negative IC penalized 2x), exponential MaxDD penalty `exp(3×maxdd)`, rejection of models with IC < -30%, (3) **Fixed SharpeRatioLoss**: true Sharpe ratio `-mean/std` with running std for gradient stability (was approximation `-mean + γ×var`), (4) **Enhanced regularization**: dropout 0.6→0.75, weight_decay 0.01→0.05, batch_size 32→64, learning_rate 0.001→0.0005, (5) **Increased rolling window** for Supervised optimal weights: 12→24 months. **All modes kept**: Binary, E2E, Sup, Multi, Fair Ensemble. |
| 2 | 2026-02-01 | **Fixed percentage formatting issues**: (1) **Total Return double conversion fixed**: removed `*100` in `comparison_runner.py` (now stored as decimal like 0.0835 for 8.35%). Removed `/100` in `export_results.py` and `holdout_plots.py`. (2) **Sharpe ratio no longer displayed as percentage**: Sharpe is a ratio (e.g., 0.85, 1.20), not a percentage. Removed all `*100` and `%` formatting for Sharpe across `comparison_runner.py`, `holdout_plots.py`, and `app.py`. Now displays as plain number with 2 decimals. (3) IC, MaxDD, Total Return, and Score remain as percentages. |
| 2 | 2026-02-01 | **Added Streamlit dashboard for interactive results exploration**: New `dashboard/app.py` with interactive filters (Strategy, Allocation, Type, Horizon), adjustable score weights, results table with averages, multiple visualizations (bar charts, heatmaps, scatter plots), pivot tables, and benchmark comparisons. Run with `streamlit run dashboard/app.py`. Added `dashboard/export_results.py` for caching results. |
| 3 | 2026-02-01 | **Centralized notebook imports with reload function**: New imports cell at top of notebook with all module imports and `reload_all()` function. Removed duplicate imports from other cells. After modifying source code, call `reload_all()` to refresh modules without kernel restart. |
| 4 | 2026-02-01 | **Added Fair Ensemble for unbiased comparison**: New `train_fair_ensemble_models()` trains N models on same data (2000-2021) with different seeds, eliminating data quantity bias in WF Ensemble. Added `run_bias_analysis()` to decompose effects (data quantity, seed variance, pure ensemble). Step 3 now evaluates **Final vs Fair Ensemble vs WF Ensemble**. Fair Ensemble is the recommended production ensemble. |
| 5 | 2026-02-01 | **Refactored benchmarks for factor allocation relevance**: Removed Market (Mkt-RF) benchmark (not relevant for factor timing). Added 3 new benchmarks: (1) **Risk Parity** (inverse volatility weighting), (2) **Factor Momentum** (allocate to positive trailing 12M factors), (3) **Best Single Factor** (100% in best in-sample Sharpe factor). Kept Equal-Weight 6F and 50/50 Cyc/Def. Updated `holdout_plots.py` color scheme. |
| 6 | 2026-02-01 | **Improved holdout visualizations with benchmarks**: (1) Bar chart split into 2×2 grid (Sup, E2E, Multi, Binary) sorted by max(Sharpe), (2) Added 3rd heatmap showing Delta (Ensemble - Final), (3) New cumulative returns chart with Top 3 models vs benchmarks (solid lines) and benchmarks (dashed), (4) Combined summary table with Top 3 models + benchmarks. |
| 7 | 2026-02-01 | **Simplified notebook using externalized modules**: Reduced notebook from ~1400 lines to ~200 lines. Created `src/visualization/` (walk_forward_plots.py, holdout_plots.py, colormaps.py) and `src/pipelines/three_step_pipeline.py` (ThreeStepEvaluation class). Notebook now uses `plot_all_walk_forward()`, `plot_all_holdout()`, and `ThreeStepEvaluation` for clean orchestration. |
| 8 | 2026-02-01 | **Added composite score to all summary tables**: Score = 0.4×Sharpe + 0.3×IC + 0.3×MaxDD (normalized). Added `compute_composite_score()` function. Updated walk-forward summary, holdout summary, and returns summary to display Sharpe, IC, MaxDD, Score, and Rank columns. |
| 9 | 2026-01-31 | **Fixed Binary mode horizon differentiation**: Binary backtest now uses prediction probability as weights (`pred * cyclical + (1-pred) * defensive`) instead of hard threshold (`>0.5`). This allows different horizons to produce different results because prediction magnitude matters, not just direction. Fixed in `main_strategy.py` and `comparison_runner.py`. |
| 10 | 2026-01-31 | **Fixed horizon seed bug**: Seed was `42 + window_id`, causing all horizons (1M, 3M, 6M, 12M) to have identical initialization → identical Binary predictions. Fixed: seed now includes horizon (`42 + i*100 + horizon`). Also added walk-forward visualizations (heatmaps, cumulative returns by year). |

---

## Current State

Factor allocation strategy using **Point-in-Time FRED-MD** macroeconomic data with **MICRO Transformer** model.

- **Status**: Development/Research
- **Model**: MICRO Transformer only (12k params, d_model=32, 1 layer, 1 head)
- **Pipeline**: 16 combinations (2 strategies × 2 allocations × 4 horizons)
  - Strategies: E2E (3-phase), Supervised
  - Allocations: Binary (2F), Multi (6F)
  - Horizons: 1M, 3M, 6M, 12M
- **Evaluation Methodology** (rigorous 3-step):
  1. **Walk-forward (2017-2021)**: N models, non-overlapping test periods, holdout reserved
  2. **Final model**: Trained on 2000-2021 (all data except holdout) for production
  3. **Holdout comparison (2022+)**: **Final vs Fair Ensemble vs WF Ensemble**
     - **Fair Ensemble**: 5 models on same data (2000-2021), different seeds
     - **WF Ensemble**: N models from walk-forward (biased: different data quantities)
- **Enhanced regularization** (reduces overfitting):
  - `dropout=0.75` (was 0.6)
  - `weight_decay=0.05` (was 0.01)
  - `learning_rate=0.0005` (was 0.001)
  - `batch_size=64` (was 32)
  - **Early stopping** enabled (patience=5)
  - **Rolling window**: 24 months (was 12) for Supervised optimal weights
- **Improved training**:
  - **SharpeRatioLoss**: True Sharpe ratio `-mean/std` with running std for gradient stability
  - **EarlyStopping + ModelCheckpoint**: Auto-stop training, restore best weights
- **Defaults**: `RUN_TUNING=False`, `USE_FEATURE_SELECTION=False`, `train_fair_ensemble=True`
- **Main features**:
  - Point-in-time data loading exclusively (305 FRED-MD vintages)
  - 6-factor allocation (cyclical, defensive, value, growth, quality, momentum)
  - 3-phase training (Binary → Regression → Sharpe optimization)
  - **Multi-horizon support**: 4 horizons (1M, 3M, 6M, 12M)
  - **Feature selection & PCA**: Reduce 112 indicators via mutual info or PCA
  - **Walk-forward HP tuning**: Auto-tune hyperparameters per window (Optuna/random search)
  - **Model persistence**: Walk-forward models saved for ensemble
  - **Fair Ensemble** (new): Unbiased comparison using same training data
  - **Bias analysis** (optional): Decompose data quantity, seed variance, and ensemble effects
- **Key finding**: Multi-factor (6F) outperforms Binary (2F)
- **Notebook**: Simplified ~200 lines using externalized modules (was ~1400 lines)
- **New modules**: `src/visualization/` (plots), `src/pipelines/` (ThreeStepEvaluation)
- **Dashboard**: Streamlit app (`dashboard/app.py`) for interactive exploration with filters, adjustable score weights, and visualizations
- **Visualizations**: 4 heatmaps (return × year × horizon), 2×2 cumulative returns grid, total returns bar chart, max drawdown bar chart, holdout comparison with Delta heatmap
- **Benchmarks**: Equal-Weight 6F, 50/50 Cyc/Def, Risk Parity, Factor Momentum, Best Single Factor
- **Composite scoring**: Score = 0.35×Sharpe + 0.25×IC + 0.30×MaxDD + 0.10×Return
  - **Asymmetric IC penalty**: Negative IC penalized 2x, models with IC < -30% rejected
  - **Exponential MaxDD penalty**: `exp(3×maxdd)` more severe for large drawdowns
- **Benchmark reference lines**: All bar plots show best benchmark as dashed purple line for easy comparison

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
