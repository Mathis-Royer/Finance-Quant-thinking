# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 1 | 2026-02-02 | **Added factor allocation area charts to dashboard**: (1) New `allocation_weights` and `factor_columns` fields in `HoldoutResult` dataclass. (2) Modified `evaluate_on_holdout()` and `ensemble_predict()` to capture weights. (3) Added Binary mode weights storage in `backtest()`. (4) Export weights as JSON to parquet via `export_results.py`. (5) New `FACTOR_COLORS` and `FACTOR_ORDER` in `colormaps.py`. (6) New `render_factor_allocation_charts()` in `app.py` showing stacked area charts for top 3 models factor allocation evolution over holdout period. **Requires**: Re-run notebook Step 3 + re-export to see charts. |
| 2 | 2026-02-01 | **Changed winner determination from Sharpe to Score**: (1) `flatten_holdout_results()` now computes and includes composite score. (2) `print_model_comparison_summary()` uses `score` instead of `sharpe`. (3) `plot_winner_distribution()` determines winner by Score (pie chart title updated). (4) `plot_final_vs_ensemble_scatter()` now plots Score on both axes with percentage formatting. Winner uses composite Score = 0.35×Sharpe + 0.25×IC + 0.30×MaxDD + 0.10×Return for fairer multi-metric comparison. |
| 1 | 2026-02-01 | **Added statistical analysis module and notebook integration**: (1) New `src/utils/statistics.py` with **Kelly Criterion** (`compute_kelly_fraction`, `compute_kelly_weights`), **Bootstrap CIs** (`bootstrap_sharpe_ratio`, `bootstrap_metric` with n=1000), **Significance Tests** (Lo 2002 for single Sharpe, Jobson-Korkie with Memmel for comparison). (2) New `print_statistical_analysis()` in `analysis.py` for formatted output of top models with CIs, p-values, and benchmark comparison. (3) **Notebook integration**: New cell after holdout results calls `print_statistical_analysis()` to show statistical rigor of results. |
| 2 | 2026-02-01 | **Integrated new features into training pipeline**: Updated `training_strategies.py`: (1) New config `loss_type='sortino'` (default) uses `SortinoLoss` in `EndToEndTrainer.train_phase3()`. (2) New config `early_stopping_type='composite'` (default) uses `CompositeEarlyStopping` in **BOTH** `SupervisedTrainer._train_phase3()` AND `EndToEndTrainer.train_phase3()`. (3) **RoPE active by default** (`use_rope=True`). (4) **Fixed bug**: SortinoLoss only takes 3 args (was incorrectly passed 4). All features now active by default in training pipeline. |
| 2 | 2026-02-01 | **Added SortinoLoss, CompositeEarlyStopping, and RoPE**: (1) `SortinoLoss`: penalizes only downside volatility. (2) `CompositeEarlyStopping`: stops on composite score (Sharpe+IC+MaxDD). (3) `RotaryPositionalEmbedding`: encodes relative positions in attention. |
| 3 | 2026-02-01 | **Fixed data leakage in feature selection**: (1) **PCA leakage warning**: Added deprecation warning to `PCAFeatureReducer.fit_transform()` recommending separate `fit(X_train)` then `transform(X)` calls. Updated `fit()` docstring to emphasize training-only data. (2) **Feature selection forward-looking bias**: Fixed `_align_with_targets()` to lag features by 1 month (features[M] predict target[M+1]) preventing look-ahead bias in indicator selection via mutual information. |
| 2 | 2026-02-01 | **Centralized utilities and removed code duplication**: (1) **New `src/utils/keys.py`**: Centralized `unpack_key()` function replacing 4 duplicate implementations across holdout_plots.py, walk_forward_plots.py, export_results.py, and notebook. (2) **New `src/utils/constants.py`**: Shared constants `MODEL_TYPE_ABBREV`, `CONFIG_SUFFIX`, `STRATEGY_ABBREV`, `ALLOCATION_ABBREV`, `format_model_label()` replacing hardcoded dicts scattered across app.py. (3) **New `src/utils/analysis.py`**: Externalized analysis functions `print_best_models_table()`, `print_model_comparison_summary()`, `compare_model_types()`, `find_best_models()` from notebook. (4) **Dashboard simplified**: Removed duplicated `compute_score()` (~70 lines), now uses `compute_composite_score` from comparison_runner. Replaced 4 local `type_abbrev`/`config_abbrev` defs with imports. (5) **Notebook simplified**: Replaced ~100 lines of inline analysis code with calls to `print_best_models_table()` and `print_model_comparison_summary()`. Added utils.keys, utils.constants, utils.analysis to reload_all(). |
| 2 | 2026-02-01 | **Integrated multi-config visualization (FS/HPT) into existing charts**: (1) **Grouped bar charts**: `plot_total_returns_bar()` and `plot_max_drawdown_bar()` now show configs as grouped bars when `config_filter=None`, (2) **Grid heatmaps**: `plot_sharpe_heatmaps()` and `plot_return_heatmaps()` show n_combos × n_configs grid when multiple configs, (3) **Summary tables**: `print_holdout_summary_table()` and `print_year_summary_table()` add Config column when multiple configs, (4) **Cumulative returns**: Multi-config mode uses config colors with horizon linestyles, (5) **New `CONFIG_COLORS` and `CONFIG_ORDER`** in `colormaps.py` for consistent config styling (Blue=baseline, Green=fs, Yellow=hpt, Red=fs+hpt), (6) **Notebook updated** to use `config_filter=None` instead of looping over configs. (7) **Dashboard app.py updated**: Labels include config suffix (-FS/-HPT/-FS+HPT), Pivot Analysis has "Group By" option (Horizon or Config), Config Comparison chart now shows 3 metrics (Sharpe, IC, MaxDD) matching Model Type Comparison style, Sharpe bar chart height dynamic based on number of models. |
| 2 | 2026-02-01 | **Enhanced training with regularization and score improvements**: (1) **New EarlyStopping + ModelCheckpoint** in `src/utils/training_utils.py`, (2) **Improved composite score**: asymmetric IC penalty (negative IC penalized 2x), exponential MaxDD penalty `exp(3×maxdd)`, rejection of models with IC < -30%, (3) **Fixed SharpeRatioLoss**: true Sharpe ratio `-mean/std` with running std for gradient stability (was approximation `-mean + γ×var`), (4) **Enhanced regularization**: dropout 0.6→0.75, weight_decay 0.01→0.05, batch_size 32→64, learning_rate 0.001→0.0005, (5) **Increased rolling window** for Supervised optimal weights: 12→24 months. **All modes kept**: Binary, E2E, Sup, Multi, Fair Ensemble. |
| 2 | 2026-02-01 | **Fixed percentage formatting issues**: (1) **Total Return double conversion fixed**: removed `*100` in `comparison_runner.py` (now stored as decimal like 0.0835 for 8.35%). Removed `/100` in `export_results.py` and `holdout_plots.py`. (2) **Sharpe ratio no longer displayed as percentage**: Sharpe is a ratio (e.g., 0.85, 1.20), not a percentage. Removed all `*100` and `%` formatting for Sharpe across `comparison_runner.py`, `holdout_plots.py`, and `app.py`. Now displays as plain number with 2 decimals. (3) IC, MaxDD, Total Return, and Score remain as percentages. |
| 2 | 2026-02-01 | **Added Streamlit dashboard for interactive results exploration**: New `dashboard/app.py` with interactive filters (Strategy, Allocation, Type, Horizon), adjustable score weights, results table with averages, multiple visualizations (bar charts, heatmaps, scatter plots), pivot tables, and benchmark comparisons. Run with `streamlit run dashboard/app.py`. Added `dashboard/export_results.py` for caching results. |
| 3 | 2026-02-01 | **Centralized notebook imports with reload function**: New imports cell at top of notebook with all module imports and `reload_all()` function. Removed duplicate imports from other cells. After modifying source code, call `reload_all()` to refresh modules without kernel restart. |
| 4 | 2026-02-01 | **Added Fair Ensemble for unbiased comparison**: New `train_fair_ensemble_models()` trains N models on same data (2000-2021) with different seeds, eliminating data quantity bias in WF Ensemble. Added `run_bias_analysis()` to decompose effects (data quantity, seed variance, pure ensemble). Step 3 now evaluates **Final vs Fair Ensemble vs WF Ensemble**. Fair Ensemble is the recommended production ensemble. |
| 5 | 2026-02-01 | **Refactored benchmarks for factor allocation relevance**: Removed Market (Mkt-RF) benchmark (not relevant for factor timing). Added 3 new benchmarks: (1) **Risk Parity** (inverse volatility weighting), (2) **Factor Momentum** (allocate to positive trailing 12M factors), (3) **Best Single Factor** (100% in best in-sample Sharpe factor). Kept Equal-Weight 6F and 50/50 Cyc/Def. Updated `holdout_plots.py` color scheme. |
| 6 | 2026-02-01 | **Improved holdout visualizations with benchmarks**: (1) Bar chart split into 2×2 grid (Sup, E2E, Multi, Binary) sorted by max(Sharpe), (2) Added 3rd heatmap showing Delta (Ensemble - Final), (3) New cumulative returns chart with Top 3 models vs benchmarks (solid lines) and benchmarks (dashed), (4) Combined summary table with Top 3 models + benchmarks. |
| 7 | 2026-02-01 | **Simplified notebook using externalized modules**: Reduced notebook from ~1400 lines to ~200 lines. Created `src/visualization/` (walk_forward_plots.py, holdout_plots.py, colormaps.py) and `src/pipelines/three_step_pipeline.py` (ThreeStepEvaluation class). Notebook now uses `plot_all_walk_forward()`, `plot_all_holdout()`, and `ThreeStepEvaluation` for clean orchestration. |
| 8 | 2026-02-01 | **Added composite score to all summary tables**: Score = 0.4×Sharpe + 0.3×IC + 0.3×MaxDD (normalized). Added `compute_composite_score()` function. Updated walk-forward summary, holdout summary, and returns summary to display Sharpe, IC, MaxDD, Score, and Rank columns. |

---

## Current State

Factor allocation strategy using **Point-in-Time FRED-MD** macroeconomic data with **MICRO Transformer** model.

- **Status**: Development/Research
- **Model**: MICRO Transformer only (12k params, d_model=32, 1 layer, 1 head)
- **Pipeline**: Up to 64 combinations (2 strategies × 2 allocations × 4 horizons × 4 configs)
  - Strategies: E2E (3-phase), Supervised
  - Allocations: Binary (2F), Multi (6F)
  - Horizons: 1M, 3M, 6M, 12M
  - **Configs** (new): baseline, fs (feature selection), hpt (HP tuning), fs+hpt (both)
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
  - **SortinoLoss** (default): Penalizes only downside volatility, not upside
  - **CompositeEarlyStopping** (default): Stops on composite score (Sharpe + IC + MaxDD)
  - **RoPE** (default): Rotary Position Embeddings for relative position encoding
  - **EarlyStopping + ModelCheckpoint**: Auto-stop training, restore best weights
- **Statistical analysis** (`src/utils/statistics.py`):
  - **Kelly Criterion**: Optimal position sizing based on edge/variance
  - **Bootstrap CI**: Confidence intervals for Sharpe, returns, drawdown
  - **Significance Tests**: Lo (2002) for single Sharpe, Jobson-Korkie for comparison
- **Defaults**: `configs_to_run=["baseline"]`, `train_fair_ensemble=True`, `n_features=30`, `hp_tuning_trials=15`
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
- **New modules**:
  - `src/visualization/` (plots)
  - `src/pipelines/` (ThreeStepEvaluation)
  - `src/utils/keys.py` (centralized key unpacking)
  - `src/utils/constants.py` (shared abbreviations and labels)
  - `src/utils/analysis.py` (holdout result analysis functions)
  - `src/utils/statistics.py` (Kelly, Bootstrap CI, significance tests)
- **Dashboard**: Streamlit app (`dashboard/app.py`) for interactive exploration with filters, adjustable score weights, and visualizations
- **Visualizations**: Multi-config integrated charts (grouped bars, grid heatmaps with configs as columns), cumulative returns with config colors, summary tables with Config column, **config comparison chart** for side-by-side analysis, **factor allocation area charts** showing weight evolution for top 3 models
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
