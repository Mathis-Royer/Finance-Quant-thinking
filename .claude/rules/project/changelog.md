# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 1 | 2026-01-31 | **Refactored notebook**: Extracted training logic to `comparison_runner.py`. Notebook now has 8 clean cells with clear sections. 16 combinations (E2E/Sup × Binary/Multi × 4 horizons). |
| 2 | 2026-01-31 | **Multi-horizon support**: Added 4 prediction horizons (1M, 3M, 6M, 12M) for Multi-factor mode. Each horizon optimizes Sharpe on cumulative returns. New `MultiHorizonStrategy` class. |
| 3 | 2026-01-31 | **Transformer-only codebase**: Removed all non-Transformer models (LR, LSTM, GradientBoosting, Naive). Deleted `baseline_models.py` and `initial_validation.py` |
| 4 | 2026-01-31 | **Multi-factor allocation**: Backtest uses all 6 factors (cyclical, defensive, value, growth, quality, momentum) with weighted allocation |
| 5 | 2026-01-31 | **MICRO Transformer**: Reduced model (d_model=32, 1 layer, 1 head, 12k params) to address overfitting in walk-forward |
| 6 | 2026-01-31 | **Additive embeddings fix**: Changed `MacroTokenEmbedding` from concatenation to addition (BERT-like approach) |
| 7 | 2026-01-31 | **Transformer walk-forward**: Added OOS validation with 3-phase retraining on each window |
| 8 | 2026-01-31 | **Point-in-Time loader**: Implemented `PointInTimeFREDMDLoader` using 305 FRED-MD vintage files |
| 9 | 2026-01-31 | **Walk-forward validation**: Added proper temporal cross-validation with expanding windows |

---

## Current State

Factor allocation strategy using **Point-in-Time FRED-MD** macroeconomic data with **MICRO Transformer** model.

- **Status**: Development/Research
- **Model**: MICRO Transformer only (12k params, d_model=32, 1 layer, 1 head)
- **Main features**:
  - Point-in-time data loading exclusively (305 FRED-MD vintages)
  - 6-factor allocation (cyclical, defensive, value, growth, quality, momentum)
  - 3-phase training (Binary → Regression → Sharpe optimization)
  - Walk-forward validation with expanding windows
  - **Multi-horizon support**: 4 horizons (1M, 3M, 6M, 12M) for Multi-factor mode
- **Key finding**: Multi-factor (6F) outperforms Binary (2F): Sharpe +0.83 vs +0.56, IC +0.10 vs -0.10
- **Notebook**: 8 clean cells using `comparison_runner.py` for 16-combination comparison

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
