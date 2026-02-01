# Notebook Refactoring Rules

> **Scope:** These rules apply specifically to Jupyter notebooks in this project.

## Core Principles

### 1. Check Existing Functions First

**Before creating any new function in a notebook cell:**

1. **Search existing modules** for similar functionality:
   ```bash
   grep -rn "function_name\|similar_keyword" src/
   ```

2. **Check these files in order:**
   - `src/utils/metrics.py` - Performance metrics (Sharpe, IC, drawdown, returns)
   - `src/utils/walk_forward.py` - Walk-forward validation utilities
   - `src/comparison_runner.py` - Training, evaluation, orchestration
   - `src/main_strategy.py` - Strategy class methods
   - `src/features/feature_engineering.py` - Feature creation utilities
   - `src/features/feature_selection.py` - Indicator selection, PCA

3. **If function exists:** Import and use it
4. **If function is close but not exact:** Extend/modify the existing function (update all callers)
5. **If function doesn't exist:** Create it in the appropriate `src/` module, NOT in the notebook

### 2. Notebook Content Guidelines

**Notebooks should ONLY contain:**
- Imports and configuration
- High-level orchestration calls (1-2 lines per step)
- Visualization calls (not visualization logic)
- Results display and interpretation
- Markdown documentation

**Notebooks should NOT contain:**
- Function definitions (except trivial lambdas for display)
- Data processing logic
- Model training loops
- Metric calculations
- Plot creation logic

### 3. Where to Put New Code

| Code Type | Target Location |
|-----------|-----------------|
| Metrics (Sharpe, IC, returns) | `src/utils/metrics.py` |
| Walk-forward utilities | `src/utils/walk_forward.py` |
| Training orchestration | `src/comparison_runner.py` |
| Visualization functions | `src/visualization/` (create if needed) |
| Data loading | `src/data/` |
| Feature engineering | `src/features/feature_engineering.py` |
| Configuration | `src/config.py` or YAML file |

### 4. Refactoring Checklist

Before adding code to a notebook:

- [ ] Searched `src/` for existing similar function
- [ ] If creating new function, placed it in appropriate `src/` module
- [ ] Function is imported, not defined inline
- [ ] Notebook cell is < 20 lines (excluding imports)
- [ ] No duplicate logic with existing `src/` code

### 5. Visualization Standards

All visualization code should be in `src/visualization/`:
- Create modular functions with clear inputs
- Return figure objects (let notebook control display)
- Use consistent styling via shared utilities
- Separate data preparation from plotting

---

## Quick Reference: Existing Utilities

### src/utils/metrics.py
- `PerformanceMetrics` dataclass
- `compute_information_coefficient()`
- `compute_accuracy()`
- `compute_auc()`

### src/comparison_runner.py
- `set_seed()` - Reproducibility
- `prepare_data()` - Multi-horizon targets
- `train_e2e_model()` / `train_supervised_model()`
- `run_combination_walk_forward()` - Walk-forward for 1 combo
- `train_final_model()` - Train on all non-holdout data
- `evaluate_on_holdout()` - Holdout evaluation
- `ensemble_predict()` - Average predictions from N models
- `compute_composite_score()` - Weighted ranking score

### src/utils/walk_forward.py
- `WalkForwardValidator.create_expanding_windows()`
- `WalkForwardValidator.run_all_windows()`
- `WalkForwardValidator.aggregate_results()`
