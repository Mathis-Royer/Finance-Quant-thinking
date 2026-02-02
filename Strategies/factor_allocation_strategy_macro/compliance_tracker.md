# Compliance Tracker: Strategy Document vs Current Implementation

This document tracks the divergence between the strategy specification (`factor_allocation_strategy_macro.md`) and the current implementation. It serves as a living log of what is implemented, what is intentionally deferred, and what remains to be done.

---

## Legend

| Status | Meaning |
|--------|---------|
| **DONE** | Fully implemented as specified |
| **DEFERRED** | Intentionally not implemented (current scope/resources) |
| **PARTIAL** | Partially implemented, needs completion |
| **TODO** | Not yet implemented, should be done |

---

## Summary

| Category | Count |
|----------|-------|
| DONE | 83 |
| DEFERRED | 4 |
| PARTIAL | 0 |
| TODO | 0 |

---

## Section 1: Objective and Data

### 1.1 Objective

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 1.1.1 | Neural network with macro data | **DONE** | `FactorAllocationTransformer` with FRED-MD | |
| 1.1.2 | Output allocation weights | **DONE** | Softmax weights for 6 factors | cyclical, defensive, value, growth, quality, momentum |
| 1.1.3 | Maximize risk-adjusted return | **DONE** | `SortinoLoss` (default) or `SharpeRatioLoss` | Sortino preferred: penalizes downside only |
| 1.1.4 | Multiple time horizons | **DONE** | 1M, 3M, 6M, 12M all implemented | One model per horizon |
| 1.1.5 | Multiple regions | **DEFERRED** | US only | Focus on US first; Europe/Japan planned |

### 1.2 Input Data

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 1.2.1 | Point-in-Time FRED-MD | **DONE** | 305 vintage files (1999-2024) | Prevents look-ahead bias from data revisions |
| 1.2.2 | 112 macro indicators | **DONE** | All FRED-MD indicators loaded | |
| 1.2.3 | 12-month sequence | **DONE** | seq_len=12 | Captures seasonality; longer sequences overfit |
| 1.2.4 | Market context (spreads, VIX) | **DONE** | Term spread, credit spread, VIX | |
| 1.2.5 | Feature selection (optional) | **DONE** | `IndicatorSelector` with mutual info | 30 features default; reduces noise |

### 1.3 Input Encoding

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 1.3.1 | Additive embeddings | **DONE** | Sum not concat | Reduces parameters; BERT-like approach |
| 1.3.2 | Indicator embedding | **DONE** | d_embed=32 per indicator | |
| 1.3.3 | Category embedding | **DONE** | 8 macro categories | |
| 1.3.4 | Temporal encoding | **DONE** | RoPE (Rotary Positional Embeddings) | Encodes relative positions |
| 1.3.5 | LayerNorm fusion | **DONE** | `LayerNorm(E_total + Linear(numericals))` | |

---

## Section 2: Model Architecture

### 2.1 MICRO Transformer

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 2.1.1 | Minimal parameters | **DONE** | ~12,000 params | Rule: params << samples (~300) |
| 2.1.2 | d_model=32 | **DONE** | Smallest capturing relationships | Larger (64, 128) showed overfitting |
| 2.1.3 | 1 layer | **DONE** | Single TransformerBlock | Data can't support deeper models |
| 2.1.4 | 1 attention head | **DONE** | Single head | Multiple heads add unused capacity |
| 2.1.5 | d_ff=64 | **DONE** | 2×d_model | Standard ratio |
| 2.1.6 | dropout=0.75 | **DONE** | Extremely high | Necessary for ~300 samples |

### 2.2 Attention Mechanisms

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 2.2.1 | RoPE embeddings | **DONE** | `RotaryPositionalEmbedding` class | Encodes relative positions in time series |
| 2.2.2 | Causal masking | **DONE** | `torch.tril` mask | Prevents future leakage |
| 2.2.3 | Temporal decay | **DONE** | Learnable `decay_rate` | Recent data more relevant; `exp(-decay × distance)` |

### 2.3 Output Heads

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 2.3.1 | Binary head | **DONE** | `Linear(d_model, 1)` → sigmoid | Phase 1 classification |
| 2.3.2 | Regression head | **DONE** | `Linear(d_model, 1)` | Phase 2 regression |
| 2.3.3 | Allocation head | **DONE** | `Linear(d_model, 6)` → softmax | Phase 3 allocation |
| 2.3.4 | Execution gate | **DONE** | `ExecutionGate` with threshold | Reduces turnover; 5% default |

---

## Section 3: Training Strategy

### 3.1 Training Approaches

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 3.1.1 | E2E 3-phase training | **DONE** | Binary → Regression → Sharpe | Progressive: avoids local minima |
| 3.1.2 | Supervised training | **DONE** | `SupervisedTrainer` with forward targets | Optimal weights on FORWARD returns (T+1 to T+h) |
| 3.1.3 | Choice of training | **DONE** | Both available | E2E: discovery; Sup: stability |
| 3.1.4 | Phase 3-only ablation | **DONE** | `skip_phase1_phase2=True` | Test curriculum learning necessity |
| 3.1.5 | Equal epochs | **DONE** | E2E=70, Supervised=70 | Fair comparison between strategies |

### 3.2 Loss Functions

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 3.2.1 | SortinoLoss (default) | **DONE** | `-mean(R) / downside_std(R)` | Only penalizes downside volatility |
| 3.2.2 | SharpeRatioLoss | **DONE** | `-mean(R) / std(R)` | For comparison/benchmarking |
| 3.2.3 | Phase 1: Cross-Entropy | **DONE** | `nn.BCELoss()` | Binary warm-up |
| 3.2.4 | Phase 2: MSE | **DONE** | `nn.MSELoss()` | Regression bridge |
| 3.2.5 | Turnover penalty | **DONE** | Calibrated λ | Reduces transaction costs |

### 3.3 Early Stopping

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 3.3.1 | CompositeEarlyStopping | **DONE** | Sharpe+IC+MaxDD+Return | Multi-metric; loss alone insufficient |
| 3.3.2 | Score weights | **DONE** | 0.35, 0.25, 0.30, 0.10 | Sharpe primary; MaxDD prevents catastrophe |
| 3.3.3 | Patience | **DONE** | 5 epochs | Balances convergence and overfitting |
| 3.3.4 | ModelCheckpoint | **DONE** | Best weights restored | Prevents late overfitting |

### 3.4 Regularization

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 3.4.1 | High dropout | **DONE** | 0.75 | Forces redundancy; prevents memorization |
| 3.4.2 | Weight decay | **DONE** | 0.05 | Strong L2 penalty |
| 3.4.3 | Conservative LR | **DONE** | 0.0005 | Stable with high dropout |
| 3.4.4 | Large batch size | **DONE** | 64 | Stable gradients (~300 total samples) |

---

## Section 4: Evaluation Methodology

### 4.1 3-Step Protocol

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 4.1.1 | Step 1: Walk-forward | **DONE** | 3-5 expanding windows | Multiple OOS estimates; non-overlapping |
| 4.1.2 | Step 2: Final model | **DONE** | Train on 2000-2021 | Production model; all non-holdout data |
| 4.1.3 | Step 3: Holdout eval | **DONE** | 2022+ reserved | True OOS; never seen in training |
| 4.1.4 | Fair Ensemble | **DONE** | 5 models, same data, different seeds | Eliminates data-quantity bias |
| 4.1.5 | WF Ensemble | **DONE** | Average of walk-forward models | Reference for bias quantification |

### 4.2 Evaluation Metrics

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 4.2.1 | Sharpe Ratio | **DONE** | `mean(R)/std(R) × √12` | Industry standard |
| 4.2.2 | Information Coefficient | **DONE** | Spearman correlation | Measures directional skill |
| 4.2.3 | Max Drawdown | **DONE** | `min(wealth/peak - 1)` | Tail risk measure |
| 4.2.4 | Total Return | **DONE** | Cumulative return | Absolute performance |
| 4.2.5 | Accuracy (Binary) | **DONE** | Classification accuracy | Only for Binary mode |

### 4.3 Benchmarks

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 4.3.1 | Equal-Weight 6F | **DONE** | 1/6 each factor | Naive diversification baseline |
| 4.3.2 | 50/50 Cyc/Def | **DONE** | Static allocation | Simple macro rule |
| 4.3.3 | Risk Parity | **DONE** | Inverse volatility | Volatility-based timing |
| 4.3.4 | Factor Momentum | **DONE** | Positive trailing 12M | Trend-following |
| 4.3.5 | Best Single Factor | **DONE** | 100% best in-sample | Perfect hindsight |

---

## Section 5: Composite Scoring

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 5.1 | Multi-metric score | **DONE** | `0.35×Sharpe + 0.25×IC + 0.30×MaxDD + 0.10×Return` | Single metrics insufficient |
| 5.2 | Asymmetric IC penalty | **DONE** | Negative IC penalized 2× | Inverted predictions are dangerous |
| 5.3 | Exponential MaxDD | **DONE** | `exp(3 × maxdd)` | Large drawdowns exponentially worse |
| 5.4 | Rejection criterion | **DONE** | IC < -30% → score=0 | Reject systematically wrong models |
| 5.5 | Normalization | **DONE** | All metrics to [0, 1] | Comparable scales |

---

## Section 6: Statistical Analysis

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 6.1 | Kelly Criterion | **DONE** | `f* = (μ - r) / σ²` | Optimal position sizing |
| 6.2 | Half-Kelly | **DONE** | Conservative default | More robust to estimation errors |
| 6.3 | Bootstrap CI | **DONE** | 1000 resamples | Non-parametric; handles fat tails |
| 6.4 | Lo (2002) test | **DONE** | Single Sharpe significance | Accounts for autocorrelation |
| 6.5 | Jobson-Korkie test | **DONE** | Sharpe comparison | Tests if two Sharpes differ |

---

## Section 7: Configuration Options

### 7.1 Feature Selection

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 7.1.1 | Mutual info selection | **DONE** | `IndicatorSelector` class | Non-linear dependency measure |
| 7.1.2 | 1-month feature lag | **DONE** | features[M] → target[M+1] | Prevents look-ahead in selection |
| 7.1.3 | 30 features default | **DONE** | Configurable | Balances retention vs noise |
| 7.1.4 | PCA option | **DONE** | `PCAFeatureReducer` | Alternative dimensionality reduction |

### 7.2 Hyperparameter Tuning

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 7.2.1 | Walk-forward HP tuning | **DONE** | Per-window optimization | Prevents overfitting to period |
| 7.2.2 | 15 trials default | **DONE** | Configurable | Balances exploration vs compute |
| 7.2.3 | Tuned params | **DONE** | LR, dropout, weight_decay | Most impactful for small data |

### 7.3 Config Combinations

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 7.3.1 | baseline config | **DONE** | No FS, no HPT | Pure model performance |
| 7.3.2 | fs config | **DONE** | Feature selection only | Noise reduction |
| 7.3.3 | hpt config | **DONE** | HP tuning only | Hyperparameter optimization |
| 7.3.4 | fs+hpt config | **DONE** | Both FS and HPT | Maximum optimization |
| 7.3.5 | 64 total combos | **DONE** | 2×2×4×4 | All combinations testable |

---

## Section 8: Dashboard and Visualization

| # | Requirement | Status | Current State | Justification |
|---|-------------|--------|---------------|---------------|
| 8.1 | Streamlit dashboard | **DONE** | `dashboard/app.py` | Interactive exploration |
| 8.2 | Filters (Strategy, Allocation, etc.) | **DONE** | Multi-select dropdowns | Hypothesis testing without code |
| 8.3 | Adjustable score weights | **DONE** | Sliders with normalization | Sensitivity analysis |
| 8.4 | Factor allocation charts | **DONE** | Stacked area charts | Weight evolution visualization |
| 8.5 | Benchmark comparison | **DONE** | Reference lines in plots | Context for performance |
| 8.6 | Results export | **DONE** | Parquet cache | Dashboard loads cached results |

---

## Section 9: Bias Prevention

| # | Problem | Status | Solution | Justification |
|---|---------|--------|----------|---------------|
| 9.1 | Look-ahead bias (data) | **DONE** | Point-in-Time FRED-MD | Uses data as-of date, not revised |
| 9.2 | Look-ahead bias (features) | **DONE** | 1-month lag in selection | features[M] → target[M+1] |
| 9.3 | Data-quantity bias | **DONE** | Fair Ensemble | Same data, different seeds |
| 9.4 | Holdout contamination | **DONE** | Fixed date never used | 2022+ reserved from all training |
| 9.5 | Overlapping test periods | **DONE** | Non-overlapping windows | Prevents autocorrelation inflation |
| 9.6 | Overfitting | **DONE** | MICRO architecture + regularization | 12k params, dropout=0.75 |
| 9.7 | Return alignment bias | **DONE** | Forward shift in cumulative_returns | returns[i+1:i+1+h] not returns[i:i+h] |
| 9.8 | Supervised target bias | **DONE** | Forward optimal weights | Targets computed on future returns, not past |
| 9.9 | Training time bias | **DONE** | Equal epochs (70 each) | Fair E2E vs Supervised comparison |

---

## Intentional Deviations

| Decision | Status | Rationale |
|----------|--------|-----------|
| US only (not Europe/Japan) | **DEFERRED** | Focus on US first; regional models planned |
| 50-100 tokens (spec) → 12 tokens | **DEFERRED** | 12 months prevents overfitting; longer sequences tested, worse results |
| True Relative Positional Encoding | **DEFERRED** | RoPE provides adequate relative position encoding |
| Weekly data granularity | **DEFERRED** | Monthly FRED-MD is primary source; weekly would require different data |

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `src/main_strategy.py` | Core strategy class | **DONE** |
| `src/comparison_runner.py` | Training and evaluation runner | **DONE** |
| `src/pipelines/three_step_pipeline.py` | 3-step evaluation orchestrator | **DONE** |
| `src/models/transformer.py` | Transformer with RoPE | **DONE** |
| `src/models/training_strategies.py` | E2E and Supervised trainers | **DONE** |
| `src/utils/walk_forward.py` | Walk-forward validation | **DONE** |
| `src/utils/statistics.py` | Kelly, Bootstrap, significance tests | **DONE** |
| `src/utils/analysis.py` | Holdout analysis functions | **DONE** |
| `src/features/feature_selection.py` | Feature selection with lag | **DONE** |
| `src/visualization/` | All plotting functions | **DONE** |
| `dashboard/app.py` | Streamlit dashboard | **DONE** |

---

## Update History

| Date | Change |
|------|--------|
| 2026-02-02 | **Critical fixes**: (1) Fixed return alignment bug - cumulative_returns now uses forward shift `[i+1:i+1+h]`. (2) Supervised training now uses FORWARD optimal weights (not past 24M). (3) Equal epochs (70) for E2E and Supervised. (4) Added `skip_phase1_phase2` option for ablation tests. |
| 2026-02-02 | Major update: Added 3-step evaluation, Fair Ensemble, composite scoring, statistical analysis, dashboard, feature selection, HP tuning, RoPE, SortinoLoss, CompositeEarlyStopping. Updated strategy document to v3.0. |
| 2026-01-31 | Added periodicity encoding, pre-training, baseline regularization, turnover calibration |
| 2026-01-31 | Implemented Supervised and End-to-End training strategies |
| 2026-01-31 | Initial compliance tracker created |
