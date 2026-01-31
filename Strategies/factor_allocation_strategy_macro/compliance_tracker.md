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
| DONE | 42 |
| DEFERRED | 6 |
| PARTIAL | 1 |
| TODO | 0 |

---

## Section 1.1: Objective

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.1.1 | Neural network with macro data as input | **DONE** | `FactorAllocationTransformer` uses FRED-MD data | |
| 1.1.2 | Output allocation weights across equity categories | **DONE** | Phase 3 outputs softmax weights for 6 factors | cyclical, defensive, value, growth, quality, momentum |
| 1.1.3 | Maximize Sharpe ratio | **DONE** | `SharpeRatioLoss` in Phase 3 | `Loss = -E[R] + γ×Var[R] + λ×turnover` |
| 1.1.4 | Multiple time horizons (1w, 1m, 3m, 6m, 1y) | **DEFERRED** | Only 1-month implemented | **Decision**: One model per timeframe. Start with 1-month, add others later |
| 1.1.5 | One model per geographic region | **DEFERRED** | Only US region | **Decision**: Focus on US first, expand later |

---

## Section 1.2: Input Data

### 1.2.1 Sequence Structure

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.2.1.1 | Sequence of 50-100 macroeconomic tokens | **DEFERRED** | 12 months (12 tokens) | **Decision**: Reduced for overfitting prevention |
| 1.2.1.2 | Data name (unique identifier) | **DONE** | `indicator_ids` → embedding | 112 indicators from FRED-MD |
| 1.2.1.3 | Publication type (consensus, revision, estimate) | **DONE** | `pub_type_ids` → 6 types | |
| 1.2.1.4 | Importance score (1-3) | **DONE** | `importance` → linear projection | |
| 1.2.1.5 | Normalized value | **DONE** | `normalized_value` from FRED-MD | |
| 1.2.1.6 | 5-period moving average | **DONE** | `ma5` calculated | |
| 1.2.1.7 | Standardized surprise | **DONE** | `surprise` calculated | |
| 1.2.1.8 | Country/Region | **DONE** | `country_ids` → embedding | |
| 1.2.1.9 | Periodicity | **DONE** | `periodicity_ids` → embedding | Added `Periodicity` enum (daily, weekly, monthly, quarterly, irregular) |
| 1.2.1.10 | Temporal information (days offset) | **DONE** | `days_offset` → sinusoidal encoding | |

### 1.2.2 Market Context Data

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.2.2.1 | Credit spread (HY - IG) | **DONE** | `credit_spread` in market context | |
| 1.2.2.2 | Yield curve slope (10Y - 2Y) | **DONE** | `yield_curve` in market context | |
| 1.2.2.3 | VIX | **DONE** | `vix` in market context | |

### 1.2.3 Data Quality

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.2.3.1 | Point-in-time databases only | **DONE** | `PointInTimeFREDMDLoader` with 305 vintages | Excellent implementation |
| 1.2.3.2 | Vintages meticulously reconstructed | **DONE** | Uses actual FRED-MD vintage files 1999-2024 | No revision bias |

---

## Section 1.3: Input Encoding

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.3.1 | Additive embeddings (sum not concat) | **DONE** | `e_total = e_identity + e_type + e_category + e_country + e_periodicity + e_importance + e_temporal` | BERT-like approach, now includes periodicity |
| 1.3.2 | E_identity (32-64 dim) | **DONE** | `d_embed=32` | |
| 1.3.3 | E_type (8-16 dim) | **DONE** | Same d_embed for addition | |
| 1.3.4 | E_importance (8 dim) | **DONE** | Linear projection to d_embed | |
| 1.3.5 | E_temporal (sinusoidal) | **DONE** | `SinusoidalPositionalEncoding` class | |
| 1.3.6 | E_category (16 dim) | **DONE** | Category embedding | 8 categories |
| 1.3.7 | E_country (8-16 dim) | **DONE** | Country embedding | |
| 1.3.8 | X_token = LayerNorm(Linear(concat(E_total, numericals))) | **DONE** | Exact formula implemented | |

---

## Section 1.4: Model Architecture

### 1.4.1 Base Configuration

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.4.1.1 | 2-4 layers | **DONE** | 2 layers | Minimum for limited data |
| 1.4.1.2 | Embedding dimension 64-128 | **DONE** | d_model=64 | Minimum for limited data |
| 1.4.1.3 | 2-4 attention heads | **DONE** | 2 heads | Minimum for limited data |
| 1.4.1.4 | Dropout 0.3-0.5 | **DONE** | 0.5 | Maximum for regularization |

### 1.4.2 Attention Mechanisms

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.4.2.1 | Relative positional embeddings | **PARTIAL** | Sinusoidal + temporal decay | Not true RPE, but adequate substitute |
| 1.4.2.2 | Causal masking | **DONE** | `torch.tril` mask | Prevents future leakage |
| 1.4.2.3 | Soft masks based on temporal distance | **DONE** | Learnable `decay_rate` parameter | `exp(-decay × distance)` |

### 1.4.3 Economic Knowledge Injection

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.4.3.1 | Embeddings structured by category | **DONE** | 8 macro categories | |
| 1.4.3.2 | Skip connections (residual) | **DONE** | Pre-norm residual connections | |
| 1.4.3.3 | Embedding pre-training (auxiliary task) | **DONE** | `pretraining.py` module | `EmbeddingPretrainer` classifies indicators by category |
| 1.4.3.4 | Baseline regularization (ridge penalty) | **DONE** | `BaselineRegularization` class | Penalizes deviation from simple linear model |

---

## Section 1.5: Model Output and Loss

### 1.5.1 Output

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.5.1.1 | Allocation weights summing to 1 | **DONE** | `F.softmax(allocation_head(pooled), dim=-1)` | |
| 1.5.1.2 | Gated execution layer | **DONE** | `ExecutionGate` with threshold | `use_soft_threshold=True` |

### 1.5.2 Progressive Loss Function

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.5.2.1 | Phase 1: Cross-Entropy (binary) | **DONE** | `nn.BCELoss()` | Cyclicals vs defensives |
| 1.5.2.2 | Phase 2: MSE (regression) | **DONE** | `nn.MSELoss()` | Outperformance score |
| 1.5.2.3 | Phase 3: -Sharpe + turnover | **DONE** | `SharpeRatioLoss` | Differentiable approximation |
| 1.5.2.4 | λ calibrated to transaction costs | **DONE** | `calibrate_turnover_penalty()` function | Based on transaction cost bps, expected turnover, holding period |

---

## Section 1.6: Validation and Backtesting

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.6.1 | Walk-forward with expanding window | **DONE** | `WalkForwardValidator`, 6 windows | |
| 1.6.2 | No classical cross-validation | **DONE** | Only temporal splits | Correct approach |
| 1.6.3 | Final holdout reserved | **DEFERRED** | No separate holdout | All data used in walk-forward |
| 1.6.4 | Test all time horizons | **DEFERRED** | Only 1-month | See 1.1.4 decision |

---

## Section 1.7: Progressive Development

| # | Requirement | Status | Current State | Notes |
|---|-------------|--------|---------------|-------|
| 1.7.1 | Step 1: Naive baseline (momentum) | **DONE** | Acc: 0.4310, IC: -0.0403 | |
| 1.7.2 | Step 2: Logistic Regression | **DONE** | Acc: 0.5172, IC: 0.0671 | |
| 1.7.3 | Step 3: LightGBM | **DONE** | Acc: 0.5172, IC: -0.0506 | |
| 1.7.4 | Step 4: LSTM | **DONE** | Acc: 0.5345, IC: -0.1518 | |
| 1.7.5 | Step 5: Transformer | **DONE** | Acc: 0.5749, IC: 0.1561 | |
| 1.7.6 | Initial test: cyclicals vs defensives, 1m | **DONE** | Exact test case | |

---

## Section 2: Problems with Solutions

| # | Problem | Status | Implementation |
|---|---------|--------|----------------|
| 2.1 | Transaction costs & churning | **DONE** | ExecutionGate with threshold |
| 2.2 | Market context integration | **DONE** | MarketContextEmbedding |
| 2.3 | Look-ahead bias | **DONE** | Point-in-time FRED-MD |
| 2.4 | Efficient sequence encoding | **DONE** | Additive embeddings |
| 2.5 | Architecture for limited data | **DONE** | Minimal config |
| 2.6 | Loss function progression | **DONE** | 3-phase training |
| 2.7 | Economic knowledge injection | **DONE** | Pre-training + baseline regularization |
| 2.8 | Temporal validation | **DONE** | Walk-forward |
| 2.9 | Over-engineering risk | **DONE** | Progressive development |

---

## Training Strategies

Two training strategies are now implemented in `training_strategies.py`:

### 1. Supervised Strategy (`SupervisedTrainer`)
- Computes optimal weights w* using rolling window Sharpe maximization
- Uses scipy optimization with constraints (sum=1, bounds)
- Trains model to regress toward optimal weights
- **Pros**: Clear, interpretable targets
- **Cons**: High variance with limited data; hindsight bias

### 2. End-to-End Strategy (`EndToEndTrainer`)
- Uses differentiable Sharpe ratio loss
- Aggregates over batches for stable gradients
- Includes baseline regularization and calibrated turnover penalty
- **Pros**: Better generalization; batch-level stability
- **Cons**: Less interpretable

---

## Current Walk-Forward Results

| Model | In-Sample Acc | In-Sample IC | OOS Acc | OOS IC |
|-------|---------------|--------------|---------|--------|
| Naive Baseline | 0.4310 | -0.0403 | - | - |
| Logistic Regression | 0.5172 | 0.0671 | 0.4684 | 0.0596 |
| LightGBM | 0.5172 | -0.0506 | - | - |
| LSTM | 0.5345 | -0.1518 | - | - |
| **Transformer** | **0.5749** | **0.1561** | **0.4183** | **-0.1110** |

**Observation**: Significant IC degradation in OOS (0.1561 → -0.1110) confirms overfitting risk warning from Section 3.5.

---

## Intentional Deviations Summary

| Decision | Rationale |
|----------|-----------|
| US only (not Europe/Japan) | Focus on one region first, expand later |
| 1-month horizon only | One model per timeframe, start with 1-month |
| 12 tokens (not 50-100) | Prevent overfitting with limited data |
| No final holdout | Walk-forward provides OOS estimates |

---

## Recently Completed (2026-01-31)

| Item | Implementation |
|------|----------------|
| Periodicity encoding (1.2.1.9) | Added `Periodicity` enum, `periodicity_embedding` in `MacroTokenEmbedding`, updated feature engineering |
| Embedding pre-training (1.4.3.3) | New `pretraining.py` module with `EmbeddingPretrainer` and `pretrain_embeddings()` |
| Baseline regularization (1.4.3.4) | `BaselineRegularization` class in `transformer.py` |
| Turnover penalty calibration (1.5.2.4) | `calibrate_turnover_penalty()` function with formula based on transaction costs |
| Supervised training strategy | `SupervisedTrainer` with `compute_optimal_weights()` and rolling window optimization |
| End-to-End training strategy | `EndToEndTrainer` with 3-phase training and all improvements |

---

## Remaining Items

| Priority | Item | Status | Notes |
|----------|------|--------|-------|
| LOW | True relative positional encoding (1.4.2.1) | **PARTIAL** | Current sinusoidal + temporal decay is adequate |

---

## Update History

| Date | Change |
|------|--------|
| 2026-01-31 | Initial compliance tracker created |
| 2026-01-31 | Added periodicity encoding, pre-training, baseline regularization, turnover calibration |
| 2026-01-31 | Implemented Supervised and End-to-End training strategies |
