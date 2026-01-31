# Neural Network Strategy for Factor Allocation Based on Macroeconomic Data

## Executive Summary

This document describes a dynamic allocation strategy across equity factor styles using a Transformer-based model fed by point-in-time macroeconomic data. The goal is to maximize risk-adjusted performance by anticipating style rotations (cyclical, defensive, value, growth, quality, momentum) from macro signals.

**Current Implementation Status**: Research/Development phase with a MICRO Transformer architecture (12k parameters) trained on Point-in-Time FRED-MD data (305 vintage files, 112 indicators).

---

## 1. Detailed Strategy

### 1.1 Objective

The objective is to develop a neural network model that takes monthly macroeconomic data as input and produces allocation weights across 6 factor categories, with the goal of maximizing risk-adjusted performance over different time horizons (1 month, 3 months, 6 months, 12 months).

**Target Factor Categories**:

| Factor | Description | Proxy ETFs |
|--------|-------------|------------|
| Cyclical | Economically sensitive sectors | XLY, XLI, XLF |
| Defensive | Stable, low-beta sectors | XLP, XLU, XLV |
| Value | Low P/E, high dividend stocks | IWD, VTV |
| Growth | High growth expectations | IWF, VUG |
| Quality | High profitability, low debt | QUAL |
| Momentum | Recent outperformers | MTUM |

### 1.2 Input Data

#### 1.2.1 Point-in-Time FRED-MD Data

The model uses **Point-in-Time FRED-MD** macroeconomic data exclusively to avoid look-ahead bias. FRED-MD provides monthly vintages of 112+ macroeconomic indicators.

**Data Pipeline**:
- **Source**: FRED-MD vintage files (305 files from 1999-08 to 2024-12)
- **Loader**: `PointInTimeFREDMDLoader` with publication lag handling
- **Transformations**: FRED-MD standard transformations (log, diff, etc.)

**Key Indicators Categories**:

| Category | Examples | Count |
|----------|----------|-------|
| Output & Income | Industrial Production, Real GDP | ~15 |
| Labor Market | Unemployment, Payrolls, Hours | ~30 |
| Consumption | Retail Sales, Consumer Sentiment | ~10 |
| Housing | Housing Starts, Building Permits | ~10 |
| Money & Credit | M1, M2, Consumer Credit | ~15 |
| Interest Rates | Fed Funds, Treasury Yields | ~20 |
| Prices | CPI, PPI, PCE | ~20 |
| Stock Market | S&P 500, Dividend Yield | ~5 |

#### 1.2.2 Market Context Data

Market condition indicators are integrated as additional features:

| Indicator | Role | Calculation |
|-----------|------|-------------|
| Term Spread | Yield curve slope | 10Y - 3M Treasury |
| Default Spread | Credit risk proxy | BAA - AAA spread |
| VIX (when available) | Implied volatility | CBOE VIX level |

#### 1.2.3 Sequence Structure

The model input is a sequence of 12 monthly observations (1 year lookback), where each timestep contains all available macro indicators for that month.

| Attribute | Dimension | Description |
|-----------|-----------|-------------|
| Macro features | 112 | FRED-MD indicators (transformed) |
| Market context | 3-5 | Spreads, volatility measures |
| Momentum features | 4 windows | 1M, 3M, 6M, 12M momentum |
| Total per timestep | ~120 | All features concatenated |

### 1.3 Input Encoding

The encoding uses an **additive embeddings approach** (BERT-like) for macro tokens:

```
E_total = E_indicator + E_temporal + E_category
```

| Embedding | Current Dimension | Description |
|-----------|-------------------|-------------|
| E_indicator | 32 | Learned embedding per indicator |
| E_temporal | 32 | Sinusoidal positional encoding |
| E_category | 32 | Macro category embedding |

Numerical values are projected and added:

```
X_token = LayerNorm(E_total + Linear(numerical_values))
```

### 1.4 Model Architecture

#### 1.4.1 MICRO Transformer Configuration

The model uses a deliberately minimal Transformer to prevent overfitting with limited data:

| Parameter | Current Value | Rationale |
|-----------|---------------|-----------|
| d_model | 32 | Minimal embedding dimension |
| num_layers | 1 | Single attention layer |
| num_heads | 1 | Single attention head |
| d_ff | 64 | Feedforward dimension |
| dropout | 0.6 | High dropout for regularization |
| sequence_length | 12 | 12 months lookback |
| **Total parameters** | **~12,000** | Intentionally small |

#### 1.4.2 Architecture Components

```
Input (batch, seq_len=12, num_indicators=112)
    ↓
MacroTokenEmbedding (additive embeddings)
    ↓
TransformerEncoder (1 layer, 1 head)
    ↓
Mean Pooling (across sequence)
    ↓
Output Head:
  - Binary mode: 2 logits (cyclical vs defensive)
  - Multi-factor mode: 6 allocation weights (softmax)
```

### 1.5 Training Strategy

#### 1.5.1 Two Training Approaches

**End-to-End (E2E) 3-Phase Training**:

| Phase | Loss Function | Target | Epochs |
|-------|---------------|--------|--------|
| Phase 1 | Cross-Entropy | Binary: cyclical > defensive? | 20 |
| Phase 2 | MSE | Regression on outperformance score | 15 |
| Phase 3 | -Sharpe approximation | Maximize risk-adjusted returns | 15 |

**Supervised Training**:
- Compute optimal weights w* using rolling Sharpe maximization
- Train model to predict w* directly using MSE loss

#### 1.5.2 Multi-Horizon Support

All models **rebalance monthly** (data is monthly). The difference is **what each model optimizes for** during training:

| Horizon | Rebalancing | Phase 3 Optimization Target |
|---------|-------------|---------------------------|
| 1M | Monthly | 1-month forward return |
| 3M | Monthly | 3-month cumulative forward return |
| 6M | Monthly | 6-month cumulative forward return |
| 12M | Monthly | 12-month cumulative forward return |

**Cumulative return formula**: `(1+r1) × (1+r2) × ... × (1+rN) - 1`

#### 1.5.3 Allocation Modes

**Binary (2F) Mode**:
- Output: 2 weights (cyclical, defensive)
- Target: Binary classification (1 if cyclical > defensive)
- Metric: Accuracy reported

**Multi-factor (6F) Mode**:
- Output: 6 weights (cyclical, defensive, value, growth, quality, momentum)
- Target: Softmax allocation weights
- No accuracy (regression task)

### 1.6 Validation and Backtesting

#### 1.6.1 Walk-Forward Validation Protocol

Classical cross-validation is **strictly excluded**. The protocol uses walk-forward validation with expanding windows:

| Window | Train | Validation | Test |
|--------|-------|------------|------|
| 1 | 2000-2013 | 2014-2016 | 2017-2024 |
| 2 | 2000-2014 | 2015-2017 | 2018-2024 |
| 3 | 2000-2015 | 2016-2018 | 2019-2024 |

**Horizon-aware windows**: For longer horizons, a lookahead buffer is added to prevent data leakage.

#### 1.6.2 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted return | > 0.5 |
| Information Coefficient (IC) | Prediction-return correlation | > 0.05 |
| Max Drawdown | Largest peak-to-trough decline | > -0.25 |
| Accuracy (Binary only) | Classification accuracy | > 55% |

#### 1.6.3 Composite Ranking Score

For multi-criteria model selection:

```
Score = 0.4 × Sharpe_norm + 0.3 × IC_norm + 0.3 × MaxDD_norm
```

Each metric normalized to [0, 1]. Higher score = better model.

---

## 2. Current Implementation

### 2.1 Comparison Framework

**16 Combinations Tested**:

| Strategy | Allocation | Horizons |
|----------|------------|----------|
| E2E (3-phase) | Binary (2F) | 1M, 3M, 6M, 12M |
| E2E (3-phase) | Multi-factor (6F) | 1M, 3M, 6M, 12M |
| Supervised | Binary (2F) | 1M, 3M, 6M, 12M |
| Supervised | Multi-factor (6F) | 1M, 3M, 6M, 12M |

### 2.2 Sample Counts by Horizon

| Horizon | Samples | Loss vs 1M |
|---------|---------|------------|
| 1M | 287 | - |
| 3M | 285 | -1% |
| 6M | 282 | -2% |
| 12M | 276 | -4% |

### 2.3 Key Files

| File | Purpose |
|------|---------|
| `src/main_strategy.py` | Core `FactorAllocationStrategy` class |
| `src/comparison_runner.py` | 16-combination comparison runner |
| `src/multi_horizon_strategy.py` | Multi-horizon orchestrator |
| `src/data/point_in_time_loader.py` | Point-in-Time FRED-MD loader |
| `src/data/factor_data_loader.py` | Factor returns loader |
| `src/models/transformer.py` | `FactorAllocationTransformer` model |
| `src/models/embeddings.py` | `MacroTokenEmbedding` (additive) |
| `src/models/training_strategies.py` | E2E and Supervised trainers |
| `src/utils/walk_forward.py` | Walk-forward validation |
| `notebooks/factor_allocation_demo.ipynb` | Main comparison notebook |

---

## 3. Results and Findings

### 3.1 Current Performance (16 Combinations)

Best results from the comparison (example run):

| Strategy | Allocation | Horizon | Sharpe | IC | Max DD |
|----------|------------|---------|--------|-----|--------|
| Sup | Multi | 6M | +0.99 | +0.14 | -0.04 |
| Sup | Multi | 3M | +1.01 | -0.19 | -0.05 |
| E2E | Multi | 1M | +0.93 | +0.02 | -0.11 |
| E2E | Binary | 3M | +0.78 | +0.26 | -0.35 |

### 3.2 Key Findings

1. **Multi-factor (6F) outperforms Binary (2F)**:
   - Average Sharpe: +0.90 vs +0.70
   - Lower drawdowns with diversified allocation

2. **Supervised training slightly better than E2E**:
   - Average Sharpe: +0.82 vs +0.77
   - More stable training dynamics

3. **3M horizon shows best average Sharpe**:
   - Intermediate horizons capture momentum effects
   - 1M may be too noisy, 12M loses signal

4. **Best composite score**: Sup + Multi @ 6M
   - Balances Sharpe, IC, and drawdown

---

## 4. Problems Identified with Solutions

### 4.1 Transaction Costs and Churning

**Problem**: Frequent allocation changes erode returns through transaction costs.

**Solution Implemented**: Execution threshold in backtest. Only execute allocation changes above a configurable threshold (default: 5% change).

### 4.2 Look-Ahead Bias

**Problem**: Using revised data not available at decision time.

**Solution Implemented**: Point-in-Time FRED-MD loader with 305 vintage files. Each decision uses only data that was actually available at that date, with configurable publication lag.

### 4.3 Overfitting with Limited Data

**Problem**: ~287 monthly observations is extremely limited for deep learning.

**Solution Implemented**:
- MICRO architecture (12k params, 1 layer, 1 head)
- High dropout (0.6)
- Walk-forward validation to detect temporal overfitting
- Progressive training (start simple, add complexity)

### 4.4 Sequence Encoding Efficiency

**Problem**: How to represent multi-dimensional macro tokens efficiently?

**Solution Implemented**: Additive embeddings (BERT-like) where categorical embeddings are summed, not concatenated. Reduces parameter count significantly.

---

## 5. Unsolved Problems and Limitations

### 5.1 Fundamental Data Scarcity

~287 monthly observations (24 years) is inherently limited. The model has seen only 2-3 major crises and 5-6 economic cycles. Generalization to new regimes remains uncertain.

### 5.2 Non-Stationarity

The relationship between macro indicators and factor performance evolves. Post-2008 zero-rate era changed many dynamics. The model may learn obsolete patterns.

### 5.3 Market Efficiency

Macroeconomic data is public and widely analyzed. Any predictable signal may already be priced in. The model must find non-obvious interactions to add value.

### 5.4 Single Region

Current implementation focuses on US data only. Regional models for Europe, Japan, etc. are designed but not yet implemented.

---

## 6. Recommendations and Next Steps

### 6.1 Immediate Priorities

1. **Walk-forward OOS evaluation**: Run proper out-of-sample validation across multiple windows
2. **Transaction cost sensitivity**: Test with realistic transaction costs (10-50 bps)
3. **Regime analysis**: Analyze performance across different market regimes

### 6.2 Future Development

1. **Regional expansion**: Implement Europe and Japan models
2. **Feature selection**: Identify most predictive indicators
3. **Ensemble methods**: Combine multiple horizon models
4. **Alternative architectures**: Test LSTM, attention-only variants

### 6.3 Stop Criteria

| Condition | Decision |
|-----------|----------|
| OOS Sharpe < 0.3 consistently | Reassess strategy fundamentals |
| IC near zero across all horizons | Signal may not exist |
| Large train-test gap (>0.5 Sharpe) | Overfitting despite regularization |

---

*Document updated: 2026-01-31*
*Version: 2.0*
*Status: Research/Development*
