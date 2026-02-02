# Neural Network Strategy for Factor Allocation Based on Macroeconomic Data

## Executive Summary

This document describes a dynamic allocation strategy across equity factor styles using a Transformer-based model fed by point-in-time macroeconomic data. The goal is to maximize risk-adjusted performance by anticipating style rotations (cyclical, defensive, value, growth, quality, momentum) from macro signals.

**Current Implementation Status**: Research/Development phase with a MICRO Transformer architecture (~12k parameters) trained on Point-in-Time FRED-MD data (305 vintage files, 112 indicators). Features rigorous 3-step evaluation methodology with statistical analysis.

---

## 1. Strategy Overview

### 1.1 Objective

Develop a neural network model that takes monthly macroeconomic data as input and produces allocation weights across 6 factor categories, maximizing risk-adjusted performance over different time horizons (1M, 3M, 6M, 12M).

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

The model uses **Point-in-Time FRED-MD** macroeconomic data exclusively.

**Why Point-in-Time?** Standard economic data is revised after initial release. Using revised data (available only in hindsight) creates look-ahead bias. Point-in-Time data uses the actual values that were available at each decision date.

**Data Pipeline**:
- **Source**: FRED-MD vintage files (305 files from 1999-08 to 2024-12)
- **Loader**: `PointInTimeFREDMDLoader` with publication lag handling
- **Transformations**: FRED-MD standard transformations (log, diff, etc.)

**Key Indicator Categories** (112 total):

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

#### 1.2.2 Sequence Structure

The model input is a sequence of 12 monthly observations (1 year lookback).

**Why 12 months?** Captures seasonal patterns and medium-term trends. Longer sequences (24-36 months) showed diminishing returns and increased overfitting risk with limited data.

| Attribute | Dimension | Description |
|-----------|-----------|-------------|
| Macro features | 112 | FRED-MD indicators (transformed) |
| Market context | 3-5 | Term spread, credit spread, VIX |
| Total per timestep | ~120 | All features concatenated |

### 1.3 Input Encoding

The encoding uses an **additive embeddings approach** (BERT-like):

```
E_total = E_indicator + E_temporal + E_category + E_region + E_periodicity
```

**Why additive (not concatenated)?** Concatenation would create a huge embedding vector (32×5 = 160 dims). Addition projects all semantic information into a shared 32-dimensional space, dramatically reducing parameters while preserving relationships.

| Embedding | Dimension | Description |
|-----------|-----------|-------------|
| E_indicator | 32 | Learned embedding per indicator |
| E_temporal | 32 | Positional encoding (RoPE) |
| E_category | 32 | Macro category embedding |

Numerical values are projected and added:
```
X_token = LayerNorm(E_total + Linear(numerical_values))
```

---

## 2. Model Architecture

### 2.1 MICRO Transformer Design Philosophy

**The Core Problem**: We have only ~300 monthly samples (25 years of data). Standard Transformers (GPT-2: 117M params, BERT: 110M params) are designed for millions to billions of tokens. Using such architectures would guarantee overfitting.

**Our Solution**: A deliberately minimal "MICRO" Transformer.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| d_model | 32 | Smallest dimension capturing indicator relationships. Larger (64, 128) showed overfitting. |
| num_layers | 1 | Single layer forces direct relationships without complex abstractions. |
| num_heads | 1 | Multiple heads add capacity the data can't support. |
| d_ff | 64 | 2×d_model, standard ratio. |
| dropout | 0.75 | Extremely high by DL standards, but necessary for tiny datasets. |
| **Total params** | **~12,000** | Rule: params << samples × features |

**Why not use simpler models (linear, XGBoost)?** We tested them. The Transformer's attention mechanism captures temporal dependencies and indicator interactions that simpler models miss. But the Transformer must be constrained to prevent memorization.

### 2.2 Rotary Positional Embeddings (RoPE)

**Why RoPE instead of sinusoidal/learned positional embeddings?**

- **Sinusoidal**: Encodes absolute positions ("this is position 3"). For time series, relative positions matter more ("3 months apart").
- **Learned**: Would overfit with 12-token sequences and limited data.
- **RoPE**: Encodes relative positions directly in the dot-product attention. "3 months apart" is the same relationship regardless of where it occurs in the sequence.

RoPE applies rotation matrices to query and key vectors, preserving relative distance information in attention scores.

### 2.3 Temporal Decay in Attention

**Why temporal decay?** Recent data is more relevant for factor timing than distant data. A macro release from last month matters more than one from 11 months ago.

**Implementation**: Learnable decay parameter `exp(-decay_rate × temporal_distance)` applied to attention scores. The model learns optimal decay rate during training.

**Why combined with causal masking?** Causal mask prevents information leakage (future→past). Temporal decay adds soft weighting within valid timesteps.

### 2.4 Architecture Flow

```
Input (batch, seq_len=12, num_indicators=112)
    ↓
MacroTokenEmbedding (additive embeddings + RoPE)
    ↓
TransformerBlock (Pre-LN, 1 layer, 1 head, temporal decay)
    ↓
Mean Pooling (across sequence)
    ↓
Output Head:
  - Binary mode: 2 weights (cyclical, defensive)
  - Multi-factor mode: 6 weights (softmax over all factors)
```

---

## 3. Training Strategy

### 3.1 Two Training Approaches

We implement two distinct training strategies. Each has trade-offs.

#### 3.1.1 End-to-End (E2E) 3-Phase Training

**Why 3 phases instead of direct Sharpe optimization?**

Sharpe loss landscape has flat regions and local minima. Random initialization often gets stuck. Progressive training guides weights to a good region before the complex optimization.

| Phase | Loss | Target | Why |
|-------|------|--------|-----|
| Phase 1 | Cross-Entropy | Binary: cyclical > defensive? | Warm-up with simple task. Clear gradients establish basic macro→factor relationships. |
| Phase 2 | MSE | Regression on excess return | Bridge to continuous outputs. Stable gradients before complex Sharpe landscape. |
| Phase 3 | Sortino | Maximize risk-adjusted return | Direct portfolio optimization with weights now in reasonable region. |

#### 3.1.2 Supervised Training

**Approach**: Compute optimal weights w* using **FORWARD returns** (Sharpe maximization via scipy SLSQP), then train model to predict w* via MSE loss.

**Critical: Forward-Looking Targets**
- For each date T, optimal weights are computed on returns from **T+1 to T+horizon**
- The model learns: "given features at T, predict weights optimal for the NEXT period"
- This is NOT using past performance to predict future (which would be naive)

**Why forward returns?** The model should learn to predict what WILL BE optimal, not what WAS optimal in the past. Using past 24M returns would teach the model to extrapolate trends, which doesn't work in non-stationary markets.

**When to use Supervised vs E2E?**
- **Supervised**: More stable, interpretable targets. Ground-truth optimal weights.
- **E2E**: Can discover patterns beyond optimal-weight approximation. Harder to train.

#### 3.1.3 Ablation Test: Phase 3 Only

**Option**: `skip_phase1_phase2=True` runs E2E with only Phase 3 (no curriculum learning).

**Purpose**: Test if the 3-phase curriculum is beneficial or just adds complexity. If Phase 3-only performs similarly, the curriculum may be unnecessary.

### 3.2 Loss Functions

#### 3.2.1 SortinoLoss (Default)

**Why Sortino over Sharpe?**

Sharpe ratio penalizes ALL volatility:
```
Sharpe = mean(R) / std(R)
```

For investments, upside volatility is *desirable*. A model that generates occasional large gains should not be penalized for it.

Sortino only penalizes downside deviation:
```
Sortino = mean(R) / downside_std(R)
```
where `downside_std = std(R where R < target)`.

**When does Sharpe make sense?** For benchmarking and industry comparison. For training, Sortino is superior.

#### 3.2.2 SharpeRatioLoss

Used as alternative when comparing to industry standards:
```
Loss = -mean(R) / running_std(R) × √12 + λ×turnover
```

Running std (exponential moving average) provides stable gradients. `√12` annualizes for monthly data.

### 3.3 CompositeEarlyStopping

**Why not stop on validation loss alone?**

Loss can plateau while actual trading performance varies. A model with good loss might have poor IC (no directional skill) or large drawdowns (tail risk).

**Composite stopping monitors multiple metrics**:
- **Sharpe**: Overall risk-adjusted return (weight: 0.35)
- **IC**: Predictive accuracy/directional skill (weight: 0.25)
- **MaxDD**: Tail risk/catastrophic loss avoidance (weight: 0.30)
- **Return**: Absolute performance bonus (weight: 0.10)

Stops when composite score doesn't improve for 5 epochs (patience).

### 3.4 Regularization Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| dropout | 0.75 | Each step sees only 25% of activations, forcing redundancy |
| weight_decay | 0.05 | Strong L2 penalty prevents memorization |
| learning_rate | 0.0005 | Conservative LR for stable convergence with high dropout |
| batch_size | 64 | Larger batches for stable gradients (~300 total samples) |

### 3.5 Fair Epoch Comparison

**Problem**: E2E has 3 phases (30+20+20=70 epochs) while Supervised had only 20 epochs. Unfair comparison.

**Solution**: `epochs_supervised=70` (same total as E2E).

| Strategy | Epochs | Breakdown |
|----------|--------|-----------|
| E2E | 70 | Phase1: 30, Phase2: 20, Phase3: 20 |
| Supervised | 70 | Single phase with MSE loss |

This ensures performance differences are due to training approach, not training time.

### 3.6 Multi-Horizon Support

All models **rebalance monthly** (data is monthly). The difference is the **optimization target**:

| Horizon | Optimization Target | Trade-off |
|---------|---------------------|-----------|
| 1M | 1-month forward return | High noise, captures short-term signals |
| 3M | 3-month cumulative return | Balanced noise/signal |
| 6M | 6-month cumulative return | Smoother signal, slower adaptation |
| 12M | 12-month cumulative return | Long-term trends, loses short-term signals |

**Cumulative return formula**: `(1+r₁) × (1+r₂) × ... × (1+rₙ) - 1`

---

## 4. Evaluation Methodology

### 4.1 Rigorous 3-Step Protocol

**Why 3 steps instead of simple train/test split?**

| Approach | Problem |
|----------|---------|
| Simple train/test | Single OOS estimate, high variance |
| Walk-forward only | Models trained on different data amounts (bias) |
| **3-step** | Multiple OOS windows + unbiased final comparison |

#### Step 1: Walk-Forward Validation (2000-2021)

**Purpose**: Generate multiple out-of-sample performance estimates.

**Why expanding (not sliding) windows?** Limited data. Sliding windows would discard precious early samples.

**Why non-overlapping test periods?** Overlapping tests inflate OOS estimates due to autocorrelation in returns.

```
Window 1: Train(2000-2014) → Val(2015-2017) → Test(2018-2021)
Window 2: Train(2000-2015) → Val(2016-2018) → Test(2019-2021)
Window 3: Train(2000-2016) → Val(2017-2019) → Test(2020-2021)
```

**Critical**: Holdout period (2022+) is NEVER used in walk-forward, even indirectly through early stopping.

#### Step 2: Final Model Training (2000-2021)

**Purpose**: Train production-ready model on all non-holdout data.

**Why train a separate final model?** WF models were trained on varying data amounts (Window 1 has less data than Window 3). Final model uses all available data for deployment.

**Why different seed (999)?** Ensures final model is independent of WF models.

#### Step 3: Holdout Evaluation (2022+)

**Purpose**: True out-of-sample comparison of three model types.

| Model Type | Description | Bias |
|------------|-------------|------|
| Final | Single model trained on 2000-2021 | None |
| Fair Ensemble | 5 models on same data (2000-2021), different seeds | None (seed only) |
| WF Ensemble | Models from walk-forward windows | Data-quantity bias |

### 4.2 Fair Ensemble vs WF Ensemble

**Why is WF Ensemble biased?**

Walk-forward windows have different training data amounts:
- Window 1: 14 years training
- Window 3: 16 years training
- Window 5: 18 years training

Performance differences could be due to data quantity, not model quality.

**Why Fair Ensemble?**

Trains 5 models on **identical** data (2000-2021) with different seeds (42, 142, 242, 342, 442). This isolates:
- Seed variance (random initialization effects)
- Pure ensemble benefit (averaging reduces variance)

**Why keep WF Ensemble?** Shows real-world walk-forward scenario and quantifies the data-quantity bias.

### 4.3 Evaluation Metrics

| Metric | Formula | Target | Why |
|--------|---------|--------|-----|
| Sharpe Ratio | mean(R)/std(R) × √12 | > 0.5 | Industry standard risk-adjusted return |
| Information Coefficient | Spearman(predictions, returns) | > 0.05 | Directional skill (does model predict direction?) |
| Max Drawdown | min(wealth/peak - 1) | > -25% | Tail risk (worst loss from peak) |
| Total Return | ∏(1+rᵢ) - 1 | > 0 | Absolute performance |

---

## 5. Composite Scoring System

### 5.1 Why Composite Score?

Single metrics are insufficient:
- **Sharpe alone**: Ignores predictive skill (IC) and tail risk (MaxDD)
- **IC alone**: Ignores magnitude of returns
- **MaxDD alone**: Ignores average performance

### 5.2 Score Formula

```
Score = 0.35×Sharpe_norm + 0.25×IC_score + 0.30×MaxDD_norm + 0.10×Return_bonus - IC_penalty
```

### 5.3 Asymmetric IC Penalty

**Why penalize negative IC 2× more?**

| IC Value | Interpretation | Treatment |
|----------|----------------|-----------|
| Positive | Model predicts correctly | Valuable |
| Zero | Random predictions | No value |
| Negative | Inverted predictions | Dangerous |

Negative IC is *worse* than zero because it could be exploited in reverse. An adversary could take opposite positions.

**Formula**: `IC_penalty = 2 × max(-IC, 0)` for IC < 0

**Rejection criterion**: IC < -30% → score = 0 (model rejected)

### 5.4 Exponential MaxDD Penalty

**Why exponential instead of linear?**

Linear assumes -20% is 2× worse than -10%. Reality:
- -10%: Uncomfortable but manageable
- -20%: Margin calls, fund redemptions, career risk
- -30%: Fund closure, reputational damage

**Formula**: `MaxDD_score = exp(3 × maxdd)` where maxdd is negative

| MaxDD | Score |
|-------|-------|
| -5% | 0.86 |
| -10% | 0.74 |
| -15% | 0.64 |
| -20% | 0.55 |
| -30% | 0.41 |

---

## 6. Statistical Analysis

### 6.1 Why Statistical Rigor?

Backtest results can be lucky. With ~300 samples, variance is high. Before deploying capital:
- Need confidence intervals (how certain are we?)
- Need significance tests (is this better than random?)

### 6.2 Kelly Criterion

**Purpose**: Optimal position sizing given edge and variance.

**Formula**: `f* = (μ - r) / σ²` (edge over variance)

**Why half-Kelly?** Full Kelly maximizes log-wealth but has high variance. Half-Kelly reduces variance significantly with small return sacrifice. More robust to estimation errors.

### 6.3 Bootstrap Confidence Intervals

**Why bootstrap instead of parametric?**

Return distributions are non-normal (fat tails, skew). Parametric CIs assuming normality underestimate uncertainty.

**Method**: Resample returns with replacement 1000 times, compute Sharpe for each, take percentile CIs.

**Example output**: "Sharpe = 0.6 [95% CI: 0.3, 0.9]" is more useful than "Sharpe = 0.6".

### 6.4 Significance Tests

**Lo (2002) for single Sharpe**: Tests if Sharpe ≠ 0 accounting for autocorrelation in returns. Standard t-test assumes independence, overstating significance.

**Jobson-Korkie for comparison**: Tests if two Sharpes are significantly different, accounting for correlation between strategies.

---

## 7. Configuration Options

### 7.1 Feature Selection

**Why feature selection?** 112 FRED-MD indicators may include noise. Selection improves signal-to-noise ratio.

**Method**: Mutual information (default)
- **Why mutual info over correlation?** Captures non-linear dependencies. Correlation only measures linear relationships.

**Default**: 30 features (empirically balances retention vs noise)

**Critical: 1-Month Feature Lag**

Without lag: Selection sees features[M] → target[M] (look-ahead bias)
With lag: Selection sees features[M] → target[M+1] (realistic)

Selection is fit ONCE on training data, then applied to all windows.

### 7.2 Hyperparameter Tuning

**Why walk-forward HP tuning?** Default hyperparameters may not be optimal for all horizons/allocations. Walk-forward tuning optimizes per window, preventing overfitting to specific periods.

**Trials**: 15 (balances exploration vs compute cost)

**Tuned parameters**: learning_rate, dropout, weight_decay

### 7.3 Configuration Combinations

| Config | Feature Selection | HP Tuning | Use Case |
|--------|-------------------|-----------|----------|
| baseline | No | No | Pure model performance |
| fs | Yes | No | Noise reduction |
| hpt | No | Yes | Hyperparameter optimization |
| fs+hpt | Yes | Yes | Maximum optimization |

**Total combinations**: 2 strategies × 2 allocations × 4 horizons × 4 configs = **64**

---

## 8. Benchmarks

### 8.1 Why Multiple Benchmarks?

No single benchmark captures all aspects of factor timing. Model should beat *relevant* benchmarks.

**Removed: Market (Mkt-RF)**. This strategy is about *factor timing*, not market vs risk-free. Market benchmark is irrelevant.

### 8.2 Benchmark Strategies

| Benchmark | Logic | Tests |
|-----------|-------|-------|
| Equal-Weight 6F | 1/6 each factor | Model vs naive diversification |
| 50/50 Cyc/Def | Static 50% cyclical + 50% defensive | Model vs simple macro allocation |
| Risk Parity | Inverse volatility weighting | Model vs volatility-based timing |
| Factor Momentum | Equal weight to positive trailing 12M factors | Model vs trend-following |
| Best Single Factor | 100% in best in-sample Sharpe factor | Model vs perfect hindsight |

---

## 9. Streamlit Dashboard

### 9.1 Why a Dashboard?

- Notebooks are good for development but poor for exploration
- Interactive filters enable hypothesis testing without code changes
- Stakeholders can explore results without running code

### 9.2 Key Features

- **Filters**: Strategy, Allocation, Horizon, Config, Model Type
- **Adjustable score weights**: Test sensitivity of rankings
- **Factor allocation charts**: Visualize weight evolution over holdout period
- **Benchmark comparison**: Context for model performance
- **Statistical summary**: Bootstrap CIs, significance tests

**How to run**: `streamlit run dashboard/app.py`

---

## 10. Problems Identified with Solutions

### 10.1 Transaction Costs and Churning

**Problem**: Frequent allocation changes erode returns.

**Solution**: Execution threshold. Only execute changes above 5% to reduce turnover.

### 10.2 Look-Ahead Bias

**Problem**: Using revised data not available at decision time.

**Solution**: Point-in-Time FRED-MD loader (305 vintages). Each decision uses only data actually available at that date.

### 10.3 Overfitting with Limited Data

**Problem**: ~300 samples is extremely limited for deep learning.

**Solutions**:
- MICRO architecture (12k params)
- High dropout (0.75)
- Strong weight decay (0.05)
- Walk-forward validation
- Early stopping on composite score

### 10.4 Feature Selection Look-Ahead Bias

**Problem**: Selecting features using full dataset means seeing future information.

**Solution**: Lag features by 1 month. Selection uses features[M] → target[M+1].

### 10.5 Walk-Forward Data-Quantity Bias

**Problem**: WF ensemble combines models trained on different data amounts.

**Solution**: Fair Ensemble trains N models on same data, isolating seed variance from data quantity.

### 10.6 Return Alignment (Critical Fix)

**Problem**: For E2E Phase 3 and Supervised training, returns must be **forward-looking** (T+1 to T+horizon), not contemporaneous (T to T+horizon-1).

**Why this matters**: If returns at index i use data from time i (contemporaneous), the model learns correlations that don't exist at prediction time. The model sees features at T and should optimize for returns starting at T+1.

**Solution**: All cumulative return calculations use `returns[i+1:i+1+horizon]` (forward shift).

| Component | Before (Bug) | After (Fixed) |
|-----------|--------------|---------------|
| `cumulative_returns[h=1]` | `returns[i]` | `returns[i+1]` |
| `cumulative_returns[h>1]` | `returns[i:i+h]` | `returns[i+1:i+1+h]` |
| Supervised targets | Past 24M returns | Forward h-month returns |

---

## 11. Unsolved Problems and Limitations

### 11.1 Fundamental Data Scarcity

~300 monthly observations (25 years) is inherently limited. The model has seen only 2-3 major crises and 5-6 economic cycles. Generalization to new regimes remains uncertain.

### 11.2 Non-Stationarity

The relationship between macro indicators and factor performance evolves. Post-2008 zero-rate era changed many dynamics. The model may learn obsolete patterns.

### 11.3 Market Efficiency

Macroeconomic data is public and widely analyzed. Any predictable signal may already be priced in. The model must find non-obvious interactions to add value.

### 11.4 Single Region

Current implementation focuses on US data only. Regional models for Europe, Japan, etc. are designed but not yet implemented.

---

## 12. Key Files

| File | Purpose |
|------|---------|
| `src/main_strategy.py` | Core `FactorAllocationStrategy` class |
| `src/comparison_runner.py` | Comparison runner with all training functions |
| `src/pipelines/three_step_pipeline.py` | 3-step evaluation orchestrator |
| `src/data/point_in_time_loader.py` | Point-in-Time FRED-MD loader |
| `src/models/transformer.py` | `FactorAllocationTransformer` with RoPE |
| `src/models/training_strategies.py` | E2E and Supervised trainers |
| `src/utils/walk_forward.py` | Walk-forward validation |
| `src/utils/statistics.py` | Kelly, Bootstrap CI, significance tests |
| `src/utils/analysis.py` | Holdout result analysis functions |
| `src/visualization/` | All plotting functions |
| `dashboard/app.py` | Streamlit dashboard |
| `notebooks/factor_allocation_quick.ipynb` | Main evaluation notebook |

---

## 13. Recommendations and Next Steps

### 13.1 Immediate Priorities

1. **Transaction cost sensitivity**: Test with realistic costs (10-50 bps)
2. **Regime analysis**: Performance across different market regimes
3. **Statistical significance**: Ensure all results pass significance tests

### 13.2 Future Development

1. **Regional expansion**: Europe and Japan models
2. **Alternative architectures**: Test LSTM, attention-only variants
3. **Ensemble methods**: Combine multiple horizon models
4. **Real-time deployment**: Production pipeline with live data

### 13.3 Stop Criteria

| Condition | Decision |
|-----------|----------|
| OOS Sharpe < 0.3 consistently | Reassess strategy fundamentals |
| IC near zero across all horizons | Signal may not exist |
| Large train-test gap (>0.5 Sharpe) | Overfitting despite regularization |
| Significance tests fail | Results may be luck |

---

*Document updated: 2026-02-02*
*Version: 3.0*
*Status: Research/Development*
