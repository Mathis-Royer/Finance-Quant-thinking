# Neural Network Strategy for Factor Allocation Based on Macroeconomic Data

## Executive Summary

This document describes a dynamic allocation strategy across equity categories (factor styles) using a Transformer-based model fed by real-time macroeconomic data. The goal is to maximize risk-adjusted performance by anticipating style rotations (value, growth, cyclical, defensive, etc.) from macro signals. A separate model and portfolio is maintained for each geographic region, allowing the allocation of the appropriate portfolio to be adjusted based on the region of the incoming data.

---

## 1. Detailed Strategy

### 1.1 Objective

The objective is to develop a neural network model that takes daily fundamental macroeconomic data as input and produces allocation weights across equity categories, with the goal of maximizing performance and the Sharpe ratio over different time horizons (1 week, 1 month, 3 months, 6 months, 1 year).

The performance of each category is calculated as the capitalization-weighted average of the stocks composing that category. Target categories are defined by established factor indices (MSCI Factor Indices, S&P Pure Style Indices) to ensure a standardized definition and consistent historical track record.

A key architectural decision is to train one model per geographic region (US, Europe, Japan, etc.), each managing its own dedicated portfolio. When new macroeconomic data arrives, the system identifies the relevant region and updates only the corresponding regional model and portfolio. This regional separation ensures that the model learns relationships specific to each economic zone, since the link between macro indicators and factor performance can differ significantly across regions.

### 1.2 Input Data

#### 1.2.1 Sequence Structure

The model input is a sequence of 50 to 100 macroeconomic tokens, updated as soon as new data becomes available. The new data is placed at the head of the sequence, with the oldest tokens being dropped to maintain a fixed length.

Each token in the sequence contains the following information:

| Attribute | Description | Type |
|-----------|-------------|------|
| Data name | Unique indicator identifier (e.g., PMI_Manufacturing, CPI_Core) | Categorical |
| Publication type | Consensus, consensus revision, estimate, estimate revision | Categorical |
| Importance | Importance score of the indicator (1-3) | Numerical |
| Normalized value | Value normalized by the indicator's historical distribution | Numerical |
| 5-period moving average | Average of the last 5 publications (for revisions) | Numerical |
| Standardized surprise | Deviation from expectation divided by the historical standard deviation of surprises | Numerical |
| Country/Region | Country or economic zone concerned | Categorical |
| Periodicity | Publication frequency (can be included in the name if relevant) | Categorical |
| Temporal information | Time offset relative to the present moment (in days) | Numerical |

The standardized surprise is calculated according to the data type: for a consensus, it is the difference from the last estimate revision; for an estimate, it is the difference from the last consensus revision; for an estimate revision, it is the difference from the estimate. Only the most recent revision of each type is kept in the sequence at any given time.

#### 1.2.2 Market Context Data

In addition to macro data, market condition indicators are integrated as supplementary tokens in the sequence:

| Indicator | Role | Calculation |
|-----------|------|-------------|
| Credit spread | Proxy for credit stress and risk aversion | HY - IG or HY - Treasury |
| Yield curve slope | Proxy for growth expectations | 10Y rate - 2Y rate |
| VIX | Proxy for implied volatility and market fear | CBOE VIX index level |

These indicators allow the model to contextualize the interpretation of macro data according to the current market regime (risk-on vs risk-off).

#### 1.2.3 Data Quality Requirements

All data must come from point-in-time databases to avoid any look-ahead bias. Vintages must be meticulously reconstructed to reflect exactly the information available at each decision date. The type structure (consensus, revision, estimate) inherently allows for distinguishing different informational states.

### 1.3 Input Encoding

The encoding uses an additive embeddings approach for each macroeconomic token:

```
E_total = E_identity + E_type + E_importance + E_temporal + E_category + E_country
```

| Embedding | Suggested Dimension | Description |
|-----------|---------------------|-------------|
| E_identity | 32-64 | Learned embedding specific to each indicator. The model learns to position similar indicators close together in latent space |
| E_type | 8-16 | Embedding of the publication type (consensus, flash, revised, final) |
| E_importance | 8 | Projection of the importance score via a linear layer |
| E_temporal | 16-32 | Continuous sinusoidal positional encoding based on the number of days since today |
| E_category | 16 | Thematic category embedding (economic activity, employment, inflation, consumption, monetary policy) |
| E_country | 8-16 | Country or economic zone embedding |

Continuous numerical values (normalized value, standardized surprise, moving average) are concatenated with the embeddings and then passed through a linear layer followed by normalization:

```
X_token = LayerNorm(Linear(concat(E_total, [val_norm, surprise, MA5])))
```

### 1.4 Model Architecture

#### 1.4.1 Base Configuration

The model uses a simplified Transformer architecture with the following hyperparameters:

| Parameter | Recommended Value |
|-----------|-------------------|
| Number of layers | 2-4 |
| Embedding dimension | 64-128 |
| Number of attention heads | 2-4 |
| Dropout | 0.3-0.5 |

Complexity is deliberately limited given the relative scarcity of training data.

#### 1.4.2 Attention Mechanisms

The model uses relative positional embeddings rather than absolute ones, since the absolute position in the sequence matters less than the temporal relationships between tokens. The attention mechanism must capture several types of relationships:

| Relationship Type | Description |
|-------------------|-------------|
| Intra-indicator temporal | Relationship between different publications of the same indicator |
| Inter-indicator contemporaneous | Relationship between different indicators at the same period |
| Causal | Masking preventing access to future data |

A standard causal mask is applied to ensure the model cannot use future information. Soft masks based on temporal distance can be added so that attention naturally decreases with temporal distance.

#### 1.4.3 Incorporating Prior Economic Knowledge

Several mechanisms allow injecting economic knowledge into the model:

| Method | Implementation |
|--------|----------------|
| Embedding structure | Embeddings by macro category impose proximity between thematically related indicators |
| Skip connections | Allow the model to easily learn a linear relationship (econometric baseline) |
| Embedding pre-training | Auxiliary task of classifying indicators by category |
| Baseline regularization | Loss term penalizing deviations from a simple econometric model (ridge regression) |

### 1.5 Model Output and Loss Function

#### 1.5.1 Output: Allocation Weights with Gated Execution

The model outputs allocation weights that sum to one across the target factor categories. However, to address the problem of excessive turnover and transaction costs, a separate decision layer determines whether to actually execute the new allocation or maintain the current one. This gating mechanism compares the difference between the newly predicted weights and the current portfolio weights against a threshold. If the predicted change is below this threshold, the current allocation is retained; if it exceeds the threshold, the new allocation is implemented.

This two-stage architecture (prediction followed by gated execution) decouples the prediction task from the execution decision, allowing the model to be trained on pure prediction quality while the execution threshold can be calibrated separately based on actual transaction costs. The threshold can be implemented as a hard cutoff or as a soft-thresholding function (non-differentiable or differentiable depending on training needs).

#### 1.5.2 Loss Function: Progressive Approach

The training strategy follows a progression in complexity:

| Phase | Loss Function | Target |
|-------|---------------|--------|
| Phase 1 (concept validation) | Cross-Entropy | Binary classification: do cyclicals outperform defensives? |
| Phase 2 (refinement) | MSE | Relative outperformance score (regression) |
| Phase 3 (production) | -Sharpe approximation + λ × turnover | Allocation maximizing Sharpe with change penalty |

Phase 1 uses the following formulation:

```
Loss = CrossEntropy(predicted, y)
where y = 1 if R_cyclicals > R_defensives, 0 otherwise
```

Phase 3 will use a differentiable approximation of the Sharpe ratio with a turnover penalty:

```
Loss = -E[R] + γ × Var[R] + λ × ||w_t - w_{t-1}||₁
```

The coefficient λ must be calibrated based on actual transaction costs.

### 1.6 Validation and Backtesting

#### 1.6.1 Protocol: Walk-Forward Validation Only

Classical cross-validation is strictly excluded because it violates the temporal order of the data. The protocol uses walk-forward validation with an expanding window:

| Window | Train | Validation | Test |
|--------|-------|------------|------|
| 1 | 2000-2014 | 2015-2017 | 2018-2024 |
| 2 | 2000-2015 | 2016-2018 | 2019-2024 |
| 3 | 2000-2016 | 2017-2019 | 2020-2024 |

Performance is averaged across multiple windows. A final holdout is reserved and will be consulted only once to avoid data snooping.

#### 1.6.2 Tested Time Horizons

| Horizon | Signal Characteristic |
|---------|----------------------|
| 1 week | Sensitive to macro surprises |
| 1 month | Momentum and expectation revision |
| 3 months | Intermediate trends |
| 6 months | Economic cycle |
| 1 year | Absolute level and complete cycle |

### 1.7 Progressive Development Approach

Given the difficulty of the problem and the potentially low signal-to-noise ratio, the development strategy follows a step-by-step progression with validation at each level:

| Step | Model | Objective |
|------|-------|-----------|
| 1 | Naive baseline (style momentum) | Establish a benchmark to beat |
| 2 | Logistic Regression / Ridge | Validate the existence of a predictive signal |
| 3 | Gradient Boosting (XGBoost, LightGBM) | Capture non-linear interactions |
| 4 | Simple LSTM/GRU (1-2 layers, 32-64 units) | Test the value of sequential modeling |
| 5 | Minimal Transformer | Exploit attention if previous steps show signal |

Progression to the next step is justified only if the current step shows significant improvement over the baseline.

The initial test case focuses on a simplified configuration: predicting the relative outperformance of "cyclicals vs defensives" at 1 month.

---

## 2. Problems Identified with Solutions

### 2.1 Transaction Costs and Churning

**Problem**: The model may recommend frequent allocation changes whose transaction costs exceed potential gains. This tendency is amplified by direct performance optimization without consideration of market frictions.

**Solution adopted**: The model outputs allocation weights (summing to one), but a separate gating layer determines whether to actually execute the new allocation. This layer compares the magnitude of the proposed change against a threshold calibrated to transaction costs. If the change is below the threshold, the current allocation is maintained; if above, the new weights are implemented. This architecture decouples prediction quality from execution decisions and allows the action threshold to be adjusted independently of model training.

### 2.2 Market Context Integration

**Problem**: The market context (risk-on vs risk-off) influences how styles react to macro data. The same signal can have different implications depending on the market regime.

**Solution adopted**: Integrate market condition indicators as supplementary features in the input sequence:

| Indicator | Role |
|-----------|------|
| Credit spread (HY - IG or HY - Treasury) | Proxy for credit stress and risk aversion |
| Yield curve slope (10Y - 2Y) | Proxy for economic growth expectations |
| VIX | Proxy for implied volatility and market fear |

These indicators are treated as special "market" tokens in the sequence, allowing the attention mechanism to weight macro data differently according to context.

### 2.3 Look-Ahead Bias in Data

**Problem**: Using revised data or "final" consensus values that were not available at the time of the investment decision leads to artificially inflated backtest performance that is not reproducible in production.

**Solution adopted**: Point-in-time databases are used exclusively, with sources providing historical vintages (Bloomberg, Refinitiv, ALFRED from the Fed). Vintages are meticulously reconstructed to document precisely the information available at each decision date. The type structure (consensus/revision/estimate) in the input data inherently reflects the informational state.

### 2.4 Efficient Sequence Encoding

**Problem**: How to efficiently represent the multiple dimensions of information for each macro token (identity, type, temporality, category, numerical values) without dimensionality explosion?

**Solution adopted**: Additive embeddings architecture where different categorical embeddings are summed (not concatenated), then combined with numerical features via a linear projection. This approach is proven (similar to BERT) and avoids the unnecessary complexity of embeddings of embeddings.

### 2.5 Appropriate Architecture for Limited Data

**Problem**: Transformers are designed for massive corpora, while historical macro data is inherently limited (a few thousand temporal points at most).

**Solution adopted**: Minimalist Transformer configuration (2-4 layers, 64-128 dimensions, 2-4 heads, high dropout 0.3-0.5) with fewer than 10k parameters to start. Gradual increase in complexity only if the data justifies it.

### 2.6 Loss Function and Prediction Target

**Problem**: Directly predicting the Sharpe ratio or optimal weights is difficult because these targets are unstable and the solution space is vast.

**Solution adopted**: Progressive approach starting with simple binary classification (which category outperforms?) with Cross-Entropy, then evolving toward score regression, and finally toward differentiable Sharpe optimization once the concept is validated.

### 2.7 Injecting Economic Knowledge

**Problem**: How to benefit the model from established knowledge in financial economics rather than learning everything from scratch with limited data?

**Solution adopted**: Combination of several mechanisms including embeddings structured by category so that thematically related indicators are represented as close together, skip connections so the model can easily learn simple linear relationships, regularization toward baseline with a penalty on deviations from a reference econometric model, and pre-training with an auxiliary task of classifying indicators.

### 2.8 Temporal Validation Protocol

**Problem**: Standard cross-validation mixes past and future, creating unrealistic backtest performance.

**Solution adopted**: Exclusive walk-forward validation with expanding window, repeated over several periods. A final holdout is reserved for ultimate evaluation to avoid data snooping.

### 2.9 Problem Complexity and Over-Engineering Risk

**Problem**: The risk of developing a complex model when the underlying signal is weak or non-existent.

**Solution adopted**: Progressive development approach with validation at each step. Starting with simple models (regression, gradient boosting) to confirm the existence of a signal before investing in the Transformer architecture.

---

## 3. Unsolved Problems

### 3.1 Fundamental Scarcity of Macroeconomic Data

**Problem description**: Transformers are designed to be trained on corpora of millions or even billions of tokens. Even with 30 years of macro data history (which is optimistic for some indicators), we have approximately 360 monthly points or 1,560 weeks. A sequence of 50-100 macro tokens multiplied by a few thousand temporal points remains extremely limited for a deep learning model.

**Implications**: The model will have seen very few examples of each "type" of economic situation. Economic regimes can be counted on one hand in the available history: 5-6 complete economic cycles, 2-3 major crises (2008, 2020, 2022). The ability to generalize to new regimes is therefore structurally limited.

**Why this problem has no simple solution**: Unlike computer vision, we cannot artificially generate realistic macro data through data augmentation. Synthetic data risks injecting biases or non-existent patterns. Using data from other countries poses transferability problems since the relationship between macro indicators and style performance can differ significantly across geographic zones.

**Partial mitigation**: The progressive approach (starting simple) allows detecting whether a signal exists before investing in a complex model. Aggressive regularization limits overfitting. But the fundamental constraint of data quantity remains unavoidable.

### 3.2 Non-Stationarity of Macro-Style Relationships

**Problem description**: The relationship between macroeconomic indicators and relative style performance evolves structurally over time. For example, the sensitivity of the "Value" style to interest rates has changed significantly since the 2008 crisis with the era of zero rates. Today's "Growth" companies (mostly tech) do not have the same profile as those of the 1990s.

**Implications**: A model trained on historical data learns patterns that may no longer be valid. A Transformer can theoretically capture these evolutions via temporal attention, but with so little data, it cannot reliably learn regime changes. The risk is learning obsolete relationships.

**Why this problem has no simple solution**: We cannot know a priori which relationships have changed and which remain stable. Weighting recent data more heavily further reduces the effective size of the training sample. Regime change detection methods themselves require a lot of data.

### 3.3 Time Horizon and Causality Problem

**Problem description**: The strategy proposes testing several horizons (1 week to 1 year), but the causal relationship between macro and styles works differently depending on the horizon:

| Horizon | Signal Nature |
|---------|---------------|
| Short term (1 week) | Surprise matters (deviation vs consensus) |
| Medium term (1-3 months) | Momentum and expectation revision matter |
| Long term (6-12 months) | Absolute level and positioning in the economic cycle matter |

**Implications**: The same model with the same inputs will have difficulty capturing these very different dynamics. The signal we seek to extract is not of the same nature depending on the horizon. Training a single multi-horizon model risks producing mediocre results across all horizons.

**Why this problem has no simple solution**: Developing separate models per horizon further divides the available data. The relevant features differ by horizon, but we do not know a priori which ones. The optimal architecture may also vary.

### 3.4 Informational Efficiency of Markets

**Problem description**: Macroeconomic data is public and scrutinized by thousands of analysts and trading algorithms. Information is integrated into prices within seconds for surprises, within days for second-order implications. The proposed strategy does not aim for an execution speed advantage.

**Implications**: For an ML model to extract systematic alpha from public data, it must develop superior interpretation capability compared to market participants. This is an ambitious and not guaranteed objective, especially with a model trained only on historical data.

**Why this problem has no simple solution**: The potential advantage of the model would be to capture complex non-linear interactions or sequence effects that human analysts neglect. But these effects, if they exist, are probably subtle and difficult to distinguish from statistical noise with the available data.

### 3.5 Overfitting Risk

**Problem description**: With many input features (multiple indicators × attributes per token) and relatively few independent observations (a few thousand temporal points at most), the model risks memorizing idiosyncratic noise from historical data rather than learning generalizable relationships.

**Available mitigations**: Aggressive regularization (high dropout 0.3-0.5, weight decay, strict early stopping), complexity reduction (very small model with 1-2 layers, 32-64 dimensions), walk-forward validation for detecting temporal overfitting, and parsimonious feature selection (limiting to 10-15 truly informative indicators).

**Why this problem remains concerning**: Even with these mitigations, the overfitting risk remains high given the fundamental imbalance between the potential richness of the representation and the poverty of the data. Regularization techniques reduce the problem but do not eliminate it. The model can overfit in subtle ways (for example, by learning patterns specific to certain historical regimes that are not reproducible).

---

## 4. Recommendations and Next Steps

### 4.1 Development Priorities

The first priority is to validate the existence of a signal using simple methods (logistic regression, random forest) on the simplified test case (cyclicals vs defensives, 1-month horizon). The second priority is to build the point-in-time database with meticulous vintage reconstruction for priority macro indicators. The third priority is to define success metrics: target Information Coefficient, minimum improvement over momentum baseline to justify moving to the next step. The fourth priority is to develop the walk-forward backtesting infrastructure before starting experimentation.

### 4.2 Stop Criteria

| Condition | Decision |
|-----------|----------|
| IC close to zero with simple regression | Reassess the fundamental strategy |
| No significant improvement over momentum baseline | Do not proceed to Transformer |
| Systematic overfitting despite regularization | Drastically simplify the model or reduce features |

### 4.3 Points of Vigilance

The real added value will probably come less from the model architecture than from the quality of the data (rigorous point-in-time, economically sensible feature engineering) and the rigor of backtesting (strict walk-forward, realistic transaction costs, uncontaminated final holdout).

---

*Document generated on: [date]*
*Version: 1.0*
