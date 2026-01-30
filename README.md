# Hedge Quant

## Overview

This repository serves as a personal archive documenting my self-taught journey into market finance and quantitative finance. It contains theoretical reflections, research summaries, and experimental strategy implementations developed as part of my learning process.

The goal is to consolidate knowledge gained from academic papers, market observations, and hands-on experimentation with trading strategies—ranging from conceptual designs to executable code.

## Repository Structure

```
Hedge_quant/
├── README.md                              # This file
├── Financial_thinking.md                  # Theoretical reflections and research notes
├── Quant_strategies.md                    # Executive summary of all strategies (index)
├── investment_strategies_classification.md # Classification of strategies by timing dependency
├── src/                                   # Shared utilities and common code
└── Strategies/                            # Individual strategy implementations
    └── <strategy_name>/
        ├── <strategy_name>.md             # Detailed strategy documentation
        └── src/                           # Strategy-specific code
```

## Main Files

### [Financial_thinking.md](Financial_thinking.md)

A sourced synthesis of concepts explored during my learning:
- Why beating broad indices is structurally difficult (arithmetic of active management)
- The "blockbuster" phenomenon: how a small fraction of stocks drives wealth creation
- Why cap-weighted indices like the S&P 500 are tough benchmarks
- Challenges of tactical allocation and style rotation
- Efficient market hypothesis and factor models

Each claim is backed by academic references (Sharpe, Bessembinder, Fama-French, AQR research, etc.).

### [Quant_strategies.md](Quant_strategies.md)

An index file providing an executive summary of each quantitative strategy explored in the `/Strategies` folder. For each strategy, it includes:
- **Objective**: What the strategy attempts to achieve
- **Approach**: High-level methodology
- **Implementation status**:
  - *Minimal example*: Strategy requires data or infrastructure beyond personal scale (e.g., institutional-grade point-in-time databases, real-time feeds)
  - *Full implementation*: Strategy is testable with publicly available data or synthetic data
- **Rationale**: Why the chosen implementation level

### [investment_strategies_classification.md](investment_strategies_classification.md)

A classification of algorithmic investment strategies by timing dependency, from ultra-high frequency (microseconds) to strategies without timing dependency. For each category, it provides:
- **Required precision**: Time scale needed for execution
- **Suited profile**: Type of investor who can realistically implement the strategy
- **Strategy tables**: Description and edge source for each strategy
- **Infrastructure costs**: When applicable, estimated investment required
- **Verdict**: Accessibility assessment for retail investors

Key insight: a retail investor's edge is not speed but **discipline**, **long horizon**, and **absence of institutional constraints**.

## Strategies

| Strategy | Status | Description |
|----------|--------|-------------|
| [Factor Allocation (Macro)](Strategies/factor_allocation_strategy_macro/) | Full implementation (synthetic data) | Transformer-based dynamic allocation across equity factors using macroeconomic signals |

## Disclaimer

This repository is for **educational and personal learning purposes only**. The strategies and analyses presented here:
- Are not financial advice
- Have not been validated with real capital
- May contain errors or oversimplifications
- Should not be used for actual trading without extensive additional research and risk management