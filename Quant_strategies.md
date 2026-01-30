# Quantitative Strategies Index

This document provides an executive summary of each quantitative strategy explored in this repository. For detailed documentation, refer to the individual strategy folders in `/Strategies`.

---

## Strategy Classification

Strategies are classified by implementation status:

| Status | Meaning |
|--------|---------|
| **Full implementation** | Complete, testable code using publicly available or synthetic data |
| **Minimal example** | Conceptual code demonstrating core logic; full testing requires unavailable resources (institutional data, infrastructure, capital) |
| **Concept only** | Documented idea without implementation |

---

## 1. Factor Allocation Based on Macroeconomic Data

**Folder**: [`/Strategies/factor_allocation_strategy_macro`](Strategies/factor_allocation_strategy_macro/)

### Executive Summary

A dynamic allocation strategy across equity factor styles (value, growth, cyclical, defensive) using a Transformer-based neural network fed by macroeconomic data. The model learns to anticipate factor rotations from macro signals (PMI, CPI, employment data, etc.) and outputs allocation weights to maximize risk-adjusted returns.

### Implementation Status: Full implementation (with synthetic data)

**Why full implementation?**
- Core architecture (Transformer, embeddings, data pipeline) can be built and tested
- Synthetic data generator allows validation of model training and inference
- Code structure demonstrates production-ready patterns

**Why synthetic data?**
- Point-in-time macroeconomic databases with proper vintage reconstruction are expensive and institutionally gated
- Real-time economic data feeds require subscriptions
- Factor index returns with sufficient history require licensed data (MSCI, S&P)

**What would be needed for live testing:**
- Bloomberg/Refinitiv terminal or equivalent for point-in-time macro data
- Factor index data subscription (MSCI Factor Indices, S&P Pure Style)
- Infrastructure for real-time data ingestion and model inference

---

## Adding New Strategies

When adding a new strategy to this repository:

1. Create a folder in `/Strategies/<strategy_name>/`
2. Add `<strategy_name>.md` with detailed documentation
3. Add code in `src/` subfolder
4. Update this file with an executive summary following the template above
5. Update the strategy table in `README.md`
