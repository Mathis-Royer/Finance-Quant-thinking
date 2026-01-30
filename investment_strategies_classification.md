# Classification of Algorithmic Investment Strategies by Timing Dependency

## Overview

This document classifies algorithmic investment strategies according to their timing dependency, from most critical to least critical. The objective is to identify strategies suited to different investor profiles based on their reaction capacity.

---

## 1. Ultra-High Frequency (microseconds to milliseconds)

**Required precision**: 5-10 microseconds for latency arbitrage, 2-3 milliseconds for the fastest traders  
**Suited profile**: Institutions with dedicated infrastructure only (2% of firms represent 73% of volume)

| Strategy | Description | Edge source |
|----------|-------------|-------------|
| Market making | Provide liquidity by continuously capturing the bid-ask spread | Bid-ask spread, speed |
| Latency arbitrage | Exploit price differences between markets before correction (5-10 μs window) | Infrastructure |
| Statistical arbitrage (microstructure) | Detect and exploit patterns in order flow | Order patterns |

**Infrastructure costs**:
- Colocation: $1,000 - $5,000/month
- FPGA: > $500,000
- Custom ASICs: ~$30 million/chip
- **Total initial investment: $5 - $25 million** (NIST estimate)

**Verdict**: ❌ Inaccessible. Requires colocation, FPGA, investments in millions. Human reaction (200 ms) is 100x too slow.

---

## 2. High Frequency (milliseconds to seconds)

**Required precision**: Milliseconds (prices react to macroeconomic news in 5 ms)  
**Suited profile**: Institutional trading desks, or professionals with dedicated VPS ($40-100/month for 0.3-5 ms latency)

| Strategy | Description | Edge source |
|----------|-------------|-------------|
| Technical indicator scalping | Take multiple small profits on very short price movements | Reactivity |
| News/breaking news trading | Automatically react to announcements via NLP before the market | NLP + speed |
| Intraday momentum | Follow directional flows detected in real-time during the session | Flow detection |

**Important nuance**: Some intraday strategies are accessible with modest infrastructure. The Zarattini, Aziz & Barbon (2024) study shows that intraday momentum on SPY generates a Sharpe ratio of 1.33 (19.6% annualized return) with limited infrastructure.

**Verdict**: ⚠️ Partially accessible. True HFT requires $500k-2M minimum, but some intraday strategies are possible with a VPS.

---

## 3. Medium Frequency (hours to days)

**Required precision**: Minutes to hours (end-of-day execution is the academic standard)  
**Suited profile**: Active professional traders

| Strategy | Description | Edge source |
|----------|-------------|-------------|
| Technical swing trading | Capture price movements over several days via technical analysis | Chart patterns |
| Intraday mean reversion | Bet on price mean reversion after extreme movements | Mean reversion |
| Event-driven (earnings) | Exploit price movements around earnings announcements | Post-announcement reaction |
| Gap trading | Exploit price gaps between close and next opening | Overnight gap exploitation |

**Edge reality**:
- Renaissance Technologies (the best hedge fund) has a success rate of only 50.75%
- PEAD (Post-Earnings Announcement Drift) **generates no alpha after transaction costs** (Zhang, Cai & Keasey, 2014)
- Typical transaction costs: 30-75 bps, up to 200-300 bps for large/illiquid orders
- Pairs trading: Sharpe of 1.44 (1997-2002) degraded to 0.9 (2003-2007)

**Verdict**: ⚠️ Accessible to professionals, but highly competitive. **Transaction costs can entirely eliminate alpha.**

---

## 4. Low Frequency (days to weeks)

**Required precision**: End-of-day to weekly  
**Suited profile**: Non-HFT professionals, advanced retail investors

| Strategy | Description | Edge source |
|----------|-------------|-------------|
| Trend following (daily) | Follow market trends identified on daily data (documented over 110 years by AQR) | Trend persistence |
| Pairs trading | Exploit mean reversion of the spread between two correlated assets (Sharpe 1.35-2.9) | Spread mean reversion |
| Cross-sectional momentum | Overweight assets that performed best over 3-12 months (~12-13% annual in long-short) | Relative ranking |
| Options strategies (volatility) | Exploit the gap between implied and realized volatility | Vol implied mispricing |

**Positive points**:
- AQR's "A Century of Evidence on Trend-Following" study demonstrates profitability over 110 years
- Actual trading costs are **5-10x lower** than academic estimates (0.15-0.35% vs 1-2%)
- Modern commission-free trading makes monthly rebalancing economically viable
- Butler University: "the strategy can easily be implemented on the Internet"

**Verdict**: ✅ **Interesting zone**. Execution can be done at end-of-day. Minute-level timing is not critical.

---

## 5. Very Low Frequency (weeks to months)

**Required precision**: Monthly to quarterly  
**Suited profile**: Serious retail investors, long-term managers

| Strategy | Description | Edge source | Optimal frequency |
|----------|-------------|-------------|-------------------|
| Factor investing | Maintain systematic exposure to factors (Value, Momentum, Quality, etc.) | Risk premiums | Quarterly |
| Tactical asset allocation | Adjust allocation between asset classes according to economic regimes | Economic regimes | Monthly |
| Quantitative value investing | Select undervalued assets relative to their fundamentals | Valuation | Optimal rebalancing: 6 months |
| Dynamic risk parity | Allocate capital so each asset class contributes equally to risk | Risk management | Monthly |
| Quality/Dividend | Select companies with solid balance sheets and stable dividends | Fundamental stability | 4-6 months |

**Academic documentation**:
- Momentum premium: ~12-13% annual, "one of the strongest anomalies" (AQR)
- Quality premium: "the only factor that has truly held across all periods" (CFA Institute)
- Low volatility premium: 12% annual alpha between extreme deciles (Frazzini & Pedersen, 2014)
- Robeco data over 160 years (1866-2020): premiums "do not decline out-of-sample"

**Important warnings**:
- **58% post-publication decay** of returns in the USA (McLean & Pontiff, 2016)
- But **no significant decline** across 38 international markets (Jacobs & Müller, 2020)
- Value's "lost decade" (2010-2019): 7 percentage points underperformance over 10 years

**Verdict**: ✅ **Optimal zone for serious retail investors or non-HFT professionals**. Execution over several days does not destroy the edge. Watch out for factor cyclicality.

---

## 6. Ultra-Low Frequency (months to years)

**Required precision**: Annual or none  
**Suited profile**: All investors

| Strategy | Description | Edge source |
|----------|-------------|-------------|
| Strategic allocation | Define a target allocation between asset classes according to risk profile | Long-term diversification |
| Buy & hold with rebalancing | Maintain a target allocation by periodically rebalancing | Discipline, taxation |
| Dollar-cost averaging | Regularly invest a fixed amount to smooth entry price | Entry price smoothing |

**What research says**:
- Vanguard (2022): monthly/quarterly rebalancing **does not improve** results vs annual rebalancing
- Optimal approach: annual rebalancing + 5% drift threshold
- DCA: **underperforms lump-sum investing in 67-75% of cases** (Vanguard 2023, Northwestern Mutual)
- But DCA offers a **real behavioral benefit** by reducing regret risk

**Verdict**: ✅ Suited for all. No active edge sought, but robust and simple. DCA is justified for risk-averse investors.

---

## 7. Strategies Without Timing Dependency

**Required precision**: None  
**Suited profile**: All investors

These strategies involve no temporal prediction. They aim to optimize portfolio construction without trying to "beat the market" through timing.

| Strategy | Description | Value source | Documented gain |
|----------|-------------|--------------|-----------------|
| Covariance matrix estimation | Use statistical/ML methods to better estimate correlations between assets (Ledoit-Wolf, shrinkage + RMT) | Better diversification | +5-6% in estimation accuracy |
| Asset clustering for diversification | Group assets by actual similarity via Hierarchical Risk Parity (López de Prado, 2016) | Identification of true decorrelations | Lower out-of-sample volatility |
| Portfolio personalization by constraints | Optimize allocation according to specific constraints (taxation, horizon, liquidity, ESG) | Personalized efficiency | Variable by constraints |
| Tax-loss harvesting | Realize tax losses to reduce taxation without modifying exposure | Tax optimization | **0.82-1.08% annual alpha** (Chaudhuri et al., 2020) |
| Mechanical rebalancing | Periodically bring portfolio back to target allocation without prediction | Discipline, automatic "buy low sell high" | A few dozen bps |

**Verdict**: ✅ Accessible to all. These strategies offer **measurable and documented value** without requiring market prediction.

---

## Conclusion

Timing dependency is the determining factor of a strategy's accessibility. The more precise timing a strategy requires, the more it needs costly infrastructure and reactivity impossible for retail investors.

**The Grossman-Stiglitz paradox** provides the theoretical framework: markets cannot be perfectly efficient because information acquisition has a cost. Edge therefore necessarily exists where information is costly to acquire and arbitrage costly to execute:
- HFT strategies exploit **infrastructure costs**
- Factor strategies exploit **patient capital costs**
- Non-timing strategies exploit **tax and behavioral inefficiencies**

The strategies best suited for retail investors are those that:
1. Operate on long horizons (weeks to months)
2. Do not depend on execution speed
3. Exploit persistent phenomena (risk premiums, behavioral biases)
4. Or involve no timing (pure portfolio optimization)

**Often underestimated critical factor**: transaction costs can entirely eliminate the alpha of apparently profitable strategies.

A retail investor's edge is not speed but **discipline**, **long horizon**, and **absence of institutional constraints** (no quarterly reporting, no benchmark pressure).
