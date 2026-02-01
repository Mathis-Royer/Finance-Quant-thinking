"""
Statistical utilities for portfolio analysis.

This module provides:
- Kelly Criterion position sizing
- Bootstrap confidence intervals
- Statistical significance tests for Sharpe ratios
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class KellyCriterionResult:
    """
    Result of Kelly Criterion calculation.

    :param kelly_fraction: Optimal fraction to bet/invest
    :param half_kelly: Conservative half-Kelly fraction
    :param edge: Expected excess return (mean - risk_free)
    :param variance: Variance of returns
    """
    kelly_fraction: float
    half_kelly: float
    edge: float
    variance: float


def compute_kelly_fraction(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    max_leverage: float = 1.0,
) -> KellyCriterionResult:
    """
    Compute Kelly Criterion optimal position size.

    The Kelly formula is: f* = edge / variance = (μ - r) / σ²

    In practice, half-Kelly is often used for more conservative sizing.

    :param returns (np.ndarray): Array of returns (not percentages)
    :param risk_free_rate (float): Risk-free rate per period
    :param max_leverage (float): Maximum allowed leverage

    :return result (KellyCriterionResult): Kelly fractions and components
    """
    mean_return = np.mean(returns)
    variance = np.var(returns, ddof=1)

    # Edge is excess return over risk-free rate
    edge = mean_return - risk_free_rate

    # Kelly fraction: edge / variance
    if variance > 1e-10:
        kelly = edge / variance
    else:
        kelly = 0.0

    # Clip to reasonable range
    kelly = np.clip(kelly, -max_leverage, max_leverage)
    half_kelly = kelly / 2

    return KellyCriterionResult(
        kelly_fraction=kelly,
        half_kelly=half_kelly,
        edge=edge,
        variance=variance,
    )


def compute_kelly_weights(
    factor_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    use_half_kelly: bool = True,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute Kelly-optimal weights for multiple factors.

    :param factor_returns (np.ndarray): Returns matrix [n_periods, n_factors]
    :param risk_free_rate (float): Risk-free rate per period
    :param use_half_kelly (bool): Use conservative half-Kelly
    :param min_weight (float): Minimum weight per factor
    :param max_weight (float): Maximum weight per factor

    :return weights (np.ndarray): Normalized weights [n_factors]
    """
    n_factors = factor_returns.shape[1]
    raw_weights = np.zeros(n_factors)

    for i in range(n_factors):
        result = compute_kelly_fraction(
            factor_returns[:, i],
            risk_free_rate=risk_free_rate,
        )
        raw_weights[i] = result.half_kelly if use_half_kelly else result.kelly_fraction

    # Clip and normalize
    raw_weights = np.clip(raw_weights, min_weight, max_weight)

    # Handle all-negative case
    if raw_weights.sum() <= 0:
        return np.ones(n_factors) / n_factors

    # Normalize to sum to 1
    weights = raw_weights / raw_weights.sum()

    return weights


@dataclass
class BootstrapCI:
    """
    Bootstrap confidence interval result.

    :param estimate: Point estimate of the statistic
    :param ci_lower: Lower bound of confidence interval
    :param ci_upper: Upper bound of confidence interval
    :param std_error: Standard error of the estimate
    :param confidence_level: Confidence level (e.g., 0.95)
    """
    estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    confidence_level: float


def bootstrap_sharpe_ratio(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 12,
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for Sharpe ratio.

    :param returns (np.ndarray): Array of returns
    :param n_bootstrap (int): Number of bootstrap samples
    :param confidence_level (float): Confidence level (0.90, 0.95, 0.99)
    :param risk_free_rate (float): Risk-free rate per period
    :param annualize (bool): Annualize the Sharpe ratio
    :param periods_per_year (int): Number of periods per year (12 for monthly)

    :return ci (BootstrapCI): Confidence interval result
    """
    n_samples = len(returns)
    excess_returns = returns - risk_free_rate

    # Point estimate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    sharpe = mean_excess / (std_excess + 1e-10)
    if annualize:
        sharpe *= np.sqrt(periods_per_year)

    # Bootstrap
    bootstrap_sharpes = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample with replacement
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_returns = excess_returns[sample_idx]

        sample_mean = np.mean(sample_returns)
        sample_std = np.std(sample_returns, ddof=1)
        sample_sharpe = sample_mean / (sample_std + 1e-10)
        if annualize:
            sample_sharpe *= np.sqrt(periods_per_year)

        bootstrap_sharpes[i] = sample_sharpe

    # Compute confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)
    std_error = np.std(bootstrap_sharpes)

    return BootstrapCI(
        estimate=sharpe,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        confidence_level=confidence_level,
    )


def bootstrap_metric(
    data: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> BootstrapCI:
    """
    Generic bootstrap confidence interval for any metric.

    :param data (np.ndarray): Input data array
    :param metric_func (callable): Function that computes the metric from data
    :param n_bootstrap (int): Number of bootstrap samples
    :param confidence_level (float): Confidence level

    :return ci (BootstrapCI): Confidence interval result
    """
    n_samples = len(data)

    # Point estimate
    estimate = metric_func(data)

    # Bootstrap
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_data = data[sample_idx]
        bootstrap_estimates[i] = metric_func(sample_data)

    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
    std_error = np.std(bootstrap_estimates)

    return BootstrapCI(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        confidence_level=confidence_level,
    )


@dataclass
class SharpeTestResult:
    """
    Result of Sharpe ratio significance test.

    :param sharpe_ratio: Estimated Sharpe ratio
    :param t_statistic: T-statistic for the test
    :param p_value: P-value (two-sided)
    :param is_significant: Whether result is significant at given alpha
    :param alpha: Significance level used
    :param n_observations: Number of observations
    """
    sharpe_ratio: float
    t_statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    n_observations: int


def test_sharpe_significance(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    alpha: float = 0.05,
    annualize: bool = True,
    periods_per_year: int = 12,
) -> SharpeTestResult:
    """
    Test if Sharpe ratio is significantly different from zero.

    Uses the Lo (2002) method for Sharpe ratio standard error:
    SE(SR) = sqrt((1 + 0.5*SR^2) / T)

    :param returns (np.ndarray): Array of returns
    :param risk_free_rate (float): Risk-free rate per period
    :param alpha (float): Significance level
    :param annualize (bool): Annualize the Sharpe ratio
    :param periods_per_year (int): Periods per year

    :return result (SharpeTestResult): Test result
    """
    n = len(returns)
    excess_returns = returns - risk_free_rate

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    sharpe = mean_excess / (std_excess + 1e-10)

    if annualize:
        sharpe *= np.sqrt(periods_per_year)

    # Lo (2002) standard error
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)

    # T-statistic
    t_stat = sharpe / se_sharpe

    # P-value (two-sided)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    return SharpeTestResult(
        sharpe_ratio=sharpe,
        t_statistic=t_stat,
        p_value=p_value,
        is_significant=p_value < alpha,
        alpha=alpha,
        n_observations=n,
    )


@dataclass
class SharpeComparisonResult:
    """
    Result of comparing two Sharpe ratios.

    :param sharpe_1: Sharpe ratio of strategy 1
    :param sharpe_2: Sharpe ratio of strategy 2
    :param difference: sharpe_1 - sharpe_2
    :param t_statistic: T-statistic for the difference
    :param p_value: P-value (two-sided)
    :param is_significant: Whether difference is significant
    :param alpha: Significance level used
    """
    sharpe_1: float
    sharpe_2: float
    difference: float
    t_statistic: float
    p_value: float
    is_significant: bool
    alpha: float


def compare_sharpe_ratios(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    risk_free_rate: float = 0.0,
    alpha: float = 0.05,
    annualize: bool = True,
    periods_per_year: int = 12,
) -> SharpeComparisonResult:
    """
    Test if two Sharpe ratios are significantly different.

    Uses the Jobson-Korkie (1981) test with Memmel (2003) correction.

    :param returns_1 (np.ndarray): Returns of strategy 1
    :param returns_2 (np.ndarray): Returns of strategy 2
    :param risk_free_rate (float): Risk-free rate per period
    :param alpha (float): Significance level
    :param annualize (bool): Annualize Sharpe ratios
    :param periods_per_year (int): Periods per year

    :return result (SharpeComparisonResult): Comparison result
    """
    n = len(returns_1)
    if len(returns_2) != n:
        raise ValueError("Both return series must have the same length")

    excess_1 = returns_1 - risk_free_rate
    excess_2 = returns_2 - risk_free_rate

    mu1, mu2 = np.mean(excess_1), np.mean(excess_2)
    s1, s2 = np.std(excess_1, ddof=1), np.std(excess_2, ddof=1)
    sr1, sr2 = mu1 / (s1 + 1e-10), mu2 / (s2 + 1e-10)

    if annualize:
        sr1 *= np.sqrt(periods_per_year)
        sr2 *= np.sqrt(periods_per_year)

    # Correlation between strategies
    rho = np.corrcoef(excess_1, excess_2)[0, 1]

    # Jobson-Korkie test statistic with Memmel correction
    var_diff = (
        2 * (1 - rho)
        + 0.5 * (sr1**2 + sr2**2 - 2 * sr1 * sr2 * rho**2)
    ) / n

    t_stat = (sr1 - sr2) / np.sqrt(var_diff + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return SharpeComparisonResult(
        sharpe_1=sr1,
        sharpe_2=sr2,
        difference=sr1 - sr2,
        t_statistic=t_stat,
        p_value=p_value,
        is_significant=p_value < alpha,
        alpha=alpha,
    )


def compute_all_metrics_with_ci(
    returns: np.ndarray,
    factor_returns: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, BootstrapCI]:
    """
    Compute all key metrics with bootstrap confidence intervals.

    :param returns (np.ndarray): Portfolio returns
    :param factor_returns (np.ndarray): Optional factor returns for IC calculation
    :param n_bootstrap (int): Number of bootstrap samples
    :param confidence_level (float): Confidence level

    :return metrics (Dict[str, BootstrapCI]): Dictionary of metric CIs
    """
    results = {}

    # Sharpe ratio
    results["sharpe"] = bootstrap_sharpe_ratio(
        returns, n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )

    # Total return
    def total_return(r):
        return np.prod(1 + r) - 1

    results["total_return"] = bootstrap_metric(
        returns, total_return, n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )

    # Max drawdown
    def max_drawdown(r):
        cumulative = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown)

    results["max_drawdown"] = bootstrap_metric(
        returns, max_drawdown, n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )

    # Volatility (annualized)
    def annualized_vol(r):
        return np.std(r, ddof=1) * np.sqrt(12)

    results["volatility"] = bootstrap_metric(
        returns, annualized_vol, n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )

    return results


def print_metrics_summary(
    metrics: Dict[str, BootstrapCI],
    title: str = "Performance Metrics with 95% CI",
) -> None:
    """
    Print formatted summary of metrics with confidence intervals.

    :param metrics (Dict[str, BootstrapCI]): Dictionary of metric CIs
    :param title (str): Title for the summary
    """
    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Metric':<20} {'Estimate':>10} {'95% CI':>20} {'SE':>10}")
    print("-" * 60)

    for name, ci in metrics.items():
        ci_str = f"[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
        print(f"{name:<20} {ci.estimate:>10.4f} {ci_str:>20} {ci.std_error:>10.4f}")

    print("=" * 60)
