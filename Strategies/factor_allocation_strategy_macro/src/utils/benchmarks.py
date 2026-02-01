"""
Benchmark computation utilities for strategy evaluation.

Provides benchmark returns for comparison:
- Equal-weight factors: 1/6 weight in each factor
- Equal-weight cyclical/defensive: 50/50 between cyclical and defensive
- Risk Parity: Inverse volatility weighting
- Factor Momentum: Allocate to factors with positive trailing returns
- Best Single Factor: 100% in historically best Sharpe factor
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.metrics import compute_sharpe_ratio, compute_max_drawdown, compute_total_return


DEFAULT_FACTOR_COLUMNS = ['cyclical', 'defensive', 'value', 'growth', 'quality', 'momentum']


@dataclass
class BenchmarkResult:
    """
    Result of benchmark computation.

    :param name (str): Benchmark name
    :param returns (np.ndarray): Monthly returns
    :param sharpe (float): Annualized Sharpe ratio
    :param total_return (float): Total cumulative return
    :param maxdd (float): Maximum drawdown
    """
    name: str
    returns: np.ndarray
    sharpe: float
    total_return: float
    maxdd: float


def _filter_by_date(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.

    :param df (pd.DataFrame): Input DataFrame with 'timestamp' column
    :param start_date (pd.Timestamp): Start date (inclusive)
    :param end_date (pd.Timestamp): End date (inclusive)

    :return filtered (pd.DataFrame): Filtered DataFrame
    """
    if start_date is not None:
        df = df[df['timestamp'] >= start_date]
    if end_date is not None:
        df = df[df['timestamp'] <= end_date]
    return df


def _get_factor_columns(
    df: pd.DataFrame,
    factor_columns: Optional[List[str]],
) -> List[str]:
    """
    Get available factor columns from DataFrame.

    :param df (pd.DataFrame): Input DataFrame
    :param factor_columns (List[str]): Requested factor columns

    :return available (List[str]): Available factor columns
    """
    if factor_columns is None:
        factor_columns = DEFAULT_FACTOR_COLUMNS
    available = [c for c in factor_columns if c in df.columns]
    if not available:
        raise ValueError(f"No factor columns found. Available: {df.columns.tolist()}")
    return available


def compute_equal_weight_factors_benchmark(
    factor_data: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    factor_columns: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Compute equal-weight all factors benchmark.

    Invests 1/6 in each factor: cyclical, defensive, value, growth, quality, momentum.

    :param factor_data (pd.DataFrame): Factor data
    :param start_date (pd.Timestamp): Start date for filtering
    :param end_date (pd.Timestamp): End date for filtering
    :param factor_columns (List[str]): Factor columns to use

    :return result (BenchmarkResult): Equal-weight factors benchmark result
    """
    df = _filter_by_date(factor_data.copy(), start_date, end_date)
    cols = _get_factor_columns(df, factor_columns)

    returns = df[cols].mean(axis=1).values

    return BenchmarkResult(
        name='Equal-Weight 6F',
        returns=returns,
        sharpe=compute_sharpe_ratio(returns, periods_per_year=12),
        total_return=compute_total_return(returns),
        maxdd=compute_max_drawdown(returns),
    )


def compute_equal_weight_cyc_def_benchmark(
    factor_data: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> BenchmarkResult:
    """
    Compute 50/50 cyclical/defensive benchmark.

    No timing, just static 50% cyclical + 50% defensive allocation.

    :param factor_data (pd.DataFrame): Factor data
    :param start_date (pd.Timestamp): Start date for filtering
    :param end_date (pd.Timestamp): End date for filtering

    :return result (BenchmarkResult): Equal-weight cyc/def benchmark result
    """
    df = _filter_by_date(factor_data.copy(), start_date, end_date)

    if 'cyclical' not in df.columns or 'defensive' not in df.columns:
        raise ValueError("factor_data must contain 'cyclical' and 'defensive' columns")

    returns = (df['cyclical'].values + df['defensive'].values) / 2

    return BenchmarkResult(
        name='50/50 Cyc/Def',
        returns=returns,
        sharpe=compute_sharpe_ratio(returns, periods_per_year=12),
        total_return=compute_total_return(returns),
        maxdd=compute_max_drawdown(returns),
    )


def compute_risk_parity_benchmark(
    factor_data: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    lookback_months: int = 12,
    factor_columns: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Compute risk parity (inverse volatility) benchmark.

    Allocates inversely proportional to trailing volatility. Lower volatility
    factors receive higher weights.

    :param factor_data (pd.DataFrame): Factor data
    :param start_date (pd.Timestamp): Start date for filtering
    :param end_date (pd.Timestamp): End date for filtering
    :param lookback_months (int): Lookback period for volatility calculation
    :param factor_columns (List[str]): Factor columns to use

    :return result (BenchmarkResult): Risk parity benchmark result
    """
    df = _filter_by_date(factor_data.copy(), start_date, end_date).reset_index(drop=True)
    cols = _get_factor_columns(df, factor_columns)

    factor_returns = df[cols].values
    n_periods = len(df)
    portfolio_returns = np.zeros(n_periods)

    for t in range(n_periods):
        if t < lookback_months:
            weights = np.ones(len(cols)) / len(cols)
        else:
            trailing_returns = factor_returns[t - lookback_months:t, :]
            volatilities = np.std(trailing_returns, axis=0, ddof=1)
            volatilities = np.maximum(volatilities, 1e-8)
            inv_vol = 1.0 / volatilities
            weights = inv_vol / inv_vol.sum()

        portfolio_returns[t] = np.dot(weights, factor_returns[t, :])

    return BenchmarkResult(
        name='Risk Parity',
        returns=portfolio_returns,
        sharpe=compute_sharpe_ratio(portfolio_returns, periods_per_year=12),
        total_return=compute_total_return(portfolio_returns),
        maxdd=compute_max_drawdown(portfolio_returns),
    )


def compute_factor_momentum_benchmark(
    factor_data: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    lookback_months: int = 12,
    factor_columns: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Compute factor momentum benchmark.

    Allocates equally among factors with positive trailing returns.
    If all factors have negative returns, allocates equally across all.

    :param factor_data (pd.DataFrame): Factor data
    :param start_date (pd.Timestamp): Start date for filtering
    :param end_date (pd.Timestamp): End date for filtering
    :param lookback_months (int): Lookback period for momentum calculation
    :param factor_columns (List[str]): Factor columns to use

    :return result (BenchmarkResult): Factor momentum benchmark result
    """
    df = _filter_by_date(factor_data.copy(), start_date, end_date).reset_index(drop=True)
    cols = _get_factor_columns(df, factor_columns)

    factor_returns = df[cols].values
    n_periods = len(df)
    n_factors = len(cols)
    portfolio_returns = np.zeros(n_periods)

    for t in range(n_periods):
        if t < lookback_months:
            weights = np.ones(n_factors) / n_factors
        else:
            trailing_returns = factor_returns[t - lookback_months:t, :]
            cumulative_returns = np.prod(1 + trailing_returns, axis=0) - 1

            positive_mask = cumulative_returns > 0
            if positive_mask.sum() > 0:
                weights = np.zeros(n_factors)
                weights[positive_mask] = 1.0 / positive_mask.sum()
            else:
                weights = np.ones(n_factors) / n_factors

        portfolio_returns[t] = np.dot(weights, factor_returns[t, :])

    return BenchmarkResult(
        name='Factor Momentum',
        returns=portfolio_returns,
        sharpe=compute_sharpe_ratio(portfolio_returns, periods_per_year=12),
        total_return=compute_total_return(portfolio_returns),
        maxdd=compute_max_drawdown(portfolio_returns),
    )


def compute_best_single_factor_benchmark(
    factor_data: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    factor_columns: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Compute best single factor benchmark.

    Uses in-sample data (before start_date) to select the factor with highest
    Sharpe ratio, then holds 100% in that factor during the evaluation period.

    :param factor_data (pd.DataFrame): Factor data (full history)
    :param start_date (pd.Timestamp): Start of evaluation period
    :param end_date (pd.Timestamp): End of evaluation period
    :param factor_columns (List[str]): Factor columns to use

    :return result (BenchmarkResult): Best single factor benchmark result
    """
    df_full = factor_data.copy()
    cols = _get_factor_columns(df_full, factor_columns)

    if start_date is not None:
        df_insample = df_full[df_full['timestamp'] < start_date]
    else:
        midpoint = len(df_full) // 2
        df_insample = df_full.iloc[:midpoint]

    if len(df_insample) < 12:
        df_insample = df_full.iloc[:max(12, len(df_full) // 4)]

    best_sharpe = -np.inf
    best_factor = cols[0]

    for col in cols:
        factor_returns = df_insample[col].values
        sharpe = compute_sharpe_ratio(factor_returns, periods_per_year=12)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_factor = col

    df_eval = _filter_by_date(df_full.copy(), start_date, end_date)
    returns = df_eval[best_factor].values

    return BenchmarkResult(
        name=f'Best Factor ({best_factor.capitalize()})',
        returns=returns,
        sharpe=compute_sharpe_ratio(returns, periods_per_year=12),
        total_return=compute_total_return(returns),
        maxdd=compute_max_drawdown(returns),
    )


def compute_all_benchmarks(
    factor_data: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    lookback_months: int = 12,
) -> Dict[str, BenchmarkResult]:
    """
    Compute all benchmarks for a given period.

    :param factor_data (pd.DataFrame): Factor data
    :param start_date (pd.Timestamp): Start date for filtering
    :param end_date (pd.Timestamp): End date for filtering
    :param lookback_months (int): Lookback for dynamic benchmarks

    :return benchmarks (Dict[str, BenchmarkResult]): Dictionary of benchmark results
    """
    benchmarks = {}

    try:
        benchmarks['equal_weight_6f'] = compute_equal_weight_factors_benchmark(
            factor_data, start_date, end_date
        )
    except ValueError as e:
        print(f"Warning: Could not compute equal-weight 6F benchmark: {e}")

    try:
        benchmarks['equal_weight_cyc_def'] = compute_equal_weight_cyc_def_benchmark(
            factor_data, start_date, end_date
        )
    except ValueError as e:
        print(f"Warning: Could not compute 50/50 cyc/def benchmark: {e}")

    try:
        benchmarks['risk_parity'] = compute_risk_parity_benchmark(
            factor_data, start_date, end_date, lookback_months
        )
    except ValueError as e:
        print(f"Warning: Could not compute risk parity benchmark: {e}")

    try:
        benchmarks['factor_momentum'] = compute_factor_momentum_benchmark(
            factor_data, start_date, end_date, lookback_months
        )
    except ValueError as e:
        print(f"Warning: Could not compute factor momentum benchmark: {e}")

    try:
        benchmarks['best_single_factor'] = compute_best_single_factor_benchmark(
            factor_data, start_date, end_date
        )
    except ValueError as e:
        print(f"Warning: Could not compute best single factor benchmark: {e}")

    return benchmarks


def get_benchmark_returns_for_period(
    factor_data: pd.DataFrame,
    holdout_dates: List[pd.Timestamp],
    lookback_months: int = 12,
) -> Dict[str, BenchmarkResult]:
    """
    Get benchmark returns aligned with holdout evaluation dates.

    :param factor_data (pd.DataFrame): Factor data
    :param holdout_dates (List[pd.Timestamp]): List of holdout period timestamps
    :param lookback_months (int): Lookback for dynamic benchmarks

    :return benchmarks (Dict[str, BenchmarkResult]): Benchmark results for holdout period
    """
    if not holdout_dates:
        return {}

    start_date = min(holdout_dates)
    end_date = max(holdout_dates)

    return compute_all_benchmarks(factor_data, start_date, end_date, lookback_months)


def print_benchmark_summary(benchmarks: Dict[str, BenchmarkResult]) -> None:
    """
    Print summary of all benchmarks.

    :param benchmarks (Dict[str, BenchmarkResult]): Dictionary of benchmark results
    """
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Benchmark':<25} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10}")
    print("-" * 70)

    for key, benchmark in benchmarks.items():
        print(f"{benchmark.name:<25} {benchmark.sharpe:>+10.4f} "
              f"{benchmark.total_return:>+11.2%} {benchmark.maxdd:>+10.4f}")

    print("=" * 70)
