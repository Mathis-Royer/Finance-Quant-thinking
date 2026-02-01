"""
Analysis utilities for holdout results.

Provides functions for:
- Comparing model types (Final vs Fair Ensemble vs WF Ensemble)
- Finding best models by criterion
- Computing win counts and deltas
- Generating summary tables
"""

from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .keys import unpack_key
from .constants import (
    MODEL_TYPE_ABBREV,
    MODEL_TYPE_ORDER,
    STRATEGY_ABBREV,
    ALLOCATION_ABBREV,
    CONFIG_SUFFIX,
)


# Type alias for holdout results
HoldoutResultKey = Union[Tuple[str, str, int], Tuple[str, str, int, str]]
HoldoutResults = Dict[HoldoutResultKey, Dict[str, Any]]


@dataclass
class WinCountResult:
    """Result of win count comparison between two model types."""

    type_a: str
    type_b: str
    type_a_wins: int
    type_b_wins: int
    ties: int
    total: int

    @property
    def type_a_win_rate(self) -> float:
        """Win rate for type_a."""
        return self.type_a_wins / max(self.total, 1)


@dataclass
class DeltaResult:
    """Result of delta comparison between two model types."""

    type_a: str
    type_b: str
    avg_delta: float
    max_delta: float
    min_delta: float
    type_a_better: int
    type_b_better: int


@dataclass
class BestModelResult:
    """Best model found by a specific criterion."""

    criterion: str
    name: str
    strategy: str
    allocation: str
    horizon: int
    config: str
    model_type: str
    sharpe: float
    ic: float
    maxdd: float
    total_return: float


def flatten_holdout_results(
    holdout_results: HoldoutResults,
) -> List[Dict[str, Any]]:
    """
    Flatten holdout results dictionary into a list of model records.

    :param holdout_results (HoldoutResults): Holdout results from ThreeStepEvaluation

    :return records (List[Dict]): List of model result dictionaries
    """
    records = []

    for key, result in holdout_results.items():
        if result is None:
            continue

        strategy, allocation, horizon, config = unpack_key(key)

        for model_type in MODEL_TYPE_ORDER:
            if model_type not in result or result[model_type] is None:
                continue

            r = result[model_type]
            t_abbrev = MODEL_TYPE_ABBREV.get(model_type, model_type[:2])
            s_abbrev = STRATEGY_ABBREV.get(strategy, strategy[:3])
            a_abbrev = ALLOCATION_ABBREV.get(allocation, allocation[0])
            config_suffix = CONFIG_SUFFIX.get(config, "") if config != "baseline" else ""

            # Compute composite score for ranking
            sharpe_norm = (r.sharpe + 1) / 4
            sharpe_norm = max(0, min(1, sharpe_norm))
            if r.ic < 0:
                ic_norm = (0.5 + r.ic) * 0.5
            else:
                ic_norm = 0.5 + r.ic
            ic_norm = max(0, min(1, ic_norm))
            maxdd_penalty = np.exp(3 * abs(r.maxdd)) - 1
            maxdd_norm = 1 - min(maxdd_penalty / 3, 1)
            return_norm = (r.total_return + 0.5) / 1.5
            return_norm = max(0, min(1, return_norm))
            score = 0.35 * sharpe_norm + 0.25 * ic_norm + 0.30 * maxdd_norm + 0.10 * return_norm

            records.append({
                "strategy": strategy,
                "allocation": allocation,
                "horizon": horizon,
                "config": config,
                "model_type": model_type,
                "name": f"{s_abbrev}-{a_abbrev}-{horizon}M-{t_abbrev}{config_suffix}",
                "sharpe": r.sharpe,
                "ic": r.ic,
                "maxdd": r.maxdd,
                "total_return": r.total_return,
                "score": score,
            })

    return records


def compare_model_types(
    holdout_summary: pd.DataFrame,
    type_a: str,
    type_b: str,
    metric: str = "sharpe",
) -> WinCountResult:
    """
    Compare win counts between two model types.

    :param holdout_summary (pd.DataFrame): Summary DataFrame with model_type column
    :param type_a (str): First model type (e.g., "Final")
    :param type_b (str): Second model type (e.g., "Fair Ensemble")
    :param metric (str): Metric to compare (sharpe, ic, maxdd)

    :return result (WinCountResult): Win count comparison result
    """
    df_a = holdout_summary[holdout_summary["model_type"] == type_a]
    df_b = holdout_summary[holdout_summary["model_type"] == type_b]

    type_a_wins = 0
    type_b_wins = 0
    ties = 0

    for _, row_a in df_a.iterrows():
        key = (
            row_a["strategy"],
            row_a["allocation"],
            row_a["horizon"],
            row_a.get("config", "baseline"),
        )

        match = df_b[
            (df_b["strategy"] == key[0])
            & (df_b["allocation"] == key[1])
            & (df_b["horizon"] == key[2])
        ]
        # Also filter by config if column exists
        if "config" in df_b.columns:
            match = match[match["config"] == key[3]]

        if len(match) > 0:
            val_a = row_a[metric]
            val_b = match.iloc[0][metric]

            # For maxdd, less negative is better
            if metric == "maxdd":
                if val_a > val_b:
                    type_a_wins += 1
                elif val_b > val_a:
                    type_b_wins += 1
                else:
                    ties += 1
            else:
                if val_a > val_b:
                    type_a_wins += 1
                elif val_b > val_a:
                    type_b_wins += 1
                else:
                    ties += 1

    return WinCountResult(
        type_a=type_a,
        type_b=type_b,
        type_a_wins=type_a_wins,
        type_b_wins=type_b_wins,
        ties=ties,
        total=type_a_wins + type_b_wins + ties,
    )


def compute_delta_stats(
    holdout_summary: pd.DataFrame,
    type_a: str,
    type_b: str,
    metric: str = "sharpe",
) -> DeltaResult:
    """
    Compute delta statistics between two model types.

    :param holdout_summary (pd.DataFrame): Summary DataFrame
    :param type_a (str): First model type
    :param type_b (str): Second model type
    :param metric (str): Metric to compare

    :return result (DeltaResult): Delta statistics
    """
    df_a = holdout_summary[holdout_summary["model_type"] == type_a]
    df_b = holdout_summary[holdout_summary["model_type"] == type_b]

    deltas = []
    type_a_better = 0
    type_b_better = 0

    for _, row_a in df_a.iterrows():
        key = (
            row_a["strategy"],
            row_a["allocation"],
            row_a["horizon"],
            row_a.get("config", "baseline"),
        )

        match = df_b[
            (df_b["strategy"] == key[0])
            & (df_b["allocation"] == key[1])
            & (df_b["horizon"] == key[2])
        ]
        if "config" in df_b.columns:
            match = match[match["config"] == key[3]]

        if len(match) > 0:
            delta = row_a[metric] - match.iloc[0][metric]
            deltas.append(delta)

            if delta > 0:
                type_a_better += 1
            elif delta < 0:
                type_b_better += 1

    return DeltaResult(
        type_a=type_a,
        type_b=type_b,
        avg_delta=float(np.mean(deltas)) if deltas else 0.0,
        max_delta=float(max(deltas)) if deltas else 0.0,
        min_delta=float(min(deltas)) if deltas else 0.0,
        type_a_better=type_a_better,
        type_b_better=type_b_better,
    )


def find_best_models(
    holdout_results: HoldoutResults,
    criteria: Optional[List[str]] = None,
) -> Dict[str, BestModelResult]:
    """
    Find best models by each criterion.

    :param holdout_results (HoldoutResults): Holdout results dictionary
    :param criteria (Optional[List[str]]): Criteria to find best for (default: sharpe, ic, maxdd)

    :return best_models (Dict): Mapping of criterion -> BestModelResult
    """
    if criteria is None:
        criteria = ["sharpe", "ic", "maxdd"]

    records = flatten_holdout_results(holdout_results)

    if not records:
        return {}

    best_models = {}

    for criterion in criteria:
        # For maxdd, less negative is better (so we still use max)
        best = max(records, key=lambda x: x[criterion])

        best_models[criterion] = BestModelResult(
            criterion=criterion,
            name=best["name"],
            strategy=best["strategy"],
            allocation=best["allocation"],
            horizon=best["horizon"],
            config=best["config"],
            model_type=best["model_type"],
            sharpe=best["sharpe"],
            ic=best["ic"],
            maxdd=best["maxdd"],
            total_return=best["total_return"],
        )

    return best_models


def print_best_models_table(
    holdout_results: HoldoutResults,
    benchmarks: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print formatted table of best models vs benchmarks.

    :param holdout_results (HoldoutResults): Holdout results dictionary
    :param benchmarks (Optional[Dict]): Optional benchmarks dictionary
    """
    best = find_best_models(holdout_results, ["sharpe", "ic", "maxdd"])

    if not best:
        print("No holdout results available.")
        return

    print("\n" + "=" * 90)
    print("BEST MODELS vs BENCHMARKS (All Models)")
    print("=" * 90)
    print(f"\n{'Name':<35} {'Sharpe':>10} {'IC':>10} {'MaxDD':>10} {'Return':>12}")
    print("-" * 90)

    print("BEST MODELS (by criterion):")
    for criterion, result in best.items():
        label = f"{criterion.capitalize()} {'min' if criterion == 'maxdd' else 'max'}:"
        print(
            f"  {label:<12} {result.name:<25} "
            f"{result.sharpe:>+10.4f} {result.ic:>+10.4f} "
            f"{result.maxdd:>+10.4f} {result.total_return:>+11.2%}"
        )

    if benchmarks:
        print("-" * 90)
        print("BENCHMARKS:")
        for name, bench in benchmarks.items():
            print(
                f"  {name:<33} {bench.sharpe:>+10.4f} "
                f"{0.0:>+10.4f} {bench.maxdd:>+10.4f} "
                f"{bench.total_return:>+11.2%}"
            )

    print("=" * 90)


def print_model_comparison_summary(
    holdout_summary: pd.DataFrame,
) -> None:
    """
    Print formatted summary of Final vs Fair Ensemble vs WF Ensemble comparisons.

    Uses composite score for winner determination instead of Sharpe ratio.

    :param holdout_summary (pd.DataFrame): Summary DataFrame with model_type column
    """
    # Final vs Fair Ensemble (fair comparison) - using score
    final_vs_fair = compare_model_types(
        holdout_summary, "Final", "Fair Ensemble", "score"
    )

    print(
        f"\nFINAL vs FAIR ENSEMBLE (fair comparison, by Score): "
        f"{final_vs_fair.type_a_wins} Final wins, "
        f"{final_vs_fair.type_b_wins} Fair Ensemble wins"
    )

    # Fair Ensemble vs WF Ensemble (data quantity effect) - using score
    fair_vs_wf = compute_delta_stats(
        holdout_summary, "Fair Ensemble", "WF Ensemble", "score"
    )

    print(
        f"FAIR vs WF ENSEMBLE (data quantity effect, by Score): "
        f"Fair {fair_vs_wf.type_a_better}, WF {fair_vs_wf.type_b_better}, "
        f"Avg Delta: {fair_vs_wf.avg_delta:+.4f} Score"
    )


def print_statistical_analysis(
    holdout_results: HoldoutResults,
    benchmarks: Optional[Dict[str, Any]] = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> None:
    """
    Print statistical analysis with bootstrap CIs and significance tests.

    Analyzes the top 3 models by composite score and compares them
    to the best benchmark using:
    - Bootstrap confidence intervals for Sharpe ratios
    - Lo (2002) significance test for individual Sharpe ratios
    - Jobson-Korkie test for comparing two Sharpe ratios

    :param holdout_results (HoldoutResults): Holdout results dictionary
    :param benchmarks (Optional[Dict]): Optional benchmarks dictionary with monthly_returns
    :param confidence_level (float): Confidence level for bootstrap CIs (default 0.95)
    :param n_bootstrap (int): Number of bootstrap samples (default 1000)
    """
    from .statistics import (
        bootstrap_sharpe_ratio,
        test_sharpe_significance,
        compare_sharpe_ratios,
    )

    # Flatten results and find top models by score
    records = flatten_holdout_results(holdout_results)
    if not records:
        print("No holdout results available for statistical analysis.")
        return

    # Compute scores and sort
    for r in records:
        # Simple composite score for ranking
        sharpe_norm = (r["sharpe"] + 1) / 4
        ic_norm = 0.5 + r["ic"] if r["ic"] >= 0 else (0.5 + r["ic"]) * 0.5
        maxdd_norm = 1 - min(abs(r["maxdd"]) * 2, 1)
        return_norm = (r["total_return"] + 0.5) / 1.5
        r["score"] = 0.35 * sharpe_norm + 0.25 * ic_norm + 0.30 * maxdd_norm + 0.10 * return_norm

    records_sorted = sorted(records, key=lambda x: x["score"], reverse=True)
    top_models = records_sorted[:3]

    # Get monthly returns for top models
    model_returns = {}
    for model in top_models:
        key = (model["strategy"], model["allocation"], model["horizon"], model["config"])
        if key in holdout_results and holdout_results[key] is not None:
            result = holdout_results[key]
            model_type = model["model_type"]
            if model_type in result and result[model_type] is not None:
                r = result[model_type]
                if hasattr(r, "monthly_returns") and r.monthly_returns is not None:
                    model_returns[model["name"]] = np.array(r.monthly_returns)

    if not model_returns:
        print("No monthly returns available for statistical analysis.")
        print("(Models need 'monthly_returns' attribute in holdout results)")
        return

    # Get best benchmark returns
    benchmark_returns = None
    benchmark_name = None
    if benchmarks:
        best_bench = max(benchmarks.items(), key=lambda x: x[1].sharpe)
        benchmark_name = best_bench[0]
        if hasattr(best_bench[1], "monthly_returns") and best_bench[1].monthly_returns is not None:
            benchmark_returns = np.array(best_bench[1].monthly_returns)

    # Print header
    print("\n" + "=" * 90)
    print("STATISTICAL ANALYSIS (Bootstrap CIs & Significance Tests)")
    print("=" * 90)
    print(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.0%}")
    print("-" * 90)

    # Analyze each top model
    print(f"\n{'Model':<35} {'Sharpe':>8} {'95% CI':>18} {'p-value':>10} {'Signif':>8}")
    print("-" * 90)

    for name, returns in model_returns.items():
        # Bootstrap CI
        ci = bootstrap_sharpe_ratio(
            returns,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )

        # Significance test (H0: Sharpe = 0)
        sig = test_sharpe_significance(returns, alpha=0.05)

        ci_str = f"[{ci.ci_lower:+.3f}, {ci.ci_upper:+.3f}]"
        signif_str = "Yes ***" if sig.p_value < 0.01 else ("Yes **" if sig.p_value < 0.05 else ("Yes *" if sig.p_value < 0.10 else "No"))

        print(f"{name:<35} {ci.estimate:>+8.3f} {ci_str:>18} {sig.p_value:>10.4f} {signif_str:>8}")

    # Benchmark comparison
    if benchmark_returns is not None and len(model_returns) > 0:
        print("-" * 90)
        print(f"\nCOMPARISON vs BEST BENCHMARK ({benchmark_name})")
        print(f"{'Model':<35} {'Delta SR':>10} {'p-value':>10} {'Signif':>10}")
        print("-" * 90)

        # Bootstrap CI for benchmark
        bench_ci = bootstrap_sharpe_ratio(benchmark_returns, n_bootstrap=n_bootstrap)
        print(f"{'Benchmark: ' + benchmark_name:<35} {bench_ci.estimate:>+10.3f}")

        for name, returns in model_returns.items():
            # Jobson-Korkie test comparing model vs benchmark
            comparison = compare_sharpe_ratios(
                returns,
                benchmark_returns,
                alpha=0.05,
            )

            signif_str = "Yes ***" if comparison.p_value < 0.01 else (
                "Yes **" if comparison.p_value < 0.05 else (
                    "Yes *" if comparison.p_value < 0.10 else "No"
                )
            )

            print(f"{name:<35} {comparison.difference:>+10.3f} {comparison.p_value:>10.4f} {signif_str:>10}")

    print("=" * 90)
    print("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    print("=" * 90)
