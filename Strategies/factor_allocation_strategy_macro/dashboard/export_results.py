"""
Export holdout results to cache for dashboard consumption.

This script is called from the notebook after running the three-step evaluation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.keys import unpack_key as _unpack_key


def export_holdout_results_to_cache(
    all_holdout_results: Dict[Union[Tuple[str, str, int], Tuple[str, str, int, str]], Dict[str, Any]],
    output_path: Path = None,
) -> Path:
    """
    Export holdout results to parquet for dashboard.

    :param all_holdout_results (Dict): Holdout results from ThreeStepEvaluation (3-tuple or 4-tuple keys)
    :param output_path (Path): Output path (default: data_cache/holdout_results.parquet)

    :return output_path (Path): Path where results were saved
    """
    if output_path is None:
        output_path = project_root / "data_cache" / "holdout_results.parquet"

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    type_map = {
        "final": "Final",
        "fair_ensemble": "Fair Ensemble",
        "wf_ensemble": "WF Ensemble",
    }

    data = []
    for key, results in all_holdout_results.items():
        if results is None:
            continue

        # Handle both 3-tuple (legacy) and 4-tuple keys
        strategy, allocation, horizon, config_name = _unpack_key(key)

        for model_type_key, model_type_name in type_map.items():
            result = results.get(model_type_key)
            if result is None:
                continue

            # Get monthly returns if available
            monthly_returns = getattr(result, 'monthly_returns', None)
            if monthly_returns is not None:
                monthly_returns_str = ",".join([f"{r:.6f}" for r in monthly_returns])
            else:
                monthly_returns_str = ""

            data.append({
                "strategy": strategy,
                "allocation": allocation,
                "horizon": horizon,
                "config": config_name,
                "model_type": model_type_name,
                "sharpe": result.sharpe,
                "ic": result.ic,
                "maxdd": result.maxdd,
                # total_return is stored as decimal (e.g., 0.0835 for 8.35%)
                "total_return": result.total_return,
                "monthly_returns": monthly_returns_str,
            })

    df = pd.DataFrame(data)

    # Compute score
    if len(df) > 0:
        from comparison_runner import compute_composite_score
        df = compute_composite_score(df, sharpe_col="sharpe", ic_col="ic", maxdd_col="maxdd")

    df.to_parquet(output_path, index=False)
    print(f"Exported {len(df)} results to {output_path}")

    return output_path


def export_walk_forward_results_to_cache(
    all_wf_results: Dict[Union[Tuple[str, str, int], Tuple[str, str, int, str]], list],
    output_path: Path = None,
) -> Path:
    """
    Export walk-forward results to parquet for dashboard.

    :param all_wf_results (Dict): Walk-forward results from ThreeStepEvaluation (3-tuple or 4-tuple keys)
    :param output_path (Path): Output path

    :return output_path (Path): Path where results were saved
    """
    if output_path is None:
        output_path = project_root / "data_cache" / "wf_results.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for key, results in all_wf_results.items():
        if not results:
            continue

        # Handle both 3-tuple (legacy) and 4-tuple keys
        strategy, allocation, horizon, config_name = _unpack_key(key)

        for i, r in enumerate(results):
            data.append({
                "strategy": strategy,
                "allocation": allocation,
                "horizon": horizon,
                "config": config_name,
                "window_id": i,
                "sharpe": r.sharpe,
                "ic": r.ic,
                "total_return": getattr(r, "total_return", 0.0),
            })

    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    print(f"Exported {len(df)} walk-forward results to {output_path}")

    return output_path


def export_benchmarks_to_cache(
    benchmarks: Dict[str, Any],
    output_path: Path = None,
) -> Path:
    """
    Export benchmark results to parquet for dashboard.

    :param benchmarks (Dict): Benchmark results from compute_all_benchmarks
    :param output_path (Path): Output path

    :return output_path (Path): Path where results were saved
    """
    if output_path is None:
        output_path = project_root / "data_cache" / "benchmarks.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for bm_key, bm_result in benchmarks.items():
        if bm_result is None:
            continue

        # Get monthly returns if available
        monthly_returns = getattr(bm_result, 'returns', None)
        if monthly_returns is not None:
            monthly_returns_str = ",".join([f"{r:.6f}" for r in monthly_returns])
        else:
            monthly_returns_str = ""

        data.append({
            "key": bm_key,
            "name": bm_result.name,
            "sharpe": bm_result.sharpe,
            "ic": 0.0,  # Benchmarks don't have IC
            "maxdd": bm_result.maxdd,
            "total_return": bm_result.total_return,
            "monthly_returns": monthly_returns_str,
        })

    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    print(f"Exported {len(df)} benchmark results to {output_path}")

    return output_path
