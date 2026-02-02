"""
Holdout evaluation visualization functions.

Provides plots for analyzing holdout results:
- Final vs Ensemble comparison bar charts
- Sharpe heatmaps by model type
- Scatter plots comparing Final and Ensemble
- Winner distribution pie charts
- Cumulative return curves for holdout period
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from visualization.colormaps import (
    create_sharpe_colormap,
    COMBINATION_COLORS,
    HORIZON_ALPHAS,
    CONFIG_COLORS,
    CONFIG_ORDER,
)
from utils.metrics import compute_total_return
from utils.keys import unpack_key as _unpack_key


# Type alias for holdout results (supports both 3-tuple and 4-tuple keys)
HoldoutResults = Dict[
    Union[Tuple[str, str, int], Tuple[str, str, int, str]],
    Dict[str, Any]
]


def _filter_results_by_config(
    results: HoldoutResults,
    config_filter: Optional[str] = None,
) -> HoldoutResults:
    """
    Filter results to only include specified config.

    :param results (HoldoutResults): Full results dictionary
    :param config_filter (Optional[str]): Config to filter (None = all)

    :return filtered (HoldoutResults): Filtered results
    """
    if config_filter is None:
        return results

    filtered = {}
    for key, value in results.items():
        _, _, _, config = _unpack_key(key)
        if config == config_filter:
            filtered[key] = value
    return filtered


def _get_available_combos(
    filtered_results: HoldoutResults,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Extract strategies, allocations, and horizons that have actual data.

    :param filtered_results (HoldoutResults): Filtered results dictionary

    :return strategies (List[str]): Available strategies in order
    :return allocations (List[str]): Available allocations in order
    :return horizons (List[int]): Available horizons sorted
    """
    strategies = set()
    allocations = set()
    horizons = set()

    for key, results in filtered_results.items():
        if results is not None:
            s, a, h, _ = _unpack_key(key)
            strategies.add(s)
            allocations.add(a)
            horizons.add(h)

    # Maintain consistent order
    strat_order = [s for s in ["E2E", "E2E-P3", "Sup"] if s in strategies]
    alloc_order = [a for a in ["Binary", "Multi"] if a in allocations]
    horiz_order = sorted(horizons)

    return strat_order, alloc_order, horiz_order


def plot_final_vs_ensemble_bars(
    all_holdout_results: HoldoutResults,
    figsize: Tuple[int, int] = (16, 12),
    benchmark_sharpe: Optional[float] = None,
    benchmark_name: str = "Market",
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot 2x2 grid of bar charts comparing Final vs Ensemble Sharpe.

    Grid layout:
    - Top-left: Sup (Supervised)
    - Top-right: E2E (End-to-End)
    - Bottom-left: Multi (Multi-factor)
    - Bottom-right: Binary

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param figsize (Tuple[int, int]): Figure size
    :param benchmark_sharpe (float): Optional benchmark Sharpe to show as horizontal line
    :param benchmark_name (str): Name for the benchmark line label
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)

    config_suffix = f" [{config_filter}]" if config_filter else ""

    # Get available strategies and allocations
    strategies, allocations, _ = _get_available_combos(filtered_results)

    # Build categories dynamically based on available data
    categories = []
    if "Sup" in strategies:
        categories.append(('Sup', 'Supervised', lambda d: d['strategy'] == 'Sup'))
    if "E2E" in strategies:
        categories.append(('E2E', 'End-to-End', lambda d: d['strategy'] == 'E2E'))
    if "E2E-P3" in strategies:
        categories.append(('E2E-P3', 'E2E Phase3 Only', lambda d: d['strategy'] == 'E2E-P3'))
    if "Multi" in allocations:
        categories.append(('Multi', 'Multi-factor', lambda d: d['allocation'] == 'Multi'))
    if "Binary" in allocations:
        categories.append(('Binary', 'Binary', lambda d: d['allocation'] == 'Binary'))

    # Handle case with no data
    if not categories:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No holdout data available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Create adaptive grid based on number of categories
    n_cats = len(categories)
    ncols = min(n_cats, 2)
    nrows = (n_cats + ncols - 1) // ncols
    # Scale figsize based on grid dimensions
    base_w, base_h = figsize[0] / 2, figsize[1] / 2
    actual_figsize = (base_w * ncols, base_h * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=actual_figsize, squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle(
        f"Holdout Sharpe: Final vs Ensemble by Category{config_suffix}",
        fontsize=14,
        fontweight='bold'
    )

    # Hide unused axes (if odd number of categories)
    for i in range(n_cats, len(axes_flat)):
        axes_flat[i].axis('off')

    # Check if multiple configs
    all_configs = set(_unpack_key(k)[3] for k in filtered_results.keys())
    show_config = config_filter is None and len(all_configs) > 1

    # Collect all data
    all_data = []
    for key, results in filtered_results.items():
        if results is None:
            continue
        s, a, h, cfg = _unpack_key(key)
        final = results.get('final')
        fair_ens = results.get('fair_ensemble')
        wf_ens = results.get('wf_ensemble')
        final_sharpe = final.sharpe if final else 0.0
        fair_sharpe = fair_ens.sharpe if fair_ens else 0.0
        wf_sharpe = wf_ens.sharpe if wf_ens else 0.0
        all_data.append({
            'strategy': s,
            'allocation': a,
            'horizon': h,
            'config': cfg,
            'config_label': f"-{cfg}" if show_config else "",
            'final': final_sharpe,
            'fair_ensemble': fair_sharpe,
            'wf_ensemble': wf_sharpe,
            'max_sharpe': max(final_sharpe, fair_sharpe, wf_sharpe),
        })

    for idx, (cat_key, cat_title, filter_fn) in enumerate(categories):
        ax = axes_flat[idx]

        # Filter and sort data for this category
        cat_data = [d for d in all_data if filter_fn(d)]
        cat_data.sort(key=lambda x: x['max_sharpe'], reverse=True)

        # Create labels based on category type (include config if multiple)
        if cat_key in ['Sup', 'E2E']:
            labels = [f"{d['allocation'][:1]}-{d['horizon']}M{d['config_label']}" for d in cat_data]
        else:
            labels = [f"{d['strategy']}-{d['horizon']}M{d['config_label']}" for d in cat_data]

        final_sharpes = [d['final'] for d in cat_data]
        fair_sharpes = [d['fair_ensemble'] for d in cat_data]
        wf_sharpes = [d['wf_ensemble'] for d in cat_data]

        x = np.arange(len(labels))
        width = 0.25  # Narrower for 3 bars

        bars1 = ax.bar(
            x - width, final_sharpes, width,
            label='Final', color='steelblue', alpha=0.8
        )
        bars2 = ax.bar(
            x, fair_sharpes, width,
            label='Fair Ens.', color='coral', alpha=0.8
        )
        bars3 = ax.bar(
            x + width, wf_sharpes, width,
            label='WF Ens.', color='forestgreen', alpha=0.8
        )

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        if benchmark_sharpe is not None:
            ax.axhline(
                y=benchmark_sharpe, color='purple', linestyle='--', linewidth=1.5,
                label=f'{benchmark_name} = {benchmark_sharpe:.2f}'
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(cat_title, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

        # Add value annotations (only for Final to avoid clutter)
        for bar in bars1:
            height = bar.get_height()
            if abs(height) > 0.01:
                ax.annotate(
                    f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2 if height > 0 else -8),
                    textcoords='offset points',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=6
                )

    plt.tight_layout()
    return fig


def plot_final_vs_ensemble_scatter(
    all_holdout_results: HoldoutResults,
    figsize: Tuple[int, int] = (10, 8),
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot scatter comparing Final vs Ensemble Score.

    Points above diagonal = Ensemble better, below = Final better.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)

    config_suffix = f" [{config_filter}]" if config_filter else ""
    fig, ax = plt.subplots(figsize=figsize)

    paired_data = []
    for key, results in filtered_results.items():
        if results is None:
            continue
        strategy, allocation, horizon, config = _unpack_key(key)
        final = results.get('final')
        fair_ens = results.get('fair_ensemble')

        if final and fair_ens:
            final_score = _compute_score(final.sharpe, final.ic, final.maxdd, final.total_return)
            fair_score = _compute_score(fair_ens.sharpe, fair_ens.ic, fair_ens.maxdd, fair_ens.total_return)
            paired_data.append({
                'combo': f"{strategy}-{allocation[:1]}-{horizon}M",
                'final': final_score,
                'fair_ensemble': fair_score,
            })

    if not paired_data:
        ax.text(0.5, 0.5, 'No paired data available', ha='center', va='center')
        return fig

    final_vals = [d['final'] for d in paired_data]
    fair_vals = [d['fair_ensemble'] for d in paired_data]

    ax.scatter(
        final_vals, fair_vals, s=100, alpha=0.7,
        c='darkgreen', edgecolors='black'
    )

    for d in paired_data:
        ax.annotate(
            d['combo'], (d['final'], d['fair_ensemble']),
            xytext=(5, 5), textcoords='offset points', fontsize=8
        )

    lims = [
        min(min(final_vals), min(fair_vals)) - 0.02,
        max(max(final_vals), max(fair_vals)) + 0.02
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Final = Fair Ens.')
    ax.fill_between(
        lims, lims, [lims[1]]*2,
        alpha=0.1, color='red', label='Fair Ens. > Final'
    )
    ax.fill_between(
        lims, [lims[0]]*2, lims,
        alpha=0.1, color='blue', label='Final > Fair Ens.'
    )

    ax.set_xlabel('Final Model Score')
    ax.set_ylabel('Fair Ensemble Score')
    ax.set_title(f'Final vs Fair Ensemble: Holdout Score (Fair Comparison){config_suffix}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    plt.tight_layout()
    return fig


def plot_sharpe_heatmaps_by_model_type(
    all_holdout_results: HoldoutResults,
    horizons: List[int] = None,
    figsize: Tuple[int, int] = (22, 6),
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot three heatmaps: Final, Ensemble, and Delta (Ensemble - Final).

    Rows = Strategy×Allocation (×Config if multiple configs), Columns = Horizons.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param horizons (List[int]): List of horizons (default: [1, 3, 6, 12])
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)

    if horizons is None:
        horizons = sorted(set(_unpack_key(key)[2] for key in filtered_results.keys()))

    # Get unique configs in results
    configs = sorted(
        set(_unpack_key(key)[3] for key in filtered_results.keys()),
        key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999
    )
    show_config = config_filter is None and len(configs) > 1

    config_suffix = f" [{config_filter}]" if config_filter else ""

    # Collect all sharpes for consistent colormap
    all_sharpes = []
    for key, results in filtered_results.items():
        if results is None:
            continue
        if results.get('final'):
            all_sharpes.append(results['final'].sharpe)
        if results.get('fair_ensemble'):
            all_sharpes.append(results['fair_ensemble'].sharpe)
        if results.get('wf_ensemble'):
            all_sharpes.append(results['wf_ensemble'].sharpe)

    global_min = min(all_sharpes) if all_sharpes else -0.5
    global_max = max(all_sharpes) if all_sharpes else 1.0
    cmap, norm = create_sharpe_colormap(global_min, global_max)

    # Get only strategies and allocations that have data
    strategies_list, allocations_list, available_horizons = _get_available_combos(filtered_results)

    # Use available horizons if not specified
    if not horizons:
        horizons = available_horizons

    # Handle case with no data
    if not strategies_list or not allocations_list:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No holdout data available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    heatmap_final = []
    heatmap_fair = []
    heatmap_wf = []
    row_labels = []

    # When multiple configs, iterate over all configs and add to rows
    configs_to_iterate = configs if show_config else [configs[0] if configs else "baseline"]

    for config in configs_to_iterate:
        for strategy in strategies_list:
            for allocation in allocations_list:
                row_final = []
                row_fair = []
                row_wf = []
                has_data = False
                for horizon in horizons:
                    # Try 4-tuple key first, then 3-tuple for backward compat
                    key = (strategy, allocation, horizon, config)
                    results = filtered_results.get(key)
                    if results is None:
                        results = filtered_results.get((strategy, allocation, horizon), {})
                    final_data = results.get('final') if results else None
                    fair_data = results.get('fair_ensemble') if results else None
                    wf_data = results.get('wf_ensemble') if results else None

                    if final_data:
                        has_data = True
                        row_final.append(final_data.sharpe)
                    else:
                        row_final.append(np.nan)
                    row_fair.append(fair_data.sharpe if fair_data else np.nan)
                    row_wf.append(wf_data.sharpe if wf_data else np.nan)

                # Only add row if it has data
                if has_data:
                    heatmap_final.append(row_final)
                    heatmap_fair.append(row_fair)
                    heatmap_wf.append(row_wf)
                    label = f"{strategy}-{allocation}"
                    if show_config:
                        label += f" [{config}]"
                    row_labels.append(label)

    # Handle case with no data after filtering
    if not row_labels:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No holdout data available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    heatmap_final_arr = np.array(heatmap_final)
    heatmap_fair_arr = np.array(heatmap_fair)
    heatmap_wf_arr = np.array(heatmap_wf)
    heatmap_delta_arr = heatmap_fair_arr - heatmap_final_arr

    # Adjust figsize if multiple configs (more rows)
    n_rows_data = len(row_labels)
    adjusted_height = max(4, n_rows_data * 1.2)
    actual_figsize = (figsize[0], adjusted_height * 2) if show_config else figsize

    # 2x2 layout: Final, Fair Ensemble, WF Ensemble, Delta
    fig, axes = plt.subplots(2, 2, figsize=actual_figsize)
    axes = axes.flatten()

    # Plot 3 model type heatmaps
    heatmaps = [
        ('Final', heatmap_final_arr),
        ('Fair Ensemble', heatmap_fair_arr),
        ('WF Ensemble', heatmap_wf_arr),
    ]

    label_fontsize = 8 if show_config else 10
    for idx, (model_type, heatmap_arr) in enumerate(heatmaps):
        ax = axes[idx]
        im = ax.imshow(heatmap_arr, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"{h}M" for h in horizons])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=label_fontsize)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Strategy + Allocation")
        ax.set_title(f"Holdout Sharpe: {model_type}{config_suffix}")

        for i in range(len(row_labels)):
            for j in range(len(horizons)):
                val = heatmap_arr[i, j]
                if np.isnan(val):
                    continue  # Skip NaN values
                text_color = 'white' if val < 0 or val > 0.8 else 'black'
                ax.text(
                    j, i, f"{val:.2f}", ha='center', va='center',
                    color=text_color, fontsize=10
                )

        plt.colorbar(im, ax=ax, label='Sharpe', format=plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Delta heatmap (Fair Ensemble - Final)
    ax_delta = axes[3]
    # Handle NaN in delta calculation
    valid_deltas = heatmap_delta_arr[~np.isnan(heatmap_delta_arr)]
    if len(valid_deltas) > 0:
        delta_max = max(abs(valid_deltas.min()), abs(valid_deltas.max()))
    else:
        delta_max = 0.1
    if delta_max == 0:
        delta_max = 0.1
    delta_cmap = plt.cm.RdBu
    delta_norm = plt.Normalize(vmin=-delta_max, vmax=delta_max)

    im_delta = ax_delta.imshow(heatmap_delta_arr, cmap=delta_cmap, norm=delta_norm, aspect='auto')
    ax_delta.set_xticks(range(len(horizons)))
    ax_delta.set_xticklabels([f"{h}M" for h in horizons])
    ax_delta.set_yticks(range(len(row_labels)))
    ax_delta.set_yticklabels(row_labels)
    ax_delta.set_xlabel("Horizon")
    ax_delta.set_ylabel("Strategy + Allocation")
    ax_delta.set_title("Delta: Fair Ens. - Final (Blue = Final better)")

    for i in range(len(row_labels)):
        for j in range(len(horizons)):
            val = heatmap_delta_arr[i, j]
            if np.isnan(val):
                continue  # Skip NaN values
            text_color = 'white' if abs(val) > delta_max * 0.6 else 'black'
            ax_delta.text(
                j, i, f"{val:+.2f}", ha='center', va='center',
                color=text_color, fontsize=10
            )

    plt.colorbar(im_delta, ax=ax_delta, label='Sharpe Delta', format=plt.FuncFormatter(lambda x, _: f"{x:+.2f}"))

    plt.tight_layout()
    return fig


def _compute_score(
    sharpe: float,
    ic: float,
    maxdd: float,
    total_return: float,
    reject_negative_ic_threshold: float = -0.3,
) -> float:
    """
    Compute composite score for a single model result.

    Uses the same formula as compute_composite_score in comparison_runner.py:
    - Sharpe: linear normalization with fixed bounds [-0.5, 1.5]
    - IC: asymmetric penalty (positive contributes, negative penalizes 2x)
    - MaxDD: exponential penalty via exp(3 * maxdd)
    - Return: binary bonus (1 if positive, 0 otherwise)
    - Models with IC < -0.3 are rejected (score = 0)

    :param sharpe: Sharpe ratio
    :param ic: Information Coefficient
    :param maxdd: Maximum drawdown (negative value)
    :param total_return: Total return
    :param reject_negative_ic_threshold: IC below this threshold = score 0

    :return score: Composite score (higher is better, in [0, 1])
    """
    weights = {'sharpe': 0.35, 'ic': 0.25, 'maxdd': 0.30, 'return': 0.10}
    sharpe_min, sharpe_max = -0.5, 1.5

    # 1. Reject if IC very negative (inverted predictions)
    if ic < reject_negative_ic_threshold:
        return 0.0

    # 2. Sharpe normalization [0, 1]
    sharpe_norm = np.clip((sharpe - sharpe_min) / (sharpe_max - sharpe_min), 0, 1)

    # 3. IC with asymmetric penalty (saturates at 100%, not 50%)
    if ic >= 0:
        ic_score = np.clip(ic, 0, 1)  # Positive IC: [0, 1.0] -> [0, 1]
        ic_penalty = 0.0
    else:
        ic_score = 0.0  # Negative IC contributes nothing positive
        ic_penalty = 2 * np.clip(-ic, 0, 1)  # Double penalty

    # 4. MaxDD with exponential penalty
    # -5% -> 0.86, -10% -> 0.74, -15% -> 0.64, -20% -> 0.55
    maxdd_score = np.exp(3 * maxdd)  # maxdd is negative

    # 5. Final score
    score = (
        weights['sharpe'] * sharpe_norm
        + weights['ic'] * ic_score
        + weights['maxdd'] * maxdd_score
        + weights['return'] * (1 if total_return > 0 else 0)
        - 0.15 * ic_penalty  # Explicit penalty for negative IC
    )

    return np.clip(score, 0, 1)


def plot_winner_distribution(
    all_holdout_results: HoldoutResults,
    figsize: Tuple[int, int] = (14, 5),
    benchmark_sharpe: Optional[float] = None,
    benchmark_name: str = "Best Benchmark",
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot winner distribution (by Score) and average Sharpe by model type.

    Winner is determined by composite Score (not Sharpe).

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param figsize (Tuple[int, int]): Figure size
    :param benchmark_sharpe (float): Optional benchmark Sharpe to show as horizontal line
    :param benchmark_name (str): Name for the benchmark line label
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)
    config_suffix = f" [{config_filter}]" if config_filter else ""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"Winner Distribution (by Score){config_suffix}", fontsize=14, fontweight='bold')

    final_wins = 0
    ensemble_wins = 0
    ties = 0

    final_sharpes = []
    fair_ensemble_sharpes = []
    wf_ensemble_sharpes = []

    for key, results in filtered_results.items():
        if results is None:
            continue
        final = results.get('final')
        # Support both old ('ensemble') and new ('fair_ensemble', 'wf_ensemble') formats
        fair_ensemble = results.get('fair_ensemble') or results.get('ensemble')
        wf_ensemble = results.get('wf_ensemble')
        if final is None:
            continue

        final_sharpes.append(final.sharpe)
        if fair_ensemble:
            fair_ensemble_sharpes.append(fair_ensemble.sharpe)
            # Compute scores for winner determination
            final_score = _compute_score(final.sharpe, final.ic, final.maxdd, final.total_return)
            fair_score = _compute_score(fair_ensemble.sharpe, fair_ensemble.ic, fair_ensemble.maxdd, fair_ensemble.total_return)
            if final_score > fair_score:
                final_wins += 1
            elif fair_score > final_score:
                ensemble_wins += 1
            else:
                ties += 1
        if wf_ensemble:
            wf_ensemble_sharpes.append(wf_ensemble.sharpe)

    # Handle case where no comparisons were made
    total = final_wins + ensemble_wins + ties

    ax1 = axes[0]
    if total > 0:
        ax1.pie(
            [final_wins, ensemble_wins, ties],
            labels=['Final Wins', 'Fair Ens. Wins', 'Ties'],
            autopct='%1.0f%%',
            colors=['steelblue', 'coral', 'gray'],
            startangle=90
        )
        ax1.set_title('Winner Distribution by Score (Final vs Fair Ensemble)')
    else:
        ax1.text(
            0.5, 0.5, 'No Fair Ensemble\n(WF Ensemble only)',
            ha='center', va='center', fontsize=12, color='gray'
        )
        ax1.set_title('Winner Distribution')
        ax1.axis('off')

    ax2 = axes[1]
    avg_data = {}
    bar_colors = []
    if final_sharpes:
        avg_data['Final'] = np.mean(final_sharpes)
        bar_colors.append('steelblue')
    if fair_ensemble_sharpes:
        avg_data['Fair Ens.'] = np.mean(fair_ensemble_sharpes)
        bar_colors.append('coral')
    if wf_ensemble_sharpes:
        avg_data['WF Ens.'] = np.mean(wf_ensemble_sharpes)
        bar_colors.append('green')

    colors = bar_colors
    ax2.bar(
        avg_data.keys(), avg_data.values(),
        color=colors, alpha=0.8
    )
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add benchmark reference line
    if benchmark_sharpe is not None:
        ax2.axhline(
            y=benchmark_sharpe, color='purple', linestyle='--', linewidth=2,
            label=f'{benchmark_name}: {benchmark_sharpe:.2f}'
        )
        ax2.legend(loc='upper right', fontsize=9)

    ax2.set_ylabel('Average Holdout Sharpe')
    ax2.set_title('Average Sharpe by Model Type')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    for i, (mt, val) in enumerate(avg_data.items()):
        ax2.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=11)

    plt.tight_layout()
    return fig


def plot_holdout_cumulative_returns_grid(
    all_holdout_results: HoldoutResults,
    horizons: List[int] = None,
    figsize: Tuple[int, int] = (18, 14),
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot 2x2 grid of cumulative returns for holdout period.

    One subplot per Strategy×Allocation combination.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param horizons (List[int]): List of horizons
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)
    config_suffix = f" [{config_filter}]" if config_filter else ""

    if horizons is None:
        horizons = sorted(set(_unpack_key(key)[2] for key in filtered_results.keys()))

    # Get available strategies and allocations
    strategies, allocations, _ = _get_available_combos(filtered_results)

    # Build available groups
    available_groups = []
    for s in strategies:
        for a in allocations:
            available_groups.append((s, a))

    # Handle case with no data
    n_groups = len(available_groups)
    if n_groups == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No holdout data available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Create adaptive grid
    ncols = min(n_groups, 2)
    nrows = (n_groups + ncols - 1) // ncols
    base_w, base_h = figsize[0] / 2, figsize[1] / 2
    actual_figsize = (base_w * ncols, base_h * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=actual_figsize, squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle(
        f"Holdout Cumulative Returns: All Models (Final + Ensemble){config_suffix}",
        fontsize=14,
        fontweight='bold'
    )

    # Hide unused axes
    for i in range(n_groups, len(axes_flat)):
        axes_flat[i].axis('off')

    horizon_colors = {1: 'blue', 3: 'green', 6: 'orange', 12: 'red'}

    # Build groups with axes
    groups = [(s, a, axes_flat[i]) for i, (s, a) in enumerate(available_groups)]

    config_to_use = config_filter or "baseline"

    for strategy, allocation, ax in groups:
        for horizon in horizons:
            # Try 4-tuple key first, then 3-tuple for backward compat
            key = (strategy, allocation, horizon, config_to_use)
            results = filtered_results.get(key)
            if results is None:
                results = filtered_results.get((strategy, allocation, horizon))

            if results is None:
                continue
            final = results.get('final')
            fair_ens = results.get('fair_ensemble')
            wf_ens = results.get('wf_ensemble')
            color = horizon_colors.get(horizon, 'black')

            if final and final.monthly_returns:
                cum_ret = np.cumprod(1 + np.array(final.monthly_returns))
                ax.plot(
                    range(len(cum_ret)), cum_ret,
                    color=color, linestyle='-', linewidth=2,
                    label=f"{horizon}M Final"
                )

            if fair_ens and fair_ens.monthly_returns:
                cum_ret = np.cumprod(1 + np.array(fair_ens.monthly_returns))
                ax.plot(
                    range(len(cum_ret)), cum_ret,
                    color=color, linestyle='--', linewidth=2, alpha=0.7,
                    label=f"{horizon}M Fair"
                )

            if wf_ens and wf_ens.monthly_returns:
                cum_ret = np.cumprod(1 + np.array(wf_ens.monthly_returns))
                ax.plot(
                    range(len(cum_ret)), cum_ret,
                    color=color, linestyle=':', linewidth=2, alpha=0.5,
                    label=f"{horizon}M WF"
                )

        ax.axhline(y=1, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f"{strategy} + {allocation}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Month (Holdout)")
        ax.set_ylabel("Cumulative Return")
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_top_models_cumulative(
    all_holdout_results: HoldoutResults,
    top_n: int = 5,
    figsize: Tuple[int, int] = (14, 8),
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot cumulative returns for top N models of each type.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param top_n (int): Number of top models per type to show
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)
    config_suffix = f" [{config_filter}]" if config_filter else ""

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Holdout Cumulative Returns: Top {top_n} per Type (Final, Fair, WF){config_suffix}",
        fontsize=14,
        fontweight='bold'
    )

    final_data = []
    fair_data = []
    wf_data = []

    for key, results in filtered_results.items():
        if results is None:
            continue
        strategy, allocation, horizon, config = _unpack_key(key)
        label = f"{strategy}-{allocation[:1]}-{horizon}M"
        if config != "baseline":
            label += f"-{config}"

        if results.get('final'):
            r = results['final']
            score = _compute_score(r.sharpe, r.ic, r.maxdd, r.total_return)
            final_data.append({
                'key': key,
                'label': label + "-F",
                'sharpe': r.sharpe,
                'score': score,
                'returns': r.monthly_returns,
            })
        if results.get('fair_ensemble'):
            r = results['fair_ensemble']
            score = _compute_score(r.sharpe, r.ic, r.maxdd, r.total_return)
            fair_data.append({
                'key': key,
                'label': label + "-FE",
                'sharpe': r.sharpe,
                'score': score,
                'returns': r.monthly_returns,
            })
        if results.get('wf_ensemble'):
            r = results['wf_ensemble']
            score = _compute_score(r.sharpe, r.ic, r.maxdd, r.total_return)
            wf_data.append({
                'key': key,
                'label': label + "-WF",
                'sharpe': r.sharpe,
                'score': score,
                'returns': r.monthly_returns,
            })

    # Sort by composite score (not Sharpe) for fairer multi-metric ranking
    final_data.sort(key=lambda x: x['score'], reverse=True)
    fair_data.sort(key=lambda x: x['score'], reverse=True)
    wf_data.sort(key=lambda x: x['score'], reverse=True)

    # Final models (solid lines)
    for d in final_data[:top_n]:
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2.5, linestyle='-',
                label=f"{d['label']} (Score={d['score']*100:.1f}%)"
            )

    # Fair Ensemble models (dashed lines)
    for d in fair_data[:top_n]:
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2.5, linestyle='--',
                label=f"{d['label']} (Score={d['score']*100:.1f}%)"
            )

    # WF Ensemble models (dotted lines)
    for d in wf_data[:top_n]:
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2, linestyle=':',
                label=f"{d['label']} (Score={d['score']*100:.1f}%)"
            )

    ax.axhline(y=1, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel("Month (Holdout)")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_top_models_with_benchmarks(
    all_holdout_results: HoldoutResults,
    benchmarks: Optional[Dict[str, Any]] = None,
    top_n: int = 3,
    figsize: Tuple[int, int] = (14, 8),
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot cumulative returns for top N models with benchmarks.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param benchmarks (Dict[str, BenchmarkResult]): Benchmark results
    :param top_n (int): Number of top models to show
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)
    config_suffix = f" [{config_filter}]" if config_filter else ""

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Holdout Cumulative Returns: Top {top_n} Models vs Benchmarks{config_suffix}",
        fontsize=14,
        fontweight='bold'
    )

    all_models = []
    for key, results in filtered_results.items():
        if results is None:
            continue
        strategy, allocation, horizon, config = _unpack_key(key)
        label = f"{strategy}-{allocation[:1]}-{horizon}M"
        if config != "baseline":
            label += f"-{config}"

        if results.get('final'):
            r = results['final']
            score = _compute_score(r.sharpe, r.ic, r.maxdd, r.total_return)
            all_models.append({
                'label': label + "-F",
                'sharpe': r.sharpe,
                'score': score,
                'returns': r.monthly_returns,
                'type': 'Final',
            })
        if results.get('fair_ensemble'):
            r = results['fair_ensemble']
            score = _compute_score(r.sharpe, r.ic, r.maxdd, r.total_return)
            all_models.append({
                'label': label + "-FE",
                'sharpe': r.sharpe,
                'score': score,
                'returns': r.monthly_returns,
                'type': 'Fair Ensemble',
            })
        if results.get('wf_ensemble'):
            r = results['wf_ensemble']
            score = _compute_score(r.sharpe, r.ic, r.maxdd, r.total_return)
            all_models.append({
                'label': label + "-WF",
                'sharpe': r.sharpe,
                'score': score,
                'returns': r.monthly_returns,
                'type': 'WF Ensemble',
            })

    # Sort by composite score (not Sharpe) for fairer multi-metric ranking
    all_models.sort(key=lambda x: x['score'], reverse=True)

    # Colors for top models (solid lines)
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    for i, d in enumerate(all_models[:top_n]):
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2.5, linestyle='-', color=model_colors[i % len(model_colors)],
                label=f"{d['label']} (Score={d['score']*100:.1f}%)"
            )

    if benchmarks:
        # Colors for benchmarks (dashed lines)
        benchmark_colors = {
            'equal_weight_6f': '#9467bd',
            'equal_weight_cyc_def': '#8c564b',
            'risk_parity': '#e377c2',
            'factor_momentum': '#17becf',
            'best_single_factor': '#bcbd22',
        }

        n_months = None
        for d in all_models[:top_n]:
            if d['returns']:
                n_months = len(d['returns'])
                break

        for bm_key, bm_result in benchmarks.items():
            if bm_result is None:
                continue
            bm_returns = bm_result.returns[:n_months] if n_months else bm_result.returns
            if len(bm_returns) > 0:
                cum_ret = np.cumprod(1 + np.array(bm_returns))
                ax.plot(
                    range(len(cum_ret)), cum_ret,
                    linewidth=2, linestyle='--',
                    color=benchmark_colors.get(bm_key, 'gray'),
                    label=f"{bm_result.name} (Sharpe={bm_result.sharpe:.2f})"
                )

    ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Month (Holdout)")
    ax.set_ylabel("Cumulative Return (Wealth)")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_holdout_summary_table(
    all_holdout_results: HoldoutResults,
    compute_score_fn: callable = None,
    config_filter: Optional[str] = "baseline",
) -> pd.DataFrame:
    """
    Print and return holdout results summary table.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param compute_score_fn (callable): Function to compute composite score
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return df (pd.DataFrame): Summary DataFrame
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)
    config_suffix = f" [{config_filter}]" if config_filter else ""

    vis_data = []
    for key, results in filtered_results.items():
        if results is None:
            continue

        strategy, allocation, horizon, config = _unpack_key(key)
        label_base = f"{strategy}-{allocation[:1]}-{horizon}M"
        if config != "baseline":
            label_base += f"-{config}"

        final = results.get('final')
        fair_ens = results.get('fair_ensemble')
        wf_ens = results.get('wf_ensemble')

        if final:
            vis_data.append({
                'label': f"{label_base}-F",
                'combination': label_base,
                'model_type': 'Final',
                'strategy': strategy,
                'allocation': allocation,
                'horizon': horizon,
                'config': config,
                'sharpe': final.sharpe,
                'total_return': final.total_return,
                'maxdd': final.maxdd,
                'ic': final.ic,
            })

        if fair_ens and fair_ens.sharpe != 0.0:
            vis_data.append({
                'label': f"{label_base}-FE",
                'combination': label_base,
                'model_type': 'Fair Ens.',
                'strategy': strategy,
                'allocation': allocation,
                'horizon': horizon,
                'config': config,
                'sharpe': fair_ens.sharpe,
                'total_return': fair_ens.total_return,
                'maxdd': fair_ens.maxdd,
                'ic': fair_ens.ic,
            })

        if wf_ens and wf_ens.sharpe != 0.0:
            vis_data.append({
                'label': f"{label_base}-WF",
                'combination': label_base,
                'model_type': 'WF Ens.',
                'strategy': strategy,
                'allocation': allocation,
                'horizon': horizon,
                'config': config,
                'sharpe': wf_ens.sharpe,
                'total_return': wf_ens.total_return,
                'maxdd': wf_ens.maxdd,
                'ic': wf_ens.ic,
            })

    vis_df = pd.DataFrame(vis_data)

    if len(vis_df) > 0 and compute_score_fn is not None:
        vis_df = compute_score_fn(
            vis_df,
            sharpe_col='sharpe',
            ic_col='ic',
            maxdd_col='maxdd',
        )
        vis_df = vis_df.sort_values('score', ascending=False)

    # Check if we have multiple configs
    show_config_col = config_filter is None and len(vis_df) > 0 and vis_df['config'].nunique() > 1

    print("\n" + "=" * 140)
    print(f"COMPLETE HOLDOUT RESULTS (with Composite Score){config_suffix}")
    print("=" * 140)

    if show_config_col:
        print(f"\n{'Label':<18} {'Strategy':<8} {'Alloc':<7} {'H':<4} {'Config':<10} {'Type':<10} "
              f"{'Sharpe':>10} {'IC':>8} {'MaxDD':>10} {'Return':>10} {'Score':>8} {'Rank':>6}")
        print("-" * 140)

        for _, row in vis_df.iterrows():
            score = row.get('score', 0)
            rank = row.get('rank', 0)
            print(f"{row['label']:<18} {row['strategy']:<8} {row['allocation']:<7} "
                  f"{row['horizon']}M{'':<2} {row['config']:<10} {row['model_type']:<10} "
                  f"{row['sharpe']:>+10.2f} {row['ic']*100:>+8.1f}% "
                  f"{row['maxdd']:>+10.1%} {row['total_return']:>+10.1%} "
                  f"{score:>8.1%} {int(rank):>6}")
    else:
        print(f"\n{'Label':<18} {'Strategy':<8} {'Alloc':<7} {'H':<4} {'Type':<10} "
              f"{'Sharpe':>10} {'IC':>8} {'MaxDD':>10} {'Return':>10} {'Score':>8} {'Rank':>6}")
        print("-" * 120)

        for _, row in vis_df.iterrows():
            score = row.get('score', 0)
            rank = row.get('rank', 0)
            print(f"{row['label']:<18} {row['strategy']:<8} {row['allocation']:<7} "
                  f"{row['horizon']}M{'':<2} {row['model_type']:<10} "
                  f"{row['sharpe']:>+10.2f} {row['ic']*100:>+8.1f}% "
                  f"{row['maxdd']:>+10.1%} {row['total_return']:>+10.1%} "
                  f"{score:>8.1%} {int(rank):>6}")

    print("=" * 140)

    return vis_df


def plot_all_holdout(
    all_holdout_results: HoldoutResults,
    horizons: List[int] = None,
    compute_score_fn: callable = None,
    benchmarks: Optional[Dict[str, Any]] = None,
    show: bool = True,
    config_filter: Optional[str] = "baseline",
) -> List[Figure]:
    """
    Create all holdout visualization figures.

    Convenience function to generate all standard plots.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param horizons (List[int]): List of horizons
    :param compute_score_fn (callable): Function to compute composite score
    :param benchmarks (Dict[str, BenchmarkResult]): Optional benchmark results
    :param show (bool): Whether to call plt.show() for each figure
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return figures (List[Figure]): List of generated figures
    """
    filtered_results = _filter_results_by_config(all_holdout_results, config_filter)
    config_suffix = f" [{config_filter}]" if config_filter else ""
    figures = []

    # Find the best benchmark by Sharpe ratio
    benchmark_sharpe = None
    benchmark_name = "Best Benchmark"
    if benchmarks:
        best_bm = max(
            [(k, v) for k, v in benchmarks.items() if v is not None],
            key=lambda x: x[1].sharpe,
            default=(None, None)
        )
        if best_bm[1] is not None:
            benchmark_sharpe = best_bm[1].sharpe
            benchmark_name = best_bm[1].name

    fig1 = plot_final_vs_ensemble_bars(
        filtered_results,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_name=benchmark_name,
        config_filter=config_filter,
    )
    figures.append(fig1)
    if show:
        plt.show()

    fig2 = plot_final_vs_ensemble_scatter(filtered_results, config_filter=config_filter)
    figures.append(fig2)
    if show:
        plt.show()

    fig3 = plot_sharpe_heatmaps_by_model_type(filtered_results, horizons, config_filter=config_filter)
    figures.append(fig3)
    if show:
        plt.show()

    fig4 = plot_winner_distribution(
        filtered_results,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_name=benchmark_name,
        config_filter=config_filter,
    )
    figures.append(fig4)
    if show:
        plt.show()

    fig5 = plot_top_models_with_benchmarks(
        filtered_results,
        benchmarks=benchmarks,
        top_n=3,
        config_filter=config_filter,
    )
    figures.append(fig5)
    if show:
        plt.show()

    print_holdout_summary_table(filtered_results, compute_score_fn, config_filter=config_filter)

    # Build comparison table: Top 3 models + Benchmarks
    print("\n" + "=" * 100)
    print(f"TOP 3 MODELS vs BENCHMARKS (Holdout Period){config_suffix}")
    print("=" * 100)
    print(f"{'Name':<25} {'Sharpe':>10} {'IC':>10} {'MaxDD':>10} {'Return':>12} {'Score':>10}")
    print("-" * 100)

    # Get top 3 models (all types)
    all_models = []
    for key, results in filtered_results.items():
        if results is None:
            continue
        strategy, allocation, horizon, config = _unpack_key(key)
        label = f"{strategy}-{allocation[:1]}-{horizon}M"
        if config != "baseline":
            label += f"-{config}"
        if results.get('final'):
            final = results['final']
            all_models.append({
                'name': label + "-F",
                'sharpe': final.sharpe,
                'ic': final.ic,
                'total_return': final.total_return,
                'maxdd': final.maxdd,
            })
        if results.get('fair_ensemble'):
            fair_ens = results['fair_ensemble']
            all_models.append({
                'name': label + "-FE",
                'sharpe': fair_ens.sharpe,
                'ic': fair_ens.ic,
                'total_return': fair_ens.total_return,
                'maxdd': fair_ens.maxdd,
            })
        if results.get('wf_ensemble'):
            wf_ens = results['wf_ensemble']
            all_models.append({
                'name': label + "-WF",
                'sharpe': wf_ens.sharpe,
                'ic': wf_ens.ic,
                'total_return': wf_ens.total_return,
                'maxdd': wf_ens.maxdd,
            })

    # Compute scores if function provided
    if compute_score_fn is not None and all_models:
        import pandas as pd
        df_models = pd.DataFrame(all_models)
        df_with_scores = compute_score_fn(df_models, total_return_col='total_return')
        for i, m in enumerate(all_models):
            m['score'] = df_with_scores.loc[df_with_scores['name'] == m['name'], 'score'].values[0]
    else:
        for m in all_models:
            m['score'] = 0.0

    all_models.sort(key=lambda x: x['score'], reverse=True)
    for m in all_models[:3]:
        print(f"{m['name']:<25} {m['sharpe']:>+10.2f} {m['ic']*100:>+10.1f}% "
              f"{m['maxdd']:>+10.1%} {m['total_return']:>+11.1%} {m['score']:>10.2%}")

    print("-" * 100)

    # Print benchmarks (no score for benchmarks)
    if benchmarks:
        for bm_key, bm_result in benchmarks.items():
            if bm_result:
                ic_val = getattr(bm_result, 'ic', 0.0) or 0.0
                print(f"{bm_result.name:<25} {bm_result.sharpe:>+10.2f} {ic_val*100:>+10.1f}% "
                      f"{bm_result.maxdd:>+10.1%} {bm_result.total_return:>+11.1%} {'N/A':>10}")

    print("=" * 100)

    return figures


def plot_config_comparison(
    all_holdout_results: HoldoutResults,
    metric: str = "sharpe",
    model_type: str = "final",
    compute_score_fn: Optional[callable] = None,
) -> Figure:
    """
    Bar chart comparing configs (baseline vs fs vs hpt vs fs+hpt).

    Groups by config, shows avg metric across all 16 base combinations.

    :param all_holdout_results (HoldoutResults): All holdout results (4-tuple keys)
    :param metric (str): Metric to compare ("sharpe", "ic", "maxdd", "score")
    :param model_type (str): Model type to use ("final", "fair_ensemble", "wf_ensemble")
    :param compute_score_fn (callable): Function to compute composite score

    :return fig (Figure): Matplotlib figure
    """
    # Collect data by config
    config_data: Dict[str, List[float]] = {}

    for key, results in all_holdout_results.items():
        if results is None:
            continue
        _, _, _, config = _unpack_key(key)

        result = results.get(model_type)
        if result is None:
            continue

        if config not in config_data:
            config_data[config] = []

        if metric == "sharpe":
            config_data[config].append(result.sharpe)
        elif metric == "ic":
            config_data[config].append(result.ic)
        elif metric == "maxdd":
            config_data[config].append(result.maxdd)
        elif metric == "score" and compute_score_fn is not None:
            # Compute score for this result
            df_temp = pd.DataFrame([{
                "sharpe": result.sharpe,
                "ic": result.ic,
                "maxdd": result.maxdd,
                "total_return": result.total_return,
            }])
            df_with_score = compute_score_fn(df_temp, total_return_col="total_return")
            config_data[config].append(df_with_score["score"].iloc[0])
        else:
            config_data[config].append(result.sharpe)

    if not config_data:
        print("No data to plot for config comparison")
        return None

    # Calculate averages
    configs = list(config_data.keys())
    avgs = [np.mean(config_data[c]) for c in configs]
    stds = [np.std(config_data[c]) for c in configs]

    # Sort by average value (descending for sharpe/ic/score, ascending for maxdd)
    if metric == "maxdd":
        sorted_idx = np.argsort(avgs)  # Less negative = better
    else:
        sorted_idx = np.argsort(avgs)[::-1]  # Higher = better

    configs = [configs[i] for i in sorted_idx]
    avgs = [avgs[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping for configs
    config_colors = {
        "baseline": "#4285F4",  # Google Blue
        "fs": "#34A853",        # Google Green
        "hpt": "#FBBC04",       # Google Yellow
        "fs+hpt": "#EA4335",    # Google Red
    }
    colors = [config_colors.get(c, "#666666") for c in configs]

    # Bar chart
    x = np.arange(len(configs))
    bars = ax.bar(x, avgs, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar, avg, std in zip(bars, avgs, stds):
        height = bar.get_height()
        if metric in ["ic", "maxdd"]:
            label = f"{avg*100:+.1f}%"
        elif metric == "score":
            label = f"{avg:.1%}"
        else:
            label = f"{avg:+.2f}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)

    metric_labels = {
        "sharpe": "Sharpe Ratio",
        "ic": "Information Coefficient",
        "maxdd": "Max Drawdown",
        "score": "Composite Score",
    }
    model_labels = {
        "final": "Final Model",
        "fair_ensemble": "Fair Ensemble",
        "wf_ensemble": "WF Ensemble",
    }
    ax.set_title(
        f"Config Comparison: {metric_labels.get(metric, metric)} ({model_labels.get(model_type, model_type)})",
        fontsize=14,
        fontweight="bold",
    )

    # Add baseline reference line
    if "baseline" in config_data:
        baseline_avg = np.mean(config_data["baseline"])
        ax.axhline(baseline_avg, color="gray", linestyle="--", alpha=0.7, label="Baseline avg")
        ax.legend(loc="best")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return fig
