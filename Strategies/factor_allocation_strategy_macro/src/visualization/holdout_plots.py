"""
Holdout evaluation visualization functions.

Provides plots for analyzing holdout results:
- Final vs Ensemble comparison bar charts
- Sharpe heatmaps by model type
- Scatter plots comparing Final and Ensemble
- Winner distribution pie charts
- Cumulative return curves for holdout period
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from visualization.colormaps import (
    create_sharpe_colormap,
    COMBINATION_COLORS,
    HORIZON_ALPHAS,
)
from utils.metrics import compute_total_return


# Type alias for holdout results
HoldoutResults = Dict[Tuple[str, str, int], Dict[str, Any]]


def plot_final_vs_ensemble_bars(
    all_holdout_results: HoldoutResults,
    figsize: Tuple[int, int] = (16, 12),
    benchmark_sharpe: Optional[float] = None,
    benchmark_name: str = "Market",
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

    :return fig (Figure): Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Holdout Sharpe: Final vs Ensemble by Category",
        fontsize=14,
        fontweight='bold'
    )

    # Collect all data
    all_data = []
    for (s, a, h), results in all_holdout_results.items():
        if results is None:
            continue
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
            'label': f"{h}M",
            'final': final_sharpe,
            'fair_ensemble': fair_sharpe,
            'wf_ensemble': wf_sharpe,
            'max_sharpe': max(final_sharpe, fair_sharpe, wf_sharpe),
        })

    # Define the 4 categories
    categories = [
        ('Sup', 'Supervised', lambda d: d['strategy'] == 'Sup'),
        ('E2E', 'End-to-End', lambda d: d['strategy'] == 'E2E'),
        ('Multi', 'Multi-factor', lambda d: d['allocation'] == 'Multi'),
        ('Binary', 'Binary', lambda d: d['allocation'] == 'Binary'),
    ]

    for idx, (cat_key, cat_title, filter_fn) in enumerate(categories):
        ax = axes[idx // 2, idx % 2]

        # Filter and sort data for this category
        cat_data = [d for d in all_data if filter_fn(d)]
        cat_data.sort(key=lambda x: x['max_sharpe'], reverse=True)

        if not cat_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(cat_title)
            continue

        # Create labels based on category type
        if cat_key in ['Sup', 'E2E']:
            labels = [f"{d['allocation'][:1]}-{d['horizon']}M" for d in cat_data]
        else:
            labels = [f"{d['strategy']}-{d['horizon']}M" for d in cat_data]

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
) -> Figure:
    """
    Plot scatter comparing Final vs Ensemble Sharpe.

    Points above diagonal = Ensemble better, below = Final better.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    paired_data = []
    for key, results in all_holdout_results.items():
        if results is None:
            continue
        strategy, allocation, horizon = key
        final = results.get('final')
        fair_ens = results.get('fair_ensemble')

        if final and fair_ens:
            paired_data.append({
                'combo': f"{strategy}-{allocation[:1]}-{horizon}M",
                'final': final.sharpe,
                'fair_ensemble': fair_ens.sharpe,
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
        min(min(final_vals), min(fair_vals)) - 0.2,
        max(max(final_vals), max(fair_vals)) + 0.2
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

    ax.set_xlabel('Final Model Sharpe')
    ax.set_ylabel('Fair Ensemble Sharpe')
    ax.set_title('Final vs Fair Ensemble: Holdout Sharpe (Fair Comparison)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    return fig


def plot_sharpe_heatmaps_by_model_type(
    all_holdout_results: HoldoutResults,
    horizons: List[int] = None,
    figsize: Tuple[int, int] = (22, 6),
) -> Figure:
    """
    Plot three heatmaps: Final, Ensemble, and Delta (Ensemble - Final).

    Rows = Strategy×Allocation, Columns = Horizons.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param horizons (List[int]): List of horizons (default: [1, 3, 6, 12])
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    if horizons is None:
        horizons = sorted(set(key[2] for key in all_holdout_results.keys()))

    # Collect all sharpes for consistent colormap
    all_sharpes = []
    for key, results in all_holdout_results.items():
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

    strategies_list = ["E2E", "Sup"]
    allocations_list = ["Binary", "Multi"]

    heatmap_final = []
    heatmap_fair = []
    heatmap_wf = []
    row_labels = []

    for strategy in strategies_list:
        for allocation in allocations_list:
            row_final = []
            row_fair = []
            row_wf = []
            for horizon in horizons:
                key = (strategy, allocation, horizon)
                results = all_holdout_results.get(key, {})
                final_data = results.get('final')
                fair_data = results.get('fair_ensemble')
                wf_data = results.get('wf_ensemble')
                row_final.append(final_data.sharpe if final_data else 0.0)
                row_fair.append(fair_data.sharpe if fair_data else 0.0)
                row_wf.append(wf_data.sharpe if wf_data else 0.0)
            heatmap_final.append(row_final)
            heatmap_fair.append(row_fair)
            heatmap_wf.append(row_wf)
            row_labels.append(f"{strategy}-{allocation}")

    heatmap_final_arr = np.array(heatmap_final)
    heatmap_fair_arr = np.array(heatmap_fair)
    heatmap_wf_arr = np.array(heatmap_wf)
    heatmap_delta_arr = heatmap_fair_arr - heatmap_final_arr

    # 2x2 layout: Final, Fair Ensemble, WF Ensemble, Delta
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Plot 3 model type heatmaps
    heatmaps = [
        ('Final', heatmap_final_arr),
        ('Fair Ensemble', heatmap_fair_arr),
        ('WF Ensemble', heatmap_wf_arr),
    ]

    for idx, (model_type, heatmap_arr) in enumerate(heatmaps):
        ax = axes[idx]
        im = ax.imshow(heatmap_arr, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"{h}M" for h in horizons])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Strategy + Allocation")
        ax.set_title(f"Holdout Sharpe: {model_type}")

        for i in range(len(row_labels)):
            for j in range(len(horizons)):
                val = heatmap_arr[i, j]
                text_color = 'white' if val < 0 or val > 0.8 else 'black'
                ax.text(
                    j, i, f"{val:.2f}", ha='center', va='center',
                    color=text_color, fontsize=10
                )

        plt.colorbar(im, ax=ax, label='Sharpe', format=plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Delta heatmap (Fair Ensemble - Final)
    ax_delta = axes[3]
    delta_max = max(abs(heatmap_delta_arr.min()), abs(heatmap_delta_arr.max()))
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
            text_color = 'white' if abs(val) > delta_max * 0.6 else 'black'
            ax_delta.text(
                j, i, f"{val:+.2f}", ha='center', va='center',
                color=text_color, fontsize=10
            )

    plt.colorbar(im_delta, ax=ax_delta, label='Sharpe Delta', format=plt.FuncFormatter(lambda x, _: f"{x:+.2f}"))

    plt.tight_layout()
    return fig


def plot_winner_distribution(
    all_holdout_results: HoldoutResults,
    figsize: Tuple[int, int] = (14, 5),
    benchmark_sharpe: Optional[float] = None,
    benchmark_name: str = "Best Benchmark",
) -> Figure:
    """
    Plot winner distribution and average Sharpe by model type.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param figsize (Tuple[int, int]): Figure size
    :param benchmark_sharpe (float): Optional benchmark Sharpe to show as horizontal line
    :param benchmark_name (str): Name for the benchmark line label

    :return fig (Figure): Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    final_wins = 0
    ensemble_wins = 0
    ties = 0

    final_sharpes = []
    fair_ensemble_sharpes = []
    wf_ensemble_sharpes = []

    for key, results in all_holdout_results.items():
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
            if final.sharpe > fair_ensemble.sharpe:
                final_wins += 1
            elif fair_ensemble.sharpe > final.sharpe:
                ensemble_wins += 1
            else:
                ties += 1
        if wf_ensemble:
            wf_ensemble_sharpes.append(wf_ensemble.sharpe)

    # Handle case where no comparisons were made
    total = final_wins + ensemble_wins + ties
    if total == 0:
        total = 1  # Avoid division by zero

    ax1 = axes[0]
    ax1.pie(
        [final_wins, ensemble_wins, ties],
        labels=['Final Wins', 'Fair Ens. Wins', 'Ties'],
        autopct='%1.0f%%',
        colors=['steelblue', 'coral', 'gray'],
        startangle=90
    )
    ax1.set_title('Winner Distribution (Final vs Fair Ensemble)')

    ax2 = axes[1]
    avg_data = {
        'Final': np.mean(final_sharpes) if final_sharpes else 0,
        'Fair Ens.': np.mean(fair_ensemble_sharpes) if fair_ensemble_sharpes else 0,
    }
    if wf_ensemble_sharpes:
        avg_data['WF Ens.'] = np.mean(wf_ensemble_sharpes)

    colors = ['steelblue', 'coral', 'green'][:len(avg_data)]
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
) -> Figure:
    """
    Plot 2x2 grid of cumulative returns for holdout period.

    One subplot per Strategy×Allocation combination.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param horizons (List[int]): List of horizons
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    if horizons is None:
        horizons = sorted(set(key[2] for key in all_holdout_results.keys()))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Holdout Cumulative Returns: All Models (Final + Ensemble)",
        fontsize=14,
        fontweight='bold'
    )

    horizon_colors = {1: 'blue', 3: 'green', 6: 'orange', 12: 'red'}

    groups = [
        ("E2E", "Binary", axes[0, 0]),
        ("E2E", "Multi", axes[0, 1]),
        ("Sup", "Binary", axes[1, 0]),
        ("Sup", "Multi", axes[1, 1]),
    ]

    for strategy, allocation, ax in groups:
        for horizon in horizons:
            key = (strategy, allocation, horizon)

            if key not in all_holdout_results or all_holdout_results[key] is None:
                continue

            results = all_holdout_results[key]
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
) -> Figure:
    """
    Plot cumulative returns for top N models of each type.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param top_n (int): Number of top models per type to show
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Holdout Cumulative Returns: Top {top_n} per Type (Final, Fair, WF)",
        fontsize=14,
        fontweight='bold'
    )

    final_data = []
    fair_data = []
    wf_data = []

    for key, results in all_holdout_results.items():
        if results is None:
            continue
        strategy, allocation, horizon = key
        label = f"{strategy}-{allocation[:1]}-{horizon}M"

        if results.get('final'):
            final_data.append({
                'key': key,
                'label': label + "-F",
                'sharpe': results['final'].sharpe,
                'returns': results['final'].monthly_returns,
            })
        if results.get('fair_ensemble'):
            fair_data.append({
                'key': key,
                'label': label + "-FE",
                'sharpe': results['fair_ensemble'].sharpe,
                'returns': results['fair_ensemble'].monthly_returns,
            })
        if results.get('wf_ensemble'):
            wf_data.append({
                'key': key,
                'label': label + "-WF",
                'sharpe': results['wf_ensemble'].sharpe,
                'returns': results['wf_ensemble'].monthly_returns,
            })

    final_data.sort(key=lambda x: x['sharpe'], reverse=True)
    fair_data.sort(key=lambda x: x['sharpe'], reverse=True)
    wf_data.sort(key=lambda x: x['sharpe'], reverse=True)

    # Final models (solid lines)
    for d in final_data[:top_n]:
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2.5, linestyle='-',
                label=f"{d['label']} (Sharpe={d['sharpe']:.2f})"
            )

    # Fair Ensemble models (dashed lines)
    for d in fair_data[:top_n]:
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2.5, linestyle='--',
                label=f"{d['label']} (Sharpe={d['sharpe']:.2f})"
            )

    # WF Ensemble models (dotted lines)
    for d in wf_data[:top_n]:
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2, linestyle=':',
                label=f"{d['label']} (Sharpe={d['sharpe']:.2f})"
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
) -> Figure:
    """
    Plot cumulative returns for top N models with benchmarks.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param benchmarks (Dict[str, BenchmarkResult]): Benchmark results
    :param top_n (int): Number of top models to show
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Holdout Cumulative Returns: Top {top_n} Models vs Benchmarks",
        fontsize=14,
        fontweight='bold'
    )

    all_models = []
    for key, results in all_holdout_results.items():
        if results is None:
            continue
        strategy, allocation, horizon = key
        label = f"{strategy}-{allocation[:1]}-{horizon}M"

        if results.get('final'):
            all_models.append({
                'label': label + "-F",
                'sharpe': results['final'].sharpe,
                'returns': results['final'].monthly_returns,
                'type': 'Final',
            })
        if results.get('fair_ensemble'):
            all_models.append({
                'label': label + "-FE",
                'sharpe': results['fair_ensemble'].sharpe,
                'returns': results['fair_ensemble'].monthly_returns,
                'type': 'Fair Ensemble',
            })
        if results.get('wf_ensemble'):
            all_models.append({
                'label': label + "-WF",
                'sharpe': results['wf_ensemble'].sharpe,
                'returns': results['wf_ensemble'].monthly_returns,
                'type': 'WF Ensemble',
            })

    all_models.sort(key=lambda x: x['sharpe'], reverse=True)

    # Colors for top models (solid lines)
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    for i, d in enumerate(all_models[:top_n]):
        if d['returns']:
            cum_ret = np.cumprod(1 + np.array(d['returns']))
            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2.5, linestyle='-', color=model_colors[i % len(model_colors)],
                label=f"{d['label']} (Sharpe={d['sharpe']:.2f})"
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
) -> pd.DataFrame:
    """
    Print and return holdout results summary table.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param compute_score_fn (callable): Function to compute composite score

    :return df (pd.DataFrame): Summary DataFrame
    """
    vis_data = []
    for key, results in all_holdout_results.items():
        if results is None:
            continue

        strategy, allocation, horizon = key
        label_base = f"{strategy}-{allocation[:1]}-{horizon}M"

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

    print("\n" + "=" * 130)
    print("COMPLETE HOLDOUT RESULTS (with Composite Score)")
    print("=" * 130)
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

    print("=" * 130)

    return vis_df


def plot_all_holdout(
    all_holdout_results: HoldoutResults,
    horizons: List[int] = None,
    compute_score_fn: callable = None,
    benchmarks: Optional[Dict[str, Any]] = None,
    show: bool = True,
) -> List[Figure]:
    """
    Create all holdout visualization figures.

    Convenience function to generate all standard plots.

    :param all_holdout_results (HoldoutResults): Holdout results dict
    :param horizons (List[int]): List of horizons
    :param compute_score_fn (callable): Function to compute composite score
    :param benchmarks (Dict[str, BenchmarkResult]): Optional benchmark results
    :param show (bool): Whether to call plt.show() for each figure

    :return figures (List[Figure]): List of generated figures
    """
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
        all_holdout_results,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_name=benchmark_name,
    )
    figures.append(fig1)
    if show:
        plt.show()

    fig2 = plot_final_vs_ensemble_scatter(all_holdout_results)
    figures.append(fig2)
    if show:
        plt.show()

    fig3 = plot_sharpe_heatmaps_by_model_type(all_holdout_results, horizons)
    figures.append(fig3)
    if show:
        plt.show()

    fig4 = plot_winner_distribution(
        all_holdout_results,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_name=benchmark_name,
    )
    figures.append(fig4)
    if show:
        plt.show()

    fig5 = plot_top_models_with_benchmarks(
        all_holdout_results,
        benchmarks=benchmarks,
        top_n=3,
    )
    figures.append(fig5)
    if show:
        plt.show()

    print_holdout_summary_table(all_holdout_results, compute_score_fn)

    # Build comparison table: Top 3 models + Benchmarks
    print("\n" + "=" * 90)
    print("TOP 3 MODELS vs BENCHMARKS (Holdout Period)")
    print("=" * 90)
    print(f"{'Name':<25} {'Sharpe':>10} {'IC':>10} {'MaxDD':>10} {'Return':>12}")
    print("-" * 90)

    # Get top 3 models (all types)
    all_models = []
    for key, results in all_holdout_results.items():
        if results is None:
            continue
        strategy, allocation, horizon = key
        label = f"{strategy}-{allocation[:1]}-{horizon}M"
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

    all_models.sort(key=lambda x: x['sharpe'], reverse=True)
    for m in all_models[:3]:
        print(f"{m['name']:<25} {m['sharpe']:>+10.2f} {m['ic']*100:>+10.1f}% "
              f"{m['maxdd']:>+10.1%} {m['total_return']:>+11.1%}")

    print("-" * 90)

    # Print benchmarks
    if benchmarks:
        for bm_key, bm_result in benchmarks.items():
            if bm_result:
                ic_val = getattr(bm_result, 'ic', 0.0) or 0.0
                print(f"{bm_result.name:<25} {bm_result.sharpe:>+10.2f} {ic_val*100:>+10.1f}% "
                      f"{bm_result.maxdd:>+10.1%} {bm_result.total_return:>+11.1%}")

    print("=" * 90)

    return figures
