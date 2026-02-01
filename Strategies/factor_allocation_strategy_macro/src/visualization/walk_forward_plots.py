"""
Walk-forward validation visualization functions.

Provides plots for analyzing walk-forward results:
- Sharpe ratio heatmaps by year and horizon
- Return heatmaps by year and horizon
- Cumulative return curves
- Total return and drawdown bar charts
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from visualization.colormaps import (
    create_sharpe_colormap,
    create_return_colormap,
    COMBINATION_COLORS,
    HORIZON_COLORS,
    HORIZON_LINESTYLES,
)
from utils.metrics import compute_total_return, compute_max_drawdown


# Type alias for walk-forward results
WFResults = Dict[Tuple[str, str, int], List[Any]]


def _get_combinations() -> List[Tuple[str, str, str]]:
    """Return list of (strategy, allocation, title) tuples."""
    return [
        ("E2E", "Binary", "E2E-Binary"),
        ("E2E", "Multi", "E2E-Multi"),
        ("Sup", "Binary", "Sup-Binary"),
        ("Sup", "Multi", "Sup-Multi"),
    ]


def plot_sharpe_heatmaps(
    all_wf_results: WFResults,
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Plot 4 heatmaps showing Sharpe ratio by year and horizon.

    One heatmap per combination (E2E-Binary, E2E-Multi, Sup-Binary, Sup-Multi).
    Rows = horizons, Columns = years.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    all_years = sorted(set(
        r.test_year for results in all_wf_results.values()
        for r in results if r.test_year
    ))
    horizons = sorted(set(key[2] for key in all_wf_results.keys()))
    combinations = _get_combinations()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Walk-Forward OOS Sharpe Ratio by Year x Horizon",
        fontsize=14,
        fontweight='bold'
    )

    for idx, (strategy, allocation, title) in enumerate(combinations):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(title, fontsize=12, fontweight='bold')

        heatmap_data = []
        for horizon in horizons:
            key = (strategy, allocation, horizon)
            row = []
            for year in all_years:
                year_results = [
                    r for r in all_wf_results.get(key, [])
                    if r.test_year == year
                ]
                if year_results:
                    row.append(year_results[0].sharpe)
                else:
                    row.append(0.0)
            heatmap_data.append(row)

        heatmap_arr = np.array(heatmap_data)
        cmap, norm = create_sharpe_colormap(heatmap_arr.min(), heatmap_arr.max())

        im = ax.imshow(heatmap_arr, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(all_years)))
        ax.set_xticklabels(all_years)
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels([f"{h}M" for h in horizons])
        ax.set_xlabel("Test Year")
        ax.set_ylabel("Horizon")

        for i in range(len(horizons)):
            for j in range(len(all_years)):
                val = heatmap_arr[i, j]
                text_color = 'white' if val < 0 or val > 2.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color=text_color, fontsize=10, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Sharpe Ratio')

    plt.tight_layout()
    return fig


def plot_return_heatmaps(
    all_wf_results: WFResults,
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Plot 4 heatmaps showing returns (%) by year and horizon.

    One heatmap per combination (E2E-Binary, E2E-Multi, Sup-Binary, Sup-Multi).
    Rows = horizons, Columns = years.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    all_years = sorted(set(
        r.test_year for results in all_wf_results.values()
        for r in results if r.test_year
    ))
    horizons = sorted(set(key[2] for key in all_wf_results.keys()))
    combinations = _get_combinations()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Walk-Forward OOS Returns (%) by Year x Horizon",
        fontsize=14,
        fontweight='bold'
    )

    for idx, (strategy, allocation, title) in enumerate(combinations):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(title, fontsize=12, fontweight='bold')

        heatmap_data = []
        for horizon in horizons:
            key = (strategy, allocation, horizon)
            row = []
            for year in all_years:
                year_results = [
                    r for r in all_wf_results.get(key, [])
                    if r.test_year == year
                ]
                if year_results and year_results[0].monthly_returns:
                    total_ret = compute_total_return(
                        year_results[0].monthly_returns
                    ) * 100
                    row.append(total_ret)
                else:
                    row.append(0.0)
            heatmap_data.append(row)

        heatmap_arr = np.array(heatmap_data)
        cmap, norm = create_return_colormap(heatmap_arr.min(), heatmap_arr.max())

        im = ax.imshow(heatmap_arr, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(all_years)))
        ax.set_xticklabels(all_years)
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels([f"{h}M" for h in horizons])
        ax.set_xlabel("Test Year")
        ax.set_ylabel("Horizon")

        for i in range(len(horizons)):
            for j in range(len(all_years)):
                val = heatmap_arr[i, j]
                text_color = 'white' if abs(val) > 15 else 'black'
                ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                        color=text_color, fontsize=10, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Return (%)')

    plt.tight_layout()
    return fig


def plot_cumulative_returns_grid(
    all_wf_results: WFResults,
    figsize: Tuple[int, int] = (16, 12),
) -> Figure:
    """
    Plot 2x2 grid of cumulative return curves.

    One subplot per combination, with one line per horizon.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    all_years = sorted(set(
        r.test_year for results in all_wf_results.values()
        for r in results if r.test_year
    ))
    horizons = sorted(set(key[2] for key in all_wf_results.keys()))
    combinations = _get_combinations()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Walk-Forward: Concatenated OOS Cumulative Returns",
        fontsize=14,
        fontweight='bold'
    )

    for idx, (strategy, allocation, title) in enumerate(combinations):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(title, fontsize=12, fontweight='bold')

        for horizon in horizons:
            key = (strategy, allocation, horizon)
            if key not in all_wf_results:
                continue

            all_returns = []
            for result in sorted(all_wf_results[key], key=lambda r: r.test_year):
                if result.monthly_returns:
                    all_returns.extend(result.monthly_returns)

            if all_returns:
                cum_ret = np.cumprod(1 + np.array(all_returns))
                color = HORIZON_COLORS[horizon]
                ls = HORIZON_LINESTYLES[horizon]
                ax.plot(
                    range(len(cum_ret)), cum_ret,
                    color=color, linestyle=ls,
                    linewidth=2.5, alpha=0.9, label=f"{horizon}M"
                )

        # Year boundaries
        months_per_year = 12
        for i in range(len(all_years) - 1):
            x_pos = (i + 1) * months_per_year
            ax.axvline(x=x_pos, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.axhline(y=1, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Month")
        ax.set_ylabel("Cumulative Return")
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_total_returns_bar(
    all_wf_results: WFResults,
    figsize: Tuple[int, int] = (14, 6),
) -> Figure:
    """
    Plot bar chart of total returns by combination.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        "Walk-Forward: Total OOS Return by Combination",
        fontsize=14,
        fontweight='bold'
    )

    labels = []
    returns = []
    colors = []

    for key in sorted(all_wf_results.keys()):
        strategy, allocation, horizon = key

        all_returns_data = []
        for result in sorted(all_wf_results[key], key=lambda r: r.test_year):
            if result.monthly_returns:
                all_returns_data.extend(result.monthly_returns)

        if all_returns_data:
            total_ret = compute_total_return(all_returns_data) * 100
            label = f"{strategy}-{allocation[:1]}-{horizon}M"
            labels.append(label)
            returns.append(total_ret)
            colors.append(COMBINATION_COLORS[(strategy, allocation)])

    x_pos = np.arange(len(labels))
    bars = ax.bar(
        x_pos, returns, color=colors, alpha=0.8,
        edgecolor='black', linewidth=0.5
    )

    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 1 if height >= 0 else -1
        ax.text(
            bar.get_x() + bar.get_width()/2, height + offset,
            f"{ret:.1f}%", ha='center', va=va, fontsize=9, fontweight='bold'
        )

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Total Return (%)")
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, label=f"{s}-{a}")
        for (s, a), c in COMBINATION_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def plot_max_drawdown_bar(
    all_wf_results: WFResults,
    figsize: Tuple[int, int] = (14, 6),
) -> Figure:
    """
    Plot bar chart of maximum drawdown by combination.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size

    :return fig (Figure): Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        "Walk-Forward: Maximum Drawdown by Combination",
        fontsize=14,
        fontweight='bold'
    )

    labels = []
    drawdowns = []
    colors = []

    for key in sorted(all_wf_results.keys()):
        strategy, allocation, horizon = key

        all_returns_data = []
        for result in sorted(all_wf_results[key], key=lambda r: r.test_year):
            if result.monthly_returns:
                all_returns_data.extend(result.monthly_returns)

        if all_returns_data:
            max_dd = compute_max_drawdown(all_returns_data) * 100
            label = f"{strategy}-{allocation[:1]}-{horizon}M"
            labels.append(label)
            drawdowns.append(max_dd)
            colors.append(COMBINATION_COLORS[(strategy, allocation)])

    x_pos = np.arange(len(labels))
    bars = ax.bar(
        x_pos, drawdowns, color=colors, alpha=0.8,
        edgecolor='black', linewidth=0.5
    )

    for bar, dd in zip(bars, drawdowns):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height - 0.5,
            f"{dd:.1f}%", ha='center', va='top',
            fontsize=9, fontweight='bold', color='white'
        )

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Maximum Drawdown (%)")
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, label=f"{s}-{a}")
        for (s, a), c in COMBINATION_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    return fig


def print_year_summary_table(
    all_wf_results: WFResults,
) -> None:
    """
    Print year-by-year summary table to console.

    :param all_wf_results (WFResults): Walk-forward results dict
    """
    all_years = sorted(set(
        r.test_year for results in all_wf_results.values()
        for r in results if r.test_year
    ))

    print("\n" + "=" * 100)
    print("WALK-FORWARD SUMMARY BY YEAR")
    print("=" * 100)

    header = f"{'Combination':<18}"
    for year in all_years:
        header += f"{year:>10}"
    header += f"{'TOTAL':>12}{'MAX DD':>12}"
    print(f"\n{header}")
    print("-" * (18 + 10*len(all_years) + 24))

    all_totals = []
    for key in sorted(all_wf_results.keys()):
        strategy, allocation, horizon = key
        label = f"{strategy}-{allocation[:1]}-{horizon}M"
        line = f"{label:<18}"

        all_rets = []
        for year in all_years:
            year_results = [r for r in all_wf_results[key] if r.test_year == year]
            if year_results and year_results[0].monthly_returns:
                ret = compute_total_return(year_results[0].monthly_returns) * 100
                line += f"{ret:>+10.1f}%"
                all_rets.extend(year_results[0].monthly_returns)
            else:
                line += f"{'N/A':>10}"

        if all_rets:
            total_ret = compute_total_return(all_rets) * 100
            max_dd = compute_max_drawdown(all_rets) * 100
            line += f"{total_ret:>+12.1f}%{max_dd:>+12.1f}%"
            all_totals.append((label, total_ret, max_dd))
        else:
            line += f"{'N/A':>12}{'N/A':>12}"

        print(line)

    print("=" * 100)
    print(f"\nWalk-forward windows: {len(all_years)} years")
    print(f"Total models trained: {len(all_wf_results) * len(all_years)}")

    if all_totals:
        all_totals.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 3 by Total Return:")
        for i, (name, ret, dd) in enumerate(all_totals[:3]):
            print(f"  {i+1}. {name}: {ret:+.1f}% (Max DD: {dd:.1f}%)")

        all_totals.sort(key=lambda x: x[2], reverse=True)
        print(f"\nTop 3 by Lowest Drawdown:")
        for i, (name, ret, dd) in enumerate(all_totals[:3]):
            print(f"  {i+1}. {name}: {dd:.1f}% (Total Ret: {ret:+.1f}%)")

    print("=" * 100)


def plot_all_walk_forward(
    all_wf_results: WFResults,
    show: bool = True,
) -> List[Figure]:
    """
    Create all walk-forward visualization figures.

    Convenience function to generate all standard plots.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param show (bool): Whether to call plt.show() for each figure

    :return figures (List[Figure]): List of generated figures
    """
    figures = []

    fig1 = plot_sharpe_heatmaps(all_wf_results)
    figures.append(fig1)
    if show:
        plt.show()

    fig2 = plot_return_heatmaps(all_wf_results)
    figures.append(fig2)
    if show:
        plt.show()

    fig3 = plot_cumulative_returns_grid(all_wf_results)
    figures.append(fig3)
    if show:
        plt.show()

    fig4 = plot_total_returns_bar(all_wf_results)
    figures.append(fig4)
    if show:
        plt.show()

    fig5 = plot_max_drawdown_bar(all_wf_results)
    figures.append(fig5)
    if show:
        plt.show()

    print_year_summary_table(all_wf_results)

    return figures
