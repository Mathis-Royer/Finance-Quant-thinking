"""
Walk-forward validation visualization functions.

Provides plots for analyzing walk-forward results:
- Sharpe ratio heatmaps by year and horizon
- Return heatmaps by year and horizon
- Cumulative return curves
- Total return and drawdown bar charts
"""

from typing import Dict, List, Tuple, Any, Optional, Union
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
    CONFIG_COLORS,
    CONFIG_ORDER,
)
from utils.metrics import compute_total_return, compute_max_drawdown
from utils.keys import unpack_key as _unpack_key


# Type alias for walk-forward results (supports both 3-tuple and 4-tuple keys)
WFResultKey = Union[Tuple[str, str, int], Tuple[str, str, int, str]]
WFResults = Dict[WFResultKey, List[Any]]


def _find_key(
    results: WFResults,
    strategy: str,
    allocation: str,
    horizon: int,
    config: str = "baseline",
) -> Optional[WFResultKey]:
    """
    Find matching key in results dict, handling both 3-tuple and 4-tuple keys.

    :param results (WFResults): Results dictionary
    :param strategy (str): Strategy name
    :param allocation (str): Allocation name
    :param horizon (int): Horizon value
    :param config (str): Config name (default: "baseline")

    :return key (Optional[WFResultKey]): Matching key or None
    """
    # Try 4-tuple first
    key_4 = (strategy, allocation, horizon, config)
    if key_4 in results:
        return key_4

    # Try 3-tuple for legacy/baseline
    if config == "baseline":
        key_3 = (strategy, allocation, horizon)
        if key_3 in results:
            return key_3

    return None


def _filter_results_by_config(
    results: WFResults,
    config_filter: Optional[str] = None,
) -> WFResults:
    """
    Filter results to only include specified config.

    :param results (WFResults): Full results dictionary
    :param config_filter (Optional[str]): Config to filter (None = all)

    :return filtered (WFResults): Filtered results
    """
    if config_filter is None:
        return results

    filtered = {}
    for key, value in results.items():
        _, _, _, config = _unpack_key(key)
        if config == config_filter:
            filtered[key] = value
    return filtered


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
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot heatmaps showing Sharpe ratio by year and horizon.

    When config_filter is specified: 2×2 grid (one per combination).
    When config_filter is None with multiple configs: n_combos × n_configs grid.
    Rows = horizons, Columns = years within each heatmap.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_wf_results, config_filter)

    all_years = sorted(set(
        r.test_year for results in filtered_results.values()
        for r in results if r.test_year
    ))
    horizons = sorted(set(_unpack_key(key)[2] for key in filtered_results.keys()))
    combinations = _get_combinations()

    # Check if multiple configs
    all_configs = sorted(set(
        _unpack_key(k)[3] for k in filtered_results.keys()
    ), key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999)
    use_grid = config_filter is None and len(all_configs) > 1

    if use_grid:
        # Multi-config grid: rows = combinations, cols = configs
        n_rows, n_cols = len(combinations), len(all_configs)
        adjusted_figsize = (n_cols * 4, n_rows * 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=adjusted_figsize)
        fig.suptitle(
            "Walk-Forward OOS Sharpe Ratio by Config",
            fontsize=14,
            fontweight='bold'
        )

        # Collect all data to get global min/max for colormap
        all_sharpes = []
        for key, results in filtered_results.items():
            for r in results:
                if hasattr(r, 'sharpe'):
                    all_sharpes.append(r.sharpe)
        global_min = min(all_sharpes) if all_sharpes else 0
        global_max = max(all_sharpes) if all_sharpes else 1
        cmap, norm = create_sharpe_colormap(global_min, global_max)

        for row_idx, (strategy, allocation, combo_title) in enumerate(combinations):
            for col_idx, config in enumerate(all_configs):
                ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]

                # Title: config on top row, combo on left column
                if row_idx == 0:
                    ax.set_title(config.upper(), fontsize=11, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{combo_title}\nHorizon", fontsize=10)
                else:
                    ax.set_ylabel("Horizon")

                heatmap_data = []
                for horizon in horizons:
                    key = _find_key(filtered_results, strategy, allocation, horizon, config)
                    row = []
                    for year in all_years:
                        year_results = [
                            r for r in filtered_results.get(key, [])
                            if r.test_year == year
                        ] if key else []
                        if year_results:
                            row.append(year_results[0].sharpe)
                        else:
                            row.append(0.0)
                    heatmap_data.append(row)

                heatmap_arr = np.array(heatmap_data)
                im = ax.imshow(heatmap_arr, cmap=cmap, norm=norm, aspect='auto')
                ax.set_xticks(range(len(all_years)))
                ax.set_xticklabels(all_years, fontsize=8)
                ax.set_yticks(range(len(horizons)))
                ax.set_yticklabels([f"{h}M" for h in horizons], fontsize=8)
                ax.set_xlabel("Year", fontsize=8)

                for i in range(len(horizons)):
                    for j in range(len(all_years)):
                        val = heatmap_arr[i, j]
                        text_color = 'white' if val < 0 or val > 2.5 else 'black'
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                                color=text_color, fontsize=7, fontweight='bold')

        # Single colorbar for all
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Sharpe Ratio', shrink=0.8)
    else:
        # Single config: 2×2 grid
        config_suffix = f" [{config_filter}]" if config_filter else ""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Walk-Forward OOS Sharpe Ratio by Year x Horizon{config_suffix}",
            fontsize=14,
            fontweight='bold'
        )

        for idx, (strategy, allocation, title) in enumerate(combinations):
            ax = axes[idx // 2, idx % 2]
            ax.set_title(title, fontsize=12, fontweight='bold')

            heatmap_data = []
            for horizon in horizons:
                key = _find_key(filtered_results, strategy, allocation, horizon, config_filter or "baseline")
                row = []
                for year in all_years:
                    year_results = [
                        r for r in filtered_results.get(key, [])
                        if r.test_year == year
                    ] if key else []
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
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot heatmaps showing returns (%) by year and horizon.

    When config_filter is specified: 2×2 grid (one per combination).
    When config_filter is None with multiple configs: n_combos × n_configs grid.
    Rows = horizons, Columns = years within each heatmap.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_wf_results, config_filter)

    all_years = sorted(set(
        r.test_year for results in filtered_results.values()
        for r in results if r.test_year
    ))
    horizons = sorted(set(_unpack_key(key)[2] for key in filtered_results.keys()))
    combinations = _get_combinations()

    # Check if multiple configs
    all_configs = sorted(set(
        _unpack_key(k)[3] for k in filtered_results.keys()
    ), key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999)
    use_grid = config_filter is None and len(all_configs) > 1

    if use_grid:
        # Multi-config grid: rows = combinations, cols = configs
        n_rows, n_cols = len(combinations), len(all_configs)
        adjusted_figsize = (n_cols * 4, n_rows * 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=adjusted_figsize)
        fig.suptitle(
            "Walk-Forward OOS Returns (%) by Config",
            fontsize=14,
            fontweight='bold'
        )

        # Collect all data to get global min/max for colormap
        all_returns = []
        for key, results in filtered_results.items():
            for r in results:
                if hasattr(r, 'monthly_returns') and r.monthly_returns:
                    all_returns.append(compute_total_return(r.monthly_returns) * 100)
        global_min = min(all_returns) if all_returns else -10
        global_max = max(all_returns) if all_returns else 10
        cmap, norm = create_return_colormap(global_min, global_max)

        for row_idx, (strategy, allocation, combo_title) in enumerate(combinations):
            for col_idx, config in enumerate(all_configs):
                ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]

                # Title: config on top row, combo on left column
                if row_idx == 0:
                    ax.set_title(config.upper(), fontsize=11, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{combo_title}\nHorizon", fontsize=10)
                else:
                    ax.set_ylabel("Horizon")

                heatmap_data = []
                for horizon in horizons:
                    key = _find_key(filtered_results, strategy, allocation, horizon, config)
                    row = []
                    for year in all_years:
                        year_results = [
                            r for r in filtered_results.get(key, [])
                            if r.test_year == year
                        ] if key else []
                        if year_results and year_results[0].monthly_returns:
                            total_ret = compute_total_return(
                                year_results[0].monthly_returns
                            ) * 100
                            row.append(total_ret)
                        else:
                            row.append(0.0)
                    heatmap_data.append(row)

                heatmap_arr = np.array(heatmap_data)
                im = ax.imshow(heatmap_arr, cmap=cmap, norm=norm, aspect='auto')
                ax.set_xticks(range(len(all_years)))
                ax.set_xticklabels(all_years, fontsize=8)
                ax.set_yticks(range(len(horizons)))
                ax.set_yticklabels([f"{h}M" for h in horizons], fontsize=8)
                ax.set_xlabel("Year", fontsize=8)

                for i in range(len(horizons)):
                    for j in range(len(all_years)):
                        val = heatmap_arr[i, j]
                        text_color = 'white' if abs(val) > 15 else 'black'
                        ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                                color=text_color, fontsize=7, fontweight='bold')

        # Single colorbar for all
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Return (%)', shrink=0.8)
    else:
        # Single config: 2×2 grid
        config_suffix = f" [{config_filter}]" if config_filter else ""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Walk-Forward OOS Returns (%) by Year x Horizon{config_suffix}",
            fontsize=14,
            fontweight='bold'
        )

        for idx, (strategy, allocation, title) in enumerate(combinations):
            ax = axes[idx // 2, idx % 2]
            ax.set_title(title, fontsize=12, fontweight='bold')

            heatmap_data = []
            for horizon in horizons:
                key = _find_key(filtered_results, strategy, allocation, horizon, config_filter or "baseline")
                row = []
                for year in all_years:
                    year_results = [
                        r for r in filtered_results.get(key, [])
                        if r.test_year == year
                    ] if key else []
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
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot 2x2 grid of cumulative return curves.

    One subplot per combination. When single config: one line per horizon.
    When multiple configs: uses config colors with horizon linestyles.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_wf_results, config_filter)

    all_years = sorted(set(
        r.test_year for results in filtered_results.values()
        for r in results if r.test_year
    ))
    horizons = sorted(set(_unpack_key(key)[2] for key in filtered_results.keys()))
    combinations = _get_combinations()

    # Check if multiple configs
    all_configs = sorted(set(
        _unpack_key(k)[3] for k in filtered_results.keys()
    ), key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999)
    use_multi_config = config_filter is None and len(all_configs) > 1

    config_suffix = f" [{config_filter}]" if config_filter else ""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Walk-Forward: Concatenated OOS Cumulative Returns{config_suffix}",
        fontsize=14,
        fontweight='bold'
    )

    for idx, (strategy, allocation, title) in enumerate(combinations):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(title, fontsize=12, fontweight='bold')

        if use_multi_config:
            # Multi-config mode: color by config, linestyle by horizon
            for config in all_configs:
                for horizon in horizons:
                    key = _find_key(filtered_results, strategy, allocation, horizon, config)
                    if key is None or key not in filtered_results:
                        continue

                    all_returns = []
                    for result in sorted(filtered_results[key], key=lambda r: r.test_year):
                        if result.monthly_returns:
                            all_returns.extend(result.monthly_returns)

                    if all_returns:
                        cum_ret = np.cumprod(1 + np.array(all_returns))
                        color = CONFIG_COLORS.get(config, '#888888')
                        ls = HORIZON_LINESTYLES[horizon]
                        ax.plot(
                            range(len(cum_ret)), cum_ret,
                            color=color, linestyle=ls,
                            linewidth=1.5, alpha=0.8, label=f"{config}-{horizon}M"
                        )
        else:
            # Single config mode: color by horizon
            config_to_use = config_filter or "baseline"
            for horizon in horizons:
                key = _find_key(filtered_results, strategy, allocation, horizon, config_to_use)
                if key is None or key not in filtered_results:
                    continue

                all_returns = []
                for result in sorted(filtered_results[key], key=lambda r: r.test_year):
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
        ax.legend(loc='upper left', fontsize=7 if use_multi_config else 10, ncol=2 if use_multi_config else 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_total_returns_bar(
    all_wf_results: WFResults,
    figsize: Tuple[int, int] = (14, 6),
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot bar chart of total returns by combination.

    When config_filter is None and multiple configs exist, shows grouped bars
    with one color per config.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_wf_results, config_filter)

    # Check if multiple configs
    all_configs = sorted(set(
        _unpack_key(k)[3] for k in filtered_results.keys()
    ), key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999)
    use_grouped = config_filter is None and len(all_configs) > 1

    config_suffix = f" [{config_filter}]" if config_filter else ""
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Walk-Forward: Total OOS Return by Combination{config_suffix}",
        fontsize=14,
        fontweight='bold'
    )

    if use_grouped:
        # Grouped bars by config
        base_combos = sorted(set(
            (s, a, h) for s, a, h, _ in [_unpack_key(k) for k in filtered_results.keys()]
        ))
        n_configs = len(all_configs)
        width = 0.8 / n_configs
        x = np.arange(len(base_combos))

        for i, config in enumerate(all_configs):
            returns = []
            for (strategy, allocation, horizon) in base_combos:
                key = _find_key(filtered_results, strategy, allocation, horizon, config)
                if key and key in filtered_results:
                    all_returns_data = []
                    for result in sorted(filtered_results[key], key=lambda r: r.test_year):
                        if result.monthly_returns:
                            all_returns_data.extend(result.monthly_returns)
                    if all_returns_data:
                        returns.append(compute_total_return(all_returns_data) * 100)
                    else:
                        returns.append(0)
                else:
                    returns.append(0)

            offset = (i - (n_configs - 1) / 2) * width
            bars = ax.bar(
                x + offset, returns, width,
                label=config, color=CONFIG_COLORS.get(config, '#888888'),
                alpha=0.85, edgecolor='black', linewidth=0.3
            )

        labels = [f"{s}-{a[:1]}-{h}M" for s, a, h in base_combos]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.legend(title="Config", loc='upper right', fontsize=9)
    else:
        # Single config (legacy behavior)
        labels = []
        returns = []
        colors = []

        for key in sorted(filtered_results.keys()):
            strategy, allocation, horizon, config = _unpack_key(key)

            all_returns_data = []
            for result in sorted(filtered_results[key], key=lambda r: r.test_year):
                if result.monthly_returns:
                    all_returns_data.extend(result.monthly_returns)

            if all_returns_data:
                total_ret = compute_total_return(all_returns_data) * 100
                label = f"{strategy}-{allocation[:1]}-{horizon}M"
                if config != "baseline":
                    label += f" ({config})"
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

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, label=f"{s}-{a}")
            for (s, a), c in COMBINATION_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel("Total Return (%)")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_max_drawdown_bar(
    all_wf_results: WFResults,
    figsize: Tuple[int, int] = (14, 6),
    config_filter: Optional[str] = "baseline",
) -> Figure:
    """
    Plot bar chart of maximum drawdown by combination.

    When config_filter is None and multiple configs exist, shows grouped bars
    with one color per config.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param figsize (Tuple[int, int]): Figure size
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return fig (Figure): Matplotlib figure
    """
    filtered_results = _filter_results_by_config(all_wf_results, config_filter)

    # Check if multiple configs
    all_configs = sorted(set(
        _unpack_key(k)[3] for k in filtered_results.keys()
    ), key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999)
    use_grouped = config_filter is None and len(all_configs) > 1

    config_suffix = f" [{config_filter}]" if config_filter else ""
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Walk-Forward: Maximum Drawdown by Combination{config_suffix}",
        fontsize=14,
        fontweight='bold'
    )

    if use_grouped:
        # Grouped bars by config
        base_combos = sorted(set(
            (s, a, h) for s, a, h, _ in [_unpack_key(k) for k in filtered_results.keys()]
        ))
        n_configs = len(all_configs)
        width = 0.8 / n_configs
        x = np.arange(len(base_combos))

        for i, config in enumerate(all_configs):
            drawdowns = []
            for (strategy, allocation, horizon) in base_combos:
                key = _find_key(filtered_results, strategy, allocation, horizon, config)
                if key and key in filtered_results:
                    all_returns_data = []
                    for result in sorted(filtered_results[key], key=lambda r: r.test_year):
                        if result.monthly_returns:
                            all_returns_data.extend(result.monthly_returns)
                    if all_returns_data:
                        drawdowns.append(compute_max_drawdown(all_returns_data) * 100)
                    else:
                        drawdowns.append(0)
                else:
                    drawdowns.append(0)

            offset = (i - (n_configs - 1) / 2) * width
            bars = ax.bar(
                x + offset, drawdowns, width,
                label=config, color=CONFIG_COLORS.get(config, '#888888'),
                alpha=0.85, edgecolor='black', linewidth=0.3
            )

        labels = [f"{s}-{a[:1]}-{h}M" for s, a, h in base_combos]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.legend(title="Config", loc='lower right', fontsize=9)
    else:
        # Single config (legacy behavior)
        labels = []
        drawdowns = []
        colors = []

        for key in sorted(filtered_results.keys()):
            strategy, allocation, horizon, config = _unpack_key(key)

            all_returns_data = []
            for result in sorted(filtered_results[key], key=lambda r: r.test_year):
                if result.monthly_returns:
                    all_returns_data.extend(result.monthly_returns)

            if all_returns_data:
                max_dd = compute_max_drawdown(all_returns_data) * 100
                label = f"{strategy}-{allocation[:1]}-{horizon}M"
                if config != "baseline":
                    label += f" ({config})"
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

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, label=f"{s}-{a}")
            for (s, a), c in COMBINATION_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel("Maximum Drawdown (%)")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def print_year_summary_table(
    all_wf_results: WFResults,
    config_filter: Optional[str] = "baseline",
) -> None:
    """
    Print year-by-year summary table to console.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)
    """
    filtered_results = _filter_results_by_config(all_wf_results, config_filter)

    all_years = sorted(set(
        r.test_year for results in filtered_results.values()
        for r in results if r.test_year
    ))

    # Check if we have multiple configs
    all_configs = set()
    for key in filtered_results.keys():
        _, _, _, config = _unpack_key(key)
        all_configs.add(config)
    show_config_col = config_filter is None and len(all_configs) > 1

    config_suffix = f" [{config_filter}]" if config_filter else ""
    line_width = 120 if show_config_col else 100
    print("\n" + "=" * line_width)
    print(f"WALK-FORWARD SUMMARY BY YEAR{config_suffix}")
    print("=" * line_width)

    if show_config_col:
        header = f"{'Combination':<18}{'Config':<10}"
    else:
        header = f"{'Combination':<25}"
    for year in all_years:
        header += f"{year:>10}"
    header += f"{'TOTAL':>12}{'MAX DD':>12}"
    print(f"\n{header}")
    header_len = (18 + 10 if show_config_col else 25) + 10*len(all_years) + 24
    print("-" * header_len)

    all_totals = []
    for key in sorted(filtered_results.keys()):
        strategy, allocation, horizon, config = _unpack_key(key)
        label = f"{strategy}-{allocation[:1]}-{horizon}M"

        if show_config_col:
            line = f"{label:<18}{config:<10}"
        else:
            if config != "baseline":
                label += f" ({config})"
            line = f"{label:<25}"

        all_rets = []
        for year in all_years:
            year_results = [r for r in filtered_results[key] if r.test_year == year]
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
            display_label = f"{label} [{config}]" if show_config_col else label
            all_totals.append((display_label, total_ret, max_dd, config))
        else:
            line += f"{'N/A':>12}{'N/A':>12}"

        print(line)

    print("=" * line_width)
    print(f"\nWalk-forward windows: {len(all_years)} years")
    print(f"Total models trained: {len(filtered_results) * len(all_years)}")

    if all_totals:
        all_totals.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 3 by Total Return:")
        for i, (name, ret, dd, _) in enumerate(all_totals[:3]):
            print(f"  {i+1}. {name}: {ret:+.1f}% (Max DD: {dd:.1f}%)")

        all_totals.sort(key=lambda x: x[2], reverse=True)
        print(f"\nTop 3 by Lowest Drawdown:")
        for i, (name, ret, dd, _) in enumerate(all_totals[:3]):
            print(f"  {i+1}. {name}: {dd:.1f}% (Total Ret: {ret:+.1f}%)")

    print("=" * line_width)


def plot_all_walk_forward(
    all_wf_results: WFResults,
    show: bool = True,
    config_filter: Optional[str] = "baseline",
) -> List[Figure]:
    """
    Create all walk-forward visualization figures.

    Convenience function to generate all standard plots.

    :param all_wf_results (WFResults): Walk-forward results dict
    :param show (bool): Whether to call plt.show() for each figure
    :param config_filter (Optional[str]): Config to show (default: "baseline", None = all)

    :return figures (List[Figure]): List of generated figures
    """
    figures = []

    fig1 = plot_sharpe_heatmaps(all_wf_results, config_filter=config_filter)
    figures.append(fig1)
    if show:
        plt.show()

    fig2 = plot_return_heatmaps(all_wf_results, config_filter=config_filter)
    figures.append(fig2)
    if show:
        plt.show()

    fig3 = plot_cumulative_returns_grid(all_wf_results, config_filter=config_filter)
    figures.append(fig3)
    if show:
        plt.show()

    fig4 = plot_total_returns_bar(all_wf_results, config_filter=config_filter)
    figures.append(fig4)
    if show:
        plt.show()

    fig5 = plot_max_drawdown_bar(all_wf_results, config_filter=config_filter)
    figures.append(fig5)
    if show:
        plt.show()

    print_year_summary_table(all_wf_results, config_filter=config_filter)

    return figures
