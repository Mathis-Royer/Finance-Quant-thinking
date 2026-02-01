"""
Visualization module for factor allocation strategy.

Provides plotting functions for:
- Walk-forward validation results
- Holdout evaluation results
- Custom colormaps for financial metrics
"""

from visualization.colormaps import (
    create_sharpe_colormap,
    create_return_colormap,
    COMBINATION_COLORS,
    HORIZON_COLORS,
    HORIZON_LINESTYLES,
    HORIZON_ALPHAS,
    CONFIG_COLORS,
    CONFIG_ORDER,
)

from visualization.walk_forward_plots import (
    plot_sharpe_heatmaps,
    plot_return_heatmaps,
    plot_cumulative_returns_grid,
    plot_total_returns_bar,
    plot_max_drawdown_bar,
    print_year_summary_table,
    plot_all_walk_forward,
)

from visualization.holdout_plots import (
    plot_final_vs_ensemble_bars,
    plot_final_vs_ensemble_scatter,
    plot_sharpe_heatmaps_by_model_type,
    plot_winner_distribution,
    plot_holdout_cumulative_returns_grid,
    plot_top_models_cumulative,
    plot_top_models_with_benchmarks,
    print_holdout_summary_table,
    plot_all_holdout,
    plot_config_comparison,
)

__all__ = [
    # Colormaps
    "create_sharpe_colormap",
    "create_return_colormap",
    "COMBINATION_COLORS",
    "HORIZON_COLORS",
    "HORIZON_LINESTYLES",
    "HORIZON_ALPHAS",
    "CONFIG_COLORS",
    "CONFIG_ORDER",
    # Walk-forward plots
    "plot_sharpe_heatmaps",
    "plot_return_heatmaps",
    "plot_cumulative_returns_grid",
    "plot_total_returns_bar",
    "plot_max_drawdown_bar",
    "print_year_summary_table",
    "plot_all_walk_forward",
    # Holdout plots
    "plot_final_vs_ensemble_bars",
    "plot_final_vs_ensemble_scatter",
    "plot_sharpe_heatmaps_by_model_type",
    "plot_winner_distribution",
    "plot_holdout_cumulative_returns_grid",
    "plot_top_models_cumulative",
    "plot_top_models_with_benchmarks",
    "print_holdout_summary_table",
    "plot_all_holdout",
    "plot_config_comparison",
]
