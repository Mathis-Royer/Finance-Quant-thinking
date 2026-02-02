"""
Custom colormaps for strategy visualization.

Provides consistent color schemes for Sharpe ratios, returns, and other metrics.
"""

from typing import Tuple
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


def create_sharpe_colormap(
    data_min: float,
    data_max: float,
) -> Tuple[LinearSegmentedColormap, Normalize]:
    """
    Create a custom Sharpe colormap with three segments.

    Color segments:
    - min to 0: dark red to red (negative Sharpe)
    - 0 to 1: red to green (low positive Sharpe)
    - 1 to max: green to dark green (high Sharpe)

    :param data_min (float): Minimum Sharpe value in data
    :param data_max (float): Maximum Sharpe value in data

    :return cmap (LinearSegmentedColormap): Custom colormap
    :return norm (Normalize): Normalization object
    """
    if data_max <= 0:
        colors = ['#8B0000', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('sharpe_neg', colors)
        norm = Normalize(vmin=data_min, vmax=data_max)
        return cmap, norm

    if data_min >= 1:
        colors = ['#00AA00', '#004400']
        cmap = LinearSegmentedColormap.from_list('sharpe_high', colors)
        norm = Normalize(vmin=data_min, vmax=data_max)
        return cmap, norm

    data_range = data_max - data_min
    pos_0 = (0 - data_min) / data_range if data_min < 0 else 0
    pos_1 = (1 - data_min) / data_range if data_max > 1 else 1

    pos_0 = max(0, min(1, pos_0))
    pos_1 = max(pos_0 + 0.01, min(1, pos_1))

    colors = []
    positions = []

    if data_min < 0:
        colors.extend(['#8B0000', '#FF0000'])
        positions.extend([0.0, pos_0])
    else:
        pos_0 = 0.0

    if data_max <= 1:
        colors.extend(['#FF0000', '#00AA00'])
        positions.extend([pos_0, 1.0])
    else:
        colors.extend(['#FF0000', '#00AA00'])
        positions.extend([pos_0, pos_1])
        colors.extend(['#00AA00', '#004400'])
        positions.extend([pos_1, 1.0])

    unique = []
    unique_colors = []
    for i, (p, c) in enumerate(zip(positions, colors)):
        if i == 0 or p > positions[i-1]:
            unique.append(p)
            unique_colors.append(c)

    cmap = LinearSegmentedColormap.from_list(
        'sharpe_custom',
        list(zip(unique, unique_colors))
    )
    norm = Normalize(vmin=data_min, vmax=data_max)

    return cmap, norm


def create_return_colormap(
    data_min: float,
    data_max: float,
) -> Tuple[LinearSegmentedColormap, TwoSlopeNorm]:
    """
    Create a diverging colormap centered at 0 for returns.

    Colors: Red (negative) -> White (zero) -> Green (positive)

    :param data_min (float): Minimum return value in data
    :param data_max (float): Maximum return value in data

    :return cmap (LinearSegmentedColormap): Custom colormap
    :return norm (TwoSlopeNorm): Two-slope normalization centered at 0
    """
    abs_max = max(abs(data_min), abs(data_max))
    if abs_max == 0:
        abs_max = 1

    cmap = LinearSegmentedColormap.from_list(
        'return_diverging',
        ['#d62728', '#ffcccc', '#ffffff', '#ccffcc', '#2ca02c']
    )
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    return cmap, norm


# Color schemes for different combination types
COMBINATION_COLORS = {
    ('E2E', 'Binary'): '#1f77b4',  # Blue
    ('E2E', 'Multi'): '#ff7f0e',   # Orange
    ('Sup', 'Binary'): '#2ca02c',  # Green
    ('Sup', 'Multi'): '#d62728',   # Red
}

# Color scheme for horizons
HORIZON_COLORS = {
    1: '#1f77b4',   # Blue
    3: '#2ca02c',   # Green
    6: '#ff7f0e',   # Orange
    12: '#d62728', # Red
}

# Line styles for horizons
HORIZON_LINESTYLES = {
    1: '-',
    3: '--',
    6: '-.',
    12: ':',
}

# Alpha values by horizon (for overlapping plots)
HORIZON_ALPHAS = {
    1: 0.4,
    3: 0.6,
    6: 0.8,
    12: 1.0,
}

# Color scheme for configs (FS/HPT analysis)
CONFIG_COLORS = {
    "baseline": "#4285F4",  # Blue (Google)
    "fs": "#34A853",        # Green
    "hpt": "#FBBC04",       # Yellow/Orange
    "fs+hpt": "#EA4335",    # Red
}

# Ordered list of configs for consistent plotting
CONFIG_ORDER = ["baseline", "fs", "hpt", "fs+hpt"]

# Color scheme for factor allocations (warm/cool palette for visual distinction)
FACTOR_COLORS = {
    "cyclical": "#e41a1c",   # Red - cyclical exposure
    "defensive": "#377eb8",  # Blue - defensive exposure
    "value": "#4daf4a",      # Green - value tilt
    "growth": "#984ea3",     # Purple - growth tilt
    "quality": "#ff7f00",    # Orange - quality factor
    "momentum": "#a65628",   # Brown - momentum factor
}

# Ordered list of factors for consistent plotting
FACTOR_ORDER = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]
