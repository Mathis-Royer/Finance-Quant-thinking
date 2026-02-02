"""
Shared constants for factor allocation strategy.

Provides consistent abbreviations, mappings, and display constants
used across notebooks, dashboard, and visualization modules.
"""

from typing import Dict, List

# Model type abbreviations (for labels)
# Keys support both internal names and display names
MODEL_TYPE_ABBREV: Dict[str, str] = {
    "final": "F",
    "fair_ensemble": "FE",
    "wf_ensemble": "WF",
    # Also support display names (used in dashboard)
    "Final": "F",
    "Fair Ensemble": "FE",
    "WF Ensemble": "WF",
}

# Strategy abbreviations
STRATEGY_ABBREV: Dict[str, str] = {
    "Sup": "Sup",
    "E2E": "E2E",
    "E2E-P3": "P3",  # Phase 3 only (ablation test for curriculum learning)
}

# Allocation abbreviations
ALLOCATION_ABBREV: Dict[str, str] = {
    "Multi": "M",
    "Binary": "B",
}

# Config suffix for labels (empty string for baseline)
CONFIG_SUFFIX: Dict[str, str] = {
    "baseline": "",
    "fs": "-FS",
    "hpt": "-HPT",
    "fs+hpt": "-FS+HPT",
}

# Model type display names (internal key -> display name)
MODEL_TYPE_DISPLAY: Dict[str, str] = {
    "final": "Final",
    "fair_ensemble": "Fair Ensemble",
    "wf_ensemble": "WF Ensemble",
}

# Ordered model types for consistent iteration
MODEL_TYPE_ORDER: List[str] = ["final", "fair_ensemble", "wf_ensemble"]

# Display order for model types (used in dashboard)
MODEL_TYPE_DISPLAY_ORDER: List[str] = ["Final", "Fair Ensemble", "WF Ensemble"]


def format_model_label(
    strategy: str,
    allocation: str,
    horizon: int,
    model_type: str,
    config: str = "baseline",
    include_config: bool = True,
) -> str:
    """
    Create a standardized short label for a model.

    Format: {strategy}-{alloc}-{horizon}M[-{config}]-{type}
    Example: "Sup-M-12M-FS-F" or "E2E-B-1M-WF"

    :param strategy (str): Strategy name
    :param allocation (str): Allocation type
    :param horizon (int): Horizon in months
    :param model_type (str): Model type (final, fair_ensemble, wf_ensemble or display names)
    :param config (str): Config name
    :param include_config (bool): Include config in label (skip suffix if baseline)

    :return label (str): Formatted label
    """
    s_abbrev = STRATEGY_ABBREV.get(strategy, strategy[:3])
    a_abbrev = ALLOCATION_ABBREV.get(allocation, allocation[0])
    t_abbrev = MODEL_TYPE_ABBREV.get(model_type, model_type[:2])

    config_part = ""
    if include_config and config != "baseline":
        config_part = CONFIG_SUFFIX.get(config, f"-{config.upper()}")

    return f"{s_abbrev}-{a_abbrev}-{horizon}M{config_part}-{t_abbrev}"
