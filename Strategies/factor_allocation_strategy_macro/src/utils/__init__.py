"""Utility modules for metrics and validation."""

from .metrics import PerformanceMetrics
from .walk_forward import WalkForwardValidator, WalkForwardWindow
from .keys import unpack_key, make_key, ResultKey
from .constants import (
    MODEL_TYPE_ABBREV,
    STRATEGY_ABBREV,
    ALLOCATION_ABBREV,
    CONFIG_SUFFIX,
    MODEL_TYPE_DISPLAY,
    MODEL_TYPE_ORDER,
    MODEL_TYPE_DISPLAY_ORDER,
    format_model_label,
)
from .hyperparameter_tuning import (
    WalkForwardTuner,
    HyperparameterSpace,
    TuningConfig,
    TuningResult,
    create_default_tuner,
)
from .benchmarks import (
    BenchmarkResult,
    compute_equal_weight_factors_benchmark,
    compute_equal_weight_cyc_def_benchmark,
    compute_risk_parity_benchmark,
    compute_factor_momentum_benchmark,
    compute_best_single_factor_benchmark,
    compute_all_benchmarks,
    get_benchmark_returns_for_period,
    print_benchmark_summary,
)
from .analysis import (
    WinCountResult,
    DeltaResult,
    BestModelResult,
    flatten_holdout_results,
    compare_model_types,
    compute_delta_stats,
    find_best_models,
    print_best_models_table,
    print_model_comparison_summary,
)

__all__ = [
    # Metrics
    "PerformanceMetrics",
    # Walk-forward
    "WalkForwardValidator",
    "WalkForwardWindow",
    # Hyperparameter tuning
    "WalkForwardTuner",
    "HyperparameterSpace",
    "TuningConfig",
    "TuningResult",
    "create_default_tuner",
    # Benchmarks
    "BenchmarkResult",
    "compute_equal_weight_factors_benchmark",
    "compute_equal_weight_cyc_def_benchmark",
    "compute_risk_parity_benchmark",
    "compute_factor_momentum_benchmark",
    "compute_best_single_factor_benchmark",
    "compute_all_benchmarks",
    "get_benchmark_returns_for_period",
    "print_benchmark_summary",
    # Keys
    "unpack_key",
    "make_key",
    "ResultKey",
    # Constants
    "MODEL_TYPE_ABBREV",
    "STRATEGY_ABBREV",
    "ALLOCATION_ABBREV",
    "CONFIG_SUFFIX",
    "MODEL_TYPE_DISPLAY",
    "MODEL_TYPE_ORDER",
    "MODEL_TYPE_DISPLAY_ORDER",
    "format_model_label",
    # Analysis
    "WinCountResult",
    "DeltaResult",
    "BestModelResult",
    "flatten_holdout_results",
    "compare_model_types",
    "compute_delta_stats",
    "find_best_models",
    "print_best_models_table",
    "print_model_comparison_summary",
]
