"""Utility modules for metrics and validation."""

from .metrics import PerformanceMetrics
from .walk_forward import WalkForwardValidator, WalkForwardWindow
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

__all__ = [
    "PerformanceMetrics",
    "WalkForwardValidator",
    "WalkForwardWindow",
    "WalkForwardTuner",
    "HyperparameterSpace",
    "TuningConfig",
    "TuningResult",
    "create_default_tuner",
    "BenchmarkResult",
    "compute_equal_weight_factors_benchmark",
    "compute_equal_weight_cyc_def_benchmark",
    "compute_risk_parity_benchmark",
    "compute_factor_momentum_benchmark",
    "compute_best_single_factor_benchmark",
    "compute_all_benchmarks",
    "get_benchmark_returns_for_period",
    "print_benchmark_summary",
]
