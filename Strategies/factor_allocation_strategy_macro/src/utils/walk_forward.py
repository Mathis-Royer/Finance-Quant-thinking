"""
Walk-forward validation for time series backtesting.

This module implements the walk-forward validation protocol as specified
in the strategy document. Classical cross-validation is STRICTLY EXCLUDED
because it violates temporal order and creates look-ahead bias.

Walk-forward validation uses expanding or rolling windows:
- Train on historical data up to time T
- Validate on data from T to T+V
- Test on data from T+V to T+V+E
- Repeat with expanding or sliding window
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics, PerformanceReport


@dataclass
class WalkForwardWindow:
    """
    Definition of a single walk-forward window.

    :param train_start (str): Training start date
    :param train_end (str): Training end date
    :param val_start (str): Validation start date
    :param val_end (str): Validation end date
    :param test_start (str): Test start date
    :param test_end (str): Test end date
    """
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


@dataclass
class WalkForwardResult:
    """
    Results from a single walk-forward window.

    :param window (WalkForwardWindow): Window definition
    :param train_metrics (PerformanceReport): Training metrics
    :param val_metrics (PerformanceReport): Validation metrics
    :param test_metrics (PerformanceReport): Test metrics
    :param predictions (np.ndarray): Test set predictions
    :param targets (np.ndarray): Test set targets
    """
    window: WalkForwardWindow
    train_metrics: PerformanceReport
    val_metrics: PerformanceReport
    test_metrics: PerformanceReport
    predictions: np.ndarray
    targets: np.ndarray


class WalkForwardValidator:
    """
    Walk-forward validation framework.

    Implements expanding window validation as specified:
    | Window | Train      | Validation | Test      |
    |--------|------------|------------|-----------|
    | 1      | 2000-2014  | 2015-2017  | 2018-2024 |
    | 2      | 2000-2015  | 2016-2018  | 2019-2024 |
    | 3      | 2000-2016  | 2017-2019  | 2020-2024 |

    :param windows (List[WalkForwardWindow]): List of validation windows
    :param final_holdout_start (str): Start of final holdout (never touched)
    """

    # Default windows as specified in strategy document
    DEFAULT_WINDOWS = [
        WalkForwardWindow("2000-01-01", "2014-12-31", "2015-01-01", "2017-12-31", "2018-01-01", "2024-12-31"),
        WalkForwardWindow("2000-01-01", "2015-12-31", "2016-01-01", "2018-12-31", "2019-01-01", "2024-12-31"),
        WalkForwardWindow("2000-01-01", "2016-12-31", "2017-01-01", "2019-12-31", "2020-01-01", "2024-12-31"),
    ]

    def __init__(
        self,
        windows: Optional[List[WalkForwardWindow]] = None,
        final_holdout_start: str = "2023-01-01",
    ):
        """
        Initialize walk-forward validator.

        :param windows (List[WalkForwardWindow]): Validation windows
        :param final_holdout_start (str): Final holdout start date
        """
        self.windows = windows or self.DEFAULT_WINDOWS
        self.final_holdout_start = final_holdout_start
        self.results: List[WalkForwardResult] = []

    def create_expanding_windows(
        self,
        data_start: str,
        data_end: str,
        initial_train_years: int = 14,
        val_years: int = 3,
        step_years: int = 1,
    ) -> List[WalkForwardWindow]:
        """
        Create expanding window schedule.

        :param data_start (str): Data start date
        :param data_end (str): Data end date
        :param initial_train_years (int): Initial training period
        :param val_years (int): Validation period length
        :param step_years (int): Step size between windows

        :return windows (List[WalkForwardWindow]): Generated windows
        """
        windows = []
        start = pd.Timestamp(data_start)
        end = pd.Timestamp(data_end)

        train_end_year = start.year + initial_train_years - 1

        while True:
            train_end = pd.Timestamp(f"{train_end_year}-12-31")
            val_start = pd.Timestamp(f"{train_end_year + 1}-01-01")
            val_end = pd.Timestamp(f"{train_end_year + val_years}-12-31")
            test_start = pd.Timestamp(f"{train_end_year + val_years + 1}-01-01")

            if test_start >= end:
                break

            window = WalkForwardWindow(
                train_start=data_start,
                train_end=str(train_end.date()),
                val_start=str(val_start.date()),
                val_end=str(val_end.date()),
                test_start=str(test_start.date()),
                test_end=data_end,
            )
            windows.append(window)

            train_end_year += step_years

        return windows

    def split_data(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        date_column: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data according to window definition.

        :param data (pd.DataFrame): Full dataset
        :param window (WalkForwardWindow): Window definition
        :param date_column (str): Date column name

        :return train (pd.DataFrame): Training data
        :return val (pd.DataFrame): Validation data
        :return test (pd.DataFrame): Test data
        """
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])

        train_mask = (
            (data[date_column] >= window.train_start) &
            (data[date_column] <= window.train_end)
        )
        val_mask = (
            (data[date_column] >= window.val_start) &
            (data[date_column] <= window.val_end)
        )
        test_mask = (
            (data[date_column] >= window.test_start) &
            (data[date_column] <= window.test_end)
        )

        return data[train_mask], data[val_mask], data[test_mask]

    def run_validation(
        self,
        model_factory: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        window: WalkForwardWindow,
    ) -> WalkForwardResult:
        """
        Run validation for a single window.

        :param model_factory (Callable): Function that returns a new model instance
        :param X_train (np.ndarray): Training features
        :param y_train (np.ndarray): Training targets
        :param X_val (np.ndarray): Validation features
        :param y_val (np.ndarray): Validation targets
        :param X_test (np.ndarray): Test features
        :param y_test (np.ndarray): Test targets
        :param window (WalkForwardWindow): Window definition

        :return result (WalkForwardResult): Validation result
        """
        # Create and train model
        model = model_factory()
        model.fit(X_train, y_train, X_val, y_val)

        # Get predictions
        train_preds = model.predict_proba(X_train)
        val_preds = model.predict_proba(X_val)
        test_preds = model.predict_proba(X_test)

        # Calculate metrics
        train_metrics = PerformanceMetrics.full_evaluation(train_preds, y_train)
        val_metrics = PerformanceMetrics.full_evaluation(val_preds, y_val)
        test_metrics = PerformanceMetrics.full_evaluation(test_preds, y_test)

        return WalkForwardResult(
            window=window,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            predictions=test_preds,
            targets=y_test,
        )

    def run_all_windows(
        self,
        model_factory: Callable,
        feature_creator: Callable,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> List[WalkForwardResult]:
        """
        Run validation across all windows.

        :param model_factory (Callable): Model creation function
        :param feature_creator (Callable): Feature creation function
        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor data
        :param market_data (pd.DataFrame): Market data
        :param target_data (pd.DataFrame): Target data

        :return results (List[WalkForwardResult]): Results for all windows
        """
        self.results = []

        for window in self.windows:
            print(f"\nRunning window: Train {window.train_start} to {window.train_end}")
            print(f"  Validation: {window.val_start} to {window.val_end}")
            print(f"  Test: {window.test_start} to {window.test_end}")

            # Split target data
            train_targets, val_targets, test_targets = self.split_data(
                target_data, window, "timestamp"
            )

            if len(train_targets) == 0 or len(val_targets) == 0 or len(test_targets) == 0:
                print("  Skipping: insufficient data")
                continue

            # Create features for each split
            X_train, y_train = feature_creator(
                macro_data, factor_data, market_data, train_targets
            )
            X_val, y_val = feature_creator(
                macro_data, factor_data, market_data, val_targets
            )
            X_test, y_test = feature_creator(
                macro_data, factor_data, market_data, test_targets
            )

            # Run validation
            result = self.run_validation(
                model_factory,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                window,
            )

            self.results.append(result)

            # Print summary
            print(f"  Train IC: {result.train_metrics.information_coefficient:.4f}")
            print(f"  Val IC: {result.val_metrics.information_coefficient:.4f}")
            print(f"  Test IC: {result.test_metrics.information_coefficient:.4f}")

        return self.results

    def aggregate_results(self) -> Dict[str, float]:
        """
        Aggregate results across all windows.

        :return aggregated (Dict): Aggregated metrics
        """
        if not self.results:
            return {}

        test_ics = [r.test_metrics.information_coefficient for r in self.results]
        test_accs = [r.test_metrics.accuracy for r in self.results]
        test_aucs = [r.test_metrics.auc for r in self.results]

        val_ics = [r.val_metrics.information_coefficient for r in self.results]

        return {
            "avg_test_ic": np.mean(test_ics),
            "std_test_ic": np.std(test_ics),
            "avg_test_accuracy": np.mean(test_accs),
            "avg_test_auc": np.mean(test_aucs),
            "avg_val_ic": np.mean(val_ics),
            "num_windows": len(self.results),
            "min_test_ic": np.min(test_ics),
            "max_test_ic": np.max(test_ics),
        }

    def check_overfitting(self) -> Dict[str, float]:
        """
        Check for signs of overfitting.

        Compares train vs val vs test performance.

        :return analysis (Dict): Overfitting analysis
        """
        if not self.results:
            return {}

        train_ics = [r.train_metrics.information_coefficient for r in self.results]
        val_ics = [r.val_metrics.information_coefficient for r in self.results]
        test_ics = [r.test_metrics.information_coefficient for r in self.results]

        return {
            "train_val_gap": np.mean(train_ics) - np.mean(val_ics),
            "val_test_gap": np.mean(val_ics) - np.mean(test_ics),
            "train_test_gap": np.mean(train_ics) - np.mean(test_ics),
            "performance_degradation": (np.mean(train_ics) - np.mean(test_ics)) / (
                np.mean(train_ics) + 1e-6
            ),
            "is_likely_overfit": (
                np.mean(train_ics) - np.mean(test_ics) > 0.1
            ),
        }

    def should_advance_to_next_model(
        self,
        baseline_ic: float,
        threshold: float = 0.02,
    ) -> Tuple[bool, str]:
        """
        Determine if current model warrants advancing to more complex model.

        Based on strategy document stop criteria:
        - IC close to zero: reassess strategy
        - No improvement over baseline: do not proceed

        :param baseline_ic (float): Baseline model IC
        :param threshold (float): Minimum improvement threshold

        :return should_advance (bool): Whether to proceed
        :return reason (str): Explanation
        """
        aggregated = self.aggregate_results()

        if not aggregated:
            return False, "No results to evaluate"

        avg_test_ic = aggregated["avg_test_ic"]

        # Check if IC is meaningful
        if abs(avg_test_ic) < 0.02:
            return False, f"IC too low ({avg_test_ic:.4f}), reassess fundamental strategy"

        # Check improvement over baseline
        improvement = avg_test_ic - baseline_ic
        if improvement < threshold:
            return False, f"Insufficient improvement over baseline ({improvement:.4f})"

        # Check for severe overfitting
        overfit = self.check_overfitting()
        if overfit.get("is_likely_overfit", False):
            return False, "Model shows signs of overfitting"

        return True, f"Model shows promise (IC: {avg_test_ic:.4f}, improvement: {improvement:.4f})"

    def generate_report(self) -> str:
        """
        Generate text report of validation results.

        :return report (str): Formatted report
        """
        if not self.results:
            return "No results available"

        lines = [
            "=" * 60,
            "WALK-FORWARD VALIDATION REPORT",
            "=" * 60,
            "",
        ]

        for i, result in enumerate(self.results):
            lines.append(f"Window {i + 1}:")
            lines.append(f"  Train: {result.window.train_start} to {result.window.train_end}")
            lines.append(f"  Val:   {result.window.val_start} to {result.window.val_end}")
            lines.append(f"  Test:  {result.window.test_start} to {result.window.test_end}")
            lines.append("")
            lines.append("  Metrics:")
            lines.append(f"    Train IC:  {result.train_metrics.information_coefficient:.4f}")
            lines.append(f"    Val IC:    {result.val_metrics.information_coefficient:.4f}")
            lines.append(f"    Test IC:   {result.test_metrics.information_coefficient:.4f}")
            lines.append(f"    Test Acc:  {result.test_metrics.accuracy:.4f}")
            lines.append(f"    Test AUC:  {result.test_metrics.auc:.4f}")
            lines.append("")

        # Aggregated results
        agg = self.aggregate_results()
        lines.append("-" * 60)
        lines.append("AGGREGATED RESULTS:")
        lines.append(f"  Average Test IC:  {agg.get('avg_test_ic', 0):.4f} +/- {agg.get('std_test_ic', 0):.4f}")
        lines.append(f"  Average Test Acc: {agg.get('avg_test_accuracy', 0):.4f}")
        lines.append(f"  Average Test AUC: {agg.get('avg_test_auc', 0):.4f}")
        lines.append("")

        # Overfitting analysis
        overfit = self.check_overfitting()
        lines.append("OVERFITTING ANALYSIS:")
        lines.append(f"  Train-Val Gap:  {overfit.get('train_val_gap', 0):.4f}")
        lines.append(f"  Val-Test Gap:   {overfit.get('val_test_gap', 0):.4f}")
        lines.append(f"  Likely Overfit: {overfit.get('is_likely_overfit', False)}")

        lines.append("=" * 60)

        return "\n".join(lines)
