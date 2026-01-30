"""
Initial Validation Pipeline: Progressive Model Testing.

This script implements the step-by-step validation approach from the strategy document.
Each model must show improvement over the previous before advancing to more complex architectures.

Progression:
| Step | Model                      | Objective                                |
|------|----------------------------|------------------------------------------|
| 1    | Naive baseline (momentum)  | Establish benchmark to beat              |
| 2    | Logistic Regression/Ridge  | Validate existence of predictive signal  |
| 3    | Gradient Boosting          | Capture non-linear interactions          |
| 4    | Simple LSTM/GRU            | Test value of sequential modeling        |
| 5    | Minimal Transformer        | Exploit attention if previous show signal|

STOP CRITERIA:
- IC close to zero with simple regression: Reassess fundamental strategy
- No significant improvement over momentum baseline: Do not proceed to Transformer
- Systematic overfitting despite regularization: Simplify or reduce features

Usage:
    python initial_validation.py --full          # Run all models
    python initial_validation.py --step 2        # Run up to step 2
    python initial_validation.py --model logistic # Run specific model

=============================================================================
"""

import argparse
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from data.synthetic_data import SyntheticDataGenerator
from data.data_loader import Region
from features.feature_engineering import FeatureEngineer, FeatureConfig
from models.baseline_models import (
    NaiveBaselineModel,
    LogisticRegressionModel,
    GradientBoostingModel,
    LSTMModel,
)
from utils.metrics import PerformanceMetrics
from utils.walk_forward import WalkForwardValidator, WalkForwardWindow


@dataclass
class ValidationResult:
    """
    Result of validating a single model.

    :param model_name (str): Name of the model
    :param step (int): Step number in progression
    :param avg_test_ic (float): Average test IC across windows
    :param avg_test_accuracy (float): Average test accuracy
    :param avg_test_auc (float): Average test AUC
    :param improvement_over_baseline (float): IC improvement vs baseline
    :param should_proceed (bool): Whether to proceed to next model
    :param reason (str): Explanation
    """
    model_name: str
    step: int
    avg_test_ic: float
    avg_test_accuracy: float
    avg_test_auc: float
    improvement_over_baseline: float
    should_proceed: bool
    reason: str


class InitialValidationPipeline:
    """
    Progressive validation pipeline for model selection.

    Implements the step-by-step approach from the strategy document,
    validating each model before proceeding to the next level of complexity.

    :param region (Region): Target geographic region
    :param seed (int): Random seed for reproducibility
    """

    # Minimum improvement thresholds
    IC_MINIMUM = 0.02  # Minimum IC to consider signal exists
    IMPROVEMENT_THRESHOLD = 0.01  # Minimum improvement over baseline

    def __init__(
        self,
        region: Region = Region.US,
        seed: int = 42,
    ):
        """
        Initialize the validation pipeline.

        :param region (Region): Target region
        :param seed (int): Random seed
        """
        self.region = region
        self.seed = seed
        np.random.seed(seed)

        # Initialize components
        self.feature_config = FeatureConfig(
            sequence_length=50,
            include_momentum=True,
            include_market_context=True,
            aggregation_windows=[1, 4, 12],
        )
        self.feature_engineer = FeatureEngineer(
            config=self.feature_config,
            region=region,
        )

        # Create shorter walk-forward windows for synthetic data
        self.validator = WalkForwardValidator(
            windows=[
                WalkForwardWindow("2000-01-01", "2014-12-31", "2015-01-01", "2017-12-31", "2018-01-01", "2020-12-31"),
                WalkForwardWindow("2000-01-01", "2016-12-31", "2017-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
            ]
        )

        # Results storage
        self.results: List[ValidationResult] = []
        self.baseline_ic: float = 0.0

    def generate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic data for validation.

        :return macro_data (pd.DataFrame): Macro data
        :return factor_data (pd.DataFrame): Factor returns
        :return market_data (pd.DataFrame): Market context
        :return target_data (pd.DataFrame): Target labels
        """
        print("Generating synthetic data...")
        generator = SyntheticDataGenerator(region=self.region, seed=self.seed)
        macro_data, factor_data, market_data = generator.generate_dataset(
            start_date="2000-01-01",
            end_date="2024-12-31",
            freq="W",
        )
        target_data = generator.create_binary_target(factor_data, horizon_weeks=4)

        print(f"  Macro observations: {len(macro_data)}")
        print(f"  Factor observations: {len(factor_data)}")
        print(f"  Target observations: {len(target_data)}")

        return macro_data, factor_data, market_data, target_data

    def create_flat_features(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create flat feature matrix for sklearn models.

        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param target_data (pd.DataFrame): Target data

        :return X (np.ndarray): Feature matrix
        :return y (np.ndarray): Target vector
        """
        X, y = self.feature_engineer.create_dataset(
            macro_data, factor_data, market_data, target_data,
            feature_type="flat",
        )
        return X, y

    def create_sequence_features(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequential features for LSTM.

        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param target_data (pd.DataFrame): Target data

        :return X (np.ndarray): Sequence features [samples, seq_len, features]
        :return y (np.ndarray): Target vector
        """
        X, y = self.feature_engineer.create_dataset(
            macro_data, factor_data, market_data, target_data,
            feature_type="sequence",
        )
        return X, y

    def run_step1_baseline(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> ValidationResult:
        """
        Step 1: Naive momentum baseline.

        Establishes the benchmark that all other models must beat.

        :param macro_data, factor_data, market_data, target_data: Input data

        :return result (ValidationResult): Validation result
        """
        print("\n" + "=" * 60)
        print("STEP 1: NAIVE BASELINE (Momentum)")
        print("=" * 60)

        def model_factory():
            return NaiveBaselineModel(lookback=4)

        def feature_creator(macro, factor, market, targets):
            X, y = self.create_flat_features(macro, factor, market, targets)
            # Add recent cyclical and defensive momentum to features
            momentum_features = []
            for _, row in targets.iterrows():
                ts = row["timestamp"]
                recent_factors = factor[factor["timestamp"] <= ts].tail(4)
                if len(recent_factors) > 0:
                    cyc_mom = recent_factors["cyclical"].sum()
                    def_mom = recent_factors["defensive"].sum()
                else:
                    cyc_mom = def_mom = 0.0
                momentum_features.append([cyc_mom, def_mom])
            momentum_features = np.array(momentum_features)
            X = np.hstack([X, momentum_features])
            return X, y

        results = self.validator.run_all_windows(
            model_factory=model_factory,
            feature_creator=feature_creator,
            macro_data=macro_data,
            factor_data=factor_data,
            market_data=market_data,
            target_data=target_data,
        )

        aggregated = self.validator.aggregate_results()
        self.baseline_ic = aggregated.get("avg_test_ic", 0.0)

        result = ValidationResult(
            model_name="Naive Baseline (Momentum)",
            step=1,
            avg_test_ic=aggregated.get("avg_test_ic", 0.0),
            avg_test_accuracy=aggregated.get("avg_test_accuracy", 0.5),
            avg_test_auc=aggregated.get("avg_test_auc", 0.5),
            improvement_over_baseline=0.0,  # This IS the baseline
            should_proceed=True,  # Always proceed from baseline
            reason="Baseline established",
        )

        self.results.append(result)
        self._print_result(result)

        return result

    def run_step2_logistic(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> ValidationResult:
        """
        Step 2: Logistic Regression / Ridge.

        Validates existence of predictive signal in macro features.

        :param macro_data, factor_data, market_data, target_data: Input data

        :return result (ValidationResult): Validation result
        """
        print("\n" + "=" * 60)
        print("STEP 2: LOGISTIC REGRESSION (Ridge)")
        print("=" * 60)

        def model_factory():
            return LogisticRegressionModel(C=0.1)

        def feature_creator(macro, factor, market, targets):
            return self.create_flat_features(macro, factor, market, targets)

        results = self.validator.run_all_windows(
            model_factory=model_factory,
            feature_creator=feature_creator,
            macro_data=macro_data,
            factor_data=factor_data,
            market_data=market_data,
            target_data=target_data,
        )

        aggregated = self.validator.aggregate_results()
        avg_ic = aggregated.get("avg_test_ic", 0.0)
        improvement = avg_ic - self.baseline_ic

        # Check stop criteria
        if abs(avg_ic) < self.IC_MINIMUM:
            should_proceed = False
            reason = f"IC too low ({avg_ic:.4f}), reassess fundamental strategy"
        elif improvement < self.IMPROVEMENT_THRESHOLD:
            should_proceed = False
            reason = f"Insufficient improvement over baseline ({improvement:.4f})"
        else:
            should_proceed = True
            reason = f"Signal detected (IC: {avg_ic:.4f}, improvement: {improvement:.4f})"

        result = ValidationResult(
            model_name="Logistic Regression (Ridge)",
            step=2,
            avg_test_ic=avg_ic,
            avg_test_accuracy=aggregated.get("avg_test_accuracy", 0.5),
            avg_test_auc=aggregated.get("avg_test_auc", 0.5),
            improvement_over_baseline=improvement,
            should_proceed=should_proceed,
            reason=reason,
        )

        self.results.append(result)
        self._print_result(result)

        # Print feature importance
        if results:
            print("\nTop feature coefficients (absolute value):")
            try:
                model = LogisticRegressionModel(C=0.1)
                X_full, y_full = self.create_flat_features(
                    macro_data, factor_data, market_data, target_data
                )
                model.fit(X_full, y_full)
                coefs = np.abs(model.get_feature_importance())
                feature_names = self.feature_engineer.get_feature_names()
                top_indices = np.argsort(coefs)[-10:][::-1]
                for idx in top_indices:
                    if idx < len(feature_names):
                        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
            except Exception as e:
                print(f"  (Could not extract feature importance: {e})")

        return result

    def run_step3_gradient_boosting(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> ValidationResult:
        """
        Step 3: Gradient Boosting (XGBoost/LightGBM).

        Tests if non-linear interactions improve performance.

        :param macro_data, factor_data, market_data, target_data: Input data

        :return result (ValidationResult): Validation result
        """
        print("\n" + "=" * 60)
        print("STEP 3: GRADIENT BOOSTING")
        print("=" * 60)

        def model_factory():
            return GradientBoostingModel(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
            )

        def feature_creator(macro, factor, market, targets):
            return self.create_flat_features(macro, factor, market, targets)

        results = self.validator.run_all_windows(
            model_factory=model_factory,
            feature_creator=feature_creator,
            macro_data=macro_data,
            factor_data=factor_data,
            market_data=market_data,
            target_data=target_data,
        )

        aggregated = self.validator.aggregate_results()
        avg_ic = aggregated.get("avg_test_ic", 0.0)
        improvement = avg_ic - self.baseline_ic

        # Check for overfitting
        overfit = self.validator.check_overfitting()

        if overfit.get("is_likely_overfit", False):
            should_proceed = False
            reason = "Systematic overfitting detected"
        elif improvement < self.IMPROVEMENT_THRESHOLD:
            should_proceed = False
            reason = f"No significant improvement ({improvement:.4f})"
        else:
            should_proceed = True
            reason = f"Non-linear patterns detected (IC: {avg_ic:.4f})"

        result = ValidationResult(
            model_name="Gradient Boosting",
            step=3,
            avg_test_ic=avg_ic,
            avg_test_accuracy=aggregated.get("avg_test_accuracy", 0.5),
            avg_test_auc=aggregated.get("avg_test_auc", 0.5),
            improvement_over_baseline=improvement,
            should_proceed=should_proceed,
            reason=reason,
        )

        self.results.append(result)
        self._print_result(result)

        # Print overfitting analysis
        print("\nOverfitting Analysis:")
        print(f"  Train-Val Gap: {overfit.get('train_val_gap', 0):.4f}")
        print(f"  Val-Test Gap: {overfit.get('val_test_gap', 0):.4f}")
        print(f"  Likely Overfit: {overfit.get('is_likely_overfit', False)}")

        return result

    def run_step4_lstm(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> ValidationResult:
        """
        Step 4: Simple LSTM/GRU.

        Tests if sequential structure adds value.

        :param macro_data, factor_data, market_data, target_data: Input data

        :return result (ValidationResult): Validation result
        """
        print("\n" + "=" * 60)
        print("STEP 4: LSTM MODEL")
        print("=" * 60)

        # LSTM needs different feature creation
        try:
            import torch
        except ImportError:
            print("PyTorch not available, skipping LSTM step")
            return ValidationResult(
                model_name="LSTM",
                step=4,
                avg_test_ic=0.0,
                avg_test_accuracy=0.5,
                avg_test_auc=0.5,
                improvement_over_baseline=0.0,
                should_proceed=False,
                reason="PyTorch not available",
            )

        # For LSTM, we need to handle validation differently
        # Create full dataset first
        X_seq, y_seq = self.create_sequence_features(
            macro_data, factor_data, market_data, target_data
        )

        # Simple time-based split for LSTM
        n_samples = len(y_seq)
        train_end = int(n_samples * 0.6)
        val_end = int(n_samples * 0.8)

        X_train = X_seq[:train_end]
        y_train = y_seq[:train_end]
        X_val = X_seq[train_end:val_end]
        y_val = y_seq[train_end:val_end]
        X_test = X_seq[val_end:]
        y_test = y_seq[val_end:]

        # Train LSTM
        print(f"Training LSTM with {X_train.shape} training samples...")
        input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            dropout=0.3,
        )

        model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=30,
            batch_size=32,
        )

        # Evaluate
        test_preds = model.predict_proba(X_test)
        test_ic = PerformanceMetrics.information_coefficient(test_preds, y_test)
        test_acc = PerformanceMetrics.accuracy(test_preds, y_test)
        test_auc = PerformanceMetrics.auc_score(test_preds, y_test)

        improvement = test_ic - self.baseline_ic

        if improvement < self.IMPROVEMENT_THRESHOLD:
            should_proceed = False
            reason = f"Sequential modeling adds no value ({improvement:.4f})"
        else:
            should_proceed = True
            reason = f"Sequential patterns help (IC: {test_ic:.4f})"

        result = ValidationResult(
            model_name="LSTM",
            step=4,
            avg_test_ic=test_ic,
            avg_test_accuracy=test_acc,
            avg_test_auc=test_auc,
            improvement_over_baseline=improvement,
            should_proceed=should_proceed,
            reason=reason,
        )

        self.results.append(result)
        self._print_result(result)

        return result

    def run_step5_transformer(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> ValidationResult:
        """
        Step 5: Minimal Transformer.

        Only run if previous steps show promise.

        :param macro_data, factor_data, market_data, target_data: Input data

        :return result (ValidationResult): Validation result
        """
        print("\n" + "=" * 60)
        print("STEP 5: MINIMAL TRANSFORMER")
        print("=" * 60)

        # Check if we should even attempt this
        if len(self.results) >= 4 and not self.results[-1].should_proceed:
            print("Previous step did not warrant proceeding to Transformer.")
            return ValidationResult(
                model_name="Transformer",
                step=5,
                avg_test_ic=0.0,
                avg_test_accuracy=0.5,
                avg_test_auc=0.5,
                improvement_over_baseline=0.0,
                should_proceed=False,
                reason="Skipped due to previous step failure",
            )

        try:
            from main_strategy import FactorAllocationStrategy, MacroDataset, collate_fn
            from torch.utils.data import DataLoader
            import torch
        except ImportError as e:
            print(f"Cannot import Transformer components: {e}")
            return ValidationResult(
                model_name="Transformer",
                step=5,
                avg_test_ic=0.0,
                avg_test_accuracy=0.5,
                avg_test_auc=0.5,
                improvement_over_baseline=0.0,
                should_proceed=False,
                reason=f"Import error: {e}",
            )

        # Initialize Transformer strategy
        strategy = FactorAllocationStrategy(region=self.region)
        strategy.config["epochs_phase1"] = 15  # Reduced for validation
        strategy.create_model()

        # Prepare data
        train_dataset, val_dataset = strategy.prepare_data(
            macro_data, factor_data, market_data, target_data
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Train phase 1
        print("Training Transformer (Phase 1)...")
        history = strategy.train_phase1(train_loader, val_loader)

        # Get final metrics
        final_acc = history["val_acc"][-1] if history["val_acc"] else 0.5

        # Approximate IC from accuracy (simplified)
        test_ic = (final_acc - 0.5) * 2 * 0.3  # Very rough approximation

        improvement = test_ic - self.baseline_ic

        if improvement < self.IMPROVEMENT_THRESHOLD:
            should_proceed = False
            reason = f"Transformer shows no additional value ({improvement:.4f})"
        else:
            should_proceed = True
            reason = f"Transformer captures patterns (Acc: {final_acc:.4f})"

        result = ValidationResult(
            model_name="Minimal Transformer",
            step=5,
            avg_test_ic=test_ic,
            avg_test_accuracy=final_acc,
            avg_test_auc=final_acc,  # Approximate
            improvement_over_baseline=improvement,
            should_proceed=should_proceed,
            reason=reason,
        )

        self.results.append(result)
        self._print_result(result)

        return result

    def run_full_validation(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
        stop_on_failure: bool = True,
    ) -> List[ValidationResult]:
        """
        Run complete progressive validation.

        :param macro_data, factor_data, market_data, target_data: Input data
        :param stop_on_failure (bool): Stop if a step fails

        :return results (List[ValidationResult]): All validation results
        """
        print("\n" + "=" * 60)
        print("PROGRESSIVE VALIDATION PIPELINE")
        print("=" * 60)
        print("Testing models in order of complexity...")
        print("Each model must show improvement to proceed to the next.")
        print("=" * 60)

        # Step 1: Baseline
        result1 = self.run_step1_baseline(macro_data, factor_data, market_data, target_data)

        # Step 2: Logistic Regression
        result2 = self.run_step2_logistic(macro_data, factor_data, market_data, target_data)
        if stop_on_failure and not result2.should_proceed:
            print("\n[STOP] Step 2 failed. Not proceeding to more complex models.")
            return self.results

        # Step 3: Gradient Boosting
        result3 = self.run_step3_gradient_boosting(macro_data, factor_data, market_data, target_data)
        if stop_on_failure and not result3.should_proceed:
            print("\n[STOP] Step 3 failed. Not proceeding to sequential models.")
            return self.results

        # Step 4: LSTM
        result4 = self.run_step4_lstm(macro_data, factor_data, market_data, target_data)
        if stop_on_failure and not result4.should_proceed:
            print("\n[STOP] Step 4 failed. Not proceeding to Transformer.")
            return self.results

        # Step 5: Transformer
        result5 = self.run_step5_transformer(macro_data, factor_data, market_data, target_data)

        return self.results

    def _print_result(self, result: ValidationResult) -> None:
        """Print a single validation result."""
        print("\n" + "-" * 40)
        print(f"Model: {result.model_name}")
        print("-" * 40)
        print(f"  Test IC:       {result.avg_test_ic:.4f}")
        print(f"  Test Accuracy: {result.avg_test_accuracy:.4f}")
        print(f"  Test AUC:      {result.avg_test_auc:.4f}")
        print(f"  Improvement:   {result.improvement_over_baseline:.4f}")
        print(f"  Proceed:       {result.should_proceed}")
        print(f"  Reason:        {result.reason}")
        print("-" * 40)

    def generate_report(self) -> str:
        """
        Generate comprehensive validation report.

        :return report (str): Formatted report
        """
        lines = [
            "",
            "=" * 70,
            "PROGRESSIVE VALIDATION REPORT",
            "=" * 70,
            "",
            f"Region: {self.region.value.upper()}",
            f"Seed: {self.seed}",
            f"Baseline IC: {self.baseline_ic:.4f}",
            "",
            "-" * 70,
            "RESULTS SUMMARY",
            "-" * 70,
            "",
            f"{'Step':<6} {'Model':<30} {'IC':<10} {'Improvement':<12} {'Proceed':}",
            "-" * 70,
        ]

        for result in self.results:
            proceed_str = "Yes" if result.should_proceed else "No"
            lines.append(
                f"{result.step:<6} {result.model_name:<30} "
                f"{result.avg_test_ic:<10.4f} {result.improvement_over_baseline:<12.4f} {proceed_str}"
            )

        lines.append("-" * 70)

        # Recommendation
        lines.append("")
        lines.append("RECOMMENDATION:")

        if not self.results:
            lines.append("  No results available.")
        else:
            final_result = self.results[-1]
            if final_result.step == 5 and final_result.should_proceed:
                lines.append("  PROCEED with Transformer model for production.")
                lines.append("  Signal exists and sequential attention adds value.")
            elif any(r.should_proceed for r in self.results if r.step >= 2):
                best_proceeding = max(
                    [r for r in self.results if r.should_proceed],
                    key=lambda r: r.avg_test_ic
                )
                lines.append(f"  USE {best_proceeding.model_name} for production.")
                lines.append(f"  Best IC achieved: {best_proceeding.avg_test_ic:.4f}")
            else:
                lines.append("  REASSESS fundamental strategy.")
                lines.append("  No model shows meaningful predictive signal.")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Progressive Model Validation")
    parser.add_argument("--full", action="store_true",
                       help="Run full validation pipeline")
    parser.add_argument("--step", type=int, default=0,
                       help="Run up to specific step (1-5)")
    parser.add_argument("--model", type=str, default="",
                       choices=["", "baseline", "logistic", "gbm", "lstm", "transformer"],
                       help="Run specific model only")
    parser.add_argument("--region", type=str, default="us",
                       choices=["us", "europe", "japan"],
                       help="Target region")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no-stop", action="store_true",
                       help="Don't stop on failure")

    args = parser.parse_args()

    # Get region
    region_map = {
        "us": Region.US,
        "europe": Region.EUROPE,
        "japan": Region.JAPAN,
    }
    region = region_map[args.region]

    # Initialize pipeline
    pipeline = InitialValidationPipeline(region=region, seed=args.seed)

    # Generate data
    macro_data, factor_data, market_data, target_data = pipeline.generate_data()

    # Run validation
    if args.full or args.step == 0:
        pipeline.run_full_validation(
            macro_data, factor_data, market_data, target_data,
            stop_on_failure=not args.no_stop,
        )
    elif args.model:
        model_steps = {
            "baseline": pipeline.run_step1_baseline,
            "logistic": pipeline.run_step2_logistic,
            "gbm": pipeline.run_step3_gradient_boosting,
            "lstm": pipeline.run_step4_lstm,
            "transformer": pipeline.run_step5_transformer,
        }
        if args.model in model_steps:
            # Run baseline first for comparison
            if args.model != "baseline":
                pipeline.run_step1_baseline(macro_data, factor_data, market_data, target_data)
            model_steps[args.model](macro_data, factor_data, market_data, target_data)
    else:
        # Run up to specific step
        steps = [
            pipeline.run_step1_baseline,
            pipeline.run_step2_logistic,
            pipeline.run_step3_gradient_boosting,
            pipeline.run_step4_lstm,
            pipeline.run_step5_transformer,
        ]
        for i in range(min(args.step, len(steps))):
            result = steps[i](macro_data, factor_data, market_data, target_data)
            if not result.should_proceed and not args.no_stop:
                break

    # Print final report
    print(pipeline.generate_report())


if __name__ == "__main__":
    main()
