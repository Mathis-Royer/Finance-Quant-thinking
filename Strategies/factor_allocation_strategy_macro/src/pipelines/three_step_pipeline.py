"""
Three-step evaluation pipeline for factor allocation strategy.

Orchestrates the rigorous 3-step evaluation methodology:
1. Walk-forward validation (2017-2021)
2. Final model training (2000-2021)
3. Holdout evaluation (2022+)

This module externalizes the notebook orchestration logic.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from comparison_runner import (
    run_combination_walk_forward,
    train_final_model,
    evaluate_on_holdout,
    ensemble_predict,
    prepare_data,
    compute_composite_score,
    WindowResult,
    HoldoutResult,
)
from features.feature_engineering import FeatureEngineer
from data.factor_data_loader import FactorDataLoader
from utils.metrics import compute_total_return, compute_max_drawdown


@dataclass
class EvaluationConfig:
    """Configuration for three-step evaluation."""
    holdout_years: int = 3
    holdout_start_date: str = "2022-01-01"
    horizons: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    strategies: List[str] = field(default_factory=lambda: ["E2E", "Sup"])
    allocations: List[str] = field(default_factory=lambda: ["Binary", "Multi"])
    save_models: bool = True
    verbose: bool = True


@dataclass
class ThreeStepResults:
    """Container for all three-step evaluation results."""
    walk_forward_results: Dict[Tuple[str, str, int], List[WindowResult]]
    final_models: Dict[Tuple[str, str, int], Dict[str, Any]]
    holdout_results: Dict[Tuple[str, str, int], Dict[str, HoldoutResult]]
    walk_forward_summary: pd.DataFrame
    holdout_summary: pd.DataFrame
    best_combination: Tuple[str, str, int]


class ThreeStepEvaluation:
    """
    Orchestrates the three-step evaluation methodology.

    Step 1: Walk-forward validation with holdout reserved
    Step 2: Train final models on all data except holdout
    Step 3: Evaluate Final vs Ensemble on holdout period
    """

    def __init__(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        indicators: List,
        feature_engineer: FeatureEngineer,
        model_config: Dict,
        factor_loader: FactorDataLoader,
        eval_config: EvaluationConfig = None,
    ):
        """
        Initialize three-step evaluation.

        :param macro_data (pd.DataFrame): Point-in-time macro data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param indicators (List): FRED-MD indicators
        :param feature_engineer (FeatureEngineer): Feature engineering instance
        :param model_config (Dict): Model configuration
        :param factor_loader (FactorDataLoader): Factor data loader
        :param eval_config (EvaluationConfig): Evaluation configuration
        """
        self.macro_data = macro_data
        self.factor_data = factor_data
        self.market_data = market_data
        self.indicators = indicators
        self.feature_engineer = feature_engineer
        self.model_config = model_config
        self.factor_loader = factor_loader
        self.eval_config = eval_config or EvaluationConfig()

        # Prepare target data
        self.targets, self.cumulative_returns = prepare_data(
            factor_loader, factor_data, self.eval_config.horizons
        )

        # Results storage
        self.wf_results: Dict[Tuple[str, str, int], List[WindowResult]] = {}
        self.final_models: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self.holdout_results: Dict[Tuple[str, str, int], Dict[str, HoldoutResult]] = {}

    def run_step1_walk_forward(self) -> Dict[Tuple[str, str, int], List[WindowResult]]:
        """
        Step 1: Run walk-forward validation for all 16 combinations.

        :return wf_results (Dict): Walk-forward results per combination
        """
        config = self.eval_config
        if config.verbose:
            print("=" * 80)
            print("STEP 1: WALK-FORWARD VALIDATION")
            print("=" * 80)
            print(f"Holdout: {config.holdout_start_date} onwards (FIXED for all horizons)")
            print(f"Running {len(config.strategies) * len(config.allocations) * len(config.horizons)} combinations...\n")

        total = len(config.strategies) * len(config.allocations) * len(config.horizons)
        current = 0

        for strategy in config.strategies:
            for allocation in config.allocations:
                for horizon in config.horizons:
                    current += 1
                    key = (strategy, allocation, horizon)

                    if config.verbose:
                        print(f"\n[{current}/{total}] {strategy} + {allocation} @ {horizon}M")

                    wf_results = run_combination_walk_forward(
                        strategy=strategy,
                        allocation=allocation,
                        horizon=horizon,
                        macro_data=self.macro_data,
                        factor_data=self.factor_data,
                        market_data=self.market_data,
                        target_data=self.targets[horizon],
                        cumulative_returns=self.cumulative_returns[horizon],
                        indicators=self.indicators,
                        feature_engineer=self.feature_engineer,
                        config=self.model_config,
                        verbose=False,
                        holdout_years=config.holdout_years,
                        save_models=config.save_models,
                    )

                    self.wf_results[key] = wf_results

                    if wf_results and config.verbose:
                        avg_sharpe = np.mean([r.sharpe for r in wf_results])
                        avg_ic = np.mean([r.ic for r in wf_results])
                        n_models = sum(1 for r in wf_results if r.model is not None)
                        print(f"  Avg Sharpe: {avg_sharpe:+.4f}, Avg IC: {avg_ic:+.4f}, Models saved: {n_models}")

        return self.wf_results

    def run_step2_final_models(self) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
        """
        Step 2: Train final models for all 16 combinations.

        :return final_models (Dict): Final models per combination
        """
        config = self.eval_config
        if config.verbose:
            print("\n" + "=" * 80)
            print("STEP 2: TRAINING FINAL MODELS")
            print("=" * 80)
            print(f"Training period: 2000-01-01 to {config.holdout_start_date[:4]}-12-31")
            print(f"Training all {len(config.strategies) * len(config.allocations) * len(config.horizons)} combinations...\n")

        total = len(config.strategies) * len(config.allocations) * len(config.horizons)
        current = 0

        for strategy in config.strategies:
            for allocation in config.allocations:
                for horizon in config.horizons:
                    current += 1
                    key = (strategy, allocation, horizon)

                    if config.verbose:
                        print(f"[{current}/{total}] Training {strategy} + {allocation} @ {horizon}M")

                    try:
                        final_model, final_strategy = train_final_model(
                            strategy=strategy,
                            allocation=allocation,
                            horizon=horizon,
                            macro_data=self.macro_data,
                            factor_data=self.factor_data,
                            market_data=self.market_data,
                            target_data=self.targets[horizon],
                            cumulative_returns=self.cumulative_returns[horizon],
                            indicators=self.indicators,
                            feature_engineer=self.feature_engineer,
                            config=self.model_config,
                            holdout_start_date=config.holdout_start_date,
                            verbose=False,
                        )

                        self.final_models[key] = {
                            'model': final_model,
                            'strategy_obj': final_strategy,
                        }
                        if config.verbose:
                            print(f"  -> Model trained successfully")

                    except Exception as e:
                        if config.verbose:
                            print(f"  -> Error: {e}")
                        self.final_models[key] = None

        return self.final_models

    def run_step3_holdout(self) -> Dict[Tuple[str, str, int], Dict[str, HoldoutResult]]:
        """
        Step 3: Evaluate Final vs Ensemble on holdout period.

        :return holdout_results (Dict): Holdout results per combination
        """
        config = self.eval_config
        if config.verbose:
            print("\n" + "=" * 80)
            print("STEP 3: HOLDOUT EVALUATION")
            print("=" * 80)
            print(f"Evaluating all {len(config.strategies) * len(config.allocations) * len(config.horizons)} combinations on holdout period...")
            print(f"Holdout start: {config.holdout_start_date}\n")

        total = len(config.strategies) * len(config.allocations) * len(config.horizons)
        current = 0

        for strategy in config.strategies:
            for allocation in config.allocations:
                for horizon in config.horizons:
                    current += 1
                    key = (strategy, allocation, horizon)
                    output_type = "binary" if allocation == "Binary" else "allocation"

                    if config.verbose:
                        print(f"\n[{current}/{total}] {strategy} + {allocation} @ {horizon}M")

                    if key not in self.final_models or self.final_models[key] is None:
                        if config.verbose:
                            print("  No final model available, skipping...")
                        continue

                    if key not in self.wf_results or not self.wf_results[key]:
                        if config.verbose:
                            print("  No walk-forward models available, skipping...")
                        continue

                    final_data = self.final_models[key]
                    wf_results = self.wf_results[key]

                    try:
                        # Evaluate Final model
                        final_holdout = evaluate_on_holdout(
                            model=final_data['model'],
                            strategy_obj=final_data['strategy_obj'],
                            macro_data=self.macro_data,
                            factor_data=self.factor_data,
                            market_data=self.market_data,
                            target_data=self.targets[horizon],
                            holdout_start_date=config.holdout_start_date,
                            output_type=output_type,
                            model_type="final",
                            verbose=config.verbose,
                        )

                        # Evaluate Ensemble
                        ensemble_models = [r.model for r in wf_results if r.model is not None]

                        if ensemble_models:
                            ensemble_holdout = ensemble_predict(
                                models=ensemble_models,
                                strategy_obj=final_data['strategy_obj'],
                                macro_data=self.macro_data,
                                factor_data=self.factor_data,
                                market_data=self.market_data,
                                target_data=self.targets[horizon],
                                holdout_start_date=config.holdout_start_date,
                                output_type=output_type,
                                verbose=config.verbose,
                            )
                        else:
                            ensemble_holdout = None

                        self.holdout_results[key] = {
                            'final': final_holdout,
                            'ensemble': ensemble_holdout,
                        }

                    except Exception as e:
                        if config.verbose:
                            print(f"  Error during evaluation: {e}")
                        self.holdout_results[key] = None

        return self.holdout_results

    def run_all(self) -> ThreeStepResults:
        """
        Run complete three-step evaluation.

        :return results (ThreeStepResults): Complete evaluation results
        """
        # Step 1
        self.run_step1_walk_forward()

        # Step 2
        self.run_step2_final_models()

        # Step 3
        self.run_step3_holdout()

        # Generate summaries
        wf_summary = self._build_walk_forward_summary()
        holdout_summary = self._build_holdout_summary()

        # Find best combination
        if len(wf_summary) > 0:
            best = wf_summary.iloc[0]
            best_combo = (best['strategy'], best['allocation'], int(best['horizon']))
        else:
            best_combo = ("E2E", "Multi", 3)

        return ThreeStepResults(
            walk_forward_results=self.wf_results,
            final_models=self.final_models,
            holdout_results=self.holdout_results,
            walk_forward_summary=wf_summary,
            holdout_summary=holdout_summary,
            best_combination=best_combo,
        )

    def _compute_true_oos_sharpe(self, results_list: List[WindowResult]) -> float:
        """Compute Sharpe from concatenated OOS returns."""
        all_rets = []
        for r in results_list:
            rets = getattr(r, 'monthly_returns', None)
            if rets:
                all_rets.extend(rets)
        if not all_rets or len(all_rets) < 2:
            return 0.0
        returns_arr = np.array(all_rets)
        if returns_arr.std() < 1e-8:
            return 0.0
        return (returns_arr.mean() / returns_arr.std()) * np.sqrt(12)

    def _compute_true_oos_maxdd(self, results_list: List[WindowResult]) -> float:
        """Compute MaxDD from concatenated OOS returns."""
        all_rets = []
        for r in results_list:
            rets = getattr(r, 'monthly_returns', None)
            if rets:
                all_rets.extend(rets)
        if not all_rets:
            return 0.0
        return compute_max_drawdown(all_rets)

    def _build_walk_forward_summary(self) -> pd.DataFrame:
        """Build walk-forward summary DataFrame with composite scores."""
        summary_data = []
        for (strategy, allocation, horizon), results in self.wf_results.items():
            if results:
                sharpes = [r.sharpe for r in results]
                ics = [r.ic for r in results]
                true_oos_sharpe = self._compute_true_oos_sharpe(results)
                true_oos_maxdd = self._compute_true_oos_maxdd(results)
                n_models = sum(1 for r in results if r.model is not None)

                summary_data.append({
                    "strategy": strategy,
                    "allocation": allocation,
                    "horizon": horizon,
                    "avg_sharpe": np.mean(sharpes),
                    "true_oos_sharpe": true_oos_sharpe,
                    "std_sharpe": np.std(sharpes),
                    "avg_ic": np.mean(ics),
                    "true_oos_maxdd": true_oos_maxdd,
                    "pct_positive": sum(1 for s in sharpes if s > 0) / len(sharpes) * 100,
                    "n_models": n_models,
                })

        summary_df = pd.DataFrame(summary_data)

        if len(summary_df) > 0:
            summary_df = compute_composite_score(
                summary_df,
                sharpe_col='true_oos_sharpe',
                ic_col='avg_ic',
                maxdd_col='true_oos_maxdd',
            )
            summary_df = summary_df.sort_values("score", ascending=False)

        return summary_df

    def _build_holdout_summary(self) -> pd.DataFrame:
        """Build holdout summary DataFrame with composite scores."""
        holdout_data = []
        for key, results in self.holdout_results.items():
            if results is None:
                continue

            strategy, allocation, horizon = key
            final = results.get('final')
            ensemble = results.get('ensemble')

            if final:
                n_samples = len(final.monthly_returns) if final.monthly_returns else 0
                holdout_data.append({
                    'strategy': strategy,
                    'allocation': allocation,
                    'horizon': horizon,
                    'model_type': 'Final',
                    'samples': n_samples,
                    'sharpe': final.sharpe,
                    'ic': final.ic,
                    'maxdd': final.maxdd,
                    'total_return': final.total_return,
                })

            if ensemble:
                holdout_data.append({
                    'strategy': strategy,
                    'allocation': allocation,
                    'horizon': horizon,
                    'model_type': 'Ensemble',
                    'samples': n_samples if final else 0,
                    'sharpe': ensemble.sharpe,
                    'ic': ensemble.ic,
                    'maxdd': ensemble.maxdd,
                    'total_return': ensemble.total_return,
                })

        holdout_df = pd.DataFrame(holdout_data)

        if len(holdout_df) > 0:
            holdout_df = compute_composite_score(
                holdout_df,
                sharpe_col='sharpe',
                ic_col='ic',
                maxdd_col='maxdd',
            )
            holdout_df = holdout_df.sort_values('score', ascending=False)

        return holdout_df


def run_three_step_evaluation(
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    indicators: List,
    feature_engineer: FeatureEngineer,
    model_config: Dict,
    factor_loader: FactorDataLoader,
    eval_config: EvaluationConfig = None,
) -> ThreeStepResults:
    """
    Convenience function to run complete three-step evaluation.

    :param macro_data (pd.DataFrame): Point-in-time macro data
    :param factor_data (pd.DataFrame): Factor returns
    :param market_data (pd.DataFrame): Market context
    :param indicators (List): FRED-MD indicators
    :param feature_engineer (FeatureEngineer): Feature engineering instance
    :param model_config (Dict): Model configuration
    :param factor_loader (FactorDataLoader): Factor data loader
    :param eval_config (EvaluationConfig): Evaluation configuration

    :return results (ThreeStepResults): Complete evaluation results
    """
    evaluator = ThreeStepEvaluation(
        macro_data=macro_data,
        factor_data=factor_data,
        market_data=market_data,
        indicators=indicators,
        feature_engineer=feature_engineer,
        model_config=model_config,
        factor_loader=factor_loader,
        eval_config=eval_config,
    )

    return evaluator.run_all()
