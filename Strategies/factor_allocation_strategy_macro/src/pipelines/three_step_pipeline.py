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
    train_fair_ensemble_models,
    evaluate_on_holdout,
    ensemble_predict,
    prepare_data,
    compute_composite_score,
    run_bias_analysis,
    WindowResult,
    HoldoutResult,
    BiasAnalysisResult,
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
    # Fair ensemble configuration
    train_fair_ensemble: bool = True
    fair_ensemble_n_models: int = 5
    fair_ensemble_base_seed: int = 42
    fair_ensemble_seed_step: int = 100
    # Bias analysis configuration
    run_bias_analysis: bool = False
    bias_cutoff_years: List[int] = field(default_factory=lambda: [2014, 2017, 2020])
    # Enhanced regularization settings
    dropout: float = 0.75           # Increased from 0.6
    weight_decay: float = 0.05      # Increased from 0.01
    learning_rate: float = 0.0005   # Reduced from 0.001
    batch_size: int = 64            # Increased from 32
    # Early stopping configuration
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001


@dataclass
class ThreeStepResults:
    """Container for all three-step evaluation results."""
    walk_forward_results: Dict[Tuple[str, str, int], List[WindowResult]]
    final_models: Dict[Tuple[str, str, int], Dict[str, Any]]
    fair_ensemble_models: Dict[Tuple[str, str, int], Dict[str, Any]]
    holdout_results: Dict[Tuple[str, str, int], Dict[str, HoldoutResult]]
    walk_forward_summary: pd.DataFrame
    holdout_summary: pd.DataFrame
    best_combination: Tuple[str, str, int]
    bias_analysis: Optional[Dict[Tuple[str, str, int], BiasAnalysisResult]] = None


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
        self.fair_ensemble_models: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self.holdout_results: Dict[Tuple[str, str, int], Dict[str, HoldoutResult]] = {}
        self.bias_results: Dict[Tuple[str, str, int], BiasAnalysisResult] = {}

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
        Step 3: Train Fair Ensemble and evaluate Final vs Fair Ensemble vs WF Ensemble.

        If train_fair_ensemble is True, trains N models on same data (2000-holdout)
        with different seeds for a fair comparison.

        :return holdout_results (Dict): Holdout results per combination
        """
        config = self.eval_config
        if config.verbose:
            print("\n" + "=" * 80)
            print("STEP 3: HOLDOUT EVALUATION")
            print("=" * 80)
            if config.train_fair_ensemble:
                print(f"Training Fair Ensemble ({config.fair_ensemble_n_models} models) + evaluating on holdout")
            else:
                print("Evaluating Final vs WF Ensemble on holdout")
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

                    final_data = self.final_models[key]
                    wf_results = self.wf_results.get(key, [])

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

                        # Train and evaluate Fair Ensemble if configured
                        fair_ensemble_holdout = None
                        if config.train_fair_ensemble:
                            if config.verbose:
                                print(f"  Training Fair Ensemble ({config.fair_ensemble_n_models} models)...")

                            fair_models, fair_strategy = train_fair_ensemble_models(
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
                                n_models=config.fair_ensemble_n_models,
                                base_seed=config.fair_ensemble_base_seed,
                                seed_step=config.fair_ensemble_seed_step,
                                holdout_start_date=config.holdout_start_date,
                                verbose=False,
                            )

                            self.fair_ensemble_models[key] = {
                                'models': fair_models,
                                'strategy_obj': fair_strategy,
                            }

                            # Evaluate Fair Ensemble
                            fair_ensemble_holdout = ensemble_predict(
                                models=fair_models,
                                strategy_obj=fair_strategy,
                                macro_data=self.macro_data,
                                factor_data=self.factor_data,
                                market_data=self.market_data,
                                target_data=self.targets[horizon],
                                holdout_start_date=config.holdout_start_date,
                                output_type=output_type,
                                model_type="fair_ensemble",
                                verbose=config.verbose,
                            )

                        # Evaluate WF Ensemble
                        wf_ensemble_holdout = None
                        ensemble_models = [r.model for r in wf_results if r.model is not None]

                        if ensemble_models:
                            wf_ensemble_holdout = ensemble_predict(
                                models=ensemble_models,
                                strategy_obj=final_data['strategy_obj'],
                                macro_data=self.macro_data,
                                factor_data=self.factor_data,
                                market_data=self.market_data,
                                target_data=self.targets[horizon],
                                holdout_start_date=config.holdout_start_date,
                                output_type=output_type,
                                model_type="wf_ensemble",
                                verbose=config.verbose,
                            )

                        self.holdout_results[key] = {
                            'final': final_holdout,
                            'fair_ensemble': fair_ensemble_holdout,
                            'wf_ensemble': wf_ensemble_holdout,
                        }

                    except Exception as e:
                        if config.verbose:
                            print(f"  Error during evaluation: {e}")
                        import traceback
                        traceback.print_exc()
                        self.holdout_results[key] = None

        return self.holdout_results

    def run_step4_bias_analysis(self) -> Dict[Tuple[str, str, int], BiasAnalysisResult]:
        """
        Step 4 (Optional): Run bias analysis for all combinations.

        Analyzes data quantity effect, seed variance, and pure ensemble effect.

        :return bias_results (Dict): Bias analysis results per combination
        """
        config = self.eval_config
        if config.verbose:
            print("\n" + "=" * 80)
            print("STEP 4: BIAS ANALYSIS")
            print("=" * 80)
            print(f"Analyzing {len(self.holdout_results)} combinations...")

        for key, holdout in self.holdout_results.items():
            if holdout is None:
                continue

            strategy, allocation, horizon = key

            # Skip if we don't have all necessary data
            if key not in self.fair_ensemble_models:
                if config.verbose:
                    print(f"  Skipping {key}: No fair ensemble models")
                continue

            final = holdout.get('final')
            fair_ensemble = holdout.get('fair_ensemble')
            wf_ensemble = holdout.get('wf_ensemble')

            if not final or not fair_ensemble:
                continue

            try:
                bias_result = run_bias_analysis(
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
                    final_result=final,
                    fair_ensemble_models=self.fair_ensemble_models[key]['models'],
                    fair_ensemble_result=fair_ensemble,
                    wf_ensemble_result=wf_ensemble,
                    cutoff_years=config.bias_cutoff_years,
                    holdout_start_date=config.holdout_start_date,
                    verbose=config.verbose,
                )

                self.bias_results[key] = bias_result

            except Exception as e:
                if config.verbose:
                    print(f"  Error analyzing {key}: {e}")

        return self.bias_results

    def run_all(self) -> ThreeStepResults:
        """
        Run complete three-step evaluation.

        :return results (ThreeStepResults): Complete evaluation results
        """
        # Step 1
        self.run_step1_walk_forward()

        # Step 2
        self.run_step2_final_models()

        # Step 3 (includes Fair Ensemble training if configured)
        self.run_step3_holdout()

        # Step 4 (optional)
        if self.eval_config.run_bias_analysis:
            self.run_step4_bias_analysis()

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
            fair_ensemble_models=self.fair_ensemble_models,
            holdout_results=self.holdout_results,
            walk_forward_summary=wf_summary,
            holdout_summary=holdout_summary,
            best_combination=best_combo,
            bias_analysis=self.bias_results if self.bias_results else None,
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
            fair_ensemble = results.get('fair_ensemble')
            wf_ensemble = results.get('wf_ensemble')

            n_samples = 0
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

            if fair_ensemble:
                holdout_data.append({
                    'strategy': strategy,
                    'allocation': allocation,
                    'horizon': horizon,
                    'model_type': 'Fair Ensemble',
                    'samples': n_samples,
                    'sharpe': fair_ensemble.sharpe,
                    'ic': fair_ensemble.ic,
                    'maxdd': fair_ensemble.maxdd,
                    'total_return': fair_ensemble.total_return,
                })

            if wf_ensemble:
                holdout_data.append({
                    'strategy': strategy,
                    'allocation': allocation,
                    'horizon': horizon,
                    'model_type': 'WF Ensemble',
                    'samples': n_samples,
                    'sharpe': wf_ensemble.sharpe,
                    'ic': wf_ensemble.ic,
                    'maxdd': wf_ensemble.maxdd,
                    'total_return': wf_ensemble.total_return,
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
