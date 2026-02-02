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
    CombinationConfig,
)
from features.feature_engineering import FeatureEngineer
from features.feature_selection import IndicatorSelector, SelectionConfig
from data.factor_data_loader import FactorDataLoader
from utils.metrics import compute_total_return, compute_max_drawdown


@dataclass
class EvaluationConfig:
    """Configuration for three-step evaluation."""
    holdout_years: int = 3
    holdout_start_date: str = "2022-01-01"
    horizons: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    strategies: List[str] = field(default_factory=lambda: ["E2E", "E2E-P3", "Sup"])
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
    dropout: float = 0.5            # Reduced from 0.75 to allow more differentiated predictions
    weight_decay: float = 0.05      # Increased from 0.01
    learning_rate: float = 0.0005   # Reduced from 0.001
    batch_size: int = 64            # Increased from 32
    # Early stopping configuration
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    # Config selection (FS/HPT axes)
    configs_to_run: List[str] = field(default_factory=lambda: ["baseline"])
    # Feature selection settings
    n_features: int = 30
    selection_method: str = "mutual_info"
    # HP tuning settings
    hp_tuning_trials: int = 15


@dataclass
class ThreeStepResults:
    """Container for all three-step evaluation results."""
    # Keys are 4-tuples: (strategy, allocation, horizon, config_name)
    walk_forward_results: Dict[Tuple[str, str, int, str], List[WindowResult]]
    final_models: Dict[Tuple[str, str, int, str], Dict[str, Any]]
    fair_ensemble_models: Dict[Tuple[str, str, int, str], Dict[str, Any]]
    holdout_results: Dict[Tuple[str, str, int, str], Dict[str, HoldoutResult]]
    walk_forward_summary: pd.DataFrame
    holdout_summary: pd.DataFrame
    best_combination: Tuple[str, str, int, str]
    bias_analysis: Optional[Dict[Tuple[str, str, int, str], BiasAnalysisResult]] = None
    # Metadata about configs used
    config_metadata: Optional[Dict[str, CombinationConfig]] = None


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

        # Results storage (4-tuple keys: strategy, allocation, horizon, config_name)
        self.wf_results: Dict[Tuple[str, str, int, str], List[WindowResult]] = {}
        self.final_models: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
        self.fair_ensemble_models: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
        self.holdout_results: Dict[Tuple[str, str, int, str], Dict[str, HoldoutResult]] = {}
        self.bias_results: Dict[Tuple[str, str, int, str], BiasAnalysisResult] = {}

        # CombinationConfig objects for each config (precomputed with FS/HPT)
        self.combo_configs: Dict[str, CombinationConfig] = {}

        # Precompute configs (fit FS selector, run HP tuning)
        self._precompute_configs()

    def _precompute_configs(self) -> None:
        """
        Precompute CombinationConfig objects for each config to run.

        - Fits IndicatorSelector once for configs with feature selection
        - Runs HP tuning independently for each config that needs it
        """
        config = self.eval_config
        configs_to_run = config.configs_to_run

        if config.verbose:
            print("=" * 80)
            print("PRECOMPUTING CONFIGS")
            print(f"Configs to run: {configs_to_run}")
            print("=" * 80)

        # Fit feature selector once (reused for "fs" and "fs+hpt")
        selector = None
        if any(c in ["fs", "fs+hpt"] for c in configs_to_run):
            if config.verbose:
                print(f"\nFitting IndicatorSelector (n_features={config.n_features}, method={config.selection_method})...")

            selector_config = SelectionConfig(
                method=config.selection_method,
                n_features=config.n_features,
            )
            selector = IndicatorSelector(selector_config)
            selector.fit(
                self.macro_data,
                self.targets[min(config.horizons)],
                self.factor_data,
            )

            if config.verbose:
                selected = selector.get_selected_indicators()
                print(f"  Selected {len(selected)} indicators")
                print(f"  Top 5: {selected[:5]}")

        # Create CombinationConfig for each config
        for config_name in configs_to_run:
            use_fs = config_name in ["fs", "fs+hpt"]
            use_hpt = config_name in ["hpt", "fs+hpt"]

            combo_config = CombinationConfig(
                use_feature_selection=use_fs,
                use_hp_tuning=use_hpt,
                n_features=config.n_features,
                selection_method=config.selection_method,
                selector=selector if use_fs else None,
            )

            # Run HP tuning if needed (independently per config)
            if use_hpt:
                if config.verbose:
                    print(f"\nRunning HP tuning for '{config_name}' config...")

                # Get data for tuning (filtered if FS is enabled)
                tuning_macro = self.macro_data
                tuning_indicators = self.indicators
                if use_fs and selector:
                    tuning_macro = selector.transform(self.macro_data)
                    selected_names = set(selector.get_selected_indicators())
                    tuning_indicators = self._filter_indicators(tuning_indicators, selected_names)

                tuned_params = self._run_hp_tuning(tuning_macro, tuning_indicators)
                combo_config.hp_tuning_params = tuned_params

                if config.verbose and tuned_params:
                    print(f"  Tuned params: {list(tuned_params.keys())}")

            self.combo_configs[config_name] = combo_config

        if config.verbose:
            print(f"\nPrecomputed {len(self.combo_configs)} configs: {list(self.combo_configs.keys())}")
            print("=" * 80)

    def _filter_indicators(self, indicators: List, selected_names: set) -> List:
        """Filter indicators list to keep only selected ones."""
        if indicators and hasattr(indicators[0], 'name'):
            return [ind for ind in indicators if ind.name in selected_names]
        return [ind for ind in indicators if ind in selected_names]

    def _run_hp_tuning(self, macro_data: pd.DataFrame, indicators: List) -> Optional[Dict[str, Any]]:
        """
        Run HP tuning and return best params.

        :param macro_data (pd.DataFrame): Macro data (possibly filtered by FS)
        :param indicators (List): Indicators (possibly filtered by FS)

        :return best_params (Dict): Tuned hyperparameters or None if tuning fails
        """
        try:
            from comparison_runner import run_with_tuning
        except ImportError:
            if self.eval_config.verbose:
                print("  Warning: run_with_tuning not available, skipping HP tuning")
            return None

        try:
            tuning_summary, _ = run_with_tuning(
                macro_data=macro_data,
                factor_data=self.factor_data,
                market_data=self.market_data,
                target_data=self.targets[min(self.eval_config.horizons)],
                indicators=indicators,
                feature_engineer=self.feature_engineer,
                config=self.model_config,
                strategy="supervised",
                n_trials=self.eval_config.hp_tuning_trials,
                optimization_metric="sharpe",
                verbose=self.eval_config.verbose,
            )
            return tuning_summary.get('best_params', {})
        except Exception as e:
            if self.eval_config.verbose:
                print(f"  Warning: HP tuning failed: {e}")
            return None

    def _get_data_for_config(self, config_name: str) -> Tuple[pd.DataFrame, List]:
        """
        Get macro_data and indicators for a given config.

        :param config_name (str): Config name ('baseline', 'fs', 'hpt', 'fs+hpt')

        :return macro_data (pd.DataFrame): Possibly filtered macro data
        :return indicators (List): Possibly filtered indicators
        """
        combo_config = self.combo_configs.get(config_name)
        if not combo_config:
            return self.macro_data, self.indicators

        if combo_config.use_feature_selection and combo_config.selector:
            filtered_macro = combo_config.selector.transform(self.macro_data)
            selected_names = set(combo_config.selector.get_selected_indicators())
            filtered_indicators = self._filter_indicators(self.indicators, selected_names)
            return filtered_macro, filtered_indicators

        return self.macro_data, self.indicators

    def run_step1_walk_forward(self) -> Dict[Tuple[str, str, int, str], List[WindowResult]]:
        """
        Step 1: Run walk-forward validation for all combinations across all configs.

        :return wf_results (Dict): Walk-forward results per combination (4-tuple keys)
        """
        config = self.eval_config
        n_base = len(config.strategies) * len(config.allocations) * len(config.horizons)
        n_configs = len(config.configs_to_run)
        total = n_base * n_configs

        if config.verbose:
            print("=" * 80)
            print("STEP 1: WALK-FORWARD VALIDATION")
            print("=" * 80)
            print(f"Holdout: {config.holdout_start_date} onwards (FIXED for all horizons)")
            print(f"Configs: {config.configs_to_run}")
            print(f"Running {total} combinations ({n_base} base × {n_configs} configs)...\n")

        current = 0

        for config_name in config.configs_to_run:
            combo_config = self.combo_configs.get(config_name)
            run_macro, run_indicators = self._get_data_for_config(config_name)

            if config.verbose:
                print(f"\n--- Config: {config_name} ---")

            for strategy in config.strategies:
                for allocation in config.allocations:
                    for horizon in config.horizons:
                        current += 1
                        key = (strategy, allocation, horizon, config_name)

                        if config.verbose:
                            print(f"\n[{current}/{total}] {strategy} + {allocation} @ {horizon}M [{config_name}]")

                        wf_results = run_combination_walk_forward(
                            strategy=strategy,
                            allocation=allocation,
                            horizon=horizon,
                            macro_data=run_macro,
                            factor_data=self.factor_data,
                            market_data=self.market_data,
                            target_data=self.targets[horizon],
                            cumulative_returns=self.cumulative_returns[horizon],
                            indicators=run_indicators,
                            feature_engineer=self.feature_engineer,
                            config=self.model_config,
                            verbose=False,
                            holdout_years=config.holdout_years,
                            save_models=config.save_models,
                            combo_config=combo_config,
                        )

                        self.wf_results[key] = wf_results

                        if wf_results and config.verbose:
                            avg_sharpe = np.mean([r.sharpe for r in wf_results])
                            avg_ic = np.mean([r.ic for r in wf_results])
                            n_models = sum(1 for r in wf_results if r.model is not None)
                            print(f"  Avg Sharpe: {avg_sharpe:+.4f}, Avg IC: {avg_ic:+.4f}, Models saved: {n_models}")

        return self.wf_results

    def run_step2_final_models(self) -> Dict[Tuple[str, str, int, str], Dict[str, Any]]:
        """
        Step 2: Train final models for all combinations across all configs.

        :return final_models (Dict): Final models per combination (4-tuple keys)
        """
        config = self.eval_config
        n_base = len(config.strategies) * len(config.allocations) * len(config.horizons)
        n_configs = len(config.configs_to_run)
        total = n_base * n_configs

        if config.verbose:
            print("\n" + "=" * 80)
            print("STEP 2: TRAINING FINAL MODELS")
            print("=" * 80)
            print(f"Training period: 2000-01-01 to {config.holdout_start_date[:4]}-12-31")
            print(f"Configs: {config.configs_to_run}")
            print(f"Training {total} combinations ({n_base} base × {n_configs} configs)...\n")

        current = 0

        for config_name in config.configs_to_run:
            combo_config = self.combo_configs.get(config_name)
            run_macro, run_indicators = self._get_data_for_config(config_name)

            if config.verbose:
                print(f"\n--- Config: {config_name} ---")

            for strategy in config.strategies:
                for allocation in config.allocations:
                    for horizon in config.horizons:
                        current += 1
                        key = (strategy, allocation, horizon, config_name)

                        if config.verbose:
                            print(f"[{current}/{total}] Training {strategy} + {allocation} @ {horizon}M [{config_name}]")

                        try:
                            final_model, final_strategy = train_final_model(
                                strategy=strategy,
                                allocation=allocation,
                                horizon=horizon,
                                macro_data=run_macro,
                                factor_data=self.factor_data,
                                market_data=self.market_data,
                                target_data=self.targets[horizon],
                                cumulative_returns=self.cumulative_returns[horizon],
                                indicators=run_indicators,
                                feature_engineer=self.feature_engineer,
                                config=self.model_config,
                                holdout_start_date=config.holdout_start_date,
                                verbose=False,
                                combo_config=combo_config,
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

    def run_step3_holdout(self) -> Dict[Tuple[str, str, int, str], Dict[str, HoldoutResult]]:
        """
        Step 3: Train Fair Ensemble and evaluate Final vs Fair Ensemble vs WF Ensemble.

        If train_fair_ensemble is True, trains N models on same data (2000-holdout)
        with different seeds for a fair comparison.

        :return holdout_results (Dict): Holdout results per combination (4-tuple keys)
        """
        config = self.eval_config
        n_base = len(config.strategies) * len(config.allocations) * len(config.horizons)
        n_configs = len(config.configs_to_run)
        total = n_base * n_configs

        if config.verbose:
            print("\n" + "=" * 80)
            print("STEP 3: HOLDOUT EVALUATION")
            print("=" * 80)
            if config.train_fair_ensemble:
                print(f"Training Fair Ensemble ({config.fair_ensemble_n_models} models) + evaluating on holdout")
            else:
                print("Evaluating Final vs WF Ensemble on holdout")
            print(f"Configs: {config.configs_to_run}")
            print(f"Holdout start: {config.holdout_start_date}\n")

        current = 0

        for config_name in config.configs_to_run:
            combo_config = self.combo_configs.get(config_name)
            run_macro, run_indicators = self._get_data_for_config(config_name)

            if config.verbose:
                print(f"\n--- Config: {config_name} ---")

            for strategy in config.strategies:
                for allocation in config.allocations:
                    for horizon in config.horizons:
                        current += 1
                        key = (strategy, allocation, horizon, config_name)
                        output_type = "binary" if allocation == "Binary" else "allocation"

                        if config.verbose:
                            print(f"\n[{current}/{total}] {strategy} + {allocation} @ {horizon}M [{config_name}]")

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
                                macro_data=run_macro,
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
                                    macro_data=run_macro,
                                    factor_data=self.factor_data,
                                    market_data=self.market_data,
                                    target_data=self.targets[horizon],
                                    cumulative_returns=self.cumulative_returns[horizon],
                                    indicators=run_indicators,
                                    feature_engineer=self.feature_engineer,
                                    config=self.model_config,
                                    n_models=config.fair_ensemble_n_models,
                                    base_seed=config.fair_ensemble_base_seed,
                                    seed_step=config.fair_ensemble_seed_step,
                                    holdout_start_date=config.holdout_start_date,
                                    verbose=False,
                                    combo_config=combo_config,
                                )

                                self.fair_ensemble_models[key] = {
                                    'models': fair_models,
                                    'strategy_obj': fair_strategy,
                                }

                                # Evaluate Fair Ensemble
                                fair_ensemble_holdout = ensemble_predict(
                                    models=fair_models,
                                    strategy_obj=fair_strategy,
                                    macro_data=run_macro,
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
                                    macro_data=run_macro,
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

    def run_step4_bias_analysis(self) -> Dict[Tuple[str, str, int, str], BiasAnalysisResult]:
        """
        Step 4 (Optional): Run bias analysis for all combinations.

        Analyzes data quantity effect, seed variance, and pure ensemble effect.

        :return bias_results (Dict): Bias analysis results per combination (4-tuple keys)
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

            strategy, allocation, horizon, config_name = key
            run_macro, run_indicators = self._get_data_for_config(config_name)

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
                    macro_data=run_macro,
                    factor_data=self.factor_data,
                    market_data=self.market_data,
                    target_data=self.targets[horizon],
                    cumulative_returns=self.cumulative_returns[horizon],
                    indicators=run_indicators,
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

        # Find best combination (4-tuple including config)
        if len(wf_summary) > 0:
            best = wf_summary.iloc[0]
            best_combo = (
                best['strategy'],
                best['allocation'],
                int(best['horizon']),
                best.get('config', 'baseline'),
            )
        else:
            best_combo = ("E2E", "Multi", 3, "baseline")

        return ThreeStepResults(
            walk_forward_results=self.wf_results,
            final_models=self.final_models,
            fair_ensemble_models=self.fair_ensemble_models,
            holdout_results=self.holdout_results,
            walk_forward_summary=wf_summary,
            holdout_summary=holdout_summary,
            best_combination=best_combo,
            bias_analysis=self.bias_results if self.bias_results else None,
            config_metadata=self.combo_configs,
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
        for key, results in self.wf_results.items():
            if results:
                # Unpack 4-tuple key
                strategy, allocation, horizon, config_name = key

                sharpes = [r.sharpe for r in results]
                ics = [r.ic for r in results]
                true_oos_sharpe = self._compute_true_oos_sharpe(results)
                true_oos_maxdd = self._compute_true_oos_maxdd(results)
                n_models = sum(1 for r in results if r.model is not None)

                summary_data.append({
                    "strategy": strategy,
                    "allocation": allocation,
                    "horizon": horizon,
                    "config": config_name,
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

            # Unpack 4-tuple key
            strategy, allocation, horizon, config_name = key
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
                    'config': config_name,
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
                    'config': config_name,
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
                    'config': config_name,
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
