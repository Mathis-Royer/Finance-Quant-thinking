"""
Multi-Horizon Factor Allocation Strategy.

This module implements multi-horizon support for factor allocation,
managing 4 independent Transformer models (1M, 3M, 6M, 12M), each
trained to maximize Sharpe ratio over its specific prediction horizon.

Key design decisions:
- All models rebalance monthly (data is monthly)
- Each model optimizes Sharpe on its horizon's cumulative returns
- Binary (2F) mode remains unchanged (single horizon)
- Multi-factor (6F) mode now supports 4 horizons

Usage:
    from multi_horizon_strategy import MultiHorizonStrategy

    mh_strategy = MultiHorizonStrategy(horizons=[1, 3, 6, 12])
    mh_strategy.prepare_targets(factor_loader, factor_data)
    mh_strategy.create_models(Region.US, indicators)
    histories = mh_strategy.train_all(macro_data, factor_data, market_data)
    results = mh_strategy.backtest_all(macro_data, factor_data, market_data)
    comparison = mh_strategy.compare_horizons(results)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.data_loader import Region
from data.factor_data_loader import FactorDataLoader
from main_strategy import FactorAllocationStrategy, MacroDataset, collate_fn
from models.training_strategies import TrainingConfig, SupervisedTrainer
from models.transformer import FactorAllocationTransformer


@dataclass
class HorizonConfig:
    """
    Configuration for a single horizon model.

    :param horizon_months (int): Prediction horizon (1, 3, 6, or 12)
    :param rolling_window_months (int): Rolling window for optimal weight computation
    :param epochs_phase1 (int): Binary classification epochs
    :param epochs_phase2 (int): Regression epochs
    :param epochs_phase3 (int): Sharpe optimization epochs
    """
    horizon_months: int
    rolling_window_months: int = 12
    epochs_phase1: int = 20
    epochs_phase2: int = 15
    epochs_phase3: int = 15


class MultiHorizonStrategy:
    """
    Orchestrator for multiple horizon-specific factor allocation models.

    Manages 4 independent Transformer models, one per horizon:
    - 1M: Monthly predictions, optimize Sharpe on 1-month returns
    - 3M: Monthly predictions, optimize Sharpe on 3-month cumulative returns
    - 6M: Monthly predictions, optimize Sharpe on 6-month cumulative returns
    - 12M: Monthly predictions, optimize Sharpe on 12-month cumulative returns

    All models rebalance monthly (data is monthly). The difference is
    what each model optimizes for during Phase 3 training.

    :param horizons (List[int]): Horizons to use (default: [1, 3, 6, 12])
    :param base_config (Dict): Base configuration shared across models
    :param model_dir (Path): Directory for model checkpoints
    :param verbose (bool): Print progress
    """

    SUPPORTED_HORIZONS = [1, 3, 6, 12]
    FACTOR_COLUMNS = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        base_config: Optional[Dict] = None,
        model_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Initialize multi-horizon strategy.

        :param horizons (List[int]): Horizons to use [1, 3, 6, 12]
        :param base_config (Dict): Base configuration shared across models
        :param model_dir (Path): Directory for model checkpoints
        :param verbose (bool): Print progress
        """
        self.horizons = horizons or self.SUPPORTED_HORIZONS
        self.base_config = base_config or {}
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.verbose = verbose

        # Validate horizons
        for h in self.horizons:
            if h not in self.SUPPORTED_HORIZONS:
                raise ValueError(f"Unsupported horizon {h}. Must be in {self.SUPPORTED_HORIZONS}")

        # Storage for per-horizon objects
        self.strategies: Dict[int, FactorAllocationStrategy] = {}
        self.horizon_configs: Dict[int, HorizonConfig] = {}
        self.targets: Dict[int, pd.DataFrame] = {}
        self.cumulative_returns: Dict[int, np.ndarray] = {}

        # Region and indicators (set in create_models)
        self.region: Optional[Region] = None
        self.fred_md_indicators: Optional[List[str]] = None

    def _get_horizon_config(self, horizon: int) -> HorizonConfig:
        """
        Get configuration for specific horizon.

        :param horizon (int): Horizon in months

        :return config (HorizonConfig): Horizon-specific configuration
        """
        return HorizonConfig(
            horizon_months=horizon,
            rolling_window_months=max(12, horizon),  # At least 12 months
        )

    def prepare_targets(
        self,
        factor_loader: FactorDataLoader,
        factor_data: pd.DataFrame,
    ) -> Dict[int, pd.DataFrame]:
        """
        Create targets for all horizons using cumulative forward returns.

        :param factor_loader (FactorDataLoader): Data loader instance
        :param factor_data (pd.DataFrame): Factor returns

        :return targets (Dict[int, pd.DataFrame]): Horizon -> target mapping
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("PREPARING MULTI-HORIZON TARGETS")
            print("=" * 60)

        self.targets = factor_loader.create_multi_horizon_targets(
            factor_data, horizons=self.horizons
        )

        # Pre-compute cumulative returns for Phase 3 training
        for horizon in self.horizons:
            if horizon == 1:
                # For 1-month, use raw returns
                self.cumulative_returns[horizon] = factor_data[self.FACTOR_COLUMNS].values
            else:
                # For longer horizons, compute cumulative returns
                self.cumulative_returns[horizon] = factor_loader.get_cumulative_factor_returns_for_horizon(
                    factor_data, horizon
                )

            if self.verbose:
                print(f"  {horizon}M: {len(self.targets[horizon])} samples, "
                      f"cumulative returns shape: {self.cumulative_returns[horizon].shape}")

        return self.targets

    def create_models(
        self,
        region: Region,
        fred_md_indicators: List[str],
    ) -> None:
        """
        Create all horizon-specific models.

        :param region (Region): Geographic region
        :param fred_md_indicators (List[str]): FRED-MD indicator names
        """
        self.region = region
        self.fred_md_indicators = fred_md_indicators

        if self.verbose:
            print("\n" + "=" * 60)
            print("CREATING MULTI-HORIZON MODELS")
            print("=" * 60)

        for horizon in self.horizons:
            config = self.horizon_configs.get(
                horizon, self._get_horizon_config(horizon)
            )

            # Merge base config with horizon-specific settings
            strategy_config = {
                **self.base_config,
                "horizon_months": horizon,
            }

            strategy = FactorAllocationStrategy(
                region=region,
                config=strategy_config,
                use_fred_md=True,
                fred_md_indicators=fred_md_indicators,
                verbose=False,  # Suppress individual model output
            )
            strategy.create_model()
            self.strategies[horizon] = strategy

            if self.verbose:
                param_count = strategy.model.count_parameters()
                print(f"  {horizon}M model: {param_count:,} parameters")

    def train_horizon(
        self,
        horizon: int,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Train single horizon model with 3-phase approach.

        :param horizon (int): Horizon in months
        :param macro_data (pd.DataFrame): Macro features
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data

        :return history (Dict): Training history for all phases
        """
        strategy = self.strategies[horizon]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING {horizon}M HORIZON MODEL")
            print(f"{'='*60}")

        # Phase 1: Binary classification
        history1 = strategy.train_phase1(train_loader, val_loader, verbose=self.verbose)

        # Phase 2: Regression
        history2 = strategy.train_phase2(train_loader, val_loader, verbose=self.verbose)

        # Phase 3: Sharpe optimization with horizon-specific cumulative returns
        factor_returns = self.cumulative_returns[horizon]
        history3 = strategy.train_phase3(
            train_loader, val_loader, factor_returns, verbose=self.verbose
        )

        return {
            "phase1": history1,
            "phase2": history2,
            "phase3": history3,
        }

    def train_all(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Train all horizon models.

        :param macro_data (pd.DataFrame): Macro features
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context

        :return all_histories (Dict): Horizon -> training history
        """
        all_histories = {}

        for horizon in self.horizons:
            target_data = self.targets[horizon]
            strategy = self.strategies[horizon]

            # Prepare data loaders
            train_ds, val_ds = strategy.prepare_data(
                macro_data, factor_data, market_data, target_data
            )

            # Use generator for reproducibility
            g = torch.Generator()
            g.manual_seed(42 + horizon)

            train_loader = DataLoader(
                train_ds,
                batch_size=strategy.config["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True,
                generator=g,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=strategy.config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )

            history = self.train_horizon(
                horizon, macro_data, factor_data, market_data,
                train_loader, val_loader
            )
            all_histories[horizon] = history

        return all_histories

    def backtest_horizon(
        self,
        horizon: int,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        test_targets: Optional[pd.DataFrame] = None,
        output_type: str = "allocation",
    ) -> Dict:
        """
        Backtest single horizon model with monthly rebalancing.

        :param horizon (int): Horizon in months
        :param macro_data (pd.DataFrame): Macro features
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param test_targets (pd.DataFrame): Test targets (default: use prepared targets)
        :param output_type (str): 'binary' or 'allocation'

        :return results (Dict): Backtest results
        """
        strategy = self.strategies[horizon]
        targets = test_targets if test_targets is not None else self.targets[horizon]

        if self.verbose:
            print(f"\n{horizon}M backtest: {len(targets)} observations")

        return strategy.backtest(
            macro_data, factor_data, market_data,
            targets, output_type=output_type,
            verbose=False  # Suppress per-period output
        )

    def backtest_all(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        output_type: str = "allocation",
    ) -> Dict[int, Dict]:
        """
        Backtest all horizon models.

        :param macro_data (pd.DataFrame): Macro features
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param output_type (str): 'binary' or 'allocation'

        :return results (Dict[int, Dict]): Horizon -> backtest results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("MULTI-HORIZON BACKTEST")
            print("=" * 60)

        results = {}
        for horizon in self.horizons:
            results[horizon] = self.backtest_horizon(
                horizon, macro_data, factor_data, market_data,
                output_type=output_type
            )
        return results

    def evaluate_horizon(
        self,
        horizon: int,
        results: Dict,
    ) -> Dict[str, float]:
        """
        Evaluate backtest results for single horizon.

        :param horizon (int): Horizon in months
        :param results (Dict): Backtest results

        :return metrics (Dict): Performance metrics
        """
        strategy = self.strategies[horizon]
        return strategy.evaluate(results, verbose=False)

    def compare_horizons(
        self,
        results: Dict[int, Dict],
    ) -> pd.DataFrame:
        """
        Compare performance across all horizons.

        :param results (Dict[int, Dict]): Results from backtest_all

        :return comparison (pd.DataFrame): Comparison table
        """
        metrics_list = []

        for horizon in self.horizons:
            res = results[horizon]
            metrics = self.evaluate_horizon(horizon, res)
            metrics["horizon"] = f"{horizon}M"
            metrics["horizon_int"] = horizon
            metrics_list.append(metrics)

        df = pd.DataFrame(metrics_list)
        df = df.sort_values("horizon_int")
        df = df.set_index("horizon")
        df = df.drop(columns=["horizon_int"])

        if self.verbose:
            print("\n" + "=" * 70)
            print("MULTI-HORIZON COMPARISON")
            print("=" * 70)
            print(df[["sharpe_ratio", "information_coefficient", "max_drawdown", "total_return"]].to_string())
            print("=" * 70)

        return df

    def save_models(self) -> None:
        """Save all horizon models with horizon suffix."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        for horizon, strategy in self.strategies.items():
            path = self.model_dir / f"transformer_{horizon}M.pt"
            strategy.save_model(str(path))

        if self.verbose:
            print(f"\nSaved {len(self.strategies)} models to {self.model_dir}")

    def load_models(self) -> None:
        """Load all horizon models."""
        loaded = 0
        for horizon in self.horizons:
            path = self.model_dir / f"transformer_{horizon}M.pt"
            if path.exists():
                self.strategies[horizon].load_model(str(path))
                loaded += 1

        if self.verbose:
            print(f"\nLoaded {loaded}/{len(self.horizons)} models from {self.model_dir}")

    def get_model(self, horizon: int) -> FactorAllocationTransformer:
        """
        Get the model for a specific horizon.

        :param horizon (int): Horizon in months

        :return model (FactorAllocationTransformer): The model
        """
        if horizon not in self.strategies:
            raise ValueError(f"No model for horizon {horizon}M. Available: {list(self.strategies.keys())}")
        return self.strategies[horizon].model

    def get_strategy(self, horizon: int) -> FactorAllocationStrategy:
        """
        Get the full strategy for a specific horizon.

        :param horizon (int): Horizon in months

        :return strategy (FactorAllocationStrategy): The strategy
        """
        if horizon not in self.strategies:
            raise ValueError(f"No strategy for horizon {horizon}M. Available: {list(self.strategies.keys())}")
        return self.strategies[horizon]
