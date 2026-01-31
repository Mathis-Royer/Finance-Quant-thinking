"""
Main Strategy: Transformer-based Factor Allocation.

This is the main entry point for the factor allocation strategy using
a Transformer model. It integrates all components:
- Data loading (synthetic for testing, real APIs for production)
- Feature engineering
- Transformer model training
- Gated execution for portfolio management
- Walk-forward backtesting

Usage:
    python main_strategy.py --mode train --region us
    python main_strategy.py --mode backtest --region us
    python main_strategy.py --mode live --region us

=============================================================================
DATA SOURCES SUMMARY (see data/data_loader.py for full details)
=============================================================================

MACROECONOMIC DATA:
- FRED/ALFRED (FREE): US economic data with point-in-time vintages
- Bloomberg Terminal ($24k/year): Comprehensive global macro
- Refinitiv Eikon ($15-22k/year): Global with vintage reconstruction
- Trading Economics ($49-299/month): Global macro with surprises

FACTOR RETURNS:
- Kenneth French Library (FREE): US Fama-French factors
- AQR Data Library (FREE): Academic quality factor data
- MSCI Factor Indices (PAID): Industry standard definitions
- Yahoo Finance (FREE): ETF proxies (IWD, IWF, XLY, XLP)

MARKET CONTEXT:
- FRED (FREE): VIX (VIXCLS), Treasury yields (DGS2, DGS10)
- Yahoo Finance (FREE): ^VIX index

=============================================================================
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data.synthetic_data import SyntheticDataGenerator
from data.data_loader import Region, MacroDataLoader
from data.fred_md_loader import FREDMDLoader, FREDMDConfig, load_fred_md_dataset
from features.feature_engineering import FeatureEngineer, FeatureConfig
from models.transformer import FactorAllocationTransformer, SharpeRatioLoss
from models.execution_gate import ExecutionGate, PortfolioManager
from utils.metrics import PerformanceMetrics
from utils.walk_forward import WalkForwardValidator, WalkForwardWindow


class MacroDataset(Dataset):
    """
    PyTorch Dataset for macro token sequences.

    :param macro_batches (List[Dict]): List of macro batch dictionaries
    :param market_contexts (List[np.ndarray]): List of market context arrays
    :param targets (np.ndarray): Target labels
    """

    def __init__(
        self,
        macro_batches: List[Dict[str, np.ndarray]],
        market_contexts: List[np.ndarray],
        targets: np.ndarray,
    ):
        """
        Initialize dataset.

        :param macro_batches (List[Dict]): Macro features per sample
        :param market_contexts (List[np.ndarray]): Market context per sample
        :param targets (np.ndarray): Target labels
        """
        self.macro_batches = macro_batches
        self.market_contexts = market_contexts
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        macro_batch = {
            k: torch.tensor(v, dtype=torch.long if "ids" in k else torch.float32)
            for k, v in self.macro_batches[idx].items()
        }
        market_context = torch.tensor(self.market_contexts[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return macro_batch, market_context, target


def collate_fn(batch: List) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    :param batch (List): List of (macro_batch, market_context, target) tuples

    :return collated (Tuple): Batched tensors
    """
    macro_batches, market_contexts, targets = zip(*batch)

    # Stack macro batches
    collated_macro = {}
    for key in macro_batches[0].keys():
        collated_macro[key] = torch.stack([b[key] for b in macro_batches])

    market_contexts = torch.stack(market_contexts)
    targets = torch.stack(targets)

    return collated_macro, market_contexts, targets


class FactorAllocationStrategy:
    """
    Main strategy class for Transformer-based factor allocation.

    Implements the full pipeline:
    1. Data preparation
    2. Model training (3-phase progressive loss)
    3. Gated execution
    4. Walk-forward backtesting

    :param region (Region): Target geographic region
    :param config (Dict): Strategy configuration
    :param use_fred_md (bool): Use FRED-MD data source
    :param fred_md_indicators (List): Custom FRED-MD indicators
    """

    def __init__(
        self,
        region: Region = Region.US,
        config: Optional[Dict] = None,
        use_fred_md: bool = False,
        fred_md_indicators: Optional[List] = None,
        verbose: bool = True,
    ):
        """
        Initialize the strategy.

        :param region (Region): Target region
        :param config (Dict): Configuration parameters
        :param use_fred_md (bool): Use FRED-MD data source
        :param fred_md_indicators (List): Custom FRED-MD indicators from loader
        :param verbose (bool): Print progress and metrics
        """
        self.region = region
        self.config = config or self._default_config()
        self.use_fred_md = use_fred_md
        self.fred_md_indicators = fred_md_indicators
        self.verbose = verbose

        # Initialize components
        self.feature_engineer = FeatureEngineer(
            config=FeatureConfig(
                sequence_length=self.config["sequence_length"],
                include_momentum=True,
                include_market_context=True,
                use_fred_md=use_fred_md,
            ),
            region=region,
            fred_md_indicators=fred_md_indicators,
        )

        self.model: Optional[FactorAllocationTransformer] = None
        self.execution_gate: Optional[ExecutionGate] = None
        self.portfolio_manager: Optional[PortfolioManager] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Using device: {self.device}")

    def _default_config(self) -> Dict:
        """
        Default strategy configuration.

        MICRO architecture optimized for ~300 monthly samples:
        - Minimal model size (<10k params) to prevent overfitting
        - Single layer, single head attention
        - Very high dropout for regularization

        :return config (Dict): Default parameters
        """
        return {
            # Data parameters
            "sequence_length": 12,  # 1 year of monthly data
            "num_factors": 6,

            # Model parameters (MICRO - heavily reduced)
            "d_model": 32,  # Reduced from 64
            "num_heads": 1,  # Single head
            "num_layers": 1,  # Single layer
            "d_ff": 64,  # Reduced from 128
            "dropout": 0.6,  # High dropout

            # Training parameters
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs_phase1": 20,
            "epochs_phase2": 15,
            "epochs_phase3": 15,
            "weight_decay": 0.02,

            # Execution parameters
            "execution_threshold": 0.05,
            "transaction_cost": 0.001,

            # Validation parameters
            "val_split": 0.2,

            # Periodicity (for metrics calculation)
            "periods_per_year": 12,  # Monthly data
        }

    def create_model(self) -> None:
        """Create and initialize the Transformer model."""
        num_indicators = self.feature_engineer.get_num_indicators()

        self.model = FactorAllocationTransformer(
            num_indicators=num_indicators,
            num_factors=self.config["num_factors"],
            d_model=self.config["d_model"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            d_ff=self.config["d_ff"],
            dropout=self.config["dropout"],
            max_seq_len=self.config["sequence_length"],
        ).to(self.device)

        self.execution_gate = ExecutionGate(
            num_factors=self.config["num_factors"],
            base_threshold=self.config["execution_threshold"],
            use_soft_threshold=True,
        ).to(self.device)

        self.portfolio_manager = PortfolioManager(
            num_factors=self.config["num_factors"],
        )

        if self.verbose:
            param_count = self.model.count_parameters()
            print(f"Model created with {param_count:,} parameters")

    def prepare_data(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> Tuple[MacroDataset, MacroDataset]:
        """
        Prepare datasets for training.

        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param target_data (pd.DataFrame): Target labels

        :return train_dataset (MacroDataset): Training dataset
        :return val_dataset (MacroDataset): Validation dataset
        """
        macro_batches = []
        market_contexts = []
        targets = []

        for _, row in target_data.iterrows():
            as_of_date = row["timestamp"]

            macro_batch, market_ctx = self.feature_engineer.create_transformer_batch(
                macro_data, factor_data, market_data, as_of_date
            )

            macro_batches.append(macro_batch)
            market_contexts.append(market_ctx)
            targets.append(row["target"])

        targets = np.array(targets)

        # Split into train/val
        n_samples = len(targets)
        n_val = int(n_samples * self.config["val_split"])
        n_train = n_samples - n_val

        train_dataset = MacroDataset(
            macro_batches[:n_train],
            market_contexts[:n_train],
            targets[:n_train],
        )

        val_dataset = MacroDataset(
            macro_batches[n_train:],
            market_contexts[n_train:],
            targets[n_train:],
        )

        return train_dataset, val_dataset

    def train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: Optional[bool] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 1: Binary classification (cyclicals vs defensives).

        Uses Cross-Entropy loss to validate the basic concept.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data
        :param verbose (bool): Override instance verbose setting

        :return history (Dict): Training history
        """
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 1: Binary Classification Training")
            print("=" * 60)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.config["epochs_phase1"]):
            # Training
            self.model.train()
            train_loss = 0.0

            for macro_batch, market_context, targets in train_loader:
                # Move to device
                macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                market_context = market_context.to(self.device)
                targets = targets.to(self.device).unsqueeze(-1)

                optimizer.zero_grad()
                outputs = self.model(macro_batch, market_context, output_type="binary")
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for macro_batch, market_context, targets in val_loader:
                    macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                    market_context = market_context.to(self.device)
                    targets = targets.to(self.device).unsqueeze(-1)

                    outputs = self.model(macro_batch, market_context, output_type="binary")
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(targets.cpu().numpy().flatten())

            val_loss /= len(val_loader)
            val_acc = PerformanceMetrics.accuracy(
                np.array(val_preds), np.array(val_targets)
            )

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs_phase1']}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        return history

    def train_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: Optional[bool] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Regression on relative outperformance score.

        Uses MSE loss for finer-grained predictions.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data
        :param verbose (bool): Override instance verbose setting

        :return history (Dict): Training history
        """
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 2: Regression Training")
            print("=" * 60)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"] * 0.1,
            weight_decay=self.config["weight_decay"],
        )

        history = {"train_loss": [], "val_loss": [], "val_ic": []}

        for epoch in range(self.config["epochs_phase2"]):
            # Training
            self.model.train()
            train_loss = 0.0

            for macro_batch, market_context, targets in train_loader:
                macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                market_context = market_context.to(self.device)
                targets = targets.to(self.device).unsqueeze(-1)

                optimizer.zero_grad()
                outputs = self.model(macro_batch, market_context, output_type="regression")
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for macro_batch, market_context, targets in val_loader:
                    macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                    market_context = market_context.to(self.device)
                    targets = targets.to(self.device).unsqueeze(-1)

                    outputs = self.model(macro_batch, market_context, output_type="regression")
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(targets.cpu().numpy().flatten())

            val_loss /= len(val_loader)
            val_ic = PerformanceMetrics.information_coefficient(
                np.array(val_preds), np.array(val_targets)
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ic"].append(val_ic)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs_phase2']}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val IC: {val_ic:.4f}")

        return history

    def train_phase3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray,
        verbose: Optional[bool] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 3: Sharpe ratio optimization with turnover penalty.

        Uses differentiable Sharpe approximation.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data
        :param factor_returns (np.ndarray): Historical factor returns
        :param verbose (bool): Override instance verbose setting

        :return history (Dict): Training history
        """
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 3: Sharpe Ratio Optimization")
            print("=" * 60)

        criterion = SharpeRatioLoss(
            gamma=1.0,
            turnover_penalty=self.config["transaction_cost"] * 10,
        )
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"] * 0.01,
            weight_decay=self.config["weight_decay"],
        )

        history = {"train_loss": [], "val_sharpe": []}

        for epoch in range(self.config["epochs_phase3"]):
            # Training
            self.model.train()
            train_loss = 0.0
            prev_weights = None

            for i, (macro_batch, market_context, _) in enumerate(train_loader):
                macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                market_context = market_context.to(self.device)

                batch_size = market_context.size(0)
                returns_idx = min(i * batch_size, len(factor_returns) - batch_size)
                batch_returns = torch.tensor(
                    factor_returns[returns_idx:returns_idx + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )

                optimizer.zero_grad()
                weights = self.model(macro_batch, market_context, output_type="allocation")

                # Only use prev_weights if batch sizes match
                use_prev = prev_weights if (prev_weights is not None and prev_weights.size(0) == weights.size(0)) else None
                loss = criterion(weights, batch_returns, use_prev)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                prev_weights = weights.detach()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs_phase3']}: "
                      f"Train Loss: {train_loss:.4f}")

        return history

    def backtest(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
        output_type: str = "allocation",
        verbose: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run walk-forward backtest.

        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param target_data (pd.DataFrame): Targets
        :param output_type (str): Model output type ('binary', 'allocation')
        :param verbose (bool): Override instance verbose setting

        :return results (Dict): Backtest results
        """
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            print("\n" + "=" * 60)
            print(f"RUNNING BACKTEST (mode: {output_type})")
            print("=" * 60)

        self.model.eval()
        self.portfolio_manager.reset()

        predictions = []
        actuals = []
        portfolio_returns = []
        timestamps = []
        all_weights = []  # Track all factor weights

        # Factor columns for multi-factor allocation
        factor_columns = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]

        with torch.no_grad():
            for _, row in target_data.iterrows():
                as_of_date = row["timestamp"]
                timestamps.append(as_of_date)

                # Create features
                macro_batch, market_ctx = self.feature_engineer.create_transformer_batch(
                    macro_data, factor_data, market_data, as_of_date
                )

                # Convert to tensors
                macro_tensor = {
                    k: torch.tensor(v, dtype=torch.long if "ids" in k else torch.float32).unsqueeze(0).to(self.device)
                    for k, v in macro_batch.items()
                }
                market_tensor = torch.tensor(market_ctx, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Get factor returns for this period
                factor_row = factor_data[factor_data["timestamp"] == as_of_date]

                if output_type == "allocation":
                    # Multi-factor allocation: use all 6 factor weights
                    output = self.model(macro_tensor, market_tensor, output_type="allocation")
                    weights = output.cpu().numpy().flatten()  # [6] weights summing to 1
                    all_weights.append(weights)

                    # Binary prediction based on cyclical vs defensive weight
                    pred = weights[0] / (weights[0] + weights[1] + 1e-8)  # cyclical / (cyclical + defensive)
                    predictions.append(pred)
                    actuals.append(row["target"])

                    # Portfolio return = weighted sum of all factor returns
                    if len(factor_row) > 0:
                        factor_returns = factor_row[factor_columns].values.flatten()
                        portfolio_return = np.dot(weights, factor_returns)
                        portfolio_returns.append(portfolio_return)
                    else:
                        portfolio_returns.append(0.0)

                elif output_type == "binary":
                    # Binary mode: only cyclical vs defensive (legacy)
                    output = self.model(macro_tensor, market_tensor, output_type="binary")
                    pred = output.cpu().numpy().flatten()[0]
                    predictions.append(pred)
                    actuals.append(row["target"])

                    # Equal weight to cyclical or defensive
                    if len(factor_row) > 0:
                        if pred > 0.5:
                            portfolio_returns.append(factor_row.iloc[0].get("cyclical", 0))
                        else:
                            portfolio_returns.append(factor_row.iloc[0].get("defensive", 0))
                    else:
                        portfolio_returns.append(0.0)

        results = {
            "predictions": np.array(predictions),
            "actuals": np.array(actuals),
            "portfolio_returns": np.array(portfolio_returns),
            "timestamps": timestamps,
        }

        # Add weights history for multi-factor mode
        if output_type == "allocation" and all_weights:
            results["weights"] = np.array(all_weights)
            results["factor_columns"] = factor_columns

        return results

    def evaluate(
        self,
        results: Dict[str, np.ndarray],
        verbose: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Evaluate backtest results.

        :param results (Dict): Backtest results
        :param verbose (bool): Override instance verbose setting

        :return metrics (Dict): Performance metrics
        """
        verbose = verbose if verbose is not None else self.verbose
        predictions = results["predictions"]
        actuals = results["actuals"]
        portfolio_returns = results.get("portfolio_returns", np.array([]))

        # Use periods_per_year=12 for monthly data
        report = PerformanceMetrics.full_evaluation(
            predictions, actuals, portfolio_returns if len(portfolio_returns) > 0 else None,
            periods_per_year=12,
        )

        metrics = {
            "information_coefficient": report.information_coefficient,
            "accuracy": report.accuracy,
            "auc": report.auc,
            "sharpe_ratio": report.sharpe_ratio,
            "total_return": report.total_return,
            "max_drawdown": report.max_drawdown,
            "win_rate": report.win_rate,
        }

        if verbose:
            print("\n" + "-" * 40)
            print("PERFORMANCE METRICS:")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
            print("-" * 40)

        # Multi-factor allocation analysis
        if "weights" in results:
            weights = results["weights"]
            factor_columns = results.get("factor_columns", available_factors)

            # Compute metrics even if not verbose
            avg_weights = weights.mean(axis=0)
            weight_std = weights.std(axis=0)

            if len(weights) > 1:
                weight_changes = np.abs(np.diff(weights, axis=0))
                avg_turnover = weight_changes.sum(axis=1).mean()
                metrics["avg_turnover"] = avg_turnover

            hhi = (weights ** 2).sum(axis=1).mean()
            metrics["concentration_hhi"] = hhi

            # Store avg weights in metrics for later access
            for i, col in enumerate(factor_columns):
                metrics[f"weight_{col}"] = avg_weights[i]

            if verbose:
                print("\nFACTOR ALLOCATION ANALYSIS:")
                print("-" * 40)

                print("Average weights:")
                for i, col in enumerate(factor_columns):
                    print(f"  {col}: {avg_weights[i]:.2%}")

                print("\nWeight volatility (std):")
                for i, col in enumerate(factor_columns):
                    print(f"  {col}: {weight_std[i]:.4f}")

                if "avg_turnover" in metrics:
                    print(f"\nAverage monthly turnover: {metrics['avg_turnover']:.2%}")

                print(f"Concentration (HHI): {hhi:.4f} (1/6 = {1/6:.4f} for equal weight)")
                print("-" * 40)

        return metrics

    def save_model(self, path: str) -> None:
        """
        Save model checkpoint with horizon metadata.

        :param path (str): Save path
        """
        from datetime import datetime

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "region": self.region.value,
            "horizon_months": self.config.get("horizon_months", 1),
            "saved_at": datetime.now().isoformat(),
        }, path)
        if self.verbose:
            horizon = self.config.get("horizon_months", 1)
            print(f"Model saved to {path} (horizon: {horizon}M)")

    def load_model(self, path: str) -> None:
        """
        Load model checkpoint with horizon validation.

        :param path (str): Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Validate horizon if specified in current config
        saved_horizon = checkpoint.get("horizon_months", 1)
        current_horizon = self.config.get("horizon_months", 1)
        if saved_horizon != current_horizon:
            print(f"Warning: Loading {saved_horizon}M model into {current_horizon}M config")

        self.config = checkpoint["config"]
        self.create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.verbose:
            print(f"Model loaded from {path} (horizon: {saved_horizon}M)")


def main():
    """Main entry point for the strategy."""
    parser = argparse.ArgumentParser(description="Factor Allocation Strategy")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "backtest", "demo"],
                       help="Run mode")
    parser.add_argument("--region", type=str, default="us",
                       choices=["us", "europe", "japan"],
                       help="Target region")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--data-source", type=str, default="synthetic",
                       choices=["synthetic", "fred-md"],
                       help="Data source (synthetic for testing, fred-md for real data)")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory to cache FRED-MD data")
    parser.add_argument("--start-date", type=str, default="1990-01-01",
                       help="Start date for data (FRED-MD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                       help="End date for data (FRED-MD)")
    parser.add_argument("--local-file", type=str, default=None,
                       help="Path to local FRED-MD CSV file (for manual download)")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get region
    region_map = {
        "us": Region.US,
        "europe": Region.EUROPE,
        "japan": Region.JAPAN,
    }
    region = region_map[args.region]

    print("=" * 60)
    print("FACTOR ALLOCATION STRATEGY - TRANSFORMER MODEL")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Region: {region.value}")
    print(f"Data Source: {args.data_source}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Load data based on source
    use_fred_md = args.data_source == "fred-md"
    fred_md_indicators = None

    if use_fred_md:
        print("\nLoading FRED-MD data...")
        macro_data, market_data, fred_md_indicators = load_fred_md_dataset(
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=args.cache_dir,
            local_file=args.local_file,
        )

        # Generate synthetic factor returns for now (FRED-MD doesn't include factor returns)
        # In production, you would load these from Kenneth French Library or similar
        print("\nGenerating synthetic factor returns for training...")
        generator = SyntheticDataGenerator(region=region, seed=args.seed)
        _, factor_data, _ = generator.generate_dataset(
            start_date=args.start_date,
            end_date=args.end_date,
            freq="ME",  # Monthly (month-end) to match FRED-MD
        )

        # Create targets using synthetic factor data
        target_data = generator.create_binary_target(factor_data, horizon_weeks=4)
        print(f"Loaded {len(macro_data)} macro observations from FRED-MD")
        print(f"Generated {len(target_data)} target observations")
    else:
        # Generate synthetic data for testing
        print("\nGenerating synthetic data...")
        generator = SyntheticDataGenerator(region=region, seed=args.seed)
        macro_data, factor_data, market_data = generator.generate_dataset(
            start_date="2000-01-01",
            end_date="2024-12-31",
            freq="W",
        )

        # Create targets
        target_data = generator.create_binary_target(factor_data, horizon_weeks=4)
        print(f"Generated {len(macro_data)} macro observations")
        print(f"Generated {len(target_data)} target observations")

    # Initialize strategy
    strategy = FactorAllocationStrategy(
        region=region,
        use_fred_md=use_fred_md,
        fred_md_indicators=fred_md_indicators,
    )
    strategy.create_model()

    if args.mode == "demo":
        # Quick demo with minimal training
        print("\nRunning quick demo...")

        # Prepare small dataset
        small_target = target_data.head(200)
        train_dataset, val_dataset = strategy.prepare_data(
            macro_data, factor_data, market_data, small_target
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Train phase 1 only (quick)
        strategy.config["epochs_phase1"] = 5
        history = strategy.train_phase1(train_loader, val_loader)

        print("\nDemo complete! Final validation accuracy: "
              f"{history['val_acc'][-1]:.4f}")

    elif args.mode == "train":
        # Full training
        print("\nPreparing data...")
        train_dataset, val_dataset = strategy.prepare_data(
            macro_data, factor_data, market_data, target_data
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=strategy.config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=strategy.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Progressive training
        history1 = strategy.train_phase1(train_loader, val_loader)
        history2 = strategy.train_phase2(train_loader, val_loader)

        # Get factor returns for phase 3
        factor_returns = factor_data[["cyclical", "defensive", "value", "growth", "quality", "momentum"]].values
        history3 = strategy.train_phase3(train_loader, val_loader, factor_returns)

        # Save model
        strategy.save_model("transformer_model.pt")

        print("\nTraining complete!")

    elif args.mode == "backtest":
        # Run backtest
        print("\nRunning backtest...")

        # Need to train first for backtest
        train_dataset, val_dataset = strategy.prepare_data(
            macro_data, factor_data, market_data, target_data
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=strategy.config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=strategy.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Quick training for backtest
        strategy.config["epochs_phase1"] = 10
        strategy.train_phase1(train_loader, val_loader)

        # Run backtest
        results = strategy.backtest(
            macro_data, factor_data, market_data, target_data
        )

        # Evaluate
        metrics = strategy.evaluate(results)


# Required for simplified backtest
available_factors = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]


if __name__ == "__main__":
    main()
