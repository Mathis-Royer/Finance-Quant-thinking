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
    """

    def __init__(
        self,
        region: Region = Region.US,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the strategy.

        :param region (Region): Target region
        :param config (Dict): Configuration parameters
        """
        self.region = region
        self.config = config or self._default_config()

        # Initialize components
        self.feature_engineer = FeatureEngineer(
            config=FeatureConfig(
                sequence_length=self.config["sequence_length"],
                include_momentum=True,
                include_market_context=True,
            ),
            region=region,
        )

        self.model: Optional[FactorAllocationTransformer] = None
        self.execution_gate: Optional[ExecutionGate] = None
        self.portfolio_manager: Optional[PortfolioManager] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _default_config(self) -> Dict:
        """
        Default strategy configuration.

        :return config (Dict): Default parameters
        """
        return {
            # Data parameters
            "sequence_length": 50,
            "num_factors": 6,

            # Model parameters (minimal as per strategy doc)
            "d_model": 64,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 128,
            "dropout": 0.4,

            # Training parameters
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs_phase1": 30,
            "epochs_phase2": 20,
            "epochs_phase3": 20,
            "weight_decay": 0.01,

            # Execution parameters
            "execution_threshold": 0.05,
            "transaction_cost": 0.001,

            # Validation parameters
            "val_split": 0.2,
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
    ) -> Dict[str, List[float]]:
        """
        Phase 1: Binary classification (cyclicals vs defensives).

        Uses Cross-Entropy loss to validate the basic concept.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data

        :return history (Dict): Training history
        """
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

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs_phase1']}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        return history

    def train_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Regression on relative outperformance score.

        Uses MSE loss for finer-grained predictions.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data

        :return history (Dict): Training history
        """
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

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs_phase2']}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val IC: {val_ic:.4f}")

        return history

    def train_phase3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Phase 3: Sharpe ratio optimization with turnover penalty.

        Uses differentiable Sharpe approximation.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data
        :param factor_returns (np.ndarray): Historical factor returns

        :return history (Dict): Training history
        """
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
        prev_weights = None

        for epoch in range(self.config["epochs_phase3"]):
            # Training
            self.model.train()
            train_loss = 0.0

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
                loss = criterion(weights, batch_returns, prev_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                prev_weights = weights.detach()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs_phase3']}: "
                      f"Train Loss: {train_loss:.4f}")

        return history

    def backtest(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
        output_type: str = "binary",
    ) -> Dict[str, np.ndarray]:
        """
        Run walk-forward backtest.

        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param target_data (pd.DataFrame): Targets
        :param output_type (str): Model output type

        :return results (Dict): Backtest results
        """
        print("\n" + "=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)

        self.model.eval()
        self.portfolio_manager.reset()

        predictions = []
        actuals = []
        portfolio_returns = []
        timestamps = []

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

                # Get prediction
                output = self.model(macro_tensor, market_tensor, output_type=output_type)
                pred = output.cpu().numpy().flatten()[0]

                predictions.append(pred)
                actuals.append(row["target"])

                # Simulate portfolio returns (simplified)
                if output_type == "binary":
                    # Long cyclical if pred > 0.5, else long defensive
                    if pred > 0.5:
                        factor_idx = available_factors.index("cyclical") if "cyclical" in available_factors else 0
                    else:
                        factor_idx = available_factors.index("defensive") if "defensive" in available_factors else 1

                    # Get factor return for this period
                    factor_row = factor_data[factor_data["timestamp"] == as_of_date]
                    if len(factor_row) > 0:
                        portfolio_returns.append(factor_row.iloc[0].get("cyclical" if pred > 0.5 else "defensive", 0))
                    else:
                        portfolio_returns.append(0.0)

        return {
            "predictions": np.array(predictions),
            "actuals": np.array(actuals),
            "portfolio_returns": np.array(portfolio_returns),
            "timestamps": timestamps,
        }

    def evaluate(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate backtest results.

        :param results (Dict): Backtest results

        :return metrics (Dict): Performance metrics
        """
        predictions = results["predictions"]
        actuals = results["actuals"]
        portfolio_returns = results.get("portfolio_returns", np.array([]))

        report = PerformanceMetrics.full_evaluation(
            predictions, actuals, portfolio_returns if len(portfolio_returns) > 0 else None
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

        print("\n" + "-" * 40)
        print("PERFORMANCE METRICS:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print("-" * 40)

        return metrics

    def save_model(self, path: str) -> None:
        """
        Save model checkpoint.

        :param path (str): Save path
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "region": self.region.value,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model checkpoint.

        :param path (str): Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint["config"]
        self.create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")


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
    print(f"Seed: {args.seed}")
    print("=" * 60)

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
    strategy = FactorAllocationStrategy(region=region)
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
