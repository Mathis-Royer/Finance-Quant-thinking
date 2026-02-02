"""
Training Strategies for Factor Allocation Model.

This module implements two training strategies:

1. SUPERVISED STRATEGY (SupervisedTrainer):
   - Computes optimal weights w* for each period using Sharpe maximization
   - Uses rolling windows for more stable estimates
   - Trains model to regress toward optimal weights
   - Better for interpretability but higher variance targets

2. END-TO-END STRATEGY (EndToEndTrainer):
   - Directly optimizes differentiable Sharpe ratio loss
   - Aggregates over batches for stable gradients
   - Includes baseline regularization and calibrated turnover penalty
   - Better for generalization with limited data

Both strategies follow the 3-phase progressive training approach from the strategy document.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.optimize import minimize

from .transformer import SharpeRatioLoss, SortinoLoss, BaselineRegularization, calibrate_turnover_penalty
from .pretraining import pretrain_embeddings, transfer_pretrained_embeddings
from utils.training_utils import EarlyStopping, CompositeEarlyStopping, ModelCheckpoint


@dataclass
class TrainingConfig:
    """
    Configuration for training strategies.

    :param learning_rate (float): Base learning rate
    :param batch_size (int): Batch size
    :param weight_decay (float): L2 regularization
    :param epochs_phase1 (int): Epochs for binary classification (E2E only)
    :param epochs_phase2 (int): Epochs for regression (E2E only)
    :param epochs_phase3 (int): Epochs for Sharpe optimization (E2E only)
    :param epochs_supervised (int): Total epochs for Supervised training (default: sum of E2E phases)
    :param use_pretraining (bool): Use embedding pre-training
    :param use_baseline_reg (bool): Use baseline regularization
    :param baseline_penalty (float): Baseline regularization weight
    :param transaction_cost_bps (float): Transaction cost in basis points
    :param rolling_window_months (int): DEPRECATED - not used
    :param gamma (float): Risk aversion for Sharpe loss (0 = pure Sharpe)
    :param horizon_months (int): Prediction horizon for Sharpe optimization (1, 3, 6, or 12)
    :param early_stopping (bool): Enable early stopping
    :param early_stopping_patience (int): Epochs to wait before stopping
    :param early_stopping_min_delta (float): Minimum improvement threshold
    :param skip_phase1_phase2 (bool): Skip Phase 1 and 2, only run Phase 3 (for ablation tests)
    """
    learning_rate: float = 0.0005
    batch_size: int = 64            # Increased from 32
    weight_decay: float = 0.05      # Increased from 0.01
    epochs_phase1: int = 30
    epochs_phase2: int = 20
    epochs_phase3: int = 20
    epochs_supervised: int = 70     # Same as E2E total (30+20+20) for fair comparison
    use_pretraining: bool = True
    use_baseline_reg: bool = True
    baseline_penalty: float = 0.1
    transaction_cost_bps: float = 10.0
    rolling_window_months: int = 24  # DEPRECATED - kept for backward compatibility
    gamma: float = 0.0              # Changed to 0 for true Sharpe
    horizon_months: int = 1
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    # Loss type: 'sharpe' or 'sortino'
    loss_type: str = "sortino"  # Default to Sortino (penalizes only downside risk)
    # Early stopping type: 'simple' or 'composite'
    early_stopping_type: str = "composite"  # Default to composite score
    # Ablation test option: skip Phase 1 and 2 in E2E
    skip_phase1_phase2: bool = False


def compute_optimal_weights(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute optimal portfolio weights that maximize Sharpe ratio.

    Uses scipy optimization with constraints:
    - Weights sum to 1
    - Weights between min_weight and max_weight

    :param returns (np.ndarray): Factor returns [n_periods, n_factors]
    :param risk_free_rate (float): Risk-free rate for Sharpe calculation
    :param min_weight (float): Minimum weight per factor
    :param max_weight (float): Maximum weight per factor

    :return weights (np.ndarray): Optimal weights [n_factors]
    """
    n_factors = returns.shape[1]

    # Handle insufficient data
    if len(returns) < 2:
        return np.ones(n_factors) / n_factors  # Equal weight

    # Mean and covariance
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    # Handle singular covariance
    if np.linalg.det(cov_matrix) < 1e-10:
        cov_matrix += np.eye(n_factors) * 1e-6

    def neg_sharpe(weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        port_return = np.dot(weights, mean_returns)
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        port_std = np.sqrt(max(port_var, 1e-10))
        sharpe = (port_return - risk_free_rate) / port_std
        return -sharpe

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Sum to 1
    ]

    # Bounds
    bounds = [(min_weight, max_weight) for _ in range(n_factors)]

    # Initial guess (equal weight)
    w0 = np.ones(n_factors) / n_factors

    # Optimize
    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100},
    )

    if result.success:
        weights = result.x
        # Ensure constraints are met
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()
        return weights
    else:
        return np.ones(n_factors) / n_factors  # Fall back to equal weight


def compute_rolling_optimal_weights(
    factor_returns: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    horizon_months: int = 1,
    _rolling_window_months: int = 12,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Compute optimal weights for each target date using FORWARD returns.

    For each date T, computes optimal weights that would maximize Sharpe
    on returns from T+1 to T+horizon. Uses mean-variance optimization
    on the monthly returns within the forward period.

    The model learns: given macro features at T, predict weights that
    would have been optimal for the NEXT horizon period.

    For h=1: Only 1 observation available, falls back to equal weights
             (cannot compute covariance from single observation)
    For h>1: Uses h monthly observations to compute mean and covariance,
             then optimizes for maximum Sharpe ratio

    :param factor_returns (pd.DataFrame): Factor returns with timestamp index
    :param target_dates (pd.DatetimeIndex): Dates for which to compute weights
    :param horizon_months (int): Prediction horizon (1, 3, 6, or 12)
    :param _rolling_window_months (int): DEPRECATED, kept for backward compatibility

    :return optimal_weights (Dict): Mapping from date to optimal weights
    """
    factor_cols = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]
    optimal_weights = {}
    n_factors = len(factor_cols)

    for target_date in target_dates:
        # Find the index of target_date in factor_returns
        date_mask = factor_returns["timestamp"] == target_date
        if not date_mask.any():
            # Target date not in factor_returns, use equal weights
            optimal_weights[target_date] = np.ones(n_factors) / n_factors
            continue

        idx = factor_returns.index[date_mask][0]

        # Get forward monthly returns for this date (returns from T+1 to T+horizon)
        forward_returns = _get_forward_monthly_returns(
            factor_returns, factor_cols, idx, horizon_months
        )

        if forward_returns is not None and len(forward_returns) >= 2:
            # Use proper Sharpe optimization with mean-variance framework
            # Constraints: weights between 5% and 50%, sum to 1
            weights = compute_optimal_weights(
                forward_returns,
                risk_free_rate=0.0,
                min_weight=0.05,
                max_weight=0.50,
            )
        elif forward_returns is not None and len(forward_returns) == 1:
            # h=1 case: only 1 observation, use softmax on single return
            # Cannot compute covariance from single observation
            weights = _softmax_weights(forward_returns.flatten(), temperature=0.1)
        else:
            # Not enough data, use equal weights
            weights = np.ones(n_factors) / n_factors

        optimal_weights[target_date] = weights

    return optimal_weights


def _softmax_weights(returns: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert returns to softmax weights with temperature scaling.

    Higher temperature = more uniform weights
    Lower temperature = more concentrated on best performer

    :param returns (np.ndarray): Factor returns
    :param temperature (float): Temperature for softmax

    :return weights (np.ndarray): Softmax weights
    """
    # Ensure returns is a plain numpy array of floats
    returns = np.asarray(returns, dtype=np.float64)
    scaled = returns / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))  # Subtract max for numerical stability
    weights = exp_scaled / exp_scaled.sum()
    return np.clip(weights, 0.05, 0.50)  # Ensure min 5%, max 50%


def _compute_forward_cumulative_returns(
    factor_returns: pd.DataFrame,
    factor_cols: list,
    horizon_months: int,
) -> pd.DataFrame:
    """
    Compute FORWARD cumulative returns for Sharpe optimization.

    For each date t, computes the cumulative return from t+1 to t+horizon.
    This ensures proper forward-looking alignment: features at T predict
    returns from T+1 to T+horizon.

    :param factor_returns (pd.DataFrame): Monthly factor returns
    :param factor_cols (list): Factor column names
    :param horizon_months (int): Horizon in months

    :return cumulative_df (pd.DataFrame): Forward cumulative returns with same index
    """
    df = factor_returns.copy()
    result = pd.DataFrame(index=df.index)
    result["timestamp"] = df["timestamp"]

    for col in factor_cols:
        returns = df[col].values
        n = len(returns)
        cumulative = np.full(n, np.nan)

        # For each date t, compute cumulative return from t+1 to t+horizon
        # This is FORWARD-looking: sample at t gets returns starting at t+1
        for i in range(n - horizon_months):
            window = returns[i + 1:i + 1 + horizon_months]  # FIXED: shift by 1
            cumulative[i] = np.prod(1 + window) - 1

        result[col] = cumulative

    return result


def _get_forward_monthly_returns(
    factor_returns: pd.DataFrame,
    factor_cols: list,
    idx: int,
    horizon_months: int,
) -> np.ndarray:
    """
    Extract the monthly returns from T+1 to T+horizon for a given date index.

    Used to compute Sharpe-optimal weights based on the actual monthly
    returns in the forward period, enabling proper mean-variance optimization.

    :param factor_returns (pd.DataFrame): Monthly factor returns
    :param factor_cols (list): Factor column names
    :param idx (int): Index of the current date T in factor_returns
    :param horizon_months (int): Horizon in months

    :return forward_returns (np.ndarray): Shape [horizon_months, n_factors]
    """
    n = len(factor_returns)

    # Check if we have enough data for the forward period
    if idx + 1 + horizon_months > n:
        return None

    # Extract monthly returns from T+1 to T+horizon
    forward_returns = factor_returns[factor_cols].iloc[idx + 1:idx + 1 + horizon_months].values

    return forward_returns


class SupervisedTrainer:
    """
    Supervised training strategy with optimal weight targets.

    This strategy:
    1. Computes optimal weights w* for each period using rolling Sharpe maximization
    2. Uses these as regression targets
    3. Trains model to predict optimal weights

    Pros:
    - Clear, interpretable targets
    - Can use standard MSE loss

    Cons:
    - Targets have high variance with limited data
    - Optimal weights computed with hindsight

    :param model (nn.Module): Model to train
    :param config (TrainingConfig): Training configuration
    :param device (torch.device): Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        """
        Initialize supervised trainer.

        :param model (nn.Module): Model to train
        :param config (TrainingConfig): Configuration
        :param device (torch.device): Device
        :param verbose (bool): Print progress during training
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cpu")
        self.verbose = verbose
        self.model.to(self.device)

    def compute_targets(
        self,
        factor_returns: pd.DataFrame,
        target_dates: List[pd.Timestamp],
        horizon_months: Optional[int] = None,
    ) -> Dict[pd.Timestamp, np.ndarray]:
        """
        Compute optimal weight targets for all dates.

        Uses cumulative returns for horizons > 1 month to compute weights
        that maximize Sharpe ratio over the specified horizon.

        :param factor_returns (pd.DataFrame): Factor returns
        :param target_dates (List[pd.Timestamp]): Target dates
        :param horizon_months (int): Override horizon (default: use config)

        :return targets (Dict): Date to optimal weights mapping
        """
        horizon = horizon_months if horizon_months is not None else self.config.horizon_months
        return compute_rolling_optimal_weights(
            factor_returns,
            pd.DatetimeIndex(target_dates),
            horizon_months=horizon,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimal_weights: Dict[pd.Timestamp, np.ndarray],
        target_dates: List[pd.Timestamp],
        verbose: Optional[bool] = None,
    ) -> Dict[str, List[float]]:
        """
        Train model with supervised learning on optimal weights.

        :param train_loader (DataLoader): Training data loader
        :param val_loader (DataLoader): Validation data loader
        :param optimal_weights (Dict): Optimal weights per date
        :param target_dates (List): Ordered list of dates
        :param verbose (bool): Override instance verbose setting

        :return history (Dict): Training history
        """
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            print("\n" + "=" * 60)
            print("SUPERVISED TRAINING (Optimal Weight Targets)")
            print("=" * 60)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history = {"train_loss": [], "val_loss": []}

        # Convert optimal weights to tensor targets
        weight_targets = []
        for date in target_dates:
            if date in optimal_weights:
                weight_targets.append(optimal_weights[date])
            else:
                weight_targets.append(np.ones(6) / 6)
        weight_targets = torch.tensor(np.array(weight_targets), dtype=torch.float32)

        # Early stopping and model checkpoint
        early_stopper = None
        composite_stopper = None
        checkpoint = None
        if self.config.early_stopping:
            if self.config.early_stopping_type == "composite":
                composite_stopper = CompositeEarlyStopping(
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.early_stopping_min_delta,
                    verbose=verbose,
                )
                checkpoint = ModelCheckpoint(mode='max')  # Maximize composite score
            else:
                early_stopper = EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.early_stopping_min_delta,
                    mode='min',  # Minimize loss
                    verbose=verbose,
                )
                checkpoint = ModelCheckpoint(mode='min')

        total_epochs = self.config.epochs_supervised
        for epoch in range(total_epochs):
            self.model.train()
            train_loss = 0.0
            batch_idx = 0

            for macro_batch, market_context, _, _ in train_loader:
                macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                market_context = market_context.to(self.device)

                # Get corresponding optimal weights
                batch_size = market_context.size(0)
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + batch_size
                target_weights = weight_targets[start_idx:end_idx].to(self.device)

                if target_weights.size(0) != batch_size:
                    batch_idx += 1
                    continue

                optimizer.zero_grad()
                pred_weights = self.model(macro_batch, market_context, output_type="allocation")
                loss = criterion(pred_weights, target_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                batch_idx += 1

            train_loss /= max(batch_idx, 1)
            history["train_loss"].append(train_loss)

            # Early stopping check
            if composite_stopper is not None:
                # Estimate metrics from training loss for composite stopping
                # Lower loss → better fit → higher estimated Sharpe
                est_sharpe = max(0, 1.0 - train_loss * 10)  # Rough estimate
                est_ic = 0.1 * (1 - train_loss)  # Rough estimate
                est_maxdd = 0.15  # Typical value
                est_return = 0.0

                score = composite_stopper.compute_composite_score(
                    est_sharpe, est_ic, est_maxdd, est_return
                )
                checkpoint(self.model, score)
                if composite_stopper(est_sharpe, est_ic, est_maxdd, est_return, epoch):
                    if verbose:
                        print(f"Composite early stopping at epoch {epoch + 1}")
                    break
            elif early_stopper is not None:
                checkpoint(self.model, train_loss)
                if early_stopper(train_loss, epoch):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{total_epochs}: Train Loss: {train_loss:.6f}")

        # Restore best model weights
        if checkpoint is not None and checkpoint.best_weights is not None:
            checkpoint.restore(self.model)
            if verbose:
                print(f"Restored best model (loss: {checkpoint.best_score:.6f})")

        return history


class EndToEndTrainer:
    """
    End-to-End training strategy with differentiable Sharpe loss.

    This strategy:
    1. Uses differentiable Sharpe ratio approximation
    2. Optimizes over batches for stable gradients
    3. Includes baseline regularization and calibrated turnover penalty

    Pros:
    - Aggregates over samples for stability
    - Learns from batch-level patterns
    - Implicit regularization through batch optimization

    Cons:
    - Less interpretable than supervised
    - Loss is approximation of Sharpe

    :param model (nn.Module): Model to train
    :param config (TrainingConfig): Training configuration
    :param device (torch.device): Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize end-to-end trainer.

        :param model (nn.Module): Model to train
        :param config (TrainingConfig): Configuration
        :param device (torch.device): Device
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Calibrate turnover penalty
        self.turnover_penalty = calibrate_turnover_penalty(
            transaction_cost_bps=self.config.transaction_cost_bps,
            expected_turnover=0.3,
            holding_period_months=1,
        )

        # Baseline regularization
        self.baseline_reg = None
        if self.config.use_baseline_reg:
            self.baseline_reg = BaselineRegularization(
                num_features=64,  # Simplified feature dim
                num_outputs=6,
                alpha=1.0,
            ).to(self.device)

    def train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Phase 1: Binary classification training.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data

        :return history (Dict): Training history
        """
        print("\n" + "=" * 60)
        print("END-TO-END PHASE 1: Binary Classification")
        print("=" * 60)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.config.epochs_phase1):
            self.model.train()
            train_loss = 0.0

            for macro_batch, market_context, targets, _ in train_loader:
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
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for macro_batch, market_context, targets, _ in val_loader:
                    macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                    market_context = market_context.to(self.device)
                    targets = targets.to(self.device).unsqueeze(-1)

                    outputs = self.model(macro_batch, market_context, output_type="binary")
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    preds = (outputs > 0.5).float()
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs_phase1}: "
                      f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        return history

    def train_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Regression training.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data

        :return history (Dict): Training history
        """
        print("\n" + "=" * 60)
        print("END-TO-END PHASE 2: Regression")
        print("=" * 60)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate * 0.1,
            weight_decay=self.config.weight_decay,
        )

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.epochs_phase2):
            self.model.train()
            train_loss = 0.0

            for macro_batch, market_context, targets, _ in train_loader:
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
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for macro_batch, market_context, targets, _ in val_loader:
                    macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                    market_context = market_context.to(self.device)
                    targets = targets.to(self.device).unsqueeze(-1)

                    outputs = self.model(macro_batch, market_context, output_type="regression")
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs_phase2}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return history

    def train_phase3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 3: Sharpe ratio optimization.

        :param train_loader (DataLoader): Training data (must include cumulative_returns)
        :param val_loader (DataLoader): Validation data
        :param factor_returns (np.ndarray): DEPRECATED - returns now come from dataset

        :return history (Dict): Training history
        """
        print("\n" + "=" * 60)
        loss_name = "Sortino" if self.config.loss_type == "sortino" else "Sharpe"
        print(f"END-TO-END PHASE 3: {loss_name} Optimization")
        print("=" * 60)
        print(f"Calibrated turnover penalty: {self.turnover_penalty:.6f}")

        # Choose loss function based on config
        if self.config.loss_type == "sortino":
            criterion = SortinoLoss(
                target_return=0.0,
                turnover_penalty=self.turnover_penalty,
            )
        else:
            criterion = SharpeRatioLoss(
                gamma=self.config.gamma,
                turnover_penalty=self.turnover_penalty,
                baseline_penalty=self.config.baseline_penalty if self.config.use_baseline_reg else 0.0,
            )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate * 0.01,
            weight_decay=self.config.weight_decay,
        )

        history = {"train_loss": []}

        # Early stopping and model checkpoint
        composite_stopper = None
        early_stopper = None
        checkpoint = None
        if self.config.early_stopping:
            if self.config.early_stopping_type == "composite":
                composite_stopper = CompositeEarlyStopping(
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.early_stopping_min_delta,
                    verbose=True,
                )
                checkpoint = ModelCheckpoint(mode='max')
            else:
                early_stopper = EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.early_stopping_min_delta,
                    mode='min',
                    verbose=True,
                )
                checkpoint = ModelCheckpoint(mode='min')

        use_sortino = self.config.loss_type == "sortino"

        for epoch in range(self.config.epochs_phase3):
            self.model.train()
            train_loss = 0.0
            prev_weights = None

            for macro_batch, market_context, _, batch_returns in train_loader:
                macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                market_context = market_context.to(self.device)
                # Use cumulative returns from dataset (properly aligned with samples)
                batch_returns = batch_returns.to(self.device)

                optimizer.zero_grad()
                weights = self.model(macro_batch, market_context, output_type="allocation")

                # Only use prev_weights if batch sizes match
                use_prev = prev_weights if (prev_weights is not None and
                                            prev_weights.size(0) == weights.size(0)) else None

                # SortinoLoss takes 3 args, SharpeRatioLoss takes 4
                if use_sortino:
                    loss = criterion(weights, batch_returns, use_prev)
                else:
                    # Compute baseline deviation if enabled (only for SharpeRatioLoss)
                    baseline_dev = None
                    if self.baseline_reg is not None:
                        pooled = market_context
                        baseline_dev = self.baseline_reg(weights, pooled)
                    loss = criterion(weights, batch_returns, use_prev, baseline_dev)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                prev_weights = weights.detach()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Early stopping check
            if composite_stopper is not None:
                # Estimate metrics from loss for composite stopping
                # Lower loss → higher Sortino/Sharpe → better
                est_sharpe = max(0, -train_loss)  # Loss is negative Sortino/Sharpe
                est_ic = 0.1 * (1 - min(train_loss, 1))
                est_maxdd = 0.15
                est_return = 0.0

                score = composite_stopper.compute_composite_score(
                    est_sharpe, est_ic, est_maxdd, est_return
                )
                checkpoint(self.model, score)
                if composite_stopper(est_sharpe, est_ic, est_maxdd, est_return, epoch):
                    print(f"Composite early stopping at epoch {epoch + 1}")
                    break
            elif early_stopper is not None:
                checkpoint(self.model, train_loss)
                if early_stopper(train_loss, epoch):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs_phase3}: "
                      f"Train Loss: {train_loss:.4f}")

        # Restore best model weights
        if checkpoint is not None and checkpoint.best_weights is not None:
            checkpoint.restore(self.model)
            print(f"Restored best model (score: {checkpoint.best_score:.4f})")

        return history

    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray = None,
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Run complete 3-phase training.

        :param train_loader (DataLoader): Training data (must include cumulative_returns)
        :param val_loader (DataLoader): Validation data
        :param factor_returns (np.ndarray): DEPRECATED - returns now come from dataset

        :return all_history (Dict): History from all phases
        """
        history1 = self.train_phase1(train_loader, val_loader)
        history2 = self.train_phase2(train_loader, val_loader)
        history3 = self.train_phase3(train_loader, val_loader)

        return {
            "phase1": history1,
            "phase2": history2,
            "phase3": history3,
        }
