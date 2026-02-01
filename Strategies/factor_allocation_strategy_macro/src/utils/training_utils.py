"""Training utilities including early stopping and model checkpointing."""

from typing import Optional, Dict
import torch
import numpy as np


class CompositeEarlyStopping:
    """
    Early stopping based on composite score combining multiple metrics.

    The composite score is: w_sharpe * sharpe + w_ic * ic - w_maxdd * |maxdd|

    This allows stopping based on overall model quality rather than a single metric.

    :param patience: Number of epochs to wait before stopping
    :param min_delta: Minimum improvement to reset counter
    :param w_sharpe: Weight for Sharpe ratio (default 0.35)
    :param w_ic: Weight for Information Coefficient (default 0.25)
    :param w_maxdd: Weight for Max Drawdown penalty (default 0.30)
    :param w_return: Weight for total return bonus (default 0.10)
    :param verbose: Print early stopping messages
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        w_sharpe: float = 0.35,
        w_ic: float = 0.25,
        w_maxdd: float = 0.30,
        w_return: float = 0.10,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.w_sharpe = w_sharpe
        self.w_ic = w_ic
        self.w_maxdd = w_maxdd
        self.w_return = w_return
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        self.should_stop = False
        self.best_epoch = 0

    def compute_composite_score(
        self,
        sharpe: float,
        ic: float,
        maxdd: float,
        total_return: float = 0.0,
    ) -> float:
        """
        Compute composite score from individual metrics.

        :param sharpe: Sharpe ratio (higher is better)
        :param ic: Information Coefficient (higher is better, can be negative)
        :param maxdd: Maximum drawdown as positive number (lower is better)
        :param total_return: Total return (higher is better)

        :return score: Composite score (higher is better)
        """
        # Normalize metrics to comparable scales
        # Sharpe: typically -1 to 3, normalize to 0-1 range
        sharpe_norm = (sharpe + 1) / 4  # maps [-1, 3] to [0, 1]
        sharpe_norm = np.clip(sharpe_norm, 0, 1)

        # IC: typically -0.3 to 0.3, normalize to 0-1 range
        # Apply asymmetric penalty: negative IC penalized 2x
        if ic < 0:
            ic_norm = 0.5 + ic  # maps [-0.5, 0] to [0, 0.5]
            ic_norm = ic_norm * 0.5  # Further penalize negative
        else:
            ic_norm = 0.5 + ic  # maps [0, 0.5] to [0.5, 1]
        ic_norm = np.clip(ic_norm, 0, 1)

        # MaxDD: typically 0 to 0.5, normalize to 0-1 range (inverted)
        # Exponential penalty for large drawdowns
        maxdd_penalty = np.exp(3 * abs(maxdd)) - 1  # Exponential penalty
        maxdd_norm = 1 - np.clip(maxdd_penalty / 3, 0, 1)

        # Return: typically -0.5 to 1.0, normalize to 0-1 range
        return_norm = (total_return + 0.5) / 1.5
        return_norm = np.clip(return_norm, 0, 1)

        # Compute weighted composite
        score = (
            self.w_sharpe * sharpe_norm
            + self.w_ic * ic_norm
            + self.w_maxdd * maxdd_norm
            + self.w_return * return_norm
        )

        return score

    def __call__(
        self,
        sharpe: float,
        ic: float,
        maxdd: float,
        total_return: float = 0.0,
        epoch: int = 0,
    ) -> bool:
        """
        Check if training should stop based on composite score.

        :param sharpe: Sharpe ratio
        :param ic: Information Coefficient
        :param maxdd: Maximum drawdown
        :param total_return: Total return
        :param epoch: Current epoch number

        :return: True if training should stop
        """
        score = self.compute_composite_score(sharpe, ic, maxdd, total_return)
        metrics = {
            "sharpe": sharpe,
            "ic": ic,
            "maxdd": maxdd,
            "return": total_return,
            "score": score,
        }

        if self.best_score is None:
            self.best_score = score
            self.best_metrics = metrics
            self.best_epoch = epoch
            return False

        improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_metrics = metrics
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience

        if self.should_stop and self.verbose:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best composite score: {self.best_score:.4f} at epoch {self.best_epoch}"
            )
            if self.best_metrics:
                print(
                    f"  Best metrics: Sharpe={self.best_metrics['sharpe']:.3f}, "
                    f"IC={self.best_metrics['ic']:.3f}, "
                    f"MaxDD={self.best_metrics['maxdd']:.3f}"
                )

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_metrics = None
        self.should_stop = False
        self.best_epoch = 0


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a validation metric and stops training when no improvement
    is observed for 'patience' consecutive epochs.

    :param patience: Number of epochs to wait before stopping
    :param min_delta: Minimum change to qualify as improvement
    :param mode: 'max' for metrics to maximize (Sharpe, IC), 'min' for loss
    :param verbose: Print early stopping messages
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, val_metric: float, epoch: int = 0) -> bool:
        """
        Check if training should stop.

        :param val_metric: Current validation metric value
        :param epoch: Current epoch number

        :return: True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_metric
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = val_metric > self.best_score + self.min_delta
        else:
            improved = val_metric < self.best_score - self.min_delta

        if improved:
            self.best_score = val_metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience

        if self.should_stop and self.verbose:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}"
            )

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save best model weights during training.

    :param mode: 'max' for metrics to maximize, 'min' for loss
    """

    def __init__(self, mode: str = "max"):
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_weights: Optional[dict] = None

    def __call__(self, model: torch.nn.Module, val_metric: float) -> bool:
        """
        Check if current model is best and save weights.

        :param model: PyTorch model
        :param val_metric: Current validation metric

        :return: True if model was saved (is best)
        """
        if self.best_score is None:
            self._save(model, val_metric)
            return True

        if self.mode == "max":
            is_better = val_metric > self.best_score
        else:
            is_better = val_metric < self.best_score

        if is_better:
            self._save(model, val_metric)
            return True

        return False

    def _save(self, model: torch.nn.Module, val_metric: float):
        """Save model weights."""
        self.best_score = val_metric
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def restore(self, model: torch.nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
