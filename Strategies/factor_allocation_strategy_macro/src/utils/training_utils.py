"""Training utilities including early stopping and model checkpointing."""

from typing import Optional
import torch


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
