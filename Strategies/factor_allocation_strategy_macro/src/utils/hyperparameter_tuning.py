"""
Walk-Forward Hyperparameter Tuning.

This module implements hyperparameter tuning that respects temporal order:
- For each walk-forward window, tune hyperparameters on train/val splits
- Use the best hyperparameters to train on full train set
- Evaluate on test set
- Never use future data for tuning decisions

Supports:
- Optuna-based Bayesian optimization (efficient)
- Random search fallback (if Optuna unavailable)
- Multiple optimization metrics (Sharpe, IC, combined)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import copy

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Using random search fallback.")

from .walk_forward import WalkForwardWindow, WalkForwardValidator
from .metrics import PerformanceMetrics


@dataclass
class HyperparameterSpace:
    """
    Define the hyperparameter search space.

    Each parameter can be:
    - float range: (min, max, log_scale)
    - int range: (min, max)
    - categorical: list of options

    :param learning_rate (Tuple): (min, max, log_scale) for learning rate
    :param dropout (Tuple): (min, max) for dropout
    :param weight_decay (Tuple): (min, max, log_scale) for weight decay
    :param batch_size (List): Categorical batch sizes
    :param d_model (List): Categorical model dimensions
    :param num_layers (List): Categorical layer counts
    :param num_heads (List): Categorical head counts
    :param epochs_phase1 (Tuple): (min, max) for phase 1 epochs
    :param epochs_phase2 (Tuple): (min, max) for phase 2 epochs
    :param epochs_phase3 (Tuple): (min, max) for phase 3 epochs
    """
    learning_rate: Tuple[float, float, bool] = (1e-5, 1e-2, True)
    dropout: Tuple[float, float] = (0.3, 0.7)
    weight_decay: Tuple[float, float, bool] = (1e-4, 1e-1, True)
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64])
    d_model: List[int] = field(default_factory=lambda: [16, 32, 64])
    num_layers: List[int] = field(default_factory=lambda: [1, 2])
    num_heads: List[int] = field(default_factory=lambda: [1, 2, 4])
    epochs_phase1: Tuple[int, int] = (10, 30)
    epochs_phase2: Tuple[int, int] = (10, 25)
    epochs_phase3: Tuple[int, int] = (10, 25)

    # Which parameters to actually tune (subset for faster search)
    tune_architecture: bool = False  # If False, only tune training params
    tune_epochs: bool = True


@dataclass
class TuningConfig:
    """
    Configuration for hyperparameter tuning.

    :param n_trials (int): Number of optimization trials per window
    :param optimization_metric (str): Metric to optimize ('sharpe', 'ic', 'combined')
    :param early_stopping_patience (int): Stop if no improvement for N trials
    :param timeout_seconds (int): Maximum time per window (None = no limit)
    :param n_jobs (int): Parallel trials (-1 = all cores, 1 = sequential)
    :param seed (int): Random seed for reproducibility
    :param verbose (bool): Print progress
    :param use_pruning (bool): Early stop bad trials (Optuna only)
    """
    n_trials: int = 20
    optimization_metric: str = "sharpe"
    early_stopping_patience: int = 5
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1
    seed: int = 42
    verbose: bool = True
    use_pruning: bool = True


@dataclass
class TuningResult:
    """
    Result from tuning a single walk-forward window.

    :param window (WalkForwardWindow): The window definition
    :param best_params (Dict): Best hyperparameters found
    :param best_val_metric (float): Best validation metric achieved
    :param test_metrics (Dict): Test set metrics with best params
    :param n_trials_completed (int): Number of trials run
    :param all_trials (List): All trial results for analysis
    """
    window: WalkForwardWindow
    best_params: Dict[str, Any]
    best_val_metric: float
    test_metrics: Dict[str, float]
    n_trials_completed: int
    all_trials: List[Dict[str, Any]] = field(default_factory=list)


class HyperparameterSampler:
    """
    Sample hyperparameters from the search space.

    Provides both Optuna-based and random sampling.
    """

    def __init__(
        self,
        space: HyperparameterSpace,
        seed: int = 42,
    ):
        """
        Initialize sampler.

        :param space (HyperparameterSpace): Search space definition
        :param seed (int): Random seed
        """
        self.space = space
        self.rng = np.random.RandomState(seed)

    def sample_optuna(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """
        Sample hyperparameters using Optuna trial.

        :param trial (optuna.Trial): Optuna trial object

        :return params (Dict): Sampled hyperparameters
        """
        params = {}

        # Learning rate (log scale)
        lr_min, lr_max, log_scale = self.space.learning_rate
        if log_scale:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", lr_min, lr_max, log=True
            )
        else:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", lr_min, lr_max
            )

        # Dropout
        drop_min, drop_max = self.space.dropout
        params["dropout"] = trial.suggest_float("dropout", drop_min, drop_max)

        # Weight decay (log scale)
        wd_min, wd_max, log_scale = self.space.weight_decay
        if log_scale:
            params["weight_decay"] = trial.suggest_float(
                "weight_decay", wd_min, wd_max, log=True
            )
        else:
            params["weight_decay"] = trial.suggest_float(
                "weight_decay", wd_min, wd_max
            )

        # Batch size (categorical)
        params["batch_size"] = trial.suggest_categorical(
            "batch_size", self.space.batch_size
        )

        # Architecture params (optional)
        if self.space.tune_architecture:
            params["d_model"] = trial.suggest_categorical(
                "d_model", self.space.d_model
            )
            params["num_layers"] = trial.suggest_categorical(
                "num_layers", self.space.num_layers
            )
            params["num_heads"] = trial.suggest_categorical(
                "num_heads", self.space.num_heads
            )

        # Epoch counts (optional)
        if self.space.tune_epochs:
            e1_min, e1_max = self.space.epochs_phase1
            params["epochs_phase1"] = trial.suggest_int(
                "epochs_phase1", e1_min, e1_max
            )
            e2_min, e2_max = self.space.epochs_phase2
            params["epochs_phase2"] = trial.suggest_int(
                "epochs_phase2", e2_min, e2_max
            )
            e3_min, e3_max = self.space.epochs_phase3
            params["epochs_phase3"] = trial.suggest_int(
                "epochs_phase3", e3_min, e3_max
            )

        return params

    def sample_random(self) -> Dict[str, Any]:
        """
        Sample hyperparameters randomly.

        :return params (Dict): Sampled hyperparameters
        """
        params = {}

        # Learning rate
        lr_min, lr_max, log_scale = self.space.learning_rate
        if log_scale:
            params["learning_rate"] = np.exp(
                self.rng.uniform(np.log(lr_min), np.log(lr_max))
            )
        else:
            params["learning_rate"] = self.rng.uniform(lr_min, lr_max)

        # Dropout
        drop_min, drop_max = self.space.dropout
        params["dropout"] = self.rng.uniform(drop_min, drop_max)

        # Weight decay
        wd_min, wd_max, log_scale = self.space.weight_decay
        if log_scale:
            params["weight_decay"] = np.exp(
                self.rng.uniform(np.log(wd_min), np.log(wd_max))
            )
        else:
            params["weight_decay"] = self.rng.uniform(wd_min, wd_max)

        # Batch size
        params["batch_size"] = self.rng.choice(self.space.batch_size)

        # Architecture
        if self.space.tune_architecture:
            params["d_model"] = self.rng.choice(self.space.d_model)
            params["num_layers"] = self.rng.choice(self.space.num_layers)
            params["num_heads"] = self.rng.choice(self.space.num_heads)

        # Epochs
        if self.space.tune_epochs:
            e1_min, e1_max = self.space.epochs_phase1
            params["epochs_phase1"] = self.rng.randint(e1_min, e1_max + 1)
            e2_min, e2_max = self.space.epochs_phase2
            params["epochs_phase2"] = self.rng.randint(e2_min, e2_max + 1)
            e3_min, e3_max = self.space.epochs_phase3
            params["epochs_phase3"] = self.rng.randint(e3_min, e3_max + 1)

        return params


class WalkForwardTuner:
    """
    Walk-forward hyperparameter tuning.

    For each walk-forward window:
    1. Split train into inner_train/inner_val (for tuning)
    2. Run N trials with different hyperparameters
    3. Select best params based on inner_val performance
    4. Retrain on full train with best params
    5. Evaluate on test (held-out)

    This respects temporal order and never uses future data for tuning.

    :param model_factory (Callable): Function to create a fresh model
    :param trainer_factory (Callable): Function to create a trainer
    :param space (HyperparameterSpace): Search space
    :param config (TuningConfig): Tuning configuration
    :param device (torch.device): Device for training
    """

    def __init__(
        self,
        model_factory: Callable[[Dict], nn.Module],
        trainer_factory: Callable[[nn.Module, Dict], Any],
        space: Optional[HyperparameterSpace] = None,
        config: Optional[TuningConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize walk-forward tuner.

        :param model_factory (Callable): Creates model from params dict
        :param trainer_factory (Callable): Creates trainer from model and params
        :param space (HyperparameterSpace): Search space
        :param config (TuningConfig): Configuration
        :param device (torch.device): Device
        """
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.space = space or HyperparameterSpace()
        self.config = config or TuningConfig()
        self.device = device or torch.device("cpu")
        self.sampler = HyperparameterSampler(self.space, self.config.seed)
        self.results: List[TuningResult] = []

    def _compute_metric(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute optimization metric.

        :param predictions (np.ndarray): Model predictions
        :param targets (np.ndarray): True targets
        :param returns (np.ndarray): Portfolio returns (for Sharpe)

        :return metric (float): Computed metric value
        """
        metric_name = self.config.optimization_metric.lower()

        if metric_name == "ic":
            # Information coefficient
            if len(predictions) < 2:
                return 0.0
            return np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

        elif metric_name == "sharpe":
            # Sharpe ratio (requires returns)
            if returns is None or len(returns) < 2:
                return 0.0
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret < 1e-8:
                return 0.0
            return mean_ret / std_ret * np.sqrt(12)  # Annualized

        elif metric_name == "combined":
            # Combined IC and Sharpe
            ic = self._compute_metric(predictions, targets, None)
            sharpe = self._compute_metric(predictions, targets, returns)
            return 0.5 * ic + 0.5 * (sharpe / 2)  # Normalize Sharpe contribution

        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _create_inner_split(
        self,
        train_data: pd.DataFrame,
        val_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into inner train/val for tuning.

        Uses temporal split (last val_ratio of train for validation).

        :param train_data (pd.DataFrame): Training data
        :param val_ratio (float): Fraction for inner validation

        :return inner_train (pd.DataFrame): Inner training data
        :return inner_val (pd.DataFrame): Inner validation data
        """
        n = len(train_data)
        split_idx = int(n * (1 - val_ratio))

        # Sort by date to ensure temporal order
        if "timestamp" in train_data.columns:
            train_data = train_data.sort_values("timestamp").reset_index(drop=True)

        inner_train = train_data.iloc[:split_idx]
        inner_val = train_data.iloc[split_idx:]

        return inner_train, inner_val

    def _run_single_trial(
        self,
        params: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray,
        base_config: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run a single hyperparameter trial.

        :param params (Dict): Hyperparameters to test
        :param train_loader (DataLoader): Training data loader
        :param val_loader (DataLoader): Validation data loader
        :param factor_returns (np.ndarray): Factor returns
        :param base_config (Dict): Base configuration to override

        :return val_metric (float): Validation metric
        :return all_metrics (Dict): All computed metrics
        """
        # Merge params with base config
        config = {**base_config, **params}

        # Create fresh model and trainer
        model = self.model_factory(config)
        model.to(self.device)
        trainer = self.trainer_factory(model, config)

        # Train (silently)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                trainer.train(train_loader, val_loader, factor_returns, verbose=False)
            except Exception:
                # Training failed, return bad metric
                return float("-inf"), {"error": True}

        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_targets = []
        all_returns = []

        with torch.no_grad():
            for i, (macro_batch, market_context, targets) in enumerate(val_loader):
                macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                market_context = market_context.to(self.device)

                # Get predictions
                outputs = model(macro_batch, market_context, output_type="allocation")
                weights = outputs.cpu().numpy()

                # Compute portfolio returns
                batch_size = weights.shape[0]
                start_idx = i * config.get("batch_size", 32)
                end_idx = min(start_idx + batch_size, len(factor_returns))

                if end_idx <= start_idx:
                    continue

                batch_returns = factor_returns[start_idx:end_idx]
                if len(batch_returns) != batch_size:
                    batch_returns = batch_returns[:batch_size]

                port_returns = np.sum(weights * batch_returns, axis=1)

                all_preds.extend(weights.mean(axis=1))  # Average weight as signal
                all_targets.extend(targets.numpy())
                all_returns.extend(port_returns)

        if len(all_preds) == 0:
            return float("-inf"), {"error": True}

        predictions = np.array(all_preds)
        targets = np.array(all_targets)
        returns = np.array(all_returns)

        # Compute metrics
        val_metric = self._compute_metric(predictions, targets, returns)
        all_metrics = {
            "ic": np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0,
            "sharpe": (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(12),
            "mean_return": np.mean(returns),
        }

        return val_metric, all_metrics

    def tune_window(
        self,
        window: WalkForwardWindow,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        factor_returns_train: np.ndarray,
        factor_returns_test: np.ndarray,
        base_config: Dict[str, Any],
    ) -> TuningResult:
        """
        Tune hyperparameters for a single walk-forward window.

        :param window (WalkForwardWindow): Window definition
        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data (for tuning)
        :param test_loader (DataLoader): Test data (held-out)
        :param factor_returns_train (np.ndarray): Train factor returns
        :param factor_returns_test (np.ndarray): Test factor returns
        :param base_config (Dict): Base configuration

        :return result (TuningResult): Tuning result
        """
        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"TUNING WINDOW: {window.train_start} to {window.train_end}")
            print(f"Validation: {window.val_start} to {window.val_end}")
            print(f"{'=' * 60}")

        all_trials = []
        best_params = None
        best_metric = float("-inf")

        if OPTUNA_AVAILABLE and self.config.n_trials > 5:
            # Use Optuna for efficient search
            result = self._tune_with_optuna(
                train_loader, val_loader, factor_returns_train, base_config
            )
            best_params = result["best_params"]
            best_metric = result["best_value"]
            all_trials = result["trials"]
        else:
            # Random search fallback
            result = self._tune_with_random_search(
                train_loader, val_loader, factor_returns_train, base_config
            )
            best_params = result["best_params"]
            best_metric = result["best_value"]
            all_trials = result["trials"]

        if self.config.verbose:
            print(f"\nBest params: {best_params}")
            print(f"Best val metric: {best_metric:.4f}")

        # Retrain with best params on full train, evaluate on test
        test_metrics = self._evaluate_on_test(
            best_params, train_loader, val_loader, test_loader,
            factor_returns_train, factor_returns_test, base_config
        )

        if self.config.verbose:
            print(f"Test metrics: {test_metrics}")

        return TuningResult(
            window=window,
            best_params=best_params,
            best_val_metric=best_metric,
            test_metrics=test_metrics,
            n_trials_completed=len(all_trials),
            all_trials=all_trials,
        )

    def _tune_with_optuna(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray,
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Tune using Optuna Bayesian optimization.

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data
        :param factor_returns (np.ndarray): Factor returns
        :param base_config (Dict): Base config

        :return result (Dict): Best params and trials
        """
        def objective(trial: optuna.Trial) -> float:
            params = self.sampler.sample_optuna(trial)
            val_metric, _ = self._run_single_trial(
                params, train_loader, val_loader, factor_returns, base_config
            )

            # Report for pruning
            if self.config.use_pruning:
                trial.report(val_metric, 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return val_metric

        # Create study
        sampler = TPESampler(seed=self.config.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # Suppress Optuna logs if not verbose
        if not self.config.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.verbose,
        )

        # Extract results
        trials = [
            {
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
            }
            for t in study.trials
        ]

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": trials,
        }

    def _tune_with_random_search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray,
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Tune using random search (fallback).

        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data
        :param factor_returns (np.ndarray): Factor returns
        :param base_config (Dict): Base config

        :return result (Dict): Best params and trials
        """
        best_params = None
        best_value = float("-inf")
        trials = []
        no_improvement_count = 0

        for i in range(self.config.n_trials):
            params = self.sampler.sample_random()
            val_metric, all_metrics = self._run_single_trial(
                params, train_loader, val_loader, factor_returns, base_config
            )

            trials.append({
                "params": params,
                "value": val_metric,
                "metrics": all_metrics,
            })

            if val_metric > best_value:
                best_value = val_metric
                best_params = params
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if self.config.verbose and (i + 1) % 5 == 0:
                print(f"  Trial {i + 1}/{self.config.n_trials}: "
                      f"metric={val_metric:.4f}, best={best_value:.4f}")

            # Early stopping
            if no_improvement_count >= self.config.early_stopping_patience:
                if self.config.verbose:
                    print(f"  Early stopping at trial {i + 1}")
                break

        return {
            "best_params": best_params or {},
            "best_value": best_value,
            "trials": trials,
        }

    def _evaluate_on_test(
        self,
        best_params: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        factor_returns_train: np.ndarray,
        factor_returns_test: np.ndarray,
        base_config: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Retrain with best params and evaluate on test set.

        :param best_params (Dict): Best hyperparameters
        :param train_loader (DataLoader): Training data
        :param val_loader (DataLoader): Validation data
        :param test_loader (DataLoader): Test data
        :param factor_returns_train (np.ndarray): Train returns
        :param factor_returns_test (np.ndarray): Test returns
        :param base_config (Dict): Base configuration

        :return metrics (Dict): Test set metrics
        """
        # Merge configs
        config = {**base_config, **best_params}

        # Create and train model
        model = self.model_factory(config)
        model.to(self.device)
        trainer = self.trainer_factory(model, config)

        # Train on full train set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train(train_loader, val_loader, factor_returns_train, verbose=False)

        # Evaluate on test
        model.eval()
        all_preds = []
        all_targets = []
        all_returns = []

        with torch.no_grad():
            for i, (macro_batch, market_context, targets) in enumerate(test_loader):
                macro_batch = {k: v.to(self.device) for k, v in macro_batch.items()}
                market_context = market_context.to(self.device)

                outputs = model(macro_batch, market_context, output_type="allocation")
                weights = outputs.cpu().numpy()

                batch_size = weights.shape[0]
                start_idx = i * config.get("batch_size", 32)
                end_idx = min(start_idx + batch_size, len(factor_returns_test))

                if end_idx <= start_idx:
                    continue

                batch_returns = factor_returns_test[start_idx:end_idx]
                if len(batch_returns) != batch_size:
                    batch_returns = batch_returns[:batch_size]

                port_returns = np.sum(weights * batch_returns, axis=1)

                all_preds.extend(weights.mean(axis=1))
                all_targets.extend(targets.numpy())
                all_returns.extend(port_returns)

        if len(all_preds) == 0:
            return {"sharpe": 0.0, "ic": 0.0, "mean_return": 0.0}

        predictions = np.array(all_preds)
        targets = np.array(all_targets)
        returns = np.array(all_returns)

        return {
            "sharpe": (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(12),
            "ic": np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0,
            "mean_return": np.mean(returns),
            "total_return": np.prod(1 + returns) - 1,
            "max_drawdown": self._compute_max_drawdown(returns),
        }

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Compute maximum drawdown from returns.

        :param returns (np.ndarray): Portfolio returns

        :return max_dd (float): Maximum drawdown (negative)
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    def tune_all_windows(
        self,
        windows: List[WalkForwardWindow],
        data_creator: Callable,
        base_config: Dict[str, Any],
    ) -> List[TuningResult]:
        """
        Tune hyperparameters across all walk-forward windows.

        :param windows (List[WalkForwardWindow]): Walk-forward windows
        :param data_creator (Callable): Function to create data loaders for a window
        :param base_config (Dict): Base configuration

        :return results (List[TuningResult]): Results for all windows
        """
        self.results = []

        for i, window in enumerate(windows):
            if self.config.verbose:
                print(f"\n{'#' * 60}")
                print(f"WINDOW {i + 1}/{len(windows)}")
                print(f"{'#' * 60}")

            # Create data loaders for this window
            loaders = data_creator(window)
            train_loader = loaders["train"]
            val_loader = loaders["val"]
            test_loader = loaders["test"]
            factor_returns_train = loaders["factor_returns_train"]
            factor_returns_test = loaders["factor_returns_test"]

            # Tune this window
            result = self.tune_window(
                window,
                train_loader,
                val_loader,
                test_loader,
                factor_returns_train,
                factor_returns_test,
                base_config,
            )

            self.results.append(result)

        return self.results

    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all windows.

        :return summary (Dict): Aggregated metrics and analysis
        """
        if not self.results:
            return {}

        # Test metrics
        test_sharpes = [r.test_metrics.get("sharpe", 0) for r in self.results]
        test_ics = [r.test_metrics.get("ic", 0) for r in self.results]
        val_metrics = [r.best_val_metric for r in self.results]

        # Best params frequency analysis
        param_freq = {}
        for r in self.results:
            for k, v in r.best_params.items():
                if k not in param_freq:
                    param_freq[k] = []
                param_freq[k].append(v)

        # Compute param statistics
        param_stats = {}
        for k, values in param_freq.items():
            if isinstance(values[0], (int, float)):
                param_stats[k] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
            else:
                # Categorical - mode
                from collections import Counter
                param_stats[k] = Counter(values).most_common(1)[0]

        return {
            "avg_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),
            "avg_test_ic": np.mean(test_ics),
            "std_test_ic": np.std(test_ics),
            "avg_val_metric": np.mean(val_metrics),
            "num_windows": len(self.results),
            "total_trials": sum(r.n_trials_completed for r in self.results),
            "param_stats": param_stats,
        }

    def get_best_overall_params(self) -> Dict[str, Any]:
        """
        Get best params based on average performance across windows.

        Computes the average value of each parameter weighted by
        validation performance.

        :return best_params (Dict): Recommended parameters
        """
        if not self.results:
            return {}

        # Weight by validation metric
        weights = np.array([max(r.best_val_metric, 0.01) for r in self.results])
        weights = weights / weights.sum()

        best_params = {}
        for key in self.results[0].best_params.keys():
            values = [r.best_params.get(key) for r in self.results]

            if all(isinstance(v, (int, float)) for v in values if v is not None):
                # Numerical: weighted average
                numeric_vals = [v for v in values if v is not None]
                if numeric_vals:
                    weighted_avg = np.average(
                        numeric_vals,
                        weights=weights[:len(numeric_vals)]
                    )
                    # Round to appropriate precision
                    if all(isinstance(v, int) for v in numeric_vals):
                        best_params[key] = int(round(weighted_avg))
                    else:
                        best_params[key] = float(weighted_avg)
            else:
                # Categorical: mode weighted by performance
                from collections import Counter
                weighted_counts = Counter()
                for v, w in zip(values, weights):
                    if v is not None:
                        weighted_counts[v] += w
                if weighted_counts:
                    best_params[key] = weighted_counts.most_common(1)[0][0]

        return best_params

    def generate_report(self) -> str:
        """
        Generate a text report of tuning results.

        :return report (str): Formatted report
        """
        if not self.results:
            return "No tuning results available."

        lines = [
            "=" * 70,
            "WALK-FORWARD HYPERPARAMETER TUNING REPORT",
            "=" * 70,
            "",
        ]

        for i, result in enumerate(self.results):
            lines.append(f"Window {i + 1}: {result.window.train_start} to {result.window.train_end}")
            lines.append(f"  Trials: {result.n_trials_completed}")
            lines.append(f"  Best Val Metric: {result.best_val_metric:.4f}")
            lines.append(f"  Test Sharpe: {result.test_metrics.get('sharpe', 0):.4f}")
            lines.append(f"  Test IC: {result.test_metrics.get('ic', 0):.4f}")
            lines.append(f"  Best Params:")
            for k, v in result.best_params.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.6f}")
                else:
                    lines.append(f"    {k}: {v}")
            lines.append("")

        # Aggregated results
        agg = self.aggregate_results()
        lines.append("-" * 70)
        lines.append("AGGREGATED RESULTS:")
        lines.append(f"  Windows: {agg.get('num_windows', 0)}")
        lines.append(f"  Total Trials: {agg.get('total_trials', 0)}")
        lines.append(f"  Avg Test Sharpe: {agg.get('avg_test_sharpe', 0):.4f} "
                     f"+/- {agg.get('std_test_sharpe', 0):.4f}")
        lines.append(f"  Avg Test IC: {agg.get('avg_test_ic', 0):.4f} "
                     f"+/- {agg.get('std_test_ic', 0):.4f}")
        lines.append("")

        # Recommended params
        lines.append("RECOMMENDED PARAMETERS (weighted average):")
        best = self.get_best_overall_params()
        for k, v in best.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")

        lines.append("=" * 70)

        return "\n".join(lines)


def create_default_tuner(
    model_factory: Callable,
    trainer_factory: Callable,
    n_trials: int = 20,
    optimization_metric: str = "sharpe",
    device: Optional[torch.device] = None,
) -> WalkForwardTuner:
    """
    Create a tuner with sensible defaults.

    :param model_factory (Callable): Model creation function
    :param trainer_factory (Callable): Trainer creation function
    :param n_trials (int): Number of trials per window
    :param optimization_metric (str): Metric to optimize
    :param device (torch.device): Device

    :return tuner (WalkForwardTuner): Configured tuner
    """
    space = HyperparameterSpace(
        learning_rate=(1e-4, 5e-3, True),
        dropout=(0.4, 0.7),
        weight_decay=(1e-3, 5e-2, True),
        batch_size=[16, 32],
        tune_architecture=False,  # Keep architecture fixed for speed
        tune_epochs=True,
    )

    config = TuningConfig(
        n_trials=n_trials,
        optimization_metric=optimization_metric,
        early_stopping_patience=7,
        verbose=True,
    )

    return WalkForwardTuner(
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        space=space,
        config=config,
        device=device,
    )
