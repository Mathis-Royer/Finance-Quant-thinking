"""
Execution gate for controlling portfolio rebalancing.

This module implements the gated execution mechanism that compares
predicted allocation changes against a threshold calibrated to
transaction costs. If changes are below threshold, current allocation
is maintained; otherwise, new allocation is implemented.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class ExecutionGate(nn.Module):
    """
    Gated execution mechanism for portfolio rebalancing.

    Decouples prediction from execution decision by comparing
    the magnitude of proposed weight changes against a threshold.

    Supports:
    - Hard threshold (non-differentiable)
    - Soft threshold (differentiable for training)
    - Adaptive threshold based on volatility regime

    :param num_factors (int): Number of factor categories
    :param base_threshold (float): Base threshold for weight changes
    :param use_soft_threshold (bool): Use differentiable soft threshold
    :param temperature (float): Temperature for soft threshold
    """

    def __init__(
        self,
        num_factors: int = 6,
        base_threshold: float = 0.05,
        use_soft_threshold: bool = False,
        temperature: float = 10.0,
    ):
        """
        Initialize execution gate.

        :param num_factors (int): Number of factor categories
        :param base_threshold (float): Base threshold (5% = 0.05)
        :param use_soft_threshold (bool): Use differentiable threshold
        :param temperature (float): Softness of threshold
        """
        super().__init__()

        self.num_factors = num_factors
        self.base_threshold = base_threshold
        self.use_soft_threshold = use_soft_threshold
        self.temperature = temperature

        # Learnable threshold adjustment (optional)
        self.threshold_adjustment = nn.Parameter(torch.zeros(1))

        # Volatility-adaptive threshold network (optional)
        self.adaptive_net = nn.Sequential(
            nn.Linear(3, 8),  # VIX, credit_spread, yield_curve
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus(),  # Ensures positive output
        )

    def forward(
        self,
        new_weights: torch.Tensor,
        current_weights: torch.Tensor,
        market_context: Optional[torch.Tensor] = None,
        use_adaptive: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine whether to execute new allocation.

        :param new_weights (torch.Tensor): Predicted weights [batch, num_factors]
        :param current_weights (torch.Tensor): Current weights [batch, num_factors]
        :param market_context (Optional[torch.Tensor]): Market context for adaptive
        :param use_adaptive (bool): Use adaptive threshold

        :return executed_weights (torch.Tensor): Weights to implement
        :return execute_signal (torch.Tensor): Binary execution signal [batch]
        """
        # Compute total weight change (L1 norm)
        weight_change = torch.abs(new_weights - current_weights).sum(dim=-1)

        # Determine threshold
        if use_adaptive and market_context is not None:
            threshold = self._adaptive_threshold(market_context)
        else:
            threshold = self.base_threshold + torch.sigmoid(self.threshold_adjustment) * 0.1

        # Decision
        if self.use_soft_threshold:
            # Differentiable soft threshold using sigmoid
            execute_prob = torch.sigmoid(
                self.temperature * (weight_change - threshold)
            )
            execute_signal = execute_prob

            # Soft blend of weights
            execute_prob = execute_prob.unsqueeze(-1)
            executed_weights = (
                execute_prob * new_weights + (1 - execute_prob) * current_weights
            )
        else:
            # Hard threshold (non-differentiable)
            execute_signal = (weight_change > threshold).float()

            # Binary selection
            execute_mask = execute_signal.unsqueeze(-1)
            executed_weights = torch.where(
                execute_mask.bool().expand_as(new_weights),
                new_weights,
                current_weights,
            )

        return executed_weights, execute_signal

    def _adaptive_threshold(self, market_context: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive threshold based on market conditions.

        Higher volatility -> higher threshold (less frequent rebalancing)

        :param market_context (torch.Tensor): [batch, 3] (VIX, spread, curve)

        :return threshold (torch.Tensor): Adaptive threshold [batch]
        """
        # Normalize inputs
        normalized = market_context / torch.tensor(
            [30.0, 5.0, 2.0], device=market_context.device
        )

        # Compute threshold adjustment
        adjustment = self.adaptive_net(normalized).squeeze(-1)

        # Base threshold + learned adjustment
        threshold = self.base_threshold + adjustment * 0.05

        return threshold

    def compute_transaction_cost(
        self,
        new_weights: torch.Tensor,
        current_weights: torch.Tensor,
        cost_per_unit: float = 0.001,
    ) -> torch.Tensor:
        """
        Estimate transaction costs for a rebalance.

        :param new_weights (torch.Tensor): New weights
        :param current_weights (torch.Tensor): Current weights
        :param cost_per_unit (float): Cost per unit of weight change

        :return cost (torch.Tensor): Estimated transaction cost
        """
        turnover = torch.abs(new_weights - current_weights).sum(dim=-1)
        cost = turnover * cost_per_unit

        return cost

    def calibrate_threshold(
        self,
        historical_returns: np.ndarray,
        transaction_cost: float = 0.001,
        target_turnover: float = 0.1,
    ) -> float:
        """
        Calibrate threshold based on historical data.

        Finds threshold that achieves target turnover while
        accounting for transaction costs.

        :param historical_returns (np.ndarray): Historical factor returns
        :param transaction_cost (float): Cost per unit of turnover
        :param target_turnover (float): Target average turnover

        :return optimal_threshold (float): Calibrated threshold
        """
        # This would require simulation - simplified version
        # In practice, run backtest with different thresholds

        # Heuristic: threshold = target_turnover * cost_sensitivity
        cost_sensitivity = 2.0  # Higher means less sensitive to costs
        optimal_threshold = target_turnover / cost_sensitivity

        return optimal_threshold


class PortfolioManager:
    """
    Manages portfolio state and executes allocation changes.

    Tracks current weights, historical weights, and rebalancing history.

    :param num_factors (int): Number of factor categories
    :param initial_weights (Optional[np.ndarray]): Initial allocation
    """

    def __init__(
        self,
        num_factors: int = 6,
        initial_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize portfolio manager.

        :param num_factors (int): Number of factor categories
        :param initial_weights (Optional[np.ndarray]): Starting weights
        """
        self.num_factors = num_factors

        if initial_weights is None:
            # Equal weight initialization
            self.current_weights = np.ones(num_factors) / num_factors
        else:
            self.current_weights = initial_weights.copy()

        self.weight_history = [self.current_weights.copy()]
        self.rebalance_history = []
        self.turnover_history = []

    def update_weights(
        self,
        new_weights: np.ndarray,
        execute: bool,
        timestamp: Optional[str] = None,
    ) -> np.ndarray:
        """
        Update portfolio weights.

        :param new_weights (np.ndarray): Proposed new weights
        :param execute (bool): Whether to execute the change
        :param timestamp (Optional[str]): Timestamp for logging

        :return executed_weights (np.ndarray): Actual weights after decision
        """
        if execute:
            turnover = np.abs(new_weights - self.current_weights).sum()
            self.turnover_history.append(turnover)
            self.current_weights = new_weights.copy()
            self.rebalance_history.append({
                "timestamp": timestamp,
                "action": "rebalance",
                "turnover": turnover,
            })
        else:
            self.turnover_history.append(0.0)
            self.rebalance_history.append({
                "timestamp": timestamp,
                "action": "hold",
                "turnover": 0.0,
            })

        self.weight_history.append(self.current_weights.copy())

        return self.current_weights.copy()

    def get_current_weights(self) -> np.ndarray:
        """
        Get current portfolio weights.

        :return weights (np.ndarray): Current weights
        """
        return self.current_weights.copy()

    def get_statistics(self) -> dict:
        """
        Get portfolio management statistics.

        :return stats (dict): Statistics dictionary
        """
        return {
            "total_rebalances": sum(
                1 for r in self.rebalance_history if r["action"] == "rebalance"
            ),
            "total_holds": sum(
                1 for r in self.rebalance_history if r["action"] == "hold"
            ),
            "avg_turnover": np.mean(self.turnover_history) if self.turnover_history else 0,
            "total_turnover": sum(self.turnover_history),
            "rebalance_rate": (
                sum(1 for r in self.rebalance_history if r["action"] == "rebalance")
                / len(self.rebalance_history)
                if self.rebalance_history
                else 0
            ),
        }

    def reset(self, initial_weights: Optional[np.ndarray] = None) -> None:
        """
        Reset portfolio to initial state.

        :param initial_weights (Optional[np.ndarray]): New initial weights
        """
        if initial_weights is None:
            self.current_weights = np.ones(self.num_factors) / self.num_factors
        else:
            self.current_weights = initial_weights.copy()

        self.weight_history = [self.current_weights.copy()]
        self.rebalance_history = []
        self.turnover_history = []
