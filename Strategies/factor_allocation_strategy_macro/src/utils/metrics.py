"""
Performance metrics for strategy evaluation.

This module provides metrics for evaluating:
- Classification performance (IC, accuracy, AUC)
- Portfolio performance (Sharpe, returns, drawdown)
- Model diagnostics (calibration, feature importance)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    """
    Complete performance report.

    :param sharpe_ratio (float): Annualized Sharpe ratio
    :param total_return (float): Cumulative return
    :param annualized_return (float): Annualized return
    :param volatility (float): Annualized volatility
    :param max_drawdown (float): Maximum drawdown
    :param win_rate (float): Percentage of positive periods
    :param information_coefficient (float): IC with targets
    :param accuracy (float): Classification accuracy
    :param auc (float): Area under ROC curve
    """
    sharpe_ratio: float
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    win_rate: float
    information_coefficient: float
    accuracy: float
    auc: float


class PerformanceMetrics:
    """
    Calculate performance metrics for strategy evaluation.

    Supports both classification metrics (for binary targets)
    and portfolio metrics (for return-based evaluation).
    """

    @staticmethod
    def information_coefficient(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Calculate Information Coefficient (rank correlation).

        IC measures the correlation between predicted and actual rankings.
        Key metric for factor timing strategies.

        :param predictions (np.ndarray): Model predictions
        :param targets (np.ndarray): Actual outcomes

        :return ic (float): Information coefficient
        """
        if len(predictions) < 2:
            return 0.0

        from scipy.stats import spearmanr

        ic, _ = spearmanr(predictions, targets)
        return ic if not np.isnan(ic) else 0.0

    @staticmethod
    def accuracy(
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        Calculate classification accuracy.

        :param predictions (np.ndarray): Predicted probabilities
        :param targets (np.ndarray): Binary targets

        :return accuracy (float): Classification accuracy
        """
        binary_preds = (predictions > threshold).astype(int)
        return (binary_preds == targets).mean()

    @staticmethod
    def auc_score(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Calculate Area Under ROC Curve.

        :param predictions (np.ndarray): Predicted probabilities
        :param targets (np.ndarray): Binary targets

        :return auc (float): AUC score
        """
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(targets, predictions)
        except (ValueError, ImportError):
            return 0.5

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 52,
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        :param returns (np.ndarray): Period returns
        :param risk_free_rate (float): Annual risk-free rate
        :param periods_per_year (int): Number of periods per year

        :return sharpe (float): Annualized Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        annualized_sharpe = sharpe * np.sqrt(periods_per_year)

        return annualized_sharpe

    @staticmethod
    def maximum_drawdown(cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        :param cumulative_returns (np.ndarray): Cumulative return series

        :return max_dd (float): Maximum drawdown (negative number)
        """
        if len(cumulative_returns) == 0:
            return 0.0

        # Convert to wealth path if necessary
        if cumulative_returns[0] < 1:
            wealth = 1 + cumulative_returns
        else:
            wealth = cumulative_returns

        # Running maximum
        running_max = np.maximum.accumulate(wealth)

        # Drawdown
        drawdown = wealth / running_max - 1

        return drawdown.min()

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Calculate percentage of positive return periods.

        :param returns (np.ndarray): Period returns

        :return win_rate (float): Fraction of positive periods
        """
        if len(returns) == 0:
            return 0.0
        return (returns > 0).mean()

    @staticmethod
    def annualized_return(
        total_return: float,
        num_periods: int,
        periods_per_year: int = 52,
    ) -> float:
        """
        Calculate annualized return from total return.

        :param total_return (float): Total cumulative return
        :param num_periods (int): Number of periods
        :param periods_per_year (int): Periods per year

        :return annualized (float): Annualized return
        """
        if num_periods == 0:
            return 0.0

        years = num_periods / periods_per_year
        if years == 0:
            return 0.0

        # Handle negative total returns
        if total_return <= -1:
            return -1.0

        return (1 + total_return) ** (1 / years) - 1

    @staticmethod
    def annualized_volatility(
        returns: np.ndarray,
        periods_per_year: int = 52,
    ) -> float:
        """
        Calculate annualized volatility.

        :param returns (np.ndarray): Period returns
        :param periods_per_year (int): Periods per year

        :return volatility (float): Annualized volatility
        """
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(periods_per_year)

    @staticmethod
    def hit_ratio(
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """
        Calculate directional hit ratio.

        Measures how often the model correctly predicts direction.

        :param predictions (np.ndarray): Predicted scores
        :param targets (np.ndarray): Actual outcomes

        :return hit_ratio (float): Fraction of correct direction predictions
        """
        pred_direction = (predictions > threshold).astype(int)
        actual_direction = (targets > threshold).astype(int)
        return (pred_direction == actual_direction).mean()

    @staticmethod
    def calculate_portfolio_returns(
        weights: np.ndarray,
        factor_returns: np.ndarray,
        transaction_costs: float = 0.001,
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate portfolio returns with transaction costs.

        :param weights (np.ndarray): Portfolio weights [T, num_factors]
        :param factor_returns (np.ndarray): Factor returns [T, num_factors]
        :param transaction_costs (float): Cost per unit of turnover

        :return returns (np.ndarray): Portfolio returns [T]
        :return total_costs (float): Total transaction costs
        """
        # Portfolio returns before costs
        gross_returns = (weights * factor_returns).sum(axis=1)

        # Calculate turnover and costs
        weight_changes = np.abs(np.diff(weights, axis=0)).sum(axis=1)
        turnover = np.concatenate([[0], weight_changes])
        costs = turnover * transaction_costs

        # Net returns
        net_returns = gross_returns - costs

        return net_returns, costs.sum()

    @classmethod
    def full_evaluation(
        cls,
        predictions: np.ndarray,
        targets: np.ndarray,
        portfolio_returns: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        periods_per_year: int = 12,
    ) -> PerformanceReport:
        """
        Generate complete performance report.

        :param predictions (np.ndarray): Model predictions
        :param targets (np.ndarray): Actual outcomes
        :param portfolio_returns (Optional[np.ndarray]): Strategy returns
        :param benchmark_returns (Optional[np.ndarray]): Benchmark returns
        :param periods_per_year (int): Number of periods per year (12=monthly, 52=weekly)

        :return report (PerformanceReport): Complete performance report
        """
        # Classification metrics
        ic = cls.information_coefficient(predictions, targets)
        acc = cls.accuracy(predictions, targets)
        auc = cls.auc_score(predictions, targets)

        # Portfolio metrics (if available)
        if portfolio_returns is not None:
            sharpe = cls.sharpe_ratio(portfolio_returns, periods_per_year=periods_per_year)
            total_ret = (1 + portfolio_returns).prod() - 1
            ann_ret = cls.annualized_return(total_ret, len(portfolio_returns), periods_per_year)
            vol = cls.annualized_volatility(portfolio_returns, periods_per_year)
            max_dd = cls.maximum_drawdown(np.cumprod(1 + portfolio_returns))
            win = cls.win_rate(portfolio_returns)
        else:
            sharpe = total_ret = ann_ret = vol = max_dd = win = 0.0

        return PerformanceReport(
            sharpe_ratio=sharpe,
            total_return=total_ret,
            annualized_return=ann_ret,
            volatility=vol,
            max_drawdown=max_dd,
            win_rate=win,
            information_coefficient=ic,
            accuracy=acc,
            auc=auc,
        )

    @staticmethod
    def compare_to_baseline(
        strategy_returns: np.ndarray,
        baseline_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compare strategy to baseline.

        :param strategy_returns (np.ndarray): Strategy returns
        :param baseline_returns (np.ndarray): Baseline returns

        :return comparison (Dict): Comparison metrics
        """
        strategy_sharpe = PerformanceMetrics.sharpe_ratio(strategy_returns)
        baseline_sharpe = PerformanceMetrics.sharpe_ratio(baseline_returns)

        strategy_total = (1 + strategy_returns).prod() - 1
        baseline_total = (1 + baseline_returns).prod() - 1

        # Excess returns
        excess = strategy_returns - baseline_returns

        return {
            "strategy_sharpe": strategy_sharpe,
            "baseline_sharpe": baseline_sharpe,
            "sharpe_improvement": strategy_sharpe - baseline_sharpe,
            "strategy_return": strategy_total,
            "baseline_return": baseline_total,
            "excess_return": strategy_total - baseline_total,
            "information_ratio": (
                np.mean(excess) / np.std(excess) * np.sqrt(52)
                if np.std(excess) > 0
                else 0.0
            ),
            "outperformance_rate": (excess > 0).mean(),
        }

    @staticmethod
    def rolling_metrics(
        returns: np.ndarray,
        window: int = 52,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate rolling performance metrics.

        :param returns (np.ndarray): Period returns
        :param window (int): Rolling window size

        :return metrics (Dict): Dictionary of rolling metrics
        """
        n = len(returns)
        if n < window:
            return {}

        rolling_sharpe = np.zeros(n - window + 1)
        rolling_vol = np.zeros(n - window + 1)
        rolling_return = np.zeros(n - window + 1)

        for i in range(n - window + 1):
            window_returns = returns[i : i + window]
            rolling_sharpe[i] = PerformanceMetrics.sharpe_ratio(window_returns)
            rolling_vol[i] = PerformanceMetrics.annualized_volatility(window_returns)
            rolling_return[i] = (1 + window_returns).prod() - 1

        return {
            "rolling_sharpe": rolling_sharpe,
            "rolling_volatility": rolling_vol,
            "rolling_return": rolling_return,
        }
