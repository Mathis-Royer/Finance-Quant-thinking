"""
Synthetic data generator for testing the factor allocation strategy.

This module generates realistic-looking macroeconomic and factor return data
for development and testing purposes. The synthetic data mimics:
- Economic cycles
- Factor rotations
- Macro-factor correlations
- Data publication patterns

WARNING: This data is ARTIFICIAL and should ONLY be used to verify code functionality.
Real backtesting requires actual historical data from point-in-time databases.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .data_loader import (
    MacroIndicator,
    MacroCategory,
    PublicationType,
    Region,
    MacroDataLoader,
)


@dataclass
class EconomicRegime:
    """
    Definition of an economic regime for data generation.

    :param name (str): Regime name (expansion, contraction, recovery, etc.)
    :param pmi_mean (float): Mean PMI level
    :param inflation_mean (float): Mean inflation level
    :param cyclical_premium (float): Expected cyclical vs defensive premium
    :param value_premium (float): Expected value vs growth premium
    :param volatility_mult (float): Volatility multiplier
    """
    name: str
    pmi_mean: float
    inflation_mean: float
    cyclical_premium: float
    value_premium: float
    volatility_mult: float


class SyntheticDataGenerator:
    """
    Generate synthetic macroeconomic and factor return data.

    The generator creates data with realistic properties:
    - Economic cycles with regime transitions
    - Correlated macro indicators
    - Factor returns that respond to macro conditions
    - Proper publication patterns with surprises

    :param region (Region): Target region
    :param seed (int): Random seed for reproducibility
    """

    # Economic regimes with typical characteristics
    REGIMES: List[EconomicRegime] = [
        EconomicRegime("expansion", 55.0, 2.5, 0.02, -0.01, 0.8),
        EconomicRegime("late_cycle", 52.0, 3.5, 0.01, 0.02, 1.0),
        EconomicRegime("contraction", 45.0, 1.5, -0.03, 0.01, 1.5),
        EconomicRegime("recovery", 50.0, 1.0, 0.04, 0.03, 1.2),
        EconomicRegime("stagflation", 48.0, 5.0, -0.02, 0.04, 1.8),
    ]

    # Transition probabilities between regimes
    TRANSITION_MATRIX = np.array([
        [0.85, 0.10, 0.02, 0.02, 0.01],  # From expansion
        [0.05, 0.70, 0.20, 0.03, 0.02],  # From late_cycle
        [0.05, 0.05, 0.60, 0.28, 0.02],  # From contraction
        [0.30, 0.05, 0.05, 0.58, 0.02],  # From recovery
        [0.10, 0.05, 0.15, 0.05, 0.65],  # From stagflation
    ])

    def __init__(self, region: Region = Region.US, seed: int = 42):
        """
        Initialize the synthetic data generator.

        :param region (Region): Target region
        :param seed (int): Random seed for reproducibility
        """
        self.region = region
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.loader = MacroDataLoader(region)

    def generate_dataset(
        self,
        start_date: str = "2000-01-01",
        end_date: str = "2024-12-31",
        freq: str = "W",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate a complete synthetic dataset.

        :param start_date (str): Start date string
        :param end_date (str): End date string
        :param freq (str): Frequency ('D' for daily, 'W' for weekly)

        :return macro_data (pd.DataFrame): Macroeconomic token data
        :return factor_data (pd.DataFrame): Factor return data
        :return market_data (pd.DataFrame): Market context data (VIX, spreads, etc.)
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_periods = len(dates)

        # Generate economic regime sequence
        regimes = self._generate_regime_sequence(n_periods)

        # Generate macro data
        macro_data = self._generate_macro_data(dates, regimes)

        # Generate factor returns (correlated with macro conditions)
        factor_data = self._generate_factor_returns(dates, regimes)

        # Generate market context
        market_data = self._generate_market_context(dates, regimes)

        return macro_data, factor_data, market_data

    def _generate_regime_sequence(self, n_periods: int) -> List[int]:
        """
        Generate sequence of economic regimes using Markov chain.

        :param n_periods (int): Number of time periods

        :return regimes (List[int]): Sequence of regime indices
        """
        regimes = [0]  # Start in expansion
        for _ in range(n_periods - 1):
            current = regimes[-1]
            next_regime = self.rng.choice(
                len(self.REGIMES),
                p=self.TRANSITION_MATRIX[current],
            )
            regimes.append(next_regime)
        return regimes

    def _generate_macro_data(
        self,
        dates: pd.DatetimeIndex,
        regimes: List[int],
    ) -> pd.DataFrame:
        """
        Generate synthetic macroeconomic data.

        :param dates (pd.DatetimeIndex): Date index
        :param regimes (List[int]): Regime sequence

        :return macro_data (pd.DataFrame): Generated macro data
        """
        indicators = self.loader.indicators
        records = []

        for idx, date in enumerate(dates):
            regime = self.REGIMES[regimes[idx]]

            for indicator in indicators:
                # Skip based on periodicity
                if not self._should_publish(indicator, date):
                    continue

                # Generate values based on indicator type and regime
                value, expected, surprise = self._generate_indicator_value(
                    indicator, regime, idx
                )

                # Calculate normalized value and MA5
                normalized = self._normalize_value(indicator, value)
                ma5 = value + self.rng.normal(0, 0.5)

                # Determine publication type
                pub_type = self._get_publication_type(indicator, date)

                records.append({
                    "timestamp": date,
                    "indicator": indicator.name,
                    "category": indicator.category.value,
                    "region": indicator.region.value,
                    "importance": indicator.importance,
                    "publication_type": pub_type.value,
                    "value": value,
                    "expected": expected,
                    "normalized_value": normalized,
                    "surprise": surprise,
                    "ma5": ma5,
                })

        return pd.DataFrame(records)

    def _generate_indicator_value(
        self,
        indicator: MacroIndicator,
        regime: EconomicRegime,
        time_idx: int,
    ) -> Tuple[float, float, float]:
        """
        Generate a single indicator value based on regime.

        :param indicator (MacroIndicator): Indicator definition
        :param regime (EconomicRegime): Current economic regime
        :param time_idx (int): Time index for autocorrelation

        :return value (float): Actual value
        :return expected (float): Expected/consensus value
        :return surprise (float): Standardized surprise
        """
        # Base value depends on category
        if indicator.category == MacroCategory.ECONOMIC_ACTIVITY:
            base = regime.pmi_mean
            std = 3.0
        elif indicator.category == MacroCategory.EMPLOYMENT:
            if "Unemployment" in indicator.name:
                base = 6.0 - (regime.pmi_mean - 50) / 5
                std = 0.5
            else:
                base = (regime.pmi_mean - 50) * 50  # NFP-like
                std = 80.0
        elif indicator.category == MacroCategory.INFLATION:
            base = regime.inflation_mean
            std = 0.3
        elif indicator.category == MacroCategory.SENTIMENT:
            base = regime.pmi_mean * 2  # Sentiment indices tend to be higher
            std = 5.0
        else:
            base = 50.0
            std = 5.0

        # Add autocorrelation and noise
        noise = self.rng.normal(0, std * regime.volatility_mult)
        value = base + noise

        # Expected value is the base plus some noise
        expected_noise = self.rng.normal(0, std * 0.3)
        expected = base + expected_noise

        # Standardized surprise
        surprise = (value - expected) / std if std > 0 else 0.0

        return value, expected, surprise

    def _normalize_value(self, indicator: MacroIndicator, value: float) -> float:
        """
        Normalize value to historical percentile (0-1).

        :param indicator (MacroIndicator): Indicator definition
        :param value (float): Raw value

        :return normalized (float): Normalized value (0-1)
        """
        # Simplified normalization using typical ranges
        if indicator.category == MacroCategory.ECONOMIC_ACTIVITY:
            return (value - 30) / 40  # PMI range ~30-70
        elif indicator.category == MacroCategory.INFLATION:
            return (value + 2) / 10  # Inflation range ~-2% to 8%
        else:
            return 0.5 + self.rng.normal(0, 0.2)

    def _should_publish(self, indicator: MacroIndicator, date: pd.Timestamp) -> bool:
        """
        Determine if indicator should be published on this date.

        :param indicator (MacroIndicator): Indicator definition
        :param date (pd.Timestamp): Current date

        :return should_publish (bool): Whether to publish
        """
        if indicator.periodicity == "monthly":
            # Monthly data typically released early in month
            return date.day <= 15
        elif indicator.periodicity == "weekly":
            # Weekly data on specific weekday
            return date.dayofweek == 3  # Thursday
        elif indicator.periodicity == "quarterly":
            return date.month in [1, 4, 7, 10] and date.day <= 15
        else:
            # Irregular - random
            return self.rng.random() < 0.1

    def _get_publication_type(
        self,
        indicator: MacroIndicator,
        date: pd.Timestamp,
    ) -> PublicationType:
        """
        Determine publication type based on indicator and date.

        :param indicator (MacroIndicator): Indicator definition
        :param date (pd.Timestamp): Publication date

        :return pub_type (PublicationType): Type of publication
        """
        # Simplified logic - mostly consensus
        r = self.rng.random()
        if r < 0.6:
            return PublicationType.CONSENSUS
        elif r < 0.8:
            return PublicationType.ESTIMATE
        elif r < 0.9:
            return PublicationType.CONSENSUS_REVISION
        else:
            return PublicationType.ESTIMATE_REVISION

    def _generate_factor_returns(
        self,
        dates: pd.DatetimeIndex,
        regimes: List[int],
    ) -> pd.DataFrame:
        """
        Generate synthetic factor returns correlated with regimes.

        :param dates (pd.DatetimeIndex): Date index
        :param regimes (List[int]): Regime sequence

        :return factor_data (pd.DataFrame): Generated factor returns
        """
        n_periods = len(dates)
        base_vol = 0.02  # Weekly volatility

        records = []
        for idx, date in enumerate(dates):
            regime = self.REGIMES[regimes[idx]]

            # Market return (common factor)
            market = self.rng.normal(0.001, base_vol * regime.volatility_mult)

            # Factor-specific returns with regime dependencies
            cyclical = market + regime.cyclical_premium / 52 + self.rng.normal(
                0, base_vol * 0.5
            )
            defensive = market - regime.cyclical_premium / 52 + self.rng.normal(
                0, base_vol * 0.4
            )
            value = market + regime.value_premium / 52 + self.rng.normal(
                0, base_vol * 0.6
            )
            growth = market - regime.value_premium / 52 + self.rng.normal(
                0, base_vol * 0.7
            )
            quality = market + self.rng.normal(0.0002, base_vol * 0.3)
            momentum = market + self.rng.normal(0.0003, base_vol * 0.8)

            records.append({
                "timestamp": date,
                "cyclical": cyclical,
                "defensive": defensive,
                "value": value,
                "growth": growth,
                "quality": quality,
                "momentum": momentum,
                "market": market,
                "regime": regime.name,
            })

        return pd.DataFrame(records)

    def _generate_market_context(
        self,
        dates: pd.DatetimeIndex,
        regimes: List[int],
    ) -> pd.DataFrame:
        """
        Generate market context indicators (VIX, spreads, etc.).

        :param dates (pd.DatetimeIndex): Date index
        :param regimes (List[int]): Regime sequence

        :return market_data (pd.DataFrame): Market context data
        """
        records = []
        vix_prev = 15.0
        spread_prev = 3.0
        curve_prev = 1.5

        for idx, date in enumerate(dates):
            regime = self.REGIMES[regimes[idx]]

            # VIX mean-reverts with regime dependency
            vix_target = 12 + 8 * (regime.volatility_mult - 0.8)
            vix = 0.9 * vix_prev + 0.1 * vix_target + self.rng.normal(0, 2)
            vix = max(10, min(80, vix))

            # Credit spread widens in contraction
            spread_target = 2.5 + 3 * (regime.volatility_mult - 0.8)
            spread = 0.95 * spread_prev + 0.05 * spread_target + self.rng.normal(0, 0.2)
            spread = max(1, min(15, spread))

            # Yield curve flattens in late cycle, steepens in recovery
            if regime.name == "late_cycle":
                curve_target = 0.5
            elif regime.name == "recovery":
                curve_target = 2.5
            else:
                curve_target = 1.5
            curve = 0.9 * curve_prev + 0.1 * curve_target + self.rng.normal(0, 0.15)
            curve = max(-1, min(3, curve))

            records.append({
                "timestamp": date,
                "vix": vix,
                "credit_spread": spread,
                "yield_curve": curve,
                "regime": regime.name,
            })

            vix_prev = vix
            spread_prev = spread
            curve_prev = curve

        return pd.DataFrame(records)

    def generate_train_test_split(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        train_end: str,
        val_end: str,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Split data into train/validation/test sets (walk-forward style).

        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor data
        :param market_data (pd.DataFrame): Market context data
        :param train_end (str): End date for training
        :param val_end (str): End date for validation

        :return splits (Dict): Dictionary with 'train', 'val', 'test' keys
        """
        train_mask = macro_data["timestamp"] <= train_end
        val_mask = (macro_data["timestamp"] > train_end) & (
            macro_data["timestamp"] <= val_end
        )
        test_mask = macro_data["timestamp"] > val_end

        factor_train_mask = factor_data["timestamp"] <= train_end
        factor_val_mask = (factor_data["timestamp"] > train_end) & (
            factor_data["timestamp"] <= val_end
        )
        factor_test_mask = factor_data["timestamp"] > val_end

        market_train_mask = market_data["timestamp"] <= train_end
        market_val_mask = (market_data["timestamp"] > train_end) & (
            market_data["timestamp"] <= val_end
        )
        market_test_mask = market_data["timestamp"] > val_end

        return {
            "train": (
                macro_data[train_mask],
                factor_data[factor_train_mask],
                market_data[market_train_mask],
            ),
            "val": (
                macro_data[val_mask],
                factor_data[factor_val_mask],
                market_data[market_val_mask],
            ),
            "test": (
                macro_data[test_mask],
                factor_data[factor_test_mask],
                market_data[market_test_mask],
            ),
        }

    def create_binary_target(
        self,
        factor_data: pd.DataFrame,
        horizon_weeks: int = 4,
    ) -> pd.DataFrame:
        """
        Create binary classification target: cyclical vs defensive outperformance.

        :param factor_data (pd.DataFrame): Factor return data
        :param horizon_weeks (int): Forward horizon in weeks

        :return targets (pd.DataFrame): DataFrame with target labels
        """
        factor_data = factor_data.copy()

        # Calculate forward returns
        factor_data["cyclical_fwd"] = (
            factor_data["cyclical"].rolling(horizon_weeks).sum().shift(-horizon_weeks)
        )
        factor_data["defensive_fwd"] = (
            factor_data["defensive"].rolling(horizon_weeks).sum().shift(-horizon_weeks)
        )

        # Binary target: 1 if cyclical outperforms defensive
        factor_data["target"] = (
            factor_data["cyclical_fwd"] > factor_data["defensive_fwd"]
        ).astype(int)

        # Relative outperformance score for regression
        factor_data["target_score"] = (
            factor_data["cyclical_fwd"] - factor_data["defensive_fwd"]
        )

        return factor_data.dropna(subset=["target"])
