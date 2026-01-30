"""
Feature engineering for macroeconomic data.

This module transforms raw macro data into features suitable for
both flat models (logistic regression, gradient boosting) and
sequential models (LSTM, Transformer).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..data.data_loader import (
    MacroDataLoader,
    MacroToken,
    MacroCategory,
    PublicationType,
    Region,
)


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.

    :param sequence_length (int): Number of tokens in sequence
    :param include_momentum (bool): Include recent factor momentum
    :param include_market_context (bool): Include VIX, spreads, etc.
    :param normalize_features (bool): Standardize features
    :param aggregation_windows (List[int]): Windows for aggregation (weeks)
    """
    sequence_length: int = 50
    include_momentum: bool = True
    include_market_context: bool = True
    normalize_features: bool = True
    aggregation_windows: List[int] = None

    def __post_init__(self):
        if self.aggregation_windows is None:
            self.aggregation_windows = [1, 4, 12]


class FeatureEngineer:
    """
    Feature engineering pipeline for macro data.

    Transforms raw data into:
    - Flat features for sklearn models
    - Sequential features for LSTM
    - Token-based features for Transformer

    :param config (FeatureConfig): Feature configuration
    :param region (Region): Target region
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        region: Region = Region.US,
    ):
        """
        Initialize feature engineer.

        :param config (FeatureConfig): Feature configuration
        :param region (Region): Target region
        """
        self.config = config or FeatureConfig()
        self.region = region
        self.loader = MacroDataLoader(region, self.config.sequence_length)

        # Feature statistics for normalization
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        # Mapping dictionaries
        self.indicator_to_idx: Dict[str, int] = {}
        self.category_to_idx: Dict[str, int] = {}
        self.pub_type_to_idx: Dict[str, int] = {}
        self.region_to_idx: Dict[str, int] = {}

        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build categorical to index mappings."""
        # Indicators
        for idx, ind in enumerate(self.loader.indicators):
            self.indicator_to_idx[ind.name] = idx

        # Categories
        for idx, cat in enumerate(MacroCategory):
            self.category_to_idx[cat.value] = idx

        # Publication types
        for idx, ptype in enumerate(PublicationType):
            self.pub_type_to_idx[ptype.value] = idx

        # Regions
        for idx, reg in enumerate(Region):
            self.region_to_idx[reg.value] = idx

    def create_flat_features(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> np.ndarray:
        """
        Create flat feature vector for sklearn models.

        Aggregates macro data into summary statistics.

        :param macro_data (pd.DataFrame): Macro token data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param as_of_date (pd.Timestamp): Point-in-time date

        :return features (np.ndarray): Feature vector
        """
        features = []

        # Filter to available data
        macro_available = macro_data[macro_data["timestamp"] <= as_of_date]
        factor_available = factor_data[factor_data["timestamp"] <= as_of_date]
        market_available = market_data[market_data["timestamp"] <= as_of_date]

        # 1. Aggregate macro features by category
        for window in self.config.aggregation_windows:
            window_data = macro_available.tail(window * 4)  # Assuming weekly

            for cat in MacroCategory:
                cat_data = window_data[window_data["category"] == cat.value]

                if len(cat_data) > 0:
                    # Mean and std of surprises
                    features.append(cat_data["surprise"].mean())
                    features.append(cat_data["surprise"].std())

                    # Mean normalized value
                    features.append(cat_data["normalized_value"].mean())

                    # Count of releases
                    features.append(len(cat_data) / window)
                else:
                    features.extend([0.0, 0.0, 0.5, 0.0])

        # 2. Recent factor momentum
        if self.config.include_momentum:
            for window in self.config.aggregation_windows:
                factor_window = factor_available.tail(window)

                if len(factor_window) > 0:
                    features.append(factor_window["cyclical"].sum())
                    features.append(factor_window["defensive"].sum())
                    features.append(factor_window["value"].sum())
                    features.append(factor_window["growth"].sum())
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])

        # 3. Market context
        if self.config.include_market_context and len(market_available) > 0:
            latest_market = market_available.iloc[-1]
            features.append(latest_market.get("vix", 15.0) / 30.0)
            features.append(latest_market.get("credit_spread", 3.0) / 5.0)
            features.append(latest_market.get("yield_curve", 1.5) / 2.0)

            # Market context changes
            if len(market_available) >= 4:
                market_4w_ago = market_available.iloc[-4]
                features.append(
                    (latest_market.get("vix", 15) - market_4w_ago.get("vix", 15)) / 10
                )
                features.append(
                    (latest_market.get("credit_spread", 3) - market_4w_ago.get("credit_spread", 3)) / 2
                )
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.5, 0.6, 0.75, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def create_sequence_features(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequential features for LSTM.

        :param macro_data (pd.DataFrame): Macro token data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param as_of_date (pd.Timestamp): Point-in-time date

        :return sequence (np.ndarray): [seq_len, features]
        :return market_context (np.ndarray): [3]
        """
        # Filter and sort
        macro_available = macro_data[macro_data["timestamp"] <= as_of_date]
        macro_available = macro_available.sort_values("timestamp", ascending=False)
        macro_sequence = macro_available.head(self.config.sequence_length)

        # Build feature matrix for each token
        seq_features = []

        for _, row in macro_sequence.iterrows():
            token_features = [
                row.get("normalized_value", 0.0),
                row.get("surprise", 0.0),
                row.get("ma5", 0.0),
                row.get("importance", 2) / 3.0,
                self.category_to_idx.get(row.get("category", ""), 0) / len(MacroCategory),
                self.pub_type_to_idx.get(row.get("publication_type", ""), 0) / len(PublicationType),
            ]
            seq_features.append(token_features)

        # Pad if necessary
        while len(seq_features) < self.config.sequence_length:
            seq_features.append([0.0] * 6)

        # Market context
        market_available = market_data[market_data["timestamp"] <= as_of_date]
        if len(market_available) > 0:
            latest = market_available.iloc[-1]
            market_context = np.array([
                latest.get("vix", 15.0),
                latest.get("credit_spread", 3.0),
                latest.get("yield_curve", 1.5),
            ], dtype=np.float32)
        else:
            market_context = np.array([15.0, 3.0, 1.5], dtype=np.float32)

        return np.array(seq_features, dtype=np.float32), market_context

    def create_transformer_batch(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Create batch for Transformer model.

        :param macro_data (pd.DataFrame): Macro token data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param as_of_date (pd.Timestamp): Point-in-time date

        :return macro_batch (Dict[str, np.ndarray]): Token features
        :return market_context (np.ndarray): Market context [3]
        """
        # Filter and sort
        macro_available = macro_data[macro_data["timestamp"] <= as_of_date]
        macro_available = macro_available.sort_values("timestamp", ascending=False)
        macro_sequence = macro_available.head(self.config.sequence_length)

        # Initialize arrays
        seq_len = min(len(macro_sequence), self.config.sequence_length)
        actual_len = len(macro_sequence)

        indicator_ids = np.zeros(self.config.sequence_length, dtype=np.int64)
        pub_type_ids = np.zeros(self.config.sequence_length, dtype=np.int64)
        category_ids = np.zeros(self.config.sequence_length, dtype=np.int64)
        country_ids = np.zeros(self.config.sequence_length, dtype=np.int64)
        importance = np.zeros(self.config.sequence_length, dtype=np.float32)
        days_offset = np.zeros(self.config.sequence_length, dtype=np.float32)
        normalized_value = np.zeros(self.config.sequence_length, dtype=np.float32)
        surprise = np.zeros(self.config.sequence_length, dtype=np.float32)
        ma5 = np.zeros(self.config.sequence_length, dtype=np.float32)

        for i, (_, row) in enumerate(macro_sequence.iterrows()):
            if i >= self.config.sequence_length:
                break

            indicator_ids[i] = self.indicator_to_idx.get(row.get("indicator", ""), 0)
            pub_type_ids[i] = self.pub_type_to_idx.get(row.get("publication_type", ""), 0)
            category_ids[i] = self.category_to_idx.get(row.get("category", ""), 0)
            country_ids[i] = self.region_to_idx.get(row.get("region", ""), 0)
            importance[i] = row.get("importance", 2)
            days_offset[i] = (row.get("timestamp", as_of_date) - as_of_date).days
            normalized_value[i] = row.get("normalized_value", 0.0)
            surprise[i] = row.get("surprise", 0.0)
            ma5[i] = row.get("ma5", 0.0)

        macro_batch = {
            "indicator_ids": indicator_ids,
            "pub_type_ids": pub_type_ids,
            "category_ids": category_ids,
            "country_ids": country_ids,
            "importance": importance,
            "days_offset": days_offset,
            "normalized_value": normalized_value,
            "surprise": surprise,
            "ma5": ma5,
        }

        # Market context
        market_available = market_data[market_data["timestamp"] <= as_of_date]
        if len(market_available) > 0:
            latest = market_available.iloc[-1]
            market_context = np.array([
                latest.get("vix", 15.0),
                latest.get("credit_spread", 3.0),
                latest.get("yield_curve", 1.5),
            ], dtype=np.float32)
        else:
            market_context = np.array([15.0, 3.0, 1.5], dtype=np.float32)

        return macro_batch, market_context

    def create_dataset(
        self,
        macro_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        target_data: pd.DataFrame,
        feature_type: str = "flat",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create complete dataset for training.

        :param macro_data (pd.DataFrame): Macro data
        :param factor_data (pd.DataFrame): Factor returns
        :param market_data (pd.DataFrame): Market context
        :param target_data (pd.DataFrame): Target labels
        :param feature_type (str): 'flat', 'sequence', or 'transformer'

        :return X (np.ndarray): Features
        :return y (np.ndarray): Targets
        """
        X_list = []
        y_list = []

        for _, row in target_data.iterrows():
            as_of_date = row["timestamp"]

            if feature_type == "flat":
                features = self.create_flat_features(
                    macro_data, factor_data, market_data, as_of_date
                )
                X_list.append(features)
            elif feature_type == "sequence":
                features, market = self.create_sequence_features(
                    macro_data, factor_data, market_data, as_of_date
                )
                # Append market context to sequence
                market_expanded = np.tile(market, (len(features), 1))
                combined = np.concatenate([features, market_expanded], axis=1)
                X_list.append(combined)

            y_list.append(row["target"])

        X = np.array(X_list)
        y = np.array(y_list)

        # Normalize if configured
        if self.config.normalize_features and feature_type == "flat":
            X = self._normalize(X, fit=True)

        return X, y

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features.

        :param X (np.ndarray): Features to normalize
        :param fit (bool): Whether to fit statistics

        :return X_normalized (np.ndarray): Normalized features
        """
        if fit:
            self.feature_means = np.nanmean(X, axis=0)
            self.feature_stds = np.nanstd(X, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0

        if self.feature_means is not None and self.feature_stds is not None:
            X = (X - self.feature_means) / self.feature_stds

        return np.nan_to_num(X, nan=0.0)

    def get_feature_names(self) -> List[str]:
        """
        Get names of flat features.

        :return names (List[str]): Feature names
        """
        names = []

        for window in self.config.aggregation_windows:
            for cat in MacroCategory:
                names.extend([
                    f"{cat.value}_surprise_mean_{window}w",
                    f"{cat.value}_surprise_std_{window}w",
                    f"{cat.value}_normalized_mean_{window}w",
                    f"{cat.value}_release_count_{window}w",
                ])

        if self.config.include_momentum:
            for window in self.config.aggregation_windows:
                names.extend([
                    f"cyclical_momentum_{window}w",
                    f"defensive_momentum_{window}w",
                    f"value_momentum_{window}w",
                    f"growth_momentum_{window}w",
                ])

        if self.config.include_market_context:
            names.extend([
                "vix_normalized",
                "credit_spread_normalized",
                "yield_curve_normalized",
                "vix_change_4w",
                "spread_change_4w",
            ])

        return names

    def get_num_indicators(self) -> int:
        """
        Get number of unique indicators.

        :return count (int): Number of indicators
        """
        return len(self.indicator_to_idx)

    def get_num_categories(self) -> int:
        """
        Get number of macro categories.

        :return count (int): Number of categories
        """
        return len(MacroCategory)
