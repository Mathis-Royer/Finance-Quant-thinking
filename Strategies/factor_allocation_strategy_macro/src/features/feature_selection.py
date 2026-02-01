"""
Feature selection and dimensionality reduction for macroeconomic data.

This module provides:
- IndicatorSelector: Select top-K indicators based on information content
- PCAFeatureReducer: Apply PCA to reduce feature dimensionality
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


@dataclass
class SelectionConfig:
    """
    Configuration for feature selection.

    :param method (str): Selection method ('mutual_info', 'correlation', 'variance')
    :param n_features (int): Number of features/indicators to select
    :param min_variance (float): Minimum variance threshold for variance method
    """
    method: str = "mutual_info"
    n_features: int = 30
    min_variance: float = 0.01


@dataclass
class PCAConfig:
    """
    Configuration for PCA dimensionality reduction.

    :param n_components (int or float): Number of components or variance ratio
    :param whiten (bool): Whether to whiten the output
    """
    n_components: Union[int, float] = 0.95
    whiten: bool = False


class IndicatorSelector:
    """
    Select top-K macroeconomic indicators based on information content.

    Supports multiple selection methods:
    - mutual_info: Mutual information with target variable
    - correlation: Absolute correlation with factor returns
    - variance: Select indicators with highest variance

    :param config (SelectionConfig): Selection configuration
    """

    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Initialize indicator selector.

        :param config (SelectionConfig): Selection configuration
        """
        self.config = config or SelectionConfig()
        self.selected_indicators: List[str] = []
        self.indicator_scores: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        macro_data: pd.DataFrame,
        target_data: pd.DataFrame,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> "IndicatorSelector":
        """
        Fit the selector by computing indicator scores.

        :param macro_data (pd.DataFrame): Macro token data with 'indicator',
                                          'timestamp', 'normalized_value' columns
        :param target_data (pd.DataFrame): Target data with 'timestamp', 'target' columns
        :param factor_data (pd.DataFrame): Optional factor returns for correlation method

        :return self (IndicatorSelector): Fitted selector
        """
        # Pivot macro data to wide format: rows=dates, cols=indicators
        pivot_data = self._pivot_macro_data(macro_data)

        # Align with targets
        aligned_data, aligned_targets = self._align_with_targets(
            pivot_data, target_data
        )

        if len(aligned_data) < 10:
            raise ValueError(
                f"Insufficient aligned samples: {len(aligned_data)}. "
                "Need at least 10 observations."
            )

        # Compute scores based on method
        if self.config.method == "mutual_info":
            scores = self._compute_mutual_info(aligned_data, aligned_targets)
        elif self.config.method == "correlation":
            if factor_data is None:
                raise ValueError("factor_data required for correlation method")
            scores = self._compute_correlation(aligned_data, factor_data)
        elif self.config.method == "variance":
            scores = self._compute_variance(aligned_data)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        # Store scores and select top indicators
        self.indicator_scores = scores
        sorted_indicators = sorted(
            scores.keys(), key=lambda x: scores[x], reverse=True
        )
        self.selected_indicators = sorted_indicators[: self.config.n_features]
        self.is_fitted = True

        return self

    def transform(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter macro data to keep only selected indicators.

        :param macro_data (pd.DataFrame): Raw macro token data

        :return filtered_data (pd.DataFrame): Filtered macro data
        """
        if not self.is_fitted:
            raise RuntimeError("IndicatorSelector must be fitted before transform")

        return macro_data[macro_data["indicator"].isin(self.selected_indicators)].copy()

    def fit_transform(
        self,
        macro_data: pd.DataFrame,
        target_data: pd.DataFrame,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Fit selector and transform data in one step.

        :param macro_data (pd.DataFrame): Macro token data
        :param target_data (pd.DataFrame): Target data
        :param factor_data (pd.DataFrame): Optional factor returns

        :return filtered_data (pd.DataFrame): Filtered macro data
        """
        self.fit(macro_data, target_data, factor_data)
        return self.transform(macro_data)

    def get_selected_indicators(self) -> List[str]:
        """
        Get list of selected indicator names.

        :return indicators (List[str]): Selected indicator names
        """
        return self.selected_indicators.copy()

    def get_indicator_rankings(self) -> pd.DataFrame:
        """
        Get indicator rankings with scores.

        :return rankings (pd.DataFrame): DataFrame with indicator, score, rank
        """
        if not self.is_fitted:
            raise RuntimeError("IndicatorSelector must be fitted first")

        data = [
            {"indicator": ind, "score": score, "selected": ind in self.selected_indicators}
            for ind, score in self.indicator_scores.items()
        ]
        df = pd.DataFrame(data)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def _pivot_macro_data(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot macro data to wide format.

        :param macro_data (pd.DataFrame): Long-format macro data

        :return pivot_df (pd.DataFrame): Wide-format data (dates × indicators)
        """
        # Use most recent value per indicator per month
        macro_data = macro_data.copy()
        macro_data["month"] = macro_data["timestamp"].dt.to_period("M")

        # Aggregate by indicator and month (take last value)
        agg_data = (
            macro_data.groupby(["month", "indicator"])["normalized_value"]
            .last()
            .reset_index()
        )

        # Pivot to wide format
        pivot_df = agg_data.pivot(
            index="month", columns="indicator", values="normalized_value"
        )

        # Fill missing values with forward fill then 0
        pivot_df = pivot_df.ffill().fillna(0)

        return pivot_df

    def _align_with_targets(
        self, pivot_data: pd.DataFrame, target_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Align pivoted macro data with target dates.

        IMPORTANT: Features are lagged by 1 month relative to targets to prevent
        forward-looking bias in feature selection. This means features from month M
        are used to predict targets from month M+1.

        :param pivot_data (pd.DataFrame): Wide-format macro data
        :param target_data (pd.DataFrame): Target data

        :return aligned_data (pd.DataFrame): Aligned features (lagged by 1 month)
        :return aligned_targets (np.ndarray): Aligned target values
        """
        target_data = target_data.copy()
        target_data["month"] = target_data["timestamp"].dt.to_period("M")

        # Lag features by 1 month: use features[M] to predict target[M+1]
        # Shift pivot_data index forward by 1 month for alignment
        pivot_shifted = pivot_data.copy()
        pivot_shifted.index = pivot_shifted.index + 1  # Period + 1 = next month

        # Find common months between shifted features and targets
        common_months = set(pivot_shifted.index) & set(target_data["month"])
        common_months = sorted(common_months)

        # Filter and align
        aligned_features = pivot_shifted.loc[common_months]
        target_lookup = target_data.set_index("month")["target"]
        aligned_targets = np.array([target_lookup[m] for m in common_months])

        return aligned_features, aligned_targets

    def _compute_mutual_info(
        self, features: pd.DataFrame, targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute mutual information between each indicator and target.

        :param features (pd.DataFrame): Feature matrix
        :param targets (np.ndarray): Target values

        :return scores (Dict[str, float]): Indicator → MI score
        """
        X = features.values
        y = targets

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Use classification MI if target is binary
        if len(np.unique(y)) == 2:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        return {
            col: float(score)
            for col, score in zip(features.columns, mi_scores)
        }

    def _compute_correlation(
        self, features: pd.DataFrame, factor_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute absolute correlation between indicators and factor returns.

        Uses average absolute correlation across all factors.

        :param features (pd.DataFrame): Feature matrix
        :param factor_data (pd.DataFrame): Factor returns

        :return scores (Dict[str, float]): Indicator → correlation score
        """
        # Align factor data with features
        factor_data = factor_data.copy()
        factor_data["month"] = factor_data["timestamp"].dt.to_period("M")

        # Get factor columns
        factor_cols = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]
        available_cols = [c for c in factor_cols if c in factor_data.columns]

        scores = {}
        for indicator in features.columns:
            indicator_series = features[indicator]

            # Compute correlation with each factor
            correlations = []
            for factor_col in available_cols:
                factor_series = factor_data.set_index("month")[factor_col]
                common_idx = set(indicator_series.index) & set(factor_series.index)
                if len(common_idx) > 10:
                    common_idx = sorted(common_idx)
                    corr = np.corrcoef(
                        indicator_series.loc[common_idx].values,
                        factor_series.loc[common_idx].values,
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            # Average absolute correlation
            scores[indicator] = np.mean(correlations) if correlations else 0.0

        return scores

    def _compute_variance(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Compute variance for each indicator.

        :param features (pd.DataFrame): Feature matrix

        :return scores (Dict[str, float]): Indicator → variance score
        """
        variances = features.var()
        return {col: float(var) for col, var in variances.items()}


class PCAFeatureReducer:
    """
    PCA-based dimensionality reduction for macro features.

    Can be applied to:
    - Flat features (aggregated macro statistics)
    - Pivoted indicator values (wide format)
    - Token embeddings (in model space)

    :param config (PCAConfig): PCA configuration
    """

    def __init__(self, config: Optional[PCAConfig] = None):
        """
        Initialize PCA reducer.

        :param config (PCAConfig): PCA configuration
        """
        self.config = config or PCAConfig()
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.n_components_fitted: int = 0
        self.explained_variance_ratio: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCAFeatureReducer":
        """
        Fit PCA on feature matrix.

        IMPORTANT: Only fit on TRAINING data to avoid data leakage.
        Use transform() separately for test/validation data.

        :param X (np.ndarray): Feature matrix [n_samples, n_features]
                               Must be training data only.

        :return self (PCAFeatureReducer): Fitted reducer
        """
        # Standardize first
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Handle NaN values
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Fit PCA
        self.pca = PCA(n_components=self.config.n_components, whiten=self.config.whiten)
        self.pca.fit(X_scaled)

        self.n_components_fitted = self.pca.n_components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA transformation.

        :param X (np.ndarray): Feature matrix [n_samples, n_features]

        :return X_reduced (np.ndarray): Reduced features [n_samples, n_components]
        """
        if not self.is_fitted:
            raise RuntimeError("PCAFeatureReducer must be fitted before transform")

        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        WARNING: This method may cause data leakage if called on the entire
        dataset (train + test). Use fit(X_train) then transform(X) separately
        to ensure proper train/test separation.

        :param X (np.ndarray): Feature matrix

        :return X_reduced (np.ndarray): Reduced features
        """
        import warnings
        warnings.warn(
            "PCAFeatureReducer.fit_transform() may cause data leakage. "
            "Use fit(X_train) then transform(X) separately for proper train/test separation.",
            UserWarning,
            stacklevel=2,
        )
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Inverse PCA transformation (for reconstruction).

        :param X_reduced (np.ndarray): Reduced features

        :return X_reconstructed (np.ndarray): Reconstructed features
        """
        if not self.is_fitted:
            raise RuntimeError("PCAFeatureReducer must be fitted before inverse_transform")

        X_scaled = self.pca.inverse_transform(X_reduced)
        return self.scaler.inverse_transform(X_scaled)

    def get_explained_variance(self) -> Tuple[np.ndarray, float]:
        """
        Get explained variance information.

        :return variance_ratio (np.ndarray): Per-component variance ratio
        :return total_variance (float): Total variance explained
        """
        if not self.is_fitted:
            raise RuntimeError("PCAFeatureReducer must be fitted first")

        return (
            self.explained_variance_ratio.copy(),
            float(np.sum(self.explained_variance_ratio)),
        )

    def get_loadings(self) -> np.ndarray:
        """
        Get PCA loadings (component weights).

        :return loadings (np.ndarray): [n_components, n_features]
        """
        if not self.is_fitted:
            raise RuntimeError("PCAFeatureReducer must be fitted first")

        return self.pca.components_.copy()


class FeatureSelectionPipeline:
    """
    Combined pipeline for indicator selection and PCA reduction.

    Applies:
    1. Indicator selection (optional)
    2. PCA on aggregated features (optional)

    :param selector_config (SelectionConfig): Indicator selection config
    :param pca_config (PCAConfig): PCA config
    :param use_selector (bool): Whether to use indicator selection
    :param use_pca (bool): Whether to use PCA
    """

    def __init__(
        self,
        selector_config: Optional[SelectionConfig] = None,
        pca_config: Optional[PCAConfig] = None,
        use_selector: bool = True,
        use_pca: bool = True,
    ):
        """
        Initialize feature selection pipeline.

        :param selector_config (SelectionConfig): Indicator selection config
        :param pca_config (PCAConfig): PCA config
        :param use_selector (bool): Whether to use indicator selection
        :param use_pca (bool): Whether to use PCA
        """
        self.use_selector = use_selector
        self.use_pca = use_pca

        self.selector = IndicatorSelector(selector_config) if use_selector else None
        self.pca_reducer = PCAFeatureReducer(pca_config) if use_pca else None

        self.is_fitted = False

    def fit(
        self,
        macro_data: pd.DataFrame,
        target_data: pd.DataFrame,
        flat_features: Optional[np.ndarray] = None,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> "FeatureSelectionPipeline":
        """
        Fit the complete pipeline.

        :param macro_data (pd.DataFrame): Macro token data
        :param target_data (pd.DataFrame): Target data
        :param flat_features (np.ndarray): Optional flat features for PCA
        :param factor_data (pd.DataFrame): Optional factor data for correlation

        :return self (FeatureSelectionPipeline): Fitted pipeline
        """
        # Step 1: Indicator selection
        if self.use_selector:
            self.selector.fit(macro_data, target_data, factor_data)

        # Step 2: PCA on flat features
        if self.use_pca and flat_features is not None:
            self.pca_reducer.fit(flat_features)

        self.is_fitted = True
        return self

    def transform_macro(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform macro data (apply indicator selection).

        :param macro_data (pd.DataFrame): Raw macro data

        :return filtered_data (pd.DataFrame): Filtered macro data
        """
        if self.use_selector:
            return self.selector.transform(macro_data)
        return macro_data

    def transform_features(self, flat_features: np.ndarray) -> np.ndarray:
        """
        Transform flat features (apply PCA).

        :param flat_features (np.ndarray): Raw flat features

        :return reduced_features (np.ndarray): PCA-reduced features
        """
        if self.use_pca:
            return self.pca_reducer.transform(flat_features)
        return flat_features

    def get_selected_indicators(self) -> Optional[List[str]]:
        """
        Get selected indicators if selector is used.

        :return indicators (List[str]): Selected indicator names
        """
        if self.use_selector:
            return self.selector.get_selected_indicators()
        return None

    def get_pca_info(self) -> Optional[Dict]:
        """
        Get PCA information if PCA is used.

        :return info (Dict): PCA summary info
        """
        if self.use_pca and self.pca_reducer.is_fitted:
            variance_ratio, total = self.pca_reducer.get_explained_variance()
            return {
                "n_components": self.pca_reducer.n_components_fitted,
                "total_variance_explained": total,
                "variance_per_component": variance_ratio.tolist(),
            }
        return None

    def summary(self) -> str:
        """
        Get summary of the pipeline.

        :return summary (str): Human-readable summary
        """
        lines = ["Feature Selection Pipeline Summary", "=" * 40]

        if self.use_selector and self.selector.is_fitted:
            lines.append(f"Indicator Selection: {len(self.selector.selected_indicators)} selected")
            lines.append(f"  Method: {self.selector.config.method}")
            top_5 = self.selector.selected_indicators[:5]
            lines.append(f"  Top 5: {', '.join(top_5)}")

        if self.use_pca and self.pca_reducer.is_fitted:
            info = self.get_pca_info()
            lines.append(f"\nPCA Reduction: {info['n_components']} components")
            lines.append(f"  Variance explained: {info['total_variance_explained']:.1%}")

        return "\n".join(lines)
