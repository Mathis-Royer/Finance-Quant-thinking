"""
Point-in-Time FRED-MD Data Loader.

This module implements point-in-time (PIT) data loading using FRED-MD vintage files.
Each vintage file contains macroeconomic data as it was known at a specific date,
eliminating lookahead bias from data revisions.

As stated in the strategy document (Section 2.3):
"Point-in-time databases are used exclusively, with sources providing historical
vintages (Bloomberg, Refinitiv, ALFRED from the Fed). Vintages are meticulously
reconstructed to document precisely the information available at each decision date."

The vintage naming conventions are:
- Old format: YYYY-MM.csv (e.g., 1999-08.csv, 2014-12.csv)
- New format: FRED-MD_YYYYmMM.csv (e.g., FRED-MD_2024m03.csv)
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
import re

from .data_loader import (
    MacroIndicator,
    MacroCategory,
    PublicationType,
    Region,
)
from .fred_md_loader import (
    FRED_MD_CATEGORY_MAP,
    FRED_MD_IMPORTANCE,
    TransformCode,
)


@dataclass
class PointInTimeConfig:
    """
    Configuration for Point-in-Time FRED-MD data loading.

    :param vintages_dir (Path): Directory containing vintage CSV files
    :param publication_lag (int): Months of publication lag (default=1)
    :param apply_transformations (bool): Whether to apply FRED-MD transformations
    :param handle_missing (str): How to handle missing values
    """
    vintages_dir: Path
    publication_lag: int = 1
    apply_transformations: bool = True
    handle_missing: str = "ffill"


class PointInTimeFREDMDLoader:
    """
    Point-in-Time loader for FRED-MD using vintage files.

    This loader ensures that for any given date, only data that was actually
    available at that time is used, eliminating lookahead bias from revisions.

    For a prediction made in month t, we use the vintage from month t,
    which contains data up to month t-1 (due to publication lag).

    :param config (PointInTimeConfig): Configuration options
    """

    def __init__(self, config: PointInTimeConfig):
        """
        Initialize the Point-in-Time loader.

        :param config (PointInTimeConfig): Configuration options
        """
        self.config = config
        self._vintage_index: Dict[str, Path] = {}
        self._vintage_dates: List[pd.Timestamp] = []
        self._cached_vintages: Dict[str, pd.DataFrame] = {}
        self._transform_codes: Optional[pd.Series] = None
        self._indicators: List[MacroIndicator] = []

        self._index_vintages()

    def _index_vintages(self) -> None:
        """
        Index all available vintage files.

        Scans the vintages directory and builds an index mapping
        vintage dates to file paths.
        """
        vintages_dir = Path(self.config.vintages_dir)
        if not vintages_dir.exists():
            raise FileNotFoundError(f"Vintages directory not found: {vintages_dir}")

        # Pattern for old format: YYYY-MM.csv
        old_pattern = re.compile(r"^(\d{4})-(\d{2})\.csv$")

        # Pattern for new format: FRED-MD_YYYYmMM.csv
        new_pattern = re.compile(r"^FRED-MD_(\d{4})m(\d{2})\.csv$")

        for file_path in vintages_dir.glob("*.csv"):
            filename = file_path.name

            # Try old format
            match = old_pattern.match(filename)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                vintage_date = f"{year:04d}-{month:02d}"
                self._vintage_index[vintage_date] = file_path
                continue

            # Try new format
            match = new_pattern.match(filename)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                vintage_date = f"{year:04d}-{month:02d}"
                self._vintage_index[vintage_date] = file_path
                continue

        # Sort vintage dates
        self._vintage_dates = sorted([
            pd.Timestamp(f"{d}-01") for d in self._vintage_index.keys()
        ])

        print(f"Indexed {len(self._vintage_index)} vintage files")
        if self._vintage_dates:
            print(f"  Coverage: {self._vintage_dates[0].strftime('%Y-%m')} "
                  f"to {self._vintage_dates[-1].strftime('%Y-%m')}")

    def get_available_vintages(self) -> List[str]:
        """
        Get list of available vintage dates.

        :return vintages (List[str]): List of vintage dates (YYYY-MM format)
        """
        return sorted(self._vintage_index.keys())

    def get_vintage_for_date(self, target_date: pd.Timestamp) -> Optional[str]:
        """
        Get the appropriate vintage to use for a given prediction date.

        For a prediction at date t, we use the vintage from month t,
        which contains data through month t-1.

        :param target_date (pd.Timestamp): Date for which prediction is made

        :return vintage_date (str): Vintage date to use (YYYY-MM format)
        """
        target_month = pd.Timestamp(target_date.year, target_date.month, 1)

        # Find the latest vintage on or before the target month
        available = [d for d in self._vintage_dates if d <= target_month]

        if not available:
            return None

        vintage = available[-1]
        return vintage.strftime("%Y-%m")

    def _load_vintage(self, vintage_date: str) -> pd.DataFrame:
        """
        Load and process a specific vintage file.

        :param vintage_date (str): Vintage date (YYYY-MM format)

        :return data (pd.DataFrame): Processed vintage data
        """
        if vintage_date in self._cached_vintages:
            return self._cached_vintages[vintage_date]

        if vintage_date not in self._vintage_index:
            raise ValueError(f"Vintage {vintage_date} not found")

        file_path = self._vintage_index[vintage_date]
        raw_data = pd.read_csv(file_path)

        # Extract transformation codes from first row
        first_row = raw_data.iloc[0]
        transform_codes = first_row.drop("sasdate", errors="ignore")
        raw_data = raw_data.iloc[1:].copy()

        # Store transform codes (use first loaded vintage as reference)
        if self._transform_codes is None:
            self._transform_codes = transform_codes

        # Parse dates - handle different date formats
        try:
            raw_data["date"] = pd.to_datetime(raw_data["sasdate"], format="%m/%d/%Y")
        except (ValueError, TypeError):
            try:
                raw_data["date"] = pd.to_datetime(raw_data["sasdate"])
            except (ValueError, TypeError):
                raw_data["date"] = pd.to_datetime(
                    raw_data["sasdate"], format="%m/%d/%Y", errors="coerce"
                )

        raw_data = raw_data.drop("sasdate", axis=1)
        raw_data = raw_data.set_index("date")
        raw_data = raw_data.sort_index()

        # Convert to numeric
        for col in raw_data.columns:
            raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

        # Handle missing values
        if self.config.handle_missing == "ffill":
            raw_data = raw_data.ffill()
        elif self.config.handle_missing == "interpolate":
            raw_data = raw_data.interpolate(method="linear")

        # Apply transformations if configured
        if self.config.apply_transformations:
            raw_data = self._apply_transformations(raw_data, transform_codes)

        self._cached_vintages[vintage_date] = raw_data
        return raw_data

    def _apply_transformations(
        self, df: pd.DataFrame, transform_codes: pd.Series
    ) -> pd.DataFrame:
        """
        Apply FRED-MD recommended transformations.

        :param df (pd.DataFrame): Raw data
        :param transform_codes (pd.Series): Transformation codes

        :return transformed (pd.DataFrame): Transformed data
        """
        transformed = pd.DataFrame(index=df.index)

        for col in df.columns:
            if col not in transform_codes.index:
                continue

            try:
                tcode = int(float(transform_codes[col]))
            except (ValueError, TypeError):
                tcode = 1

            series = df[col]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if tcode == 1:  # Level
                    transformed[col] = series
                elif tcode == 2:  # First difference
                    transformed[col] = series.diff()
                elif tcode == 3:  # Second difference
                    transformed[col] = series.diff().diff()
                elif tcode == 4:  # Log
                    transformed[col] = np.log(series.clip(lower=1e-10))
                elif tcode == 5:  # First difference of log
                    transformed[col] = np.log(series.clip(lower=1e-10)).diff()
                elif tcode == 6:  # Second difference of log
                    transformed[col] = np.log(series.clip(lower=1e-10)).diff().diff()
                elif tcode == 7:  # First difference of percent change
                    transformed[col] = series.pct_change().diff()
                else:
                    transformed[col] = series

        # Drop rows with NaN from differencing
        transformed = transformed.iloc[2:]

        return transformed

    def get_pit_data_for_date(
        self,
        target_date: pd.Timestamp,
        lookback_months: int = 60,
    ) -> Optional[pd.DataFrame]:
        """
        Get point-in-time data available for a specific prediction date.

        This returns the macro data that would have been available when
        making a prediction at target_date.

        :param target_date (pd.Timestamp): Date of prediction
        :param lookback_months (int): How many months of history to include

        :return data (pd.DataFrame): Point-in-time macro data
        """
        vintage_date = self.get_vintage_for_date(target_date)

        if vintage_date is None:
            return None

        data = self._load_vintage(vintage_date)

        # The vintage contains data up to month before vintage date
        # (due to publication lag)
        cutoff = target_date - pd.DateOffset(months=self.config.publication_lag)
        start = cutoff - pd.DateOffset(months=lookback_months)

        # Filter to relevant date range
        mask = (data.index >= start) & (data.index <= cutoff)
        return data[mask].copy()

    def create_pit_dataset(
        self,
        start_date: str,
        end_date: str,
        lookback_months: int = 60,
    ) -> pd.DataFrame:
        """
        Create a complete point-in-time dataset for backtesting.

        For each month in the date range, loads the appropriate vintage
        and extracts the data that was available at that time.

        :param start_date (str): Start date (YYYY-MM-DD)
        :param end_date (str): End date (YYYY-MM-DD)
        :param lookback_months (int): History months per observation

        :return dataset (pd.DataFrame): Point-in-time dataset with columns:
            prediction_date, vintage_used, + macro indicators
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Generate monthly prediction dates
        prediction_dates = pd.date_range(start, end, freq="MS")

        records = []
        for pred_date in prediction_dates:
            vintage = self.get_vintage_for_date(pred_date)
            if vintage is None:
                continue

            pit_data = self.get_pit_data_for_date(pred_date, lookback_months)
            if pit_data is None or len(pit_data) == 0:
                continue

            # Get most recent observation (last row)
            latest = pit_data.iloc[-1].to_dict()
            latest["prediction_date"] = pred_date
            latest["vintage_used"] = vintage
            latest["data_date"] = pit_data.index[-1]
            records.append(latest)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df = df.set_index("prediction_date")
            # Reorder columns
            meta_cols = ["vintage_used", "data_date"]
            data_cols = [c for c in df.columns if c not in meta_cols]
            df = df[meta_cols + data_cols]

        return df

    def create_pit_macro_dataframe(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Create macro token DataFrame using point-in-time data.

        Compatible with existing pipeline, but uses PIT data.

        :param start_date (str): Start date
        :param end_date (str): End date

        :return macro_data (pd.DataFrame): Macro tokens with PIT data
        """
        # Build indicator list if not done
        if not self._indicators:
            self._build_indicators()

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        prediction_dates = pd.date_range(start, end, freq="MS")

        records = []

        for pred_date in prediction_dates:
            vintage = self.get_vintage_for_date(pred_date)
            if vintage is None:
                continue

            # Get historical data available at prediction time
            pit_data = self.get_pit_data_for_date(pred_date, lookback_months=60)
            if pit_data is None or len(pit_data) == 0:
                continue

            # Latest observation date
            latest_date = pit_data.index[-1]

            for indicator in self._indicators:
                if indicator.name not in pit_data.columns:
                    continue

                value = pit_data.loc[latest_date, indicator.name]
                if pd.isna(value):
                    continue

                # Calculate normalized value using PIT historical data only
                historical = pit_data[indicator.name].dropna()
                if len(historical) > 12:
                    mean_val = historical.mean()
                    std_val = historical.std()
                    normalized = (value - mean_val) / std_val if std_val > 0 else 0.0
                else:
                    normalized = 0.0

                # Calculate surprise
                ma12 = historical.tail(12).mean() if len(historical) >= 12 else historical.mean()
                std_surprise = historical.tail(24).std() if len(historical) >= 24 else historical.std()
                surprise = (value - ma12) / std_surprise if std_surprise > 0 else 0.0

                # Calculate MA5
                ma5 = historical.tail(5).mean() if len(historical) >= 5 else value

                records.append({
                    "timestamp": pred_date,
                    "data_date": latest_date,
                    "vintage_used": vintage,
                    "indicator": indicator.name,
                    "category": indicator.category.value,
                    "region": indicator.region.value,
                    "importance": indicator.importance,
                    "periodicity": indicator.periodicity,
                    "publication_type": PublicationType.FINAL.value,
                    "value": value,
                    "normalized_value": np.clip(normalized, -4, 4) / 4,
                    "surprise": np.clip(surprise, -4, 4),
                    "ma5": ma5,
                })

        return pd.DataFrame(records)

    def create_pit_market_context(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Create market context DataFrame using point-in-time data.

        :param start_date (str): Start date
        :param end_date (str): End date

        :return market_data (pd.DataFrame): Market context with PIT data
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        prediction_dates = pd.date_range(start, end, freq="MS")

        records = []

        for pred_date in prediction_dates:
            vintage = self.get_vintage_for_date(pred_date)
            if vintage is None:
                continue

            pit_data = self.get_pit_data_for_date(pred_date, lookback_months=12)
            if pit_data is None or len(pit_data) == 0:
                continue

            latest = pit_data.iloc[-1]

            # VIX proxy
            vix = latest.get("VXOCLSx", 15.0)
            if pd.isna(vix):
                vix = 15.0

            # Credit spread
            baa = latest.get("BAA", 6.0)
            aaa = latest.get("AAA", 4.0)
            if pd.isna(baa):
                baa = 6.0
            if pd.isna(aaa):
                aaa = 4.0
            credit_spread = baa - aaa

            # Yield curve
            gs10 = latest.get("GS10", 4.0)
            fedfunds = latest.get("FEDFUNDS", 2.0)
            if pd.isna(gs10):
                gs10 = 4.0
            if pd.isna(fedfunds):
                fedfunds = 2.0
            yield_curve = gs10 - fedfunds

            records.append({
                "timestamp": pred_date,
                "vintage_used": vintage,
                "vix": vix,
                "credit_spread": credit_spread,
                "yield_curve": yield_curve,
            })

        return pd.DataFrame(records)

    def _build_indicators(self) -> None:
        """Build MacroIndicator definitions."""
        # Load any vintage to get column names
        if not self._vintage_index:
            return

        first_vintage = sorted(self._vintage_index.keys())[0]
        data = self._load_vintage(first_vintage)

        self._indicators = []
        for col in data.columns:
            category = FRED_MD_CATEGORY_MAP.get(col, MacroCategory.ECONOMIC_ACTIVITY)
            importance = FRED_MD_IMPORTANCE.get(col, 2)

            indicator = MacroIndicator(
                name=col,
                category=category,
                region=Region.US,
                importance=importance,
                periodicity="monthly",
            )
            self._indicators.append(indicator)

    def get_indicators(self) -> List[MacroIndicator]:
        """
        Get list of MacroIndicator definitions.

        :return indicators (List[MacroIndicator]): Indicator definitions
        """
        if not self._indicators:
            self._build_indicators()
        return self._indicators

    def summary(self) -> Dict:
        """
        Get summary of point-in-time data availability.

        :return summary (Dict): Summary statistics
        """
        return {
            "n_vintages": len(self._vintage_index),
            "first_vintage": self._vintage_dates[0].strftime("%Y-%m") if self._vintage_dates else None,
            "last_vintage": self._vintage_dates[-1].strftime("%Y-%m") if self._vintage_dates else None,
            "publication_lag": self.config.publication_lag,
            "apply_transformations": self.config.apply_transformations,
        }


def load_pit_fred_md_dataset(
    vintages_dir: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[MacroIndicator]]:
    """
    Convenience function to load point-in-time FRED-MD dataset.

    :param vintages_dir (str): Directory containing vintage files
    :param start_date (str): Start date (e.g., "2000-01-01")
    :param end_date (str): End date (e.g., "2024-12-31")

    :return macro_data (pd.DataFrame): PIT macro token data
    :return market_data (pd.DataFrame): PIT market context data
    :return indicators (List[MacroIndicator]): Indicator definitions
    """
    config = PointInTimeConfig(vintages_dir=Path(vintages_dir))
    loader = PointInTimeFREDMDLoader(config)

    print(f"Point-in-Time FRED-MD Loader:")
    summary = loader.summary()
    print(f"  Vintages: {summary['n_vintages']}")
    print(f"  Coverage: {summary['first_vintage']} to {summary['last_vintage']}")
    print(f"  Publication lag: {summary['publication_lag']} month(s)")

    macro_data = loader.create_pit_macro_dataframe(start_date, end_date)
    market_data = loader.create_pit_market_context(start_date, end_date)
    indicators = loader.get_indicators()

    print(f"\nLoaded PIT data for {start_date} to {end_date}:")
    print(f"  Macro observations: {len(macro_data)}")
    print(f"  Market observations: {len(market_data)}")

    if len(macro_data) > 0:
        vintages_used = macro_data["vintage_used"].nunique()
        print(f"  Unique vintages used: {vintages_used}")

    return macro_data, market_data, indicators
