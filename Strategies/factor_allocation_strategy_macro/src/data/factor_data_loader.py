"""
Factor Data Loader using Kenneth French Library.

This module downloads real factor returns from Kenneth French's Data Library:
- Fama-French 5 Factors (Mkt-RF, SMB, HML, RMW, CMA)
- Momentum Factor (Mom)
- 12 Industry Portfolios (aggregated into Cyclical/Defensive)

Data is free and updated monthly.
Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
import requests


# Base URL for Kenneth French data library
FRENCH_DATA_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


@dataclass
class FactorDataConfig:
    """
    Configuration for factor data loading.

    :param start_date (str): Start date for data (YYYY-MM-DD)
    :param end_date (str): End date for data (YYYY-MM-DD)
    :param cache_dir (Path): Directory to cache downloaded data
    :param use_cache (bool): Whether to use cached data if available
    """
    start_date: str = "1990-01-01"
    end_date: str = "2024-12-31"
    cache_dir: Optional[Path] = None
    use_cache: bool = True


# MSCI Cyclical/Defensive classification applied to French 12 Industries
CYCLICAL_INDUSTRIES = [
    "Durbl",   # Consumer Durables -> Consumer Discretionary (Cyclical)
    "Manuf",   # Manufacturing -> Industrials (Cyclical)
    "BusEq",   # Business Equipment -> Information Technology (Cyclical)
    "Shops",   # Retail/Wholesale -> Consumer Discretionary (Cyclical)
    "Money",   # Finance -> Financials (Cyclical)
]

DEFENSIVE_INDUSTRIES = [
    "NoDur",   # Consumer Non-Durables -> Consumer Staples (Defensive)
    "Utils",   # Utilities (Defensive)
    "Hlth",    # Healthcare (Defensive)
    "Enrgy",   # Energy (Defensive per MSCI)
]

# Industries with mixed/unclear classification
MIXED_INDUSTRIES = [
    "Chems",   # Chemicals -> Materials (Cyclical per MSCI, but debatable)
    "Telcm",   # Telecom -> Communication Services (Cyclical per MSCI)
    "Other",   # Everything else
]


def _download_french_zip(filename: str) -> BytesIO:
    """
    Download a zip file from Kenneth French's data library.

    :param filename (str): Name of the zip file

    :return data (BytesIO): Downloaded zip file content
    """
    url = f"{FRENCH_DATA_URL}{filename}"
    print(f"Downloading {url}...")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    return BytesIO(response.content)


def _parse_french_csv(zip_content: BytesIO, skip_footer: int = 0) -> pd.DataFrame:
    """
    Parse CSV from a Kenneth French zip file.

    :param zip_content (BytesIO): Zip file content
    :param skip_footer (int): Number of footer lines to skip

    :return df (pd.DataFrame): Parsed data
    """
    with ZipFile(zip_content) as zf:
        # Get the first CSV file in the zip
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]

        with zf.open(csv_name) as f:
            # Read raw content
            content = f.read().decode("utf-8")

    # Split into lines
    lines = content.strip().split("\n")

    # Find the header line - it's the first line that starts with comma (empty index name)
    header_idx = None
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Header line starts with comma (empty index name) followed by column names
        if line_stripped.startswith(","):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header line in file")

    # Find data lines: lines where first value is a 6-digit YYYYMM date
    # Stop at first empty line after data starts (indicates section break)
    data_lines = []
    in_data = False
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            if in_data:
                # Empty line after data started = end of section
                break
            continue
        parts = line.split(",")
        if parts and parts[0].strip():
            first_val = parts[0].strip()
            # Monthly data has 6 digits (YYYYMM) between 190001 and 209912
            if (first_val.isdigit() and len(first_val) == 6 and
                1900 <= int(first_val[:4]) <= 2099):
                data_lines.append(line)
                in_data = True
            elif in_data:
                # Non-data line after data started = end of section
                break

    if not data_lines:
        raise ValueError("No valid data lines found")

    # Build CSV with header and data
    csv_lines = [lines[header_idx]] + data_lines
    csv_str = "\n".join(csv_lines)

    # Parse the CSV
    df = pd.read_csv(
        BytesIO(csv_str.encode()),
        index_col=0,
        na_values=["-99.99", "-999", ""],
    )

    # Clean column names (remove leading/trailing spaces)
    df.columns = [col.strip() for col in df.columns]

    return df


class FactorDataLoader:
    """
    Loader for real factor returns from Kenneth French Library.

    Downloads and processes:
    1. Fama-French 5 Factors + Momentum
    2. 12 Industry Portfolios -> Cyclical/Defensive aggregation

    :param config (FactorDataConfig): Configuration for data loading
    """

    def __init__(self, config: Optional[FactorDataConfig] = None):
        """
        Initialize the factor data loader.

        :param config (FactorDataConfig): Configuration parameters
        """
        self.config = config or FactorDataConfig()
        self._ff5_factors: Optional[pd.DataFrame] = None
        self._momentum: Optional[pd.DataFrame] = None
        self._industry_portfolios: Optional[pd.DataFrame] = None
        self._factor_returns: Optional[pd.DataFrame] = None

    def load_ff5_factors(self) -> pd.DataFrame:
        """
        Load Fama-French 5 Factors from Kenneth French Library.

        Factors:
        - Mkt-RF: Market excess return
        - SMB: Small Minus Big (Size)
        - HML: High Minus Low (Value)
        - RMW: Robust Minus Weak (Profitability/Quality)
        - CMA: Conservative Minus Aggressive (Investment)

        :return ff5 (pd.DataFrame): Monthly factor returns
        """
        cache_file = None
        if self.config.cache_dir and self.config.use_cache:
            cache_file = self.config.cache_dir / "ff5_factors.parquet"
            if cache_file.exists():
                print(f"Loading cached FF5 factors from {cache_file}")
                self._ff5_factors = pd.read_parquet(cache_file)
                return self._ff5_factors

        # Download FF5 factors
        zip_content = _download_french_zip("F-F_Research_Data_5_Factors_2x3_CSV.zip")
        ff5 = _parse_french_csv(zip_content)

        # Convert index to datetime
        ff5.index = pd.to_datetime(ff5.index.astype(str), format="%Y%m")
        ff5.index.name = "timestamp"

        # Convert from percentage to decimal
        ff5 = ff5 / 100.0

        # Filter by date range
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        ff5 = ff5[(ff5.index >= start) & (ff5.index <= end)]

        self._ff5_factors = ff5

        # Cache if configured
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            ff5.to_parquet(cache_file)
            print(f"Cached FF5 factors to {cache_file}")

        print(f"Loaded FF5 factors: {len(ff5)} months, {ff5.shape[1]} factors")
        return ff5

    def load_momentum_factor(self) -> pd.DataFrame:
        """
        Load Momentum Factor from Kenneth French Library.

        :return mom (pd.DataFrame): Monthly momentum factor returns
        """
        cache_file = None
        if self.config.cache_dir and self.config.use_cache:
            cache_file = self.config.cache_dir / "momentum_factor.parquet"
            if cache_file.exists():
                print(f"Loading cached Momentum factor from {cache_file}")
                self._momentum = pd.read_parquet(cache_file)
                return self._momentum

        # Download Momentum factor
        zip_content = _download_french_zip("F-F_Momentum_Factor_CSV.zip")
        mom = _parse_french_csv(zip_content)

        # Convert index to datetime
        mom.index = pd.to_datetime(mom.index.astype(str), format="%Y%m")
        mom.index.name = "timestamp"

        # Convert from percentage to decimal
        mom = mom / 100.0

        # Rename column
        mom.columns = ["Mom"]

        # Filter by date range
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        mom = mom[(mom.index >= start) & (mom.index <= end)]

        self._momentum = mom

        # Cache if configured
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            mom.to_parquet(cache_file)
            print(f"Cached Momentum factor to {cache_file}")

        print(f"Loaded Momentum factor: {len(mom)} months")
        return mom

    def load_industry_portfolios(self) -> pd.DataFrame:
        """
        Load 12 Industry Portfolios from Kenneth French Library.

        Industries: NoDur, Durbl, Manuf, Enrgy, Chems, BusEq,
                   Telcm, Utils, Shops, Hlth, Money, Other

        :return industries (pd.DataFrame): Monthly industry returns (value-weighted)
        """
        cache_file = None
        if self.config.cache_dir and self.config.use_cache:
            cache_file = self.config.cache_dir / "industry_portfolios.parquet"
            if cache_file.exists():
                print(f"Loading cached Industry portfolios from {cache_file}")
                self._industry_portfolios = pd.read_parquet(cache_file)
                return self._industry_portfolios

        # Download 12 Industry Portfolios (value-weighted)
        zip_content = _download_french_zip("12_Industry_Portfolios_CSV.zip")
        industries = _parse_french_csv(zip_content)

        # Convert index to datetime
        industries.index = pd.to_datetime(industries.index.astype(str), format="%Y%m")
        industries.index.name = "timestamp"

        # Convert from percentage to decimal
        industries = industries / 100.0

        # Filter by date range
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        industries = industries[(industries.index >= start) & (industries.index <= end)]

        self._industry_portfolios = industries

        # Cache if configured
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            industries.to_parquet(cache_file)
            print(f"Cached Industry portfolios to {cache_file}")

        print(f"Loaded 12 Industry portfolios: {len(industries)} months")
        return industries

    def compute_cyclical_defensive(
        self,
        include_mixed: bool = False,
    ) -> pd.DataFrame:
        """
        Compute Cyclical and Defensive aggregate returns from industry portfolios.

        Uses equal-weighted average of constituent industries.
        Based on MSCI Cyclical/Defensive classification.

        :param include_mixed (bool): Include mixed industries in cyclical category

        :return agg (pd.DataFrame): Cyclical and Defensive monthly returns
        """
        if self._industry_portfolios is None:
            self.load_industry_portfolios()

        industries = self._industry_portfolios

        # Get cyclical industries (handle different column name formats)
        cyclical_cols = []
        for c in CYCLICAL_INDUSTRIES:
            matching = [col for col in industries.columns if c.lower() in col.lower()]
            cyclical_cols.extend(matching)

        if include_mixed:
            for c in MIXED_INDUSTRIES:
                matching = [col for col in industries.columns if c.lower() in col.lower()]
                cyclical_cols.extend(matching)

        # Get defensive industries
        defensive_cols = []
        for c in DEFENSIVE_INDUSTRIES:
            matching = [col for col in industries.columns if c.lower() in col.lower()]
            defensive_cols.extend(matching)

        print(f"Cyclical industries: {cyclical_cols}")
        print(f"Defensive industries: {defensive_cols}")

        # Compute equal-weighted averages
        cyclical_returns = industries[cyclical_cols].mean(axis=1)
        defensive_returns = industries[defensive_cols].mean(axis=1)

        agg = pd.DataFrame({
            "cyclical": cyclical_returns,
            "defensive": defensive_returns,
        })
        agg.index.name = "timestamp"

        return agg

    def load_all_factors(self) -> pd.DataFrame:
        """
        Load all factor returns and combine into a single DataFrame.

        Returns columns:
        - value: HML factor (High Minus Low)
        - growth: -HML (inverse of value)
        - quality: RMW factor (Robust Minus Weak)
        - momentum: Mom factor
        - cyclical: Aggregate cyclical industry returns
        - defensive: Aggregate defensive industry returns

        :return factors (pd.DataFrame): Combined monthly factor returns
        """
        cache_file = None
        if self.config.cache_dir and self.config.use_cache:
            cache_file = self.config.cache_dir / "all_factors.parquet"
            if cache_file.exists():
                print(f"Loading cached all factors from {cache_file}")
                self._factor_returns = pd.read_parquet(cache_file)
                return self._factor_returns

        # Load all components
        ff5 = self.load_ff5_factors()
        mom = self.load_momentum_factor()
        cyc_def = self.compute_cyclical_defensive()

        # Combine into single DataFrame
        factors = pd.DataFrame(index=ff5.index)

        # Map to strategy factor names
        factors["value"] = ff5["HML"]
        factors["growth"] = -ff5["HML"]  # Growth is inverse of Value
        factors["quality"] = ff5["RMW"]
        factors["momentum"] = mom["Mom"]
        factors["cyclical"] = cyc_def["cyclical"]
        factors["defensive"] = cyc_def["defensive"]

        # Add market return for reference
        factors["market"] = ff5["Mkt-RF"]

        # Add timestamp column for compatibility
        factors = factors.reset_index()

        self._factor_returns = factors

        # Cache if configured
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            factors.to_parquet(cache_file)
            print(f"Cached all factors to {cache_file}")

        print(f"\nLoaded factor returns: {len(factors)} months")
        print(f"Factors: {list(factors.columns)}")
        print(f"Date range: {factors['timestamp'].min()} to {factors['timestamp'].max()}")

        return factors

    def create_binary_target(
        self,
        factor_data: pd.DataFrame,
        horizon_months: int = 1,
    ) -> pd.DataFrame:
        """
        Create binary target: do cyclicals outperform defensives?

        :param factor_data (pd.DataFrame): Factor returns with cyclical/defensive columns
        :param horizon_months (int): Prediction horizon in months

        :return target_data (pd.DataFrame): Target labels with timestamps
        """
        df = factor_data.copy()

        # Compute forward returns
        df["cyclical_fwd"] = df["cyclical"].shift(-horizon_months)
        df["defensive_fwd"] = df["defensive"].shift(-horizon_months)

        # Binary target: 1 if cyclicals outperform defensives
        df["target"] = (df["cyclical_fwd"] > df["defensive_fwd"]).astype(int)

        # Keep relevant columns
        target_data = df[["timestamp", "target", "cyclical_fwd", "defensive_fwd"]].dropna()

        print(f"\nCreated binary targets: {len(target_data)} observations")
        print(f"Target distribution:\n{target_data['target'].value_counts()}")

        return target_data

    def compute_cumulative_factor_returns(
        self,
        factor_data: pd.DataFrame,
        horizon_months: int,
        factor_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute cumulative returns for each factor over the specified horizon.

        Cumulative return = (1+r1) * (1+r2) * ... * (1+rN) - 1

        :param factor_data (pd.DataFrame): Monthly factor returns
        :param horizon_months (int): Number of months to cumulate
        :param factor_columns (List[str]): Factor columns to process (default: all 6 factors)

        :return cumulative (pd.DataFrame): Cumulative returns aligned with original timestamps
        """
        if factor_columns is None:
            factor_columns = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]

        df = factor_data.copy()
        result = pd.DataFrame(index=df.index)
        result["timestamp"] = df["timestamp"]

        for col in factor_columns:
            if col not in df.columns:
                continue

            returns = df[col].values
            n = len(returns)
            cumulative = np.full(n, np.nan)

            for i in range(n - horizon_months + 1):
                window = returns[i:i + horizon_months]
                cumulative[i] = np.prod(1 + window) - 1

            result[f"{col}_cumulative_{horizon_months}m"] = cumulative

        return result

    def create_multi_horizon_targets(
        self,
        factor_data: pd.DataFrame,
        horizons: List[int] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Create targets for multiple prediction horizons.

        For each horizon, creates a binary target based on cumulative forward returns
        of cyclical vs defensive factors.

        :param factor_data (pd.DataFrame): Factor returns with cyclical/defensive columns
        :param horizons (List[int]): List of horizon months (default: [1, 3, 6, 12])

        :return targets (Dict[int, pd.DataFrame]): Mapping horizon -> target_data
        """
        if horizons is None:
            horizons = [1, 3, 6, 12]

        targets = {}
        df = factor_data.copy()

        for horizon in horizons:
            if horizon == 1:
                # For 1-month horizon, use simple shift (no cumulation needed)
                df_h = df.copy()
                df_h["cyclical_fwd"] = df_h["cyclical"].shift(-1)
                df_h["defensive_fwd"] = df_h["defensive"].shift(-1)
            else:
                # For longer horizons, compute cumulative forward returns
                df_h = df.copy()
                cyclical_cumulative = np.full(len(df_h), np.nan)
                defensive_cumulative = np.full(len(df_h), np.nan)

                cyclical_returns = df_h["cyclical"].values
                defensive_returns = df_h["defensive"].values

                for i in range(len(df_h) - horizon):
                    # Forward cumulative return starting at i+1 for horizon months
                    cyc_window = cyclical_returns[i + 1:i + 1 + horizon]
                    def_window = defensive_returns[i + 1:i + 1 + horizon]
                    cyclical_cumulative[i] = np.prod(1 + cyc_window) - 1
                    defensive_cumulative[i] = np.prod(1 + def_window) - 1

                df_h["cyclical_fwd"] = cyclical_cumulative
                df_h["defensive_fwd"] = defensive_cumulative

            # Binary target: 1 if cyclicals outperform defensives
            df_h["target"] = (df_h["cyclical_fwd"] > df_h["defensive_fwd"]).astype(int)

            # Keep relevant columns
            target_data = df_h[["timestamp", "target", "cyclical_fwd", "defensive_fwd"]].dropna()
            targets[horizon] = target_data

            print(f"\nHorizon {horizon}M: {len(target_data)} observations")
            print(f"  Target distribution: {dict(target_data['target'].value_counts())}")

        return targets

    def get_cumulative_factor_returns_for_horizon(
        self,
        factor_data: pd.DataFrame,
        horizon_months: int,
    ) -> np.ndarray:
        """
        Get cumulative factor returns matrix for Sharpe optimization.

        Returns a matrix where each row is the cumulative return over
        the next horizon_months for each factor.

        :param factor_data (pd.DataFrame): Monthly factor returns
        :param horizon_months (int): Prediction horizon

        :return cumulative_returns (np.ndarray): Shape [n_samples, n_factors]
        """
        factor_columns = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]
        returns = factor_data[factor_columns].values
        n_samples = len(returns)
        n_factors = len(factor_columns)

        if horizon_months == 1:
            # For 1-month, just return the raw returns (shifted by 1)
            return returns[1:, :]

        # Compute cumulative returns for each period
        cumulative = np.full((n_samples - horizon_months, n_factors), np.nan)

        for i in range(n_samples - horizon_months):
            window = returns[i + 1:i + 1 + horizon_months, :]
            cumulative[i] = np.prod(1 + window, axis=0) - 1

        return cumulative

    def summary(self) -> Dict:
        """
        Get summary of loaded data.

        :return summary (Dict): Summary statistics
        """
        if self._factor_returns is None:
            return {"status": "No data loaded"}

        factors = self._factor_returns

        return {
            "n_observations": len(factors),
            "start_date": str(factors["timestamp"].min()),
            "end_date": str(factors["timestamp"].max()),
            "factors": [c for c in factors.columns if c != "timestamp"],
            "factor_stats": {
                col: {
                    "mean": factors[col].mean(),
                    "std": factors[col].std(),
                    "min": factors[col].min(),
                    "max": factors[col].max(),
                }
                for col in factors.columns if col != "timestamp"
            },
        }


def load_factor_dataset(
    start_date: str = "1990-01-01",
    end_date: str = "2024-12-31",
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load factor data and create targets.

    :param start_date (str): Start date
    :param end_date (str): End date
    :param cache_dir (str): Cache directory path

    :return factor_data (pd.DataFrame): Factor returns
    :return target_data (pd.DataFrame): Binary targets
    """
    config = FactorDataConfig(
        start_date=start_date,
        end_date=end_date,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    loader = FactorDataLoader(config)
    factor_data = loader.load_all_factors()
    target_data = loader.create_binary_target(factor_data, horizon_months=1)

    return factor_data, target_data


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("Testing Factor Data Loader")
    print("=" * 60)

    config = FactorDataConfig(
        start_date="2000-01-01",
        end_date="2024-12-31",
    )

    loader = FactorDataLoader(config)
    factors = loader.load_all_factors()

    print("\nFactor Returns Sample:")
    print(factors.head(10))

    target_data = loader.create_binary_target(factors)

    print("\nTarget Data Sample:")
    print(target_data.head(10))

    print("\nSummary:")
    print(loader.summary())
