"""
FRED-MD Data Loader for macroeconomic data.

This module loads and processes the FRED-MD dataset, a monthly database of
macroeconomic indicators maintained by the Federal Reserve Bank of St. Louis.

FRED-MD includes ~130 monthly macroeconomic time series commonly used in
empirical macroeconomics and finance research.

Data source: https://research.stlouisfed.org/econ/mccracken/fred-databases/
Direct download: https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv

Reference:
    McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for
    macroeconomic research. Journal of Business & Economic Statistics, 34(4), 574-589.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import warnings
import requests

from .data_loader import (
    MacroIndicator,
    MacroCategory,
    PublicationType,
    Region,
)


# FRED-MD transformation codes
class TransformCode(Enum):
    """
    Transformation codes used in FRED-MD.

    1: No transformation (levels)
    2: First difference
    3: Second difference
    4: Log
    5: First difference of log
    6: Second difference of log
    7: First difference of percent change
    """
    LEVEL = 1
    DIFF = 2
    DIFF2 = 3
    LOG = 4
    LOG_DIFF = 5
    LOG_DIFF2 = 6
    PCT_CHANGE_DIFF = 7


# Mapping of FRED-MD series to macro categories
FRED_MD_CATEGORY_MAP: Dict[str, MacroCategory] = {
    # Output and Income (Group 1)
    "RPI": MacroCategory.ECONOMIC_ACTIVITY,
    "W875RX1": MacroCategory.ECONOMIC_ACTIVITY,
    "INDPRO": MacroCategory.ECONOMIC_ACTIVITY,
    "IPFPNSS": MacroCategory.ECONOMIC_ACTIVITY,
    "IPFINAL": MacroCategory.ECONOMIC_ACTIVITY,
    "IPCONGD": MacroCategory.CONSUMPTION,
    "IPDCONGD": MacroCategory.CONSUMPTION,
    "IPNCONGD": MacroCategory.CONSUMPTION,
    "IPBUSEQ": MacroCategory.ECONOMIC_ACTIVITY,
    "IPMAT": MacroCategory.ECONOMIC_ACTIVITY,
    "IPDMAT": MacroCategory.ECONOMIC_ACTIVITY,
    "IPNMAT": MacroCategory.ECONOMIC_ACTIVITY,
    "IPMANSICS": MacroCategory.ECONOMIC_ACTIVITY,
    "IPB51222S": MacroCategory.ECONOMIC_ACTIVITY,
    "IPFUELS": MacroCategory.ECONOMIC_ACTIVITY,
    "CUMFNS": MacroCategory.ECONOMIC_ACTIVITY,

    # Labor Market (Group 2)
    "HWI": MacroCategory.EMPLOYMENT,
    "HWIURATIO": MacroCategory.EMPLOYMENT,
    "CLF16OV": MacroCategory.EMPLOYMENT,
    "CE16OV": MacroCategory.EMPLOYMENT,
    "UNRATE": MacroCategory.EMPLOYMENT,
    "UEMPMEAN": MacroCategory.EMPLOYMENT,
    "UEMPLT5": MacroCategory.EMPLOYMENT,
    "UEMP5TO14": MacroCategory.EMPLOYMENT,
    "UEMP15OV": MacroCategory.EMPLOYMENT,
    "UEMP15T26": MacroCategory.EMPLOYMENT,
    "UEMP27OV": MacroCategory.EMPLOYMENT,
    "CLAIMSx": MacroCategory.EMPLOYMENT,
    "PAYEMS": MacroCategory.EMPLOYMENT,
    "USGOOD": MacroCategory.EMPLOYMENT,
    "CES1021000001": MacroCategory.EMPLOYMENT,
    "USCONS": MacroCategory.EMPLOYMENT,
    "MANEMP": MacroCategory.EMPLOYMENT,
    "DMANEMP": MacroCategory.EMPLOYMENT,
    "NDMANEMP": MacroCategory.EMPLOYMENT,
    "SRVPRD": MacroCategory.EMPLOYMENT,
    "USTPU": MacroCategory.EMPLOYMENT,
    "USWTRADE": MacroCategory.EMPLOYMENT,
    "USTRADE": MacroCategory.EMPLOYMENT,
    "USFIRE": MacroCategory.EMPLOYMENT,
    "USGOVT": MacroCategory.EMPLOYMENT,
    "CES0600000007": MacroCategory.EMPLOYMENT,
    "AWOTMAN": MacroCategory.EMPLOYMENT,
    "AWHMAN": MacroCategory.EMPLOYMENT,
    "CES0600000008": MacroCategory.EMPLOYMENT,
    "CES2000000008": MacroCategory.EMPLOYMENT,
    "CES3000000008": MacroCategory.EMPLOYMENT,

    # Housing (Group 3)
    "HOUST": MacroCategory.HOUSING,
    "HOUSTNE": MacroCategory.HOUSING,
    "HOUSTMW": MacroCategory.HOUSING,
    "HOUSTS": MacroCategory.HOUSING,
    "HOUSTW": MacroCategory.HOUSING,
    "PERMIT": MacroCategory.HOUSING,
    "PERMITNE": MacroCategory.HOUSING,
    "PERMITMW": MacroCategory.HOUSING,
    "PERMITS": MacroCategory.HOUSING,
    "PERMITW": MacroCategory.HOUSING,

    # Consumption, Orders, and Inventories (Group 4)
    "DPCERA3M086SBEA": MacroCategory.CONSUMPTION,
    "CMRMTSPLx": MacroCategory.CONSUMPTION,
    "RETAILx": MacroCategory.CONSUMPTION,
    "AMDMNOx": MacroCategory.ECONOMIC_ACTIVITY,
    "ANDENOx": MacroCategory.ECONOMIC_ACTIVITY,
    "AMDMUOx": MacroCategory.ECONOMIC_ACTIVITY,
    "BUSINVx": MacroCategory.ECONOMIC_ACTIVITY,
    "ISRATIOx": MacroCategory.ECONOMIC_ACTIVITY,
    "UMCSENTx": MacroCategory.SENTIMENT,

    # Money and Credit (Group 5)
    "M1SL": MacroCategory.MONETARY_POLICY,
    "M2SL": MacroCategory.MONETARY_POLICY,
    "M2REAL": MacroCategory.MONETARY_POLICY,
    "AMBSL": MacroCategory.MONETARY_POLICY,
    "TOTRESNS": MacroCategory.MONETARY_POLICY,
    "NONBORRES": MacroCategory.MONETARY_POLICY,
    "BUSLOANS": MacroCategory.MONETARY_POLICY,
    "REALLN": MacroCategory.MONETARY_POLICY,
    "NONREVSL": MacroCategory.MONETARY_POLICY,
    "CONSPI": MacroCategory.MONETARY_POLICY,
    "DTCOLNVHFNM": MacroCategory.MONETARY_POLICY,
    "DTCTHFNM": MacroCategory.MONETARY_POLICY,
    "INVEST": MacroCategory.MONETARY_POLICY,

    # Interest Rates and Exchange Rates (Group 6)
    "FEDFUNDS": MacroCategory.MONETARY_POLICY,
    "CP3Mx": MacroCategory.MONETARY_POLICY,
    "TB3MS": MacroCategory.MONETARY_POLICY,
    "TB6MS": MacroCategory.MONETARY_POLICY,
    "GS1": MacroCategory.MONETARY_POLICY,
    "GS5": MacroCategory.MONETARY_POLICY,
    "GS10": MacroCategory.MONETARY_POLICY,
    "AAA": MacroCategory.MONETARY_POLICY,
    "BAA": MacroCategory.MONETARY_POLICY,
    "COMPAPFFx": MacroCategory.MONETARY_POLICY,
    "TB3SMFFM": MacroCategory.MONETARY_POLICY,
    "TB6SMFFM": MacroCategory.MONETARY_POLICY,
    "T1YFFM": MacroCategory.MONETARY_POLICY,
    "T5YFFM": MacroCategory.MONETARY_POLICY,
    "T10YFFM": MacroCategory.MONETARY_POLICY,
    "AAAFFM": MacroCategory.MONETARY_POLICY,
    "BAAFFM": MacroCategory.MONETARY_POLICY,
    "TWEXMMTH": MacroCategory.TRADE,
    "EXSZUSx": MacroCategory.TRADE,
    "EXJPUSx": MacroCategory.TRADE,
    "EXUSUKx": MacroCategory.TRADE,
    "EXCAUSx": MacroCategory.TRADE,

    # Prices (Group 7)
    "WPSFD49207": MacroCategory.INFLATION,
    "WPSFD49502": MacroCategory.INFLATION,
    "WPSID61": MacroCategory.INFLATION,
    "WPSID62": MacroCategory.INFLATION,
    "OILPRICEx": MacroCategory.INFLATION,
    "PPICMM": MacroCategory.INFLATION,
    "CPIAUCSL": MacroCategory.INFLATION,
    "CPIAPPSL": MacroCategory.INFLATION,
    "CPITRNSL": MacroCategory.INFLATION,
    "CPIMEDSL": MacroCategory.INFLATION,
    "CUSR0000SAC": MacroCategory.INFLATION,
    "CUSR0000SAD": MacroCategory.INFLATION,
    "CUSR0000SAS": MacroCategory.INFLATION,
    "CPIULFSL": MacroCategory.INFLATION,
    "CUSR0000SA0L2": MacroCategory.INFLATION,
    "CUSR0000SA0L5": MacroCategory.INFLATION,
    "PCEPI": MacroCategory.INFLATION,
    "DDURRG3M086SBEA": MacroCategory.INFLATION,
    "DNDGRG3M086SBEA": MacroCategory.INFLATION,
    "DSERRG3M086SBEA": MacroCategory.INFLATION,

    # Stock Market (Group 8)
    "S&P 500": MacroCategory.SENTIMENT,
    "S&P: indust": MacroCategory.SENTIMENT,
    "S&P div yield": MacroCategory.SENTIMENT,
    "S&P PE ratio": MacroCategory.SENTIMENT,
    "VXOCLSx": MacroCategory.SENTIMENT,
}

# Importance scores for FRED-MD series (1=low, 2=medium, 3=high)
FRED_MD_IMPORTANCE: Dict[str, int] = {
    # High importance (3)
    "INDPRO": 3,
    "UNRATE": 3,
    "PAYEMS": 3,
    "CPIAUCSL": 3,
    "FEDFUNDS": 3,
    "GS10": 3,
    "M2SL": 3,
    "HOUST": 3,
    "UMCSENTx": 3,
    "S&P 500": 3,
    "VXOCLSx": 3,
    "CLAIMSx": 3,
    "RETAILx": 3,

    # Medium importance (2) - default
}


@dataclass
class FREDMDConfig:
    """
    Configuration for FRED-MD data loading.

    :param data_url (str): URL to download FRED-MD data
    :param local_file (Path): Path to local CSV file (takes precedence over URL)
    :param cache_dir (Path): Directory to cache downloaded data
    :param start_date (str): Start date for data filtering
    :param end_date (str): End date for data filtering
    :param apply_transformations (bool): Whether to apply recommended transformations
    :param handle_missing (str): How to handle missing values ('drop', 'ffill', 'interpolate')
    """
    data_url: str = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
    local_file: Optional[Path] = None
    cache_dir: Optional[Path] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    apply_transformations: bool = True
    handle_missing: str = "ffill"


class FREDMDLoader:
    """
    Loader for FRED-MD macroeconomic database.

    This class downloads, processes, and transforms the FRED-MD dataset
    for use in factor allocation models.

    :param config (FREDMDConfig): Configuration options
    """

    def __init__(self, config: Optional[FREDMDConfig] = None):
        """
        Initialize the FRED-MD loader.

        :param config (FREDMDConfig): Configuration options
        """
        self.config = config or FREDMDConfig()
        self._raw_data: Optional[pd.DataFrame] = None
        self._transform_codes: Optional[pd.Series] = None
        self._transformed_data: Optional[pd.DataFrame] = None
        self._indicators: List[MacroIndicator] = []

    def load_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load FRED-MD data from local file, cache, or remote URL.

        Priority order:
        1. Local file (if specified in config)
        2. Cached file (if exists and not force_download)
        3. Remote download

        :param force_download (bool): Force re-download even if cached

        :return data (pd.DataFrame): FRED-MD data with datetime index
        """
        cache_path = None
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.config.cache_dir / "fred_md_current.csv"

        # Priority 1: Load from local file if specified
        if self.config.local_file and Path(self.config.local_file).exists():
            print(f"Loading FRED-MD data from local file: {self.config.local_file}")
            self._raw_data = pd.read_csv(self.config.local_file)
        # Priority 2: Load from cache
        elif cache_path and cache_path.exists() and not force_download:
            print(f"Loading FRED-MD data from cache: {cache_path}")
            self._raw_data = pd.read_csv(cache_path)
        # Priority 3: Download from remote
        else:
            print(f"Downloading FRED-MD data from: {self.config.data_url}")
            # Use requests with browser-like headers to avoid 403 errors
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            try:
                response = requests.get(self.config.data_url, headers=headers, timeout=60)
                response.raise_for_status()
                self._raw_data = pd.read_csv(StringIO(response.text))

                if cache_path:
                    self._raw_data.to_csv(cache_path, index=False)
                    print(f"Cached FRED-MD data to: {cache_path}")
            except requests.exceptions.HTTPError as e:
                error_msg = (
                    f"Failed to download FRED-MD data: {e}\n\n"
                    "The St. Louis Fed servers may be blocking automated requests.\n"
                    "Please download the file manually:\n"
                    "  1. Visit: https://research.stlouisfed.org/econ/mccracken/fred-databases/\n"
                    "  2. Click 'FRED-MD' and download 'current.csv'\n"
                    "  3. Either:\n"
                    f"     - Save to cache directory: {cache_path}\n"
                    "     - Or use local_file parameter in FREDMDConfig\n"
                )
                raise RuntimeError(error_msg) from e

        # Extract transformation codes from first row
        self._extract_transform_codes()

        # Process the data
        self._process_data()

        return self._transformed_data

    def _extract_transform_codes(self) -> None:
        """Extract transformation codes from the first row of data."""
        if self._raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # First row contains transformation codes
        first_row = self._raw_data.iloc[0]
        self._transform_codes = first_row.drop("sasdate", errors="ignore")

        # Remove the transformation codes row from data
        self._raw_data = self._raw_data.iloc[1:].copy()

    def _process_data(self) -> None:
        """Process raw FRED-MD data into usable format."""
        if self._raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self._raw_data.copy()

        # Parse dates
        df["date"] = pd.to_datetime(df["sasdate"], format="%m/%d/%Y")
        df = df.drop("sasdate", axis=1)
        df = df.set_index("date")

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Apply date filters
        if self.config.start_date:
            df = df[df.index >= self.config.start_date]
        if self.config.end_date:
            df = df[df.index <= self.config.end_date]

        # Handle missing values
        if self.config.handle_missing == "ffill":
            df = df.ffill()
        elif self.config.handle_missing == "interpolate":
            df = df.interpolate(method="linear")
        elif self.config.handle_missing == "drop":
            df = df.dropna(axis=1, how="any")

        # Apply transformations if configured
        if self.config.apply_transformations:
            df = self._apply_transformations(df)

        self._transformed_data = df

        # Build indicator definitions
        self._build_indicators()

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply recommended transformations to make series stationary.

        :param df (pd.DataFrame): Raw data

        :return transformed (pd.DataFrame): Transformed data
        """
        transformed = pd.DataFrame(index=df.index)

        for col in df.columns:
            if col not in self._transform_codes.index:
                continue

            try:
                tcode = int(float(self._transform_codes[col]))
            except (ValueError, TypeError):
                tcode = 1  # Default to levels

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

    def _build_indicators(self) -> None:
        """Build MacroIndicator definitions for each FRED-MD series."""
        if self._transformed_data is None:
            return

        self._indicators = []

        for col in self._transformed_data.columns:
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
        return self._indicators

    def get_indicator_names(self) -> List[str]:
        """
        Get list of indicator names.

        :return names (List[str]): Indicator names
        """
        return [ind.name for ind in self._indicators]

    def create_macro_dataframe(self) -> pd.DataFrame:
        """
        Create macro data DataFrame in format compatible with existing pipeline.

        :return macro_data (pd.DataFrame): DataFrame with columns:
            timestamp, indicator, category, region, importance,
            publication_type, value, normalized_value, surprise, ma5
        """
        if self._transformed_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        records = []
        df = self._transformed_data

        for date in df.index:
            for indicator in self._indicators:
                if indicator.name not in df.columns:
                    continue

                value = df.loc[date, indicator.name]
                if pd.isna(value):
                    continue

                # Calculate normalized value (z-score based on historical data)
                historical = df.loc[:date, indicator.name].dropna()
                if len(historical) > 12:
                    mean_val = historical.mean()
                    std_val = historical.std()
                    normalized = (value - mean_val) / std_val if std_val > 0 else 0.0
                else:
                    normalized = 0.0

                # Calculate surprise (deviation from 12-month MA)
                ma12 = historical.tail(12).mean() if len(historical) >= 12 else historical.mean()
                std_surprise = historical.tail(24).std() if len(historical) >= 24 else historical.std()
                surprise = (value - ma12) / std_surprise if std_surprise > 0 else 0.0

                # Calculate 5-period moving average
                ma5 = historical.tail(5).mean() if len(historical) >= 5 else value

                records.append({
                    "timestamp": date,
                    "indicator": indicator.name,
                    "category": indicator.category.value,
                    "region": indicator.region.value,
                    "importance": indicator.importance,
                    "publication_type": PublicationType.FINAL.value,
                    "value": value,
                    "normalized_value": np.clip(normalized, -4, 4) / 4,  # Scale to [-1, 1]
                    "surprise": np.clip(surprise, -4, 4),
                    "ma5": ma5,
                })

        return pd.DataFrame(records)

    def create_market_context(self) -> pd.DataFrame:
        """
        Create market context DataFrame from FRED-MD data.

        Uses VXO (VIX predecessor), credit spreads, and yield curve.

        :return market_data (pd.DataFrame): DataFrame with columns:
            timestamp, vix, credit_spread, yield_curve
        """
        if self._transformed_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self._transformed_data
        records = []

        for date in df.index:
            # VIX proxy (VXO)
            vix = df.loc[date, "VXOCLSx"] if "VXOCLSx" in df.columns else 15.0
            if pd.isna(vix):
                vix = 15.0

            # Credit spread (BAA - AAA)
            baa = df.loc[date, "BAA"] if "BAA" in df.columns else 6.0
            aaa = df.loc[date, "AAA"] if "AAA" in df.columns else 4.0
            if pd.isna(baa):
                baa = 6.0
            if pd.isna(aaa):
                aaa = 4.0
            credit_spread = baa - aaa

            # Yield curve (10Y - Fed Funds)
            gs10 = df.loc[date, "GS10"] if "GS10" in df.columns else 4.0
            fedfunds = df.loc[date, "FEDFUNDS"] if "FEDFUNDS" in df.columns else 2.0
            if pd.isna(gs10):
                gs10 = 4.0
            if pd.isna(fedfunds):
                fedfunds = 2.0
            yield_curve = gs10 - fedfunds

            records.append({
                "timestamp": date,
                "vix": vix,
                "credit_spread": credit_spread,
                "yield_curve": yield_curve,
            })

        return pd.DataFrame(records)

    def get_transformed_data(self) -> pd.DataFrame:
        """
        Get the transformed FRED-MD data.

        :return data (pd.DataFrame): Transformed data with datetime index
        """
        if self._transformed_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self._transformed_data

    def get_raw_data(self) -> pd.DataFrame:
        """
        Get the raw (untransformed) FRED-MD data.

        :return data (pd.DataFrame): Raw data
        """
        if self._raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self._raw_data

    def summary(self) -> Dict:
        """
        Get summary statistics of loaded data.

        :return summary (Dict): Summary statistics
        """
        if self._transformed_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self._transformed_data

        return {
            "n_series": len(df.columns),
            "n_observations": len(df),
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date": df.index.max().strftime("%Y-%m-%d"),
            "missing_pct": (df.isna().sum().sum() / df.size * 100),
            "categories": {
                cat.value: sum(1 for ind in self._indicators if ind.category == cat)
                for cat in MacroCategory
            },
        }


def load_fred_md_dataset(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[MacroIndicator]]:
    """
    Convenience function to load FRED-MD dataset.

    :param start_date (str): Start date (e.g., "2000-01-01")
    :param end_date (str): End date (e.g., "2024-12-31")
    :param cache_dir (str): Directory to cache downloaded data
    :param local_file (str): Path to local FRED-MD CSV file

    :return macro_data (pd.DataFrame): Macro token data
    :return market_data (pd.DataFrame): Market context data
    :return indicators (List[MacroIndicator]): Indicator definitions
    """
    config = FREDMDConfig(
        start_date=start_date,
        end_date=end_date,
        cache_dir=Path(cache_dir) if cache_dir else None,
        local_file=Path(local_file) if local_file else None,
    )

    loader = FREDMDLoader(config)
    loader.load_data()

    macro_data = loader.create_macro_dataframe()
    market_data = loader.create_market_context()
    indicators = loader.get_indicators()

    print(f"Loaded FRED-MD data:")
    summary = loader.summary()
    print(f"  Series: {summary['n_series']}")
    print(f"  Observations: {summary['n_observations']}")
    print(f"  Period: {summary['start_date']} to {summary['end_date']}")
    print(f"  Missing: {summary['missing_pct']:.2f}%")

    return macro_data, market_data, indicators
