"""
Data loading utilities for macroeconomic and factor return data.

This module provides the interface for loading real-world data from various APIs.
For now, it works with synthetic data but includes documentation on production APIs.

=============================================================================
DATA SOURCES FOR PRODUCTION USE
=============================================================================

1. MACROECONOMIC DATA SOURCES
-----------------------------

a) FRED (Federal Reserve Economic Data) - FREE
   URL: https://fred.stlouisfed.org/
   API: https://api.stlouisfed.org/docs/fred/
   Coverage: 800,000+ US economic time series
   Key indicators: GDP, CPI, Unemployment, PMI, Industrial Production
   Python library: `fredapi` (pip install fredapi)
   Limitations: US-focused, no point-in-time vintages in standard API

b) ALFRED (ArchivaL FRED) - FREE
   URL: https://alfred.stlouisfed.org/
   Coverage: Point-in-time database with historical vintages
   Key feature: CRITICAL for avoiding look-ahead bias
   Python: Use FRED API with vintage dates

c) Quandl / Nasdaq Data Link - FREE TIER + PAID
   URL: https://data.nasdaq.com/
   Free tier: Limited datasets (Wiki EOD, FRED mirror)
   Paid: Premium datasets (institutional quality)
   Python library: `nasdaqdatalink` (pip install nasdaq-data-link)

d) Bloomberg Terminal API - PAID (EXPENSIVE ~$24,000/year)
   Coverage: Most comprehensive macro data with point-in-time
   Key feature: BSRCH for economic calendar, ECST for surprises
   Python: `blpapi` (requires terminal subscription)
   Best for: Production institutional systems

e) Refinitiv Eikon/DataStream - PAID (~$15,000-22,000/year)
   Coverage: Global macro data with vintage reconstruction
   Key feature: Excellent point-in-time database
   Python: `eikon` or `refinitiv.data`
   Best for: Global macro analysis

f) Trading Economics API - PAID ($49-299/month)
   URL: https://tradingeconomics.com/api/
   Coverage: 300,000+ indicators from 196 countries
   Key feature: Consensus estimates and surprises
   Python: `tradingeconomics` (pip install tradingeconomics)

g) Alpha Vantage - FREE TIER (limited) + PAID ($49.99/month)
   URL: https://www.alphavantage.co/
   Coverage: Some macro indicators, mainly equities
   Python: `alpha_vantage` (pip install alpha_vantage)

2. FACTOR RETURN DATA SOURCES
-----------------------------

a) Kenneth French Data Library - FREE
   URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
   Coverage: US Fama-French factors (value, size, momentum, etc.)
   Key feature: Long history, academic standard
   Python: `pandas_datareader` (pip install pandas-datareader)

b) MSCI Factor Indices - PAID (varies by coverage)
   URL: https://www.msci.com/factor-indexes
   Coverage: Global factor indices (value, momentum, quality, etc.)
   Key feature: Industry standard for factor definitions
   Access: Via Bloomberg/Refinitiv or direct MSCI subscription

c) S&P Dow Jones Indices - PAID (via data vendors)
   Coverage: Pure Style Indices (value, growth)
   Access: Via Bloomberg, Refinitiv, or direct subscription

d) AQR Data Library - FREE
   URL: https://www.aqr.com/Insights/Datasets
   Coverage: Factor returns, betting against beta, etc.
   Key feature: Academic research quality, free

e) Yahoo Finance - FREE (via yfinance)
   Coverage: ETF prices can proxy factor returns
   Factor ETFs: IWD (value), IWF (growth), XLY (cyclical), XLP (defensive)
   Python: `yfinance` (pip install yfinance)
   Limitations: Not point-in-time, subject to survivorship bias

3. MARKET CONTEXT DATA
----------------------

a) VIX Index - FREE
   Source: CBOE via Yahoo Finance or FRED
   Symbol: ^VIX (Yahoo) or VIXCLS (FRED)

b) Treasury Yields - FREE
   Source: FRED
   Symbols: DGS2 (2-year), DGS10 (10-year)

c) Credit Spreads - FREE/PAID
   Source: FRED (ICE BofA indices)
   Symbols: BAMLH0A0HYM2EY (HY), BAMLC0A4CBBBEY (BBB)

=============================================================================
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class PublicationType(Enum):
    """Types of macroeconomic data publications."""
    CONSENSUS = "consensus"
    CONSENSUS_REVISION = "consensus_revision"
    ESTIMATE = "estimate"
    ESTIMATE_REVISION = "estimate_revision"
    FLASH = "flash"
    FINAL = "final"


class MacroCategory(Enum):
    """Thematic categories for macroeconomic indicators."""
    ECONOMIC_ACTIVITY = "economic_activity"
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    CONSUMPTION = "consumption"
    MONETARY_POLICY = "monetary_policy"
    HOUSING = "housing"
    TRADE = "trade"
    SENTIMENT = "sentiment"


class Region(Enum):
    """Geographic regions for regional model separation."""
    US = "us"
    EUROPE = "europe"
    JAPAN = "japan"
    UK = "uk"
    CHINA = "china"
    EMERGING = "emerging"


@dataclass
class MacroIndicator:
    """
    Definition of a macroeconomic indicator.

    :param name (str): Unique identifier (e.g., PMI_Manufacturing)
    :param category (MacroCategory): Thematic category
    :param region (Region): Geographic region
    :param importance (int): Importance score 1-3
    :param periodicity (str): Publication frequency (daily, weekly, monthly)
    """
    name: str
    category: MacroCategory
    region: Region
    importance: int
    periodicity: str


@dataclass
class MacroToken:
    """
    A single macroeconomic data token in the input sequence.

    :param indicator (MacroIndicator): The indicator definition
    :param publication_type (PublicationType): Type of publication
    :param normalized_value (float): Value normalized by historical distribution
    :param surprise (float): Standardized surprise vs expectation
    :param ma5 (float): 5-period moving average
    :param days_offset (int): Days relative to current date (negative = past)
    :param timestamp (pd.Timestamp): Actual publication timestamp
    """
    indicator: MacroIndicator
    publication_type: PublicationType
    normalized_value: float
    surprise: float
    ma5: float
    days_offset: int
    timestamp: pd.Timestamp


@dataclass
class FactorReturns:
    """
    Factor category returns for a given period.

    :param timestamp (pd.Timestamp): Period end date
    :param cyclical (float): Return of cyclical factor
    :param defensive (float): Return of defensive factor
    :param value (float): Return of value factor
    :param growth (float): Return of growth factor
    :param quality (float): Return of quality factor
    :param momentum (float): Return of momentum factor
    """
    timestamp: pd.Timestamp
    cyclical: float
    defensive: float
    value: float
    growth: float
    quality: float
    momentum: float


class MacroDataLoader:
    """
    Data loader for macroeconomic and factor return data.

    In production, this class would connect to real APIs.
    For development, it works with synthetic data.

    :param region (Region): Target region for data loading
    :param sequence_length (int): Number of tokens in input sequence
    """

    # Standard macro indicators per region
    INDICATOR_DEFINITIONS: Dict[Region, List[MacroIndicator]] = {
        Region.US: [
            MacroIndicator("PMI_Manufacturing", MacroCategory.ECONOMIC_ACTIVITY, Region.US, 3, "monthly"),
            MacroIndicator("PMI_Services", MacroCategory.ECONOMIC_ACTIVITY, Region.US, 3, "monthly"),
            MacroIndicator("NFP", MacroCategory.EMPLOYMENT, Region.US, 3, "monthly"),
            MacroIndicator("Unemployment_Rate", MacroCategory.EMPLOYMENT, Region.US, 3, "monthly"),
            MacroIndicator("CPI_Core", MacroCategory.INFLATION, Region.US, 3, "monthly"),
            MacroIndicator("CPI_Headline", MacroCategory.INFLATION, Region.US, 2, "monthly"),
            MacroIndicator("PPI", MacroCategory.INFLATION, Region.US, 2, "monthly"),
            MacroIndicator("Retail_Sales", MacroCategory.CONSUMPTION, Region.US, 2, "monthly"),
            MacroIndicator("Consumer_Confidence", MacroCategory.SENTIMENT, Region.US, 2, "monthly"),
            MacroIndicator("Industrial_Production", MacroCategory.ECONOMIC_ACTIVITY, Region.US, 2, "monthly"),
            MacroIndicator("Housing_Starts", MacroCategory.HOUSING, Region.US, 2, "monthly"),
            MacroIndicator("Initial_Jobless_Claims", MacroCategory.EMPLOYMENT, Region.US, 2, "weekly"),
            MacroIndicator("Fed_Funds_Rate", MacroCategory.MONETARY_POLICY, Region.US, 3, "irregular"),
            MacroIndicator("GDP_QoQ", MacroCategory.ECONOMIC_ACTIVITY, Region.US, 3, "quarterly"),
            MacroIndicator("Trade_Balance", MacroCategory.TRADE, Region.US, 1, "monthly"),
        ],
        Region.EUROPE: [
            MacroIndicator("EU_PMI_Manufacturing", MacroCategory.ECONOMIC_ACTIVITY, Region.EUROPE, 3, "monthly"),
            MacroIndicator("EU_PMI_Services", MacroCategory.ECONOMIC_ACTIVITY, Region.EUROPE, 3, "monthly"),
            MacroIndicator("EU_CPI_Core", MacroCategory.INFLATION, Region.EUROPE, 3, "monthly"),
            MacroIndicator("EU_Unemployment", MacroCategory.EMPLOYMENT, Region.EUROPE, 2, "monthly"),
            MacroIndicator("German_IFO", MacroCategory.SENTIMENT, Region.EUROPE, 3, "monthly"),
            MacroIndicator("German_ZEW", MacroCategory.SENTIMENT, Region.EUROPE, 2, "monthly"),
            MacroIndicator("ECB_Rate", MacroCategory.MONETARY_POLICY, Region.EUROPE, 3, "irregular"),
            MacroIndicator("EU_GDP_QoQ", MacroCategory.ECONOMIC_ACTIVITY, Region.EUROPE, 3, "quarterly"),
            MacroIndicator("EU_Retail_Sales", MacroCategory.CONSUMPTION, Region.EUROPE, 2, "monthly"),
            MacroIndicator("EU_Industrial_Production", MacroCategory.ECONOMIC_ACTIVITY, Region.EUROPE, 2, "monthly"),
        ],
        Region.JAPAN: [
            MacroIndicator("JP_PMI_Manufacturing", MacroCategory.ECONOMIC_ACTIVITY, Region.JAPAN, 3, "monthly"),
            MacroIndicator("JP_CPI", MacroCategory.INFLATION, Region.JAPAN, 3, "monthly"),
            MacroIndicator("JP_Tankan", MacroCategory.SENTIMENT, Region.JAPAN, 3, "quarterly"),
            MacroIndicator("JP_Industrial_Production", MacroCategory.ECONOMIC_ACTIVITY, Region.JAPAN, 2, "monthly"),
            MacroIndicator("JP_Unemployment", MacroCategory.EMPLOYMENT, Region.JAPAN, 2, "monthly"),
            MacroIndicator("BOJ_Rate", MacroCategory.MONETARY_POLICY, Region.JAPAN, 3, "irregular"),
            MacroIndicator("JP_GDP_QoQ", MacroCategory.ECONOMIC_ACTIVITY, Region.JAPAN, 3, "quarterly"),
            MacroIndicator("JP_Trade_Balance", MacroCategory.TRADE, Region.JAPAN, 2, "monthly"),
        ],
    }

    def __init__(self, region: Region, sequence_length: int = 50):
        """
        Initialize the data loader.

        :param region (Region): Target region
        :param sequence_length (int): Number of tokens per sequence
        """
        self.region = region
        self.sequence_length = sequence_length
        self.indicators = self.INDICATOR_DEFINITIONS.get(region, [])

    def load_macro_sequence(
        self,
        as_of_date: pd.Timestamp,
        macro_data: pd.DataFrame,
    ) -> List[MacroToken]:
        """
        Load macro token sequence as of a specific date.

        :param as_of_date (pd.Timestamp): Point-in-time date
        :param macro_data (pd.DataFrame): Raw macro data with columns:
            indicator, publication_type, value, expected, timestamp

        :return tokens (List[MacroToken]): Sequence of macro tokens
        """
        # Filter to data available as of the date
        available_data = macro_data[macro_data["timestamp"] <= as_of_date].copy()
        available_data = available_data.sort_values("timestamp", ascending=False)

        tokens = []
        for _, row in available_data.head(self.sequence_length).iterrows():
            indicator_def = self._get_indicator_def(row["indicator"])
            if indicator_def is None:
                continue

            token = MacroToken(
                indicator=indicator_def,
                publication_type=PublicationType(row["publication_type"]),
                normalized_value=row.get("normalized_value", 0.0),
                surprise=row.get("surprise", 0.0),
                ma5=row.get("ma5", 0.0),
                days_offset=(row["timestamp"] - as_of_date).days,
                timestamp=row["timestamp"],
            )
            tokens.append(token)

        return tokens

    def load_factor_returns(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        factor_data: pd.DataFrame,
    ) -> List[FactorReturns]:
        """
        Load factor returns for a date range.

        :param start_date (pd.Timestamp): Start of period
        :param end_date (pd.Timestamp): End of period
        :param factor_data (pd.DataFrame): Raw factor return data

        :return returns (List[FactorReturns]): List of factor returns by period
        """
        filtered = factor_data[
            (factor_data["timestamp"] >= start_date) &
            (factor_data["timestamp"] <= end_date)
        ]

        returns = []
        for _, row in filtered.iterrows():
            factor_return = FactorReturns(
                timestamp=row["timestamp"],
                cyclical=row.get("cyclical", 0.0),
                defensive=row.get("defensive", 0.0),
                value=row.get("value", 0.0),
                growth=row.get("growth", 0.0),
                quality=row.get("quality", 0.0),
                momentum=row.get("momentum", 0.0),
            )
            returns.append(factor_return)

        return returns

    def load_market_context(
        self,
        as_of_date: pd.Timestamp,
        market_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Load market context indicators as of a specific date.

        :param as_of_date (pd.Timestamp): Point-in-time date
        :param market_data (pd.DataFrame): Market context data

        :return context (Dict[str, float]): Dictionary with VIX, credit_spread, yield_curve
        """
        available = market_data[market_data["timestamp"] <= as_of_date]
        if available.empty:
            return {"vix": 0.0, "credit_spread": 0.0, "yield_curve": 0.0}

        latest = available.iloc[-1]
        return {
            "vix": latest.get("vix", 0.0),
            "credit_spread": latest.get("credit_spread", 0.0),
            "yield_curve": latest.get("yield_curve", 0.0),
        }

    def _get_indicator_def(self, name: str) -> Optional[MacroIndicator]:
        """
        Get indicator definition by name.

        :param name (str): Indicator name

        :return indicator (Optional[MacroIndicator]): Indicator definition or None
        """
        for indicator in self.indicators:
            if indicator.name == name:
                return indicator
        return None

    def get_indicator_names(self) -> List[str]:
        """
        Get list of indicator names for this region.

        :return names (List[str]): List of indicator names
        """
        return [ind.name for ind in self.indicators]

    def get_num_indicators(self) -> int:
        """
        Get number of indicators for this region.

        :return count (int): Number of indicators
        """
        return len(self.indicators)
