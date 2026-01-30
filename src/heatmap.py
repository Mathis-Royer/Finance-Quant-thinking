#!/usr/bin/env python3
"""
S&P 500 Market heatmap (treemap) by year using Yahoo Finance API:
- Rectangle size = market capitalization
- Rectangle color = annual performance (red to green gradient)

Data is fetched from Yahoo Finance (yfinance) for S&P 500 companies.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

# Fix SSL certificate issues with curl_cffi (used by yfinance)
if os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert a hex color string to RGB tuple.

    :param hex_color (str): Hex color string (e.g., "#ff0000")

    :return (Tuple[int, int, int]): RGB values as tuple
    """
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color string.

    :param rgb (Tuple[int, int, int]): RGB values as tuple

    :return (str): Hex color string
    """
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.

    :param a (float): Start value
    :param b (float): End value
    :param t (float): Interpolation factor (0-1)

    :return (float): Interpolated value
    """
    return a + (b - a) * t


def interpolate_hex(start_hex: str, end_hex: str, t: float) -> str:
    """
    Interpolate between two hex colors.

    :param start_hex (str): Start hex color
    :param end_hex (str): End hex color
    :param t (float): Interpolation factor (0-1)

    :return (str): Interpolated hex color
    """
    t = max(0.0, min(1.0, t))
    sr, sg, sb = hex_to_rgb(start_hex)
    er, eg, eb = hex_to_rgb(end_hex)
    r = int(round(lerp(sr, er, t)))
    g = int(round(lerp(sg, eg, t)))
    b = int(round(lerp(sb, eb, t)))
    return rgb_to_hex((r, g, b))


def safe_min_max(series: pd.Series) -> Tuple[float, float]:
    """
    Get min and max of a series, handling empty series.

    :param series (pd.Series): Input series

    :return (Tuple[float, float]): Min and max values
    """
    s = series.dropna()
    if s.empty:
        return 0.0, 0.0
    return float(s.min()), float(s.max())


def get_sp500_tickers(cache_dir: str | None = None) -> pd.DataFrame:
    """
    Fetch S&P 500 tickers and sectors from Wikipedia, with optional caching.

    :param cache_dir (str | None): Directory to cache the ticker list. If None, no caching.

    :return (pd.DataFrame): DataFrame with columns: ticker, sector
    """
    import requests
    from io import StringIO

    if cache_dir:
        cache_path = os.path.join(cache_dir, "sp500_tickers.csv")
        if os.path.exists(cache_path):
            print(f"Loading S&P 500 tickers from cache: {cache_path}")
            return pd.read_csv(cache_path)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    df = df[["Symbol", "GICS Sector"]].copy()
    df.columns = ["ticker", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df = pd.DataFrame(df)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Cached S&P 500 tickers to: {cache_path}")

    return df


def fetch_market_data(tickers: List[str], year: int, cache_dir: str | None = None) -> pd.DataFrame:
    """
    Fetch market data (price history and market cap) for given tickers and year, with optional caching.

    :param tickers (List[str]): List of ticker symbols
    :param year (int): Year to fetch data for
    :param cache_dir (str | None): Directory to cache market data. If None, no caching.

    :return (pd.DataFrame): DataFrame with columns: ticker, market_cap, return
    """
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"market_data_{year}.csv")
        if os.path.exists(cache_path):
            print(f"Loading market data for {year} from cache: {cache_path}")
            return pd.read_csv(cache_path)

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    results = []
    batch_size = 50

    print(f"Fetching data for {len(tickers)} tickers for year {year}...")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_str = " ".join(batch)

        try:
            data = yf.download(
                batch_str,
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
            )

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_data = data
                    else:
                        if ticker not in data.columns.get_level_values(0):
                            continue
                        ticker_data = data[ticker]

                    if ticker_data.empty or len(ticker_data) < 2:
                        continue

                    first_close = ticker_data["Close"].dropna().iloc[0]
                    last_close = ticker_data["Close"].dropna().iloc[-1]

                    if first_close > 0:
                        annual_return = (last_close - first_close) / first_close
                    else:
                        continue

                    stock = yf.Ticker(ticker)
                    info = stock.info
                    market_cap = info.get("marketCap", None)

                    if market_cap and market_cap > 0:
                        results.append(
                            {
                                "ticker": ticker,
                                "market_cap": market_cap,
                                "return": annual_return,
                            }
                        )
                except Exception:
                    continue

        except Exception as e:
            print(f"Error fetching batch: {e}")
            continue

        print(f"  Processed {min(i + batch_size, len(tickers))}/{len(tickers)} tickers...")

    df = pd.DataFrame(results)

    if cache_dir and not df.empty:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Cached market data for {year} to: {cache_path}")

    return df


def compute_colors(df_year: pd.DataFrame) -> pd.DataFrame:
    """
    Compute colors for each stock based on return.

    Color scheme:
    - Red: return < 0%
    - Red to Green: 0% to 10%
    - Green to Dark Green: 10% to 100%
    - Dark Green to Blue: 100% to 1000%

    :param df_year (pd.DataFrame): DataFrame with market data

    :return (pd.DataFrame): DataFrame with color assignments
    """
    df = df_year.copy()

    df = df[df["market_cap"].notna() & df["return"].notna()]
    df = df[df["market_cap"] > 0]

    if df.empty:
        return df

    red_hex = "#d62728"        # Red for < 0%
    green_hex = "#2ca02c"      # Green for 10%
    dark_green_hex = "#006400" # Dark green for 100%
    blue_hex = "#1f77b4"       # Blue for 1000%

    def get_color(ret: float) -> str:
        if ret < 0:
            # Negative returns: solid red
            return red_hex
        elif ret <= 0.10:
            # 0% to 10%: red to green gradient
            t = ret / 0.10  # 0% -> 0, 10% -> 1
            return interpolate_hex(red_hex, green_hex, t)
        elif ret <= 1.0:
            # 10% to 100%: green to dark green gradient
            t = (ret - 0.10) / 0.90  # 10% -> 0, 100% -> 1
            return interpolate_hex(green_hex, dark_green_hex, t)
        else:
            # 100% to 1000%: dark green to blue gradient
            t = min(1.0, (ret - 1.0) / 9.0)  # 100% -> 0, 1000% -> 1
            return interpolate_hex(dark_green_hex, blue_hex, t)

    df["color"] = df["return"].apply(get_color)

    return df


def build_treemap(
    df: pd.DataFrame,
    year: int,
    title_prefix: str,
) -> go.Figure:
    """
    Build a Plotly treemap figure from market data.

    :param df (pd.DataFrame): DataFrame with market data and colors
    :param year (int): Year for the title
    :param title_prefix (str): Prefix for the figure title

    :return (go.Figure): Plotly treemap figure
    """
    labels = ["Market"] + df["ticker"].astype(str).tolist()
    parents = [""] + ["Market"] * len(df)
    values = [df["market_cap"].sum()] + df["market_cap"].astype(float).tolist()
    node_colors = ["#ffffff"] + df["color"].tolist()
    hover_text = [""] + [
        f"{t}<br>Market cap: ${mc:,.0f}<br>Return: {ret*100:.2f}%"
        for t, mc, ret in zip(df["ticker"], df["market_cap"], df["return"])
    ]

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=node_colors, line=dict(width=0.5, color="#ffffff")),
            hovertext=hover_text,
            hoverinfo="text",
            branchvalues="total",
        )
    )

    legend_text = (
        "Color:<br>"
        "Red = &lt; 0%<br>"
        "Red → Green = 0% → 10%<br>"
        "Green → Dark green = 10% → 100%<br>"
        "Dark green → Blue = 100% → 1000%"
    )

    fig.update_layout(
        title=f"{title_prefix} {year}",
        margin=dict(t=60, l=10, r=10, b=10),
        annotations=[
            dict(
                text=legend_text,
                x=1.0,
                y=1.0,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                borderpad=6,
                font=dict(size=12),
            )
        ],
    )
    return fig


def build_gif(frame_paths: List[str], gif_path: str, frame_duration_sec: float = 1.0) -> None:
    """
    Build a GIF from a list of PNG frames.

    :param frame_paths (List[str]): List of paths to PNG frames
    :param gif_path (str): Output path for the GIF
    :param frame_duration_sec (float): Duration of each frame in seconds
    """
    if imageio is None:
        raise ImportError("imageio is required for GIF export. Install with: pip install imageio")

    images = [imageio.imread(p) for p in frame_paths]
    os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else ".", exist_ok=True)
    # imageio expects duration in milliseconds
    imageio.mimsave(gif_path, images, duration=frame_duration_sec * 1000)


def main() -> None:
    """Main function to generate S&P 500 market heatmaps."""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Create yearly S&P 500 market heatmap treemaps using Yahoo Finance API."
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(script_dir, "out_market_heatmaps"),
        help="Output directory for HTML (and optional PNG)",
    )
    parser.add_argument(
        "--years",
        default="2024",
        help="Comma-separated list of years to export (e.g., 2022,2023,2024)",
    )
    parser.add_argument(
        "--title",
        default="S&P 500 Heatmap",
        help="Figure title prefix",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export PNG (requires plotly+kaleido installed)",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Export a GIF animation from all years (requires imageio and kaleido)",
    )
    parser.add_argument(
        "--gif-duration",
        type=float,
        default=1.0,
        help="Duration of each frame in the GIF in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(script_dir, "cache"),
        help="Directory to cache S&P 500 tickers and market data (avoids re-fetching)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-fetch data even if cache exists",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if args.no_cache:
        cache_dir = None

    print("Fetching S&P 500 ticker list from Wikipedia...")
    sp500_df = get_sp500_tickers(cache_dir=cache_dir)
    tickers = sp500_df["ticker"].tolist()

    os.makedirs(args.outdir, exist_ok=True)

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    frame_paths: List[str] = []

    for year in years:
        print(f"\n--- Processing year {year} ---")

        df_year = fetch_market_data(tickers, year, cache_dir=cache_dir)

        if df_year.empty:
            print(f"No data available for year {year}, skipping...")
            continue

        df_year = compute_colors(df_year)

        fig = build_treemap(df_year, year=year, title_prefix=args.title)

        html_path = os.path.join(args.outdir, f"sp500_heatmap_{year}.html")
        fig.write_html(html_path)

        if args.png or args.gif:
            png_path = os.path.join(args.outdir, f"sp500_heatmap_{year}.png")
            fig.write_image(png_path, scale=2)
            frame_paths.append(png_path)
            print(f"Saved: {html_path} and {png_path}")
        else:
            print(f"Saved: {html_path}")

    if args.gif:
        if not frame_paths:
            print("No frames generated, cannot create GIF.")
        else:
            gif_path = os.path.join(args.outdir, "sp500_heatmap.gif")
            build_gif(frame_paths, gif_path, args.gif_duration)
            print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()
