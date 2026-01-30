# S&P 500 Market Heatmap

Generate interactive treemap heatmaps of S&P 500 stocks using Yahoo Finance data.

Rectangle size represents market capitalization, color represents annual return performance.

## Installation

```bash
# From the project root
cd src

# Create virtual environment
python3 -m venv .venv --without-pip
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install dependencies
pip install -r requirements.txt
```

## Usage

All commands are run from the project root directory (`Hedge_quant/`).

### Basic Usage

```bash
# Generate heatmap for current year
src/.venv/bin/python src/heatmap.py --years 2024
```

### Multiple Years

```bash
# Generate heatmaps for 3 years
src/.venv/bin/python src/heatmap.py --years 2022,2023,2024
```

### 10-Year GIF Animation

```bash
# Generate 10-year animation (2015-2024)
src/.venv/bin/python src/heatmap.py --years 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 --gif

# With slower animation (1 second per frame)
src/.venv/bin/python src/heatmap.py --years 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 --gif --gif-duration 1.0
```

### Cache Management

```bash
# Force refresh data (ignore cache)
src/.venv/bin/python src/heatmap.py --years 2024 --no-cache
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--years` | `2024` | Comma-separated list of years to generate |
| `--outdir` | `src/out_market_heatmaps` | Output directory for generated files |
| `--cache-dir` | `src/cache` | Cache directory for API data (speeds up subsequent runs) |
| `--png` | `false` | Export PNG images |
| `--gif` | `false` | Export GIF animation from all years |
| `--gif-duration` | `1.0` | Duration of each frame in the GIF in seconds |
| `--no-cache` | `false` | Force re-fetch data from Yahoo Finance |
| `--title` | `S&P 500 Heatmap` | Title prefix for the figures |

## Output Files

| File | Description |
|------|-------------|
| `sp500_heatmap_<YEAR>.html` | Interactive treemap (viewable in browser) |
| `sp500_heatmap_<YEAR>.png` | Static PNG image (if `--png` or `--gif`) |
| `sp500_heatmap.gif` | Animated GIF of all years (if `--gif`) |

All files are saved to `src/out_market_heatmaps/`.

## Color Legend

| Color | Return |
|-------|--------|
| Red | < 0% |
| Red → Green | 0% → 10% |
| Green → Dark Green | 10% → 100% |
| Dark Green → Blue | 100% → 1000% |

## Data Sources

- **Ticker list**: Wikipedia (S&P 500 component list)
- **Market data**: Yahoo Finance API (via `yfinance`)

Data is cached locally after first fetch to speed up subsequent runs.

## Notes

- First run for a year takes several minutes (fetching ~500 tickers)
- Subsequent runs use cached data and are near-instant
- Yahoo Finance may have rate limits; the script handles errors gracefully
