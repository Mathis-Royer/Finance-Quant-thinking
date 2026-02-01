"""
Factor Allocation Strategy Dashboard.

Interactive Streamlit dashboard for exploring model results.
Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Import shared utilities from src
from comparison_runner import compute_composite_score as _compute_composite_score
from utils.constants import (
    MODEL_TYPE_ABBREV,
    CONFIG_SUFFIX,
    STRATEGY_ABBREV,
    ALLOCATION_ABBREV,
)
from visualization.colormaps import CONFIG_COLORS
from utils.statistics import (
    bootstrap_sharpe_ratio,
    test_sharpe_significance,
    compare_sharpe_ratios,
)

# Page configuration
st.set_page_config(
    page_title="Factor Allocation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .filter-section {
        background-color: #e8eaf6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_holdout_results() -> pd.DataFrame:
    """
    Load holdout results from saved cache or compute fresh.

    Returns DataFrame with columns:
    - strategy, allocation, horizon, model_type
    - sharpe, ic, maxdd, total_return, score, rank
    """
    cache_path = project_root / "data_cache" / "holdout_results.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df

    # If no cache, create sample data structure (user needs to run notebook first)
    st.warning("No cached results found. Please run the notebook first to generate results.")
    return create_sample_data()


def create_sample_data() -> pd.DataFrame:
    """Create sample data for demo purposes."""
    data = []
    strategies = ["Sup", "E2E"]
    allocations = ["Multi", "Binary"]
    horizons = [1, 3, 6, 12]
    configs = ["baseline"]  # Sample data only uses baseline
    model_types = ["Final", "Fair Ensemble", "WF Ensemble"]

    np.random.seed(42)

    for strategy in strategies:
        for allocation in allocations:
            for horizon in horizons:
                for config in configs:
                    for model_type in model_types:
                        base_sharpe = 0.5 if allocation == "Multi" else 0.2
                        base_sharpe += 0.1 if strategy == "Sup" else 0
                        base_sharpe += np.random.uniform(-0.3, 0.3)

                        data.append({
                            "strategy": strategy,
                            "allocation": allocation,
                            "horizon": horizon,
                            "config": config,
                            "model_type": model_type,
                            "sharpe": base_sharpe,
                            "ic": np.random.uniform(-0.3, 0.3),
                            "maxdd": np.random.uniform(-0.15, -0.02),
                            "total_return": np.random.uniform(-0.05, 0.10),
                        })

    df = pd.DataFrame(data)
    df = compute_score(df)
    return df


def compute_score(
    df: pd.DataFrame,
    sharpe_weight: float = 0.35,
    ic_weight: float = 0.25,
    maxdd_weight: float = 0.30,
    return_weight: float = 0.10,
    reject_negative_ic_threshold: float = -0.3,
) -> pd.DataFrame:
    """
    Compute composite score. Wrapper around comparison_runner.compute_composite_score().

    :param df (pd.DataFrame): DataFrame with sharpe, ic, maxdd, total_return columns
    :param sharpe_weight (float): Weight for Sharpe ratio in score
    :param ic_weight (float): Weight for IC in score
    :param maxdd_weight (float): Weight for MaxDD in score
    :param return_weight (float): Weight for return bonus in score
    :param reject_negative_ic_threshold (float): IC below this = score 0

    :return df (pd.DataFrame): DataFrame with score and rank columns added
    """
    weights = {
        "sharpe": sharpe_weight,
        "ic": ic_weight,
        "maxdd": maxdd_weight,
        "return": return_weight,
    }
    return _compute_composite_score(
        df,
        sharpe_col="sharpe",
        ic_col="ic",
        maxdd_col="maxdd",
        total_return_col="total_return" if "total_return" in df.columns else None,
        weights=weights,
        reject_negative_ic_threshold=reject_negative_ic_threshold,
    )


@st.cache_data
def load_benchmarks() -> pd.DataFrame:
    """Load benchmark results from cache or use defaults."""
    cache_path = project_root / "data_cache" / "benchmarks.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df

    # Fallback to hardcoded values (without monthly returns)
    benchmarks = [
        {"name": "Equal-Weight 6F", "sharpe": 0.4741, "ic": 0.0, "maxdd": -0.0371, "total_return": 0.0522, "monthly_returns": ""},
        {"name": "50/50 Cyc/Def", "sharpe": 0.2946, "ic": 0.0, "maxdd": -0.1482, "total_return": 0.0778, "monthly_returns": ""},
        {"name": "Risk Parity", "sharpe": 0.2015, "ic": 0.0, "maxdd": -0.0498, "total_return": 0.0208, "monthly_returns": ""},
        {"name": "Factor Momentum", "sharpe": -0.0619, "ic": 0.0, "maxdd": -0.0932, "total_return": -0.0154, "monthly_returns": ""},
        {"name": "Best Factor (Def)", "sharpe": 0.4911, "ic": 0.0, "maxdd": -0.1019, "total_return": 0.1406, "monthly_returns": ""},
    ]
    return pd.DataFrame(benchmarks)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

def render_sidebar_filters(df: pd.DataFrame) -> Dict[str, List]:
    """Render sidebar filters and return selected values."""
    st.sidebar.markdown("## Filters")

    # Strategy filter
    all_strategies = sorted(df["strategy"].unique().tolist())
    selected_strategies = st.sidebar.multiselect(
        "Strategy",
        options=all_strategies,
        default=all_strategies,
        help="E2E = End-to-End 3-phase, Sup = Supervised only"
    )

    # Allocation filter
    all_allocations = sorted(df["allocation"].unique().tolist())
    selected_allocations = st.sidebar.multiselect(
        "Allocation",
        options=all_allocations,
        default=all_allocations,
        help="Multi = 6 factors, Binary = Cyclical vs Defensive"
    )

    # Horizon filter
    all_horizons = sorted(df["horizon"].unique().tolist())
    selected_horizons = st.sidebar.multiselect(
        "Horizon",
        options=[f"{h}M" for h in all_horizons],
        default=[f"{h}M" for h in all_horizons],
        help="Sharpe optimization target horizon"
    )
    # Convert back to int
    selected_horizons_int = [int(h.replace("M", "")) for h in selected_horizons]

    # Config filter (FS/HPT axis) - only show if column exists
    selected_configs = ["baseline"]  # Default for backward compatibility
    if "config" in df.columns:
        all_configs = sorted(df["config"].unique().tolist())
        selected_configs = st.sidebar.multiselect(
            "Config",
            options=all_configs,
            default=all_configs,
            help="baseline=default, fs=feature selection, hpt=HP tuning, fs+hpt=both"
        )

    # Model Type filter
    all_types = sorted(df["model_type"].unique().tolist())
    selected_types = st.sidebar.multiselect(
        "Model Type",
        options=all_types,
        default=all_types,
        help="Final = single model, Fair Ens. = same data different seeds, WF Ens. = walk-forward models"
    )

    # Score weights
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Score Weights")
    sharpe_weight = st.sidebar.slider("Sharpe Weight", 0.0, 1.0, 0.35, 0.05)
    ic_weight = st.sidebar.slider("IC Weight", 0.0, 1.0, 0.25, 0.05)
    maxdd_weight = st.sidebar.slider("MaxDD Weight", 0.0, 1.0, 0.30, 0.05)
    return_weight = st.sidebar.slider("Return Weight", 0.0, 1.0, 0.10, 0.05)

    # Normalize weights
    total_weight = sharpe_weight + ic_weight + maxdd_weight + return_weight
    if total_weight > 0:
        sharpe_weight /= total_weight
        ic_weight /= total_weight
        maxdd_weight /= total_weight
        return_weight /= total_weight

    # Score formula info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Score uses asymmetric IC penalty (negative IC penalized 2x) "
        "and exponential MaxDD penalty. Models with IC < -30% are rejected."
    )

    return {
        "strategies": selected_strategies,
        "allocations": selected_allocations,
        "horizons": selected_horizons_int,
        "configs": selected_configs,
        "model_types": selected_types,
        "sharpe_weight": sharpe_weight,
        "ic_weight": ic_weight,
        "maxdd_weight": maxdd_weight,
        "return_weight": return_weight,
    }


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe."""
    mask = (
        df["strategy"].isin(filters["strategies"]) &
        df["allocation"].isin(filters["allocations"]) &
        df["horizon"].isin(filters["horizons"]) &
        df["model_type"].isin(filters["model_types"])
    )

    # Filter by config if column exists (backward compatible)
    if "config" in df.columns and "configs" in filters:
        mask = mask & df["config"].isin(filters["configs"])

    filtered_df = df[mask].copy()

    # Recompute scores with new weights
    if len(filtered_df) > 0:
        filtered_df = compute_score(
            filtered_df,
            sharpe_weight=filters["sharpe_weight"],
            ic_weight=filters["ic_weight"],
            maxdd_weight=filters["maxdd_weight"],
            return_weight=filters.get("return_weight", 0.10),
        )

    return filtered_df


# ============================================================
# MAIN RESULTS TABLE
# ============================================================

def render_results_table(df: pd.DataFrame, show_averages: bool = True):
    """Render the main results table with optional averages."""
    if len(df) == 0:
        st.warning("No data matches the current filters.")
        return

    # Sort by score
    display_df = df.sort_values("score", ascending=False).copy()

    # Create label column (include config if multiple configs exist)
    type_abbrev = MODEL_TYPE_ABBREV
    config_abbrev = CONFIG_SUFFIX
    has_multi_configs = "config" in display_df.columns and display_df["config"].nunique() > 1
    display_df["label"] = display_df.apply(
        lambda row: (
            f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M"
            f"{config_abbrev.get(row.get('config', 'baseline'), '') if has_multi_configs else ''}"
            f"-{type_abbrev.get(row['model_type'], row['model_type'][:2])}"
        ),
        axis=1
    )

    # Format columns - Sharpe as ratio, others as percentages
    display_df["sharpe_fmt"] = display_df["sharpe"].apply(lambda x: f"{x:+.2f}")
    display_df["ic_fmt"] = display_df["ic"].apply(lambda x: f"{x*100:+.1f}%")
    display_df["maxdd_fmt"] = display_df["maxdd"].apply(lambda x: f"{x:+.1%}")
    display_df["return_fmt"] = display_df["total_return"].apply(lambda x: f"{x:+.1%}")
    display_df["score_fmt"] = display_df["score"].apply(lambda x: f"{x:.1%}")

    # Select and rename columns for display
    columns_to_show = [
        "rank", "label", "strategy", "allocation", "horizon", "model_type",
        "sharpe_fmt", "ic_fmt", "maxdd_fmt", "return_fmt", "score_fmt"
    ]
    column_names = {
        "rank": "Rank",
        "label": "Label",
        "strategy": "Strategy",
        "allocation": "Alloc",
        "horizon": "Horizon",
        "model_type": "Type",
        "sharpe_fmt": "Sharpe",
        "ic_fmt": "IC",
        "maxdd_fmt": "MaxDD",
        "return_fmt": "Return",
        "score_fmt": "Score",
    }

    table_df = display_df[columns_to_show].rename(columns=column_names)

    # Add average row if requested
    if show_averages and len(df) > 1:
        avg_row = pd.DataFrame([{
            "Rank": "-",
            "Label": "AVERAGE",
            "Strategy": "-",
            "Alloc": "-",
            "Horizon": "-",
            "Type": "-",
            "Sharpe": f"{df['sharpe'].mean():+.2f}",
            "IC": f"{df['ic'].mean()*100:+.1f}%",
            "MaxDD": f"{df['maxdd'].mean():+.1%}",
            "Return": f"{df['total_return'].mean():+.1%}",
            "Score": f"{df['score'].mean():.1%}",
        }])
        table_df = pd.concat([table_df, avg_row], ignore_index=True)

    # Display table
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=min(35 * len(table_df) + 38, 600),
    )


def render_pivot_table(df: pd.DataFrame, metric: str = "sharpe", group_by: str = "horizon"):
    """
    Render pivot table with averages.

    :param df (pd.DataFrame): DataFrame with results
    :param metric (str): Metric to display (sharpe, ic, maxdd, total_return)
    :param group_by (str): Column to use for pivot columns (horizon or config)
    """
    if len(df) == 0:
        return

    group_label = "Horizon" if group_by == "horizon" else "Config"
    st.markdown(f"### Pivot Table: {metric.upper()} by Strategy/Allocation x {group_label}")

    # Create pivot
    pivot = df.pivot_table(
        values=metric,
        index=["strategy", "allocation"],
        columns=group_by,
        aggfunc="mean",
    )

    # Add row averages
    pivot["AVG"] = pivot.mean(axis=1)

    # Add column averages
    col_avg = pivot.mean(axis=0)
    col_avg.name = ("AVG", "")
    pivot = pd.concat([pivot, col_avg.to_frame().T])

    # Format - Sharpe as ratio, IC/MaxDD/Return as percentages
    if metric in ["maxdd", "total_return"]:
        styled = pivot.style.format("{:+.1%}").background_gradient(cmap="RdYlGn", axis=None)
    elif metric == "sharpe":
        # Sharpe is a ratio, not a percentage
        styled = pivot.style.format("{:+.2f}").background_gradient(cmap="RdYlGn", axis=None)
    else:
        # For ic, multiply by 100 to show as percentage
        pivot_pct = pivot * 100
        styled = pivot_pct.style.format("{:+.1f}%").background_gradient(cmap="RdYlGn", axis=None)

    st.dataframe(styled, use_container_width=True)


# ============================================================
# VISUALIZATIONS
# ============================================================

def render_sharpe_bar_chart(df: pd.DataFrame, benchmarks_df: pd.DataFrame = None):
    """Render Sharpe ratio bar chart with benchmark reference line."""
    if len(df) == 0:
        return

    # Dynamic height based on number of models (min 8, max 30, scale by number of rows)
    n_rows = len(df)
    fig_height = max(8, min(30, n_rows * 0.5 + 4))
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Sort by sharpe
    sorted_df = df.sort_values("sharpe", ascending=True)

    # Create labels (include config if multiple configs exist)
    type_abbrev = MODEL_TYPE_ABBREV
    config_abbrev = CONFIG_SUFFIX
    has_multi_configs = "config" in sorted_df.columns and sorted_df["config"].nunique() > 1
    labels = sorted_df.apply(
        lambda row: (
            f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M"
            f"{config_abbrev.get(row.get('config', 'baseline'), '') if has_multi_configs else ''}"
            f"-{type_abbrev.get(row['model_type'], '?')}"
        ),
        axis=1
    )

    colors = ["#d32f2f" if s < 0 else "#388e3c" for s in sorted_df["sharpe"]]

    # Plot Sharpe as ratio (not percentage)
    ax.barh(range(len(sorted_df)), sorted_df["sharpe"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=1, color="#1976d2", linewidth=1.5, linestyle=":", alpha=0.7, label="Sharpe = 1")

    # Add best benchmark reference line
    if benchmarks_df is not None and len(benchmarks_df) > 0:
        best_bm = benchmarks_df.loc[benchmarks_df["sharpe"].idxmax()]
        ax.axvline(
            x=best_bm["sharpe"], color="#9467bd", linewidth=2, linestyle="--",
            label=f"Best Benchmark: {best_bm['name']} ({best_bm['sharpe']:.2f})"
        )

    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Holdout Sharpe Ratio by Model", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_sharpe_heatmap(df: pd.DataFrame):
    """Render Sharpe heatmap by combination."""
    if len(df) == 0:
        return

    # Pivot by strategy-allocation vs horizon
    df_copy = df.copy()
    df_copy["combo"] = df_copy["strategy"] + "-" + df_copy["allocation"]

    # Group by combo and horizon, averaging across model types
    pivot = df_copy.pivot_table(
        values="sharpe",
        index="combo",
        columns="horizon",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sharpe is a ratio, not percentage
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h}M" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            text_color = "white" if abs(val) > 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Strategy + Allocation")
    ax.set_title("Average Sharpe by Combination (across model types)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Sharpe")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_model_type_comparison(df: pd.DataFrame, benchmarks_df: pd.DataFrame = None):
    """Render comparison of model types with benchmark reference lines."""
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["sharpe", "ic", "maxdd"]
    titles = ["Sharpe Ratio", "Information Coefficient", "Max Drawdown"]

    # Get best benchmark values for each metric
    best_bm_values = {}
    if benchmarks_df is not None and len(benchmarks_df) > 0:
        best_bm_values["sharpe"] = benchmarks_df["sharpe"].max()
        best_bm_values["ic"] = 0.0  # Benchmarks don't have IC
        best_bm_values["maxdd"] = benchmarks_df["maxdd"].max()  # Less negative is better

    for ax, metric, title in zip(axes, metrics, titles):
        grouped = df.groupby("model_type")[metric].mean()
        colors = ["steelblue", "coral", "forestgreen"][:len(grouped)]

        # Sharpe is a ratio, IC and MaxDD are percentages
        if metric == "sharpe":
            plot_values = grouped.values
            ylabel = title
            fmt_str = "{x:.2f}"
            val_fmt = "{:.2f}"
        elif metric == "ic":
            plot_values = grouped.values * 100
            ylabel = f"{title} (%)"
            fmt_str = "{x:.0f}%"
            val_fmt = "{:.0f}%"
        else:
            plot_values = grouped.values * 100  # maxdd is already decimal, convert to %
            ylabel = f"{title} (%)"
            fmt_str = "{x:.0f}%"
            val_fmt = "{:.0f}%"

        bars = ax.bar(grouped.index, plot_values, color=colors, alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Average {title} by Model Type", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "sharpe":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
            # Add Sharpe=1 reference line
            ax.axhline(y=1, color="#1976d2", linewidth=1.5, linestyle=":", alpha=0.7, label="Sharpe = 1")
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        # Add best benchmark reference line
        if metric in best_bm_values and metric != "ic":  # Skip IC (benchmarks have no IC)
            if metric == "sharpe":
                bm_val = best_bm_values[metric]
                ax.axhline(
                    y=bm_val, color="#9467bd", linewidth=2, linestyle="--",
                    label=f"Best Benchmark: {bm_val:.2f}"
                )
            else:
                bm_val = best_bm_values[metric] * 100
                ax.axhline(
                    y=bm_val, color="#9467bd", linewidth=2, linestyle="--",
                    label=f"Best Benchmark: {bm_val:.0f}%"
                )

        # Show legend for Sharpe (has Sharpe=1 and possibly benchmark)
        if metric == "sharpe":
            ax.legend(loc="upper right", fontsize=8)
        elif metric in best_bm_values and metric != "ic":
            ax.legend(loc="upper right", fontsize=8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if metric == "sharpe":
                label = f"{height:.2f}"
            else:
                label = f"{height:.0f}%"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_scatter_final_vs_ensemble(df: pd.DataFrame):
    """Render scatter plot comparing Final vs Fair Ensemble."""
    if len(df) == 0:
        return

    # Get paired data
    final_df = df[df["model_type"] == "Final"].copy()
    fair_df = df[df["model_type"] == "Fair Ensemble"].copy()

    if len(final_df) == 0 or len(fair_df) == 0:
        st.info("Need both Final and Fair Ensemble models for this comparison.")
        return

    # Merge on key (include config if exists)
    if "config" in final_df.columns:
        final_df["key"] = final_df["strategy"] + "-" + final_df["allocation"] + "-" + final_df["horizon"].astype(str) + "-" + final_df["config"]
        fair_df["key"] = fair_df["strategy"] + "-" + fair_df["allocation"] + "-" + fair_df["horizon"].astype(str) + "-" + fair_df["config"]
    else:
        final_df["key"] = final_df["strategy"] + "-" + final_df["allocation"] + "-" + final_df["horizon"].astype(str)
        fair_df["key"] = fair_df["strategy"] + "-" + fair_df["allocation"] + "-" + fair_df["horizon"].astype(str)

    merged = final_df.merge(fair_df, on="key", suffixes=("_final", "_fair"))

    if len(merged) == 0:
        st.info("No matching pairs found.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        merged["sharpe_final"],
        merged["sharpe_fair"],
        s=100,
        alpha=0.7,
        c="darkgreen",
        edgecolors="black",
    )

    # Add labels
    for _, row in merged.iterrows():
        ax.annotate(
            row["key"],
            (row["sharpe_final"], row["sharpe_fair"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Diagonal line
    lims = [
        min(merged["sharpe_final"].min(), merged["sharpe_fair"].min()) - 0.1,
        max(merged["sharpe_final"].max(), merged["sharpe_fair"].max()) + 0.1,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Final = Fair Ens.")
    ax.fill_between(lims, lims, [lims[1]] * 2, alpha=0.1, color="red", label="Fair Ens. > Final")
    ax.fill_between(lims, [lims[0]] * 2, lims, alpha=0.1, color="blue", label="Final > Fair Ens.")

    ax.set_xlabel("Final Model Sharpe")
    ax.set_ylabel("Fair Ensemble Sharpe")
    ax.set_title("Final vs Fair Ensemble: Holdout Sharpe", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_cumulative_returns(df: pd.DataFrame, benchmarks_df: pd.DataFrame = None, top_n: int = 3):
    """
    Render cumulative returns chart for top N models vs benchmarks.

    :param df (pd.DataFrame): DataFrame with monthly_returns column
    :param benchmarks_df (pd.DataFrame): Benchmarks DataFrame with monthly_returns
    :param top_n (int): Number of top models to show
    """
    if len(df) == 0:
        return

    # Check if monthly_returns column exists
    if "monthly_returns" not in df.columns:
        st.info("Monthly returns data not available. Re-export results from notebook to enable this chart.")
        return

    # Get top N models by Score
    top_models = df.nlargest(top_n, "score").copy()

    # Filter models that have monthly returns data
    top_models = top_models[top_models["monthly_returns"].str.len() > 0]

    if len(top_models) == 0:
        st.info("No monthly returns data available for top models.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Model colors (solid lines)
    model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    type_abbrev = MODEL_TYPE_ABBREV
    config_abbrev = CONFIG_SUFFIX
    has_multi_configs = "config" in top_models.columns and top_models["config"].nunique() > 1

    n_months = None
    for i, (_, row) in enumerate(top_models.iterrows()):
        returns_str = row["monthly_returns"]
        if not returns_str:
            continue

        # Parse monthly returns from string
        try:
            returns = [float(r) for r in returns_str.split(",")]
        except (ValueError, AttributeError):
            continue

        if len(returns) == 0:
            continue

        if n_months is None:
            n_months = len(returns)

        # Compute cumulative returns
        cum_ret = np.cumprod(1 + np.array(returns))

        config_suffix = config_abbrev.get(row.get('config', 'baseline'), '') if has_multi_configs else ''
        label = f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M{config_suffix}-{type_abbrev.get(row['model_type'], '?')}"
        ax.plot(
            range(len(cum_ret)), cum_ret,
            linewidth=2.5, linestyle="-", color=model_colors[i % len(model_colors)],
            label=f"{label} (Sharpe={row['sharpe']:.2f})"
        )

    # Add benchmarks (dashed lines)
    if benchmarks_df is not None and "monthly_returns" in benchmarks_df.columns:
        benchmark_colors = {
            "Equal-Weight 6F": "#9467bd",
            "50/50 Cyc/Def": "#8c564b",
            "Risk Parity": "#e377c2",
            "Factor Momentum": "#17becf",
            "Best Factor (Def)": "#bcbd22",
        }

        for _, bm_row in benchmarks_df.iterrows():
            returns_str = bm_row.get("monthly_returns", "")
            if not returns_str or len(returns_str) == 0:
                continue

            try:
                returns = [float(r) for r in returns_str.split(",")]
            except (ValueError, AttributeError):
                continue

            if len(returns) == 0:
                continue

            # Truncate to same length as models if needed
            if n_months is not None:
                returns = returns[:n_months]

            cum_ret = np.cumprod(1 + np.array(returns))
            bm_name = bm_row["name"]
            color = benchmark_colors.get(bm_name, "gray")

            ax.plot(
                range(len(cum_ret)), cum_ret,
                linewidth=2, linestyle="--", color=color,
                label=f"{bm_name} (Sharpe={bm_row['sharpe']:.2f})"
            )

    ax.axhline(y=1, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Month (Holdout)")
    ax.set_ylabel("Cumulative Return (Wealth)")
    ax.set_title(f"Top {top_n} Models (by Score) vs Benchmarks: Cumulative Returns", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_config_comparison(df: pd.DataFrame):
    """
    Render bar chart comparing configs (baseline vs fs vs hpt vs fs+hpt).

    Matches the style of render_model_type_comparison() with 3 columns:
    Sharpe, IC, MaxDD.

    :param df (pd.DataFrame): DataFrame with config column
    """
    if "config" not in df.columns or df["config"].nunique() <= 1:
        return

    # Group by config
    config_stats = df.groupby("config").agg({
        "sharpe": "mean",
        "ic": "mean",
        "maxdd": "mean",
    }).reset_index()

    # Sort configs in logical order
    config_order = ["baseline", "fs", "hpt", "fs+hpt"]
    config_stats["order"] = config_stats["config"].apply(
        lambda x: config_order.index(x) if x in config_order else 99
    )
    config_stats = config_stats.sort_values("order")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["sharpe", "ic", "maxdd"]
    titles = ["Sharpe Ratio", "Information Coefficient", "Max Drawdown"]

    # Config colors
    config_colors = CONFIG_COLORS
    colors = [config_colors.get(c, "#666666") for c in config_stats["config"]]

    # Get baseline values for reference lines
    baseline_values = {}
    baseline_row = config_stats[config_stats["config"] == "baseline"]
    if len(baseline_row) > 0:
        baseline_values["sharpe"] = baseline_row["sharpe"].values[0]
        baseline_values["ic"] = baseline_row["ic"].values[0]
        baseline_values["maxdd"] = baseline_row["maxdd"].values[0]

    for ax, metric, title in zip(axes, metrics, titles):
        # Sharpe is a ratio, IC and MaxDD are percentages
        if metric == "sharpe":
            plot_values = config_stats[metric].values
            ylabel = title
        elif metric == "ic":
            plot_values = config_stats[metric].values * 100
            ylabel = f"{title} (%)"
        else:
            plot_values = config_stats[metric].values * 100
            ylabel = f"{title} (%)"

        bars = ax.bar(config_stats["config"], plot_values, color=colors, alpha=0.8, edgecolor="black")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Average {title} by Config", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        if metric == "sharpe":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
            # Add Sharpe=1 reference line
            ax.axhline(y=1, color="#1976d2", linewidth=1.5, linestyle=":", alpha=0.7, label="Sharpe = 1")
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        # Add baseline reference line
        if metric in baseline_values:
            if metric == "sharpe":
                bm_val = baseline_values[metric]
                ax.axhline(
                    y=bm_val, color="gray", linewidth=2, linestyle="--",
                    label=f"Baseline: {bm_val:.2f}"
                )
            else:
                bm_val = baseline_values[metric] * 100
                ax.axhline(
                    y=bm_val, color="gray", linewidth=2, linestyle="--",
                    label=f"Baseline: {bm_val:.0f}%"
                )

        # Show legend for Sharpe (has Sharpe=1 and possibly baseline)
        if metric == "sharpe":
            ax.legend(loc="upper right", fontsize=8)
        elif metric in baseline_values:
            ax.legend(loc="upper right", fontsize=8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if metric == "sharpe":
                label = f"{height:.2f}"
            else:
                label = f"{height:.0f}%"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_benchmarks_comparison(df: pd.DataFrame, benchmarks_df: pd.DataFrame):
    """Render comparison with benchmarks."""
    if len(df) == 0:
        return

    # Metric selection
    metric_options = {
        "Sharpe Ratio": "sharpe",
        "Max Drawdown": "maxdd",
        "Total Return": "total_return",
    }
    selected_metric_label = st.selectbox(
        "Select Metric",
        options=list(metric_options.keys()),
        index=0,
        key="benchmark_metric_select",
    )
    metric_col = metric_options[selected_metric_label]

    # Get top 3 models by SCORE (always the same regardless of displayed metric)
    cols_to_select = ["strategy", "allocation", "horizon", "model_type", "sharpe", "maxdd", "total_return", "score"]
    if "config" in df.columns:
        cols_to_select.append("config")
    top_models = df.nlargest(3, "score")[cols_to_select].copy()

    type_abbrev = MODEL_TYPE_ABBREV
    config_abbrev = CONFIG_SUFFIX
    has_multi_configs = "config" in top_models.columns and top_models["config"].nunique() > 1
    top_models["name"] = top_models.apply(
        lambda row: (
            f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M"
            f"{config_abbrev.get(row.get('config', 'baseline'), '') if has_multi_configs else ''}"
            f"-{type_abbrev.get(row['model_type'], '?')}"
        ),
        axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Combine top models and benchmarks
    all_data = []
    # Sharpe is a ratio, maxdd and total_return are percentages
    multiplier = 1 if metric_col == "sharpe" else 100
    for _, row in top_models.iterrows():
        all_data.append({"name": row["name"], "value": row[metric_col] * multiplier, "type": "Model"})
    for _, row in benchmarks_df.iterrows():
        all_data.append({"name": row["name"], "value": row[metric_col] * multiplier, "type": "Benchmark"})

    # Sort: for maxdd, sort descending (least negative at top), for others ascending
    if metric_col == "maxdd":
        combined_df = pd.DataFrame(all_data).sort_values("value", ascending=False)
    else:
        combined_df = pd.DataFrame(all_data).sort_values("value", ascending=True)

    colors = ["steelblue" if t == "Model" else "gray" for t in combined_df["type"]]
    ax.barh(range(len(combined_df)), combined_df["value"], color=colors, alpha=0.8)

    ax.set_yticks(range(len(combined_df)))
    ax.set_yticklabels(combined_df["name"], fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8)
    if metric_col == "sharpe":
        ax.set_xlabel(selected_metric_label)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    else:
        ax.set_xlabel(f"{selected_metric_label} (%)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title(f"Top 3 Models (by Score) vs Benchmarks - {selected_metric_label}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="Model"),
        Patch(facecolor="gray", alpha=0.8, label="Benchmark"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def render_statistical_analysis(df: pd.DataFrame, benchmarks_df: pd.DataFrame = None, top_n: int = 3):
    """
    Render statistical analysis with bootstrap CIs and significance tests.

    :param df (pd.DataFrame): DataFrame with monthly_returns column
    :param benchmarks_df (pd.DataFrame): Benchmarks DataFrame with monthly_returns
    :param top_n (int): Number of top models to analyze
    """
    if len(df) == 0:
        st.warning("No data available for statistical analysis.")
        return

    # Check if monthly_returns column exists
    if "monthly_returns" not in df.columns:
        st.info("Monthly returns data not available. Re-export results from notebook to enable statistical analysis.")
        return

    # Get top N models by Score
    top_models = df.nlargest(top_n, "score").copy()

    # Filter models that have monthly returns data
    top_models = top_models[top_models["monthly_returns"].str.len() > 0]

    if len(top_models) == 0:
        st.info("No monthly returns data available for top models.")
        return

    # Create labels
    type_abbrev = MODEL_TYPE_ABBREV
    config_abbrev = CONFIG_SUFFIX
    has_multi_configs = "config" in top_models.columns and top_models["config"].nunique() > 1

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_bootstrap = st.slider("Bootstrap samples", 100, 5000, 1000, 100)
    with col2:
        confidence_level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)

    # Parse returns and compute statistics
    results = []
    model_returns_dict = {}

    for _, row in top_models.iterrows():
        returns_str = row["monthly_returns"]
        if not returns_str:
            continue

        try:
            returns = np.array([float(r) for r in returns_str.split(",")])
        except (ValueError, AttributeError):
            continue

        if len(returns) == 0:
            continue

        config_suffix = config_abbrev.get(row.get('config', 'baseline'), '') if has_multi_configs else ''
        label = f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M{config_suffix}-{type_abbrev.get(row['model_type'], '?')}"

        model_returns_dict[label] = returns

        # Bootstrap CI
        ci = bootstrap_sharpe_ratio(
            returns,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )

        # Significance test (H0: Sharpe = 0)
        sig = test_sharpe_significance(returns, alpha=1 - confidence_level)

        results.append({
            "Model": label,
            "Sharpe": ci.estimate,
            "CI Lower": ci.ci_lower,
            "CI Upper": ci.ci_upper,
            "Std Error": ci.std_error,
            "p-value": sig.p_value,
            "Significant": "Yes" if sig.is_significant else "No",
        })

    if not results:
        st.info("Could not compute statistics for any models.")
        return

    # Display results table
    st.markdown("### Bootstrap Confidence Intervals & Significance Tests")
    st.markdown(f"**Bootstrap samples**: {n_bootstrap} | **Confidence level**: {confidence_level:.0%}")

    results_df = pd.DataFrame(results)

    # Format for display
    styled_df = results_df.style.format({
        "Sharpe": "{:+.3f}",
        "CI Lower": "{:+.3f}",
        "CI Upper": "{:+.3f}",
        "Std Error": "{:.3f}",
        "p-value": "{:.4f}",
    }).apply(
        lambda x: ['background-color: #d4edda; color: black' if v == "Yes" else 'background-color: #f8d7da; color: black' if v == "No" else '' for v in x],
        subset=["Significant"]
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Interpretation
    st.markdown("---")
    st.markdown("#### Interpretation")

    significant_models = results_df[results_df["Significant"] == "Yes"]
    if len(significant_models) > 0:
        st.success(
            f"**{len(significant_models)}/{len(results_df)}** models have statistically significant Sharpe ratios "
            f"(p < {1-confidence_level:.2f}). We can reject the null hypothesis that Sharpe = 0."
        )
    else:
        st.warning(
            "No models have statistically significant Sharpe ratios at the current confidence level. "
            "The results could be due to chance."
        )

    # Best benchmark comparison
    if benchmarks_df is not None and "monthly_returns" in benchmarks_df.columns:
        # Find best benchmark by Sharpe
        best_bm = benchmarks_df.loc[benchmarks_df["sharpe"].idxmax()]
        bm_returns_str = best_bm.get("monthly_returns", "")

        if bm_returns_str and len(bm_returns_str) > 0:
            try:
                bm_returns = np.array([float(r) for r in bm_returns_str.split(",")])

                if len(bm_returns) > 0:
                    st.markdown("---")
                    st.markdown(f"### Comparison vs Best Benchmark ({best_bm['name']})")

                    # Compute benchmark Sharpe CI
                    bm_ci = bootstrap_sharpe_ratio(bm_returns, n_bootstrap=n_bootstrap)
                    st.markdown(f"**Benchmark Sharpe**: {bm_ci.estimate:+.3f} (95% CI: [{bm_ci.ci_lower:+.3f}, {bm_ci.ci_upper:+.3f}])")

                    comparison_results = []
                    for label, returns in model_returns_dict.items():
                        # Truncate to same length
                        min_len = min(len(returns), len(bm_returns))
                        comparison = compare_sharpe_ratios(
                            returns[:min_len],
                            bm_returns[:min_len],
                            alpha=1 - confidence_level,
                        )

                        comparison_results.append({
                            "Model": label,
                            "Model Sharpe": comparison.sharpe_1,
                            "Benchmark Sharpe": comparison.sharpe_2,
                            "Delta": comparison.difference,
                            "p-value": comparison.p_value,
                            "Significant": "Yes" if comparison.is_significant else "No",
                        })

                    comp_df = pd.DataFrame(comparison_results)
                    styled_comp = comp_df.style.format({
                        "Model Sharpe": "{:+.3f}",
                        "Benchmark Sharpe": "{:+.3f}",
                        "Delta": "{:+.3f}",
                        "p-value": "{:.4f}",
                    }).apply(
                        lambda x: ['background-color: #d4edda; color: black' if v == "Yes" else 'color: black' for v in x],
                        subset=["Significant"]
                    )

                    st.dataframe(styled_comp, use_container_width=True, hide_index=True)

                    # Interpretation
                    sig_vs_bm = comp_df[comp_df["Significant"] == "Yes"]
                    if len(sig_vs_bm) > 0:
                        better = sig_vs_bm[sig_vs_bm["Delta"] > 0]
                        worse = sig_vs_bm[sig_vs_bm["Delta"] < 0]
                        if len(better) > 0:
                            st.success(f"**{len(better)}** model(s) significantly outperform the benchmark (Jobson-Korkie test).")
                        if len(worse) > 0:
                            st.warning(f"**{len(worse)}** model(s) significantly underperform the benchmark.")
                    else:
                        st.info("No significant difference between models and benchmark at current confidence level.")

            except (ValueError, AttributeError):
                pass

    # Visual: Forest plot
    st.markdown("---")
    st.markdown("### Forest Plot: Sharpe Ratio with Confidence Intervals")

    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.8)))

    y_positions = range(len(results))
    for i, res in enumerate(results):
        color = "#2ca02c" if res["Significant"] == "Yes" else "#d62728"
        ax.errorbar(
            res["Sharpe"], i,
            xerr=[[res["Sharpe"] - res["CI Lower"]], [res["CI Upper"] - res["Sharpe"]]],
            fmt='o', color=color, capsize=5, capthick=2, markersize=8
        )

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="Sharpe = 0")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["Model"] for r in results])
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title(f"Sharpe Ratio with {confidence_level:.0%} Confidence Intervals", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right")

    # Add significance legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='Significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Not Significant'),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Methodology
    with st.expander("Methodology Details"):
        st.markdown("""
        **Bootstrap Confidence Intervals**
        - Resample returns with replacement N times
        - Compute Sharpe ratio for each bootstrap sample
        - CI bounds = percentiles of bootstrap distribution

        **Lo (2002) Significance Test**
        - H0: Sharpe ratio = 0
        - Standard error: SE = sqrt((1 + 0.5*SR²) / T)
        - T-statistic: t = SR / SE
        - P-value from t-distribution with T-1 degrees of freedom

        **Jobson-Korkie Test (for benchmark comparison)**
        - H0: Sharpe_model = Sharpe_benchmark
        - Accounts for correlation between strategies
        - Uses Memmel (2003) correction for small samples

        **Interpretation**
        - p < 0.01: Very strong evidence (***)
        - p < 0.05: Strong evidence (**)
        - p < 0.10: Moderate evidence (*)
        - p >= 0.10: Insufficient evidence
        """)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main dashboard application."""
    st.markdown('<h1 class="main-header">Factor Allocation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Interactive exploration of factor allocation model results.")

    # Load data
    df = load_holdout_results()
    benchmarks_df = load_benchmarks()

    # Sidebar filters
    filters = render_sidebar_filters(df)
    filtered_df = apply_filters(df, filters)

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Models", len(filtered_df))
    with col2:
        if len(filtered_df) > 0:
            st.metric("Avg Sharpe", f"{filtered_df['sharpe'].mean():.3f}")
        else:
            st.metric("Avg Sharpe", "-")
    with col3:
        if len(filtered_df) > 0:
            st.metric("Best Sharpe", f"{filtered_df['sharpe'].max():.3f}")
        else:
            st.metric("Best Sharpe", "-")
    with col4:
        if len(filtered_df) > 0:
            st.metric("Avg IC", f"{filtered_df['ic'].mean():.3f}")
        else:
            st.metric("Avg IC", "-")
    with col5:
        if len(filtered_df) > 0:
            st.metric("Avg MaxDD", f"{filtered_df['maxdd'].mean():.1%}")
        else:
            st.metric("Avg MaxDD", "-")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Results Table",
        "Visualizations",
        "Pivot Analysis",
        "Benchmarks",
        "Statistical Analysis"
    ])

    with tab1:
        st.markdown("### Complete Results Table")
        show_avg = st.checkbox("Show average row", value=True)
        render_results_table(filtered_df, show_averages=show_avg)

    with tab2:
        st.markdown("### Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### Sharpe by Model")
            render_sharpe_bar_chart(filtered_df, benchmarks_df)

        with viz_col2:
            st.markdown("#### Sharpe Heatmap")
            render_sharpe_heatmap(filtered_df)

        st.markdown("---")

        st.markdown("#### Model Type Comparison")
        render_model_type_comparison(filtered_df, benchmarks_df)

        # Config comparison (only show if multiple configs)
        if "config" in filtered_df.columns and filtered_df["config"].nunique() > 1:
            st.markdown("---")
            st.markdown("#### Config Comparison (FS/HPT)")
            render_config_comparison(filtered_df)

        st.markdown("---")

        st.markdown("#### Cumulative Returns (Top 3 Models vs Benchmarks)")
        render_cumulative_returns(filtered_df, benchmarks_df, top_n=3)

        st.markdown("---")

        st.markdown("#### Final vs Fair Ensemble")
        render_scatter_final_vs_ensemble(filtered_df)

    with tab3:
        st.markdown("### Pivot Analysis")

        # Metric and group by selection
        col_metric, col_group = st.columns(2)

        with col_metric:
            metric_choice = st.selectbox(
                "Select Metric",
                options=["sharpe", "ic", "maxdd", "total_return"],
                format_func=lambda x: {"sharpe": "Sharpe Ratio", "ic": "Information Coefficient", "maxdd": "Max Drawdown", "total_return": "Total Return"}[x]
            )

        with col_group:
            # Only show config option if multiple configs exist
            group_options = ["horizon"]
            group_labels = {"horizon": "by Horizon"}
            if "config" in filtered_df.columns and filtered_df["config"].nunique() > 1:
                group_options.append("config")
                group_labels["config"] = "by Config (FS/HPT)"

            group_by_choice = st.selectbox(
                "Group By",
                options=group_options,
                format_func=lambda x: group_labels[x]
            )

        render_pivot_table(filtered_df, metric=metric_choice, group_by=group_by_choice)

    with tab4:
        st.markdown("### Comparison with Benchmarks")

        st.markdown("#### Benchmarks")
        # Create display dataframe - Sharpe as ratio, others as percentages
        bm_display = benchmarks_df.copy()
        # Sharpe is a ratio, don't multiply by 100
        bm_display["ic"] = bm_display["ic"] * 100
        bm_display["maxdd"] = bm_display["maxdd"] * 100
        bm_display["total_return"] = bm_display["total_return"] * 100
        st.dataframe(
            bm_display[["name", "sharpe", "ic", "maxdd", "total_return"]].style.format({
                "sharpe": "{:+.2f}",
                "ic": "{:+.1f}%",
                "maxdd": "{:+.1f}%",
                "total_return": "{:+.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        st.markdown("#### Top Models vs Benchmarks")
        render_benchmarks_comparison(filtered_df, benchmarks_df)

    with tab5:
        st.markdown("### Statistical Analysis")
        st.markdown(
            "Bootstrap confidence intervals and significance tests for top models. "
            "Uses Lo (2002) for single Sharpe test and Jobson-Korkie for benchmark comparison."
        )

        # Number of top models to analyze
        top_n_stat = st.slider("Number of top models to analyze", 1, 10, 3)

        render_statistical_analysis(filtered_df, benchmarks_df, top_n=top_n_stat)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Factor Allocation Strategy Dashboard | MICRO Transformer | Point-in-Time FRED-MD
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
