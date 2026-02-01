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
    model_types = ["Final", "Fair Ensemble", "WF Ensemble"]

    np.random.seed(42)

    for strategy in strategies:
        for allocation in allocations:
            for horizon in horizons:
                for model_type in model_types:
                    base_sharpe = 0.5 if allocation == "Multi" else 0.2
                    base_sharpe += 0.1 if strategy == "Sup" else 0
                    base_sharpe += np.random.uniform(-0.3, 0.3)

                    data.append({
                        "strategy": strategy,
                        "allocation": allocation,
                        "horizon": horizon,
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
    sharpe_weight: float = 0.4,
    ic_weight: float = 0.3,
    maxdd_weight: float = 0.3,
    use_fixed_bounds: bool = True,
) -> pd.DataFrame:
    """
    Compute composite score for each row.

    :param df (pd.DataFrame): DataFrame with sharpe, ic, maxdd columns
    :param sharpe_weight (float): Weight for Sharpe ratio in score
    :param ic_weight (float): Weight for IC in score
    :param maxdd_weight (float): Weight for MaxDD in score
    :param use_fixed_bounds (bool): If True, use fixed normalization bounds for stable scores

    :return df (pd.DataFrame): DataFrame with score and rank columns added
    """
    df = df.copy()

    if use_fixed_bounds:
        # Fixed bounds for stable scores regardless of filtered subset
        # These bounds are based on typical ranges in factor allocation strategies
        sharpe_min, sharpe_max = -0.5, 1.0  # Typical Sharpe range
        ic_min, ic_max = -0.5, 0.5          # IC is a correlation, typically -0.5 to 0.5
        maxdd_min, maxdd_max = -0.20, 0.0   # MaxDD typically -20% to 0%
    else:
        # Relative bounds (original behavior - scores change with filters)
        sharpe_min, sharpe_max = df["sharpe"].min(), df["sharpe"].max()
        ic_min, ic_max = df["ic"].min(), df["ic"].max()
        maxdd_min, maxdd_max = df["maxdd"].min(), df["maxdd"].max()

    def safe_normalize(x: float, xmin: float, xmax: float) -> float:
        """Normalize value to [0, 1] with bounds clipping."""
        if xmax - xmin < 1e-8:
            return 0.5
        # Clip to bounds to handle outliers
        x_clipped = max(xmin, min(xmax, x))
        return (x_clipped - xmin) / (xmax - xmin)

    df["sharpe_norm"] = df["sharpe"].apply(lambda x: safe_normalize(x, sharpe_min, sharpe_max))
    df["ic_norm"] = df["ic"].apply(lambda x: safe_normalize(x, ic_min, ic_max))
    # For maxdd, less negative is better, so we normalize (more negative = lower score)
    df["maxdd_norm"] = df["maxdd"].apply(lambda x: safe_normalize(x, maxdd_min, maxdd_max))

    df["score"] = (
        sharpe_weight * df["sharpe_norm"]
        + ic_weight * df["ic_norm"]
        + maxdd_weight * df["maxdd_norm"]
    )

    df["rank"] = df["score"].rank(ascending=False).astype(int)

    # Clean up temp columns
    df = df.drop(columns=["sharpe_norm", "ic_norm", "maxdd_norm"])

    return df


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
    sharpe_weight = st.sidebar.slider("Sharpe Weight", 0.0, 1.0, 0.4, 0.1)
    ic_weight = st.sidebar.slider("IC Weight", 0.0, 1.0, 0.3, 0.1)
    maxdd_weight = st.sidebar.slider("MaxDD Weight", 0.0, 1.0, 0.3, 0.1)

    # Normalize weights
    total_weight = sharpe_weight + ic_weight + maxdd_weight
    if total_weight > 0:
        sharpe_weight /= total_weight
        ic_weight /= total_weight
        maxdd_weight /= total_weight

    # Normalization mode
    st.sidebar.markdown("---")
    use_fixed_bounds = st.sidebar.checkbox(
        "Fixed score bounds",
        value=True,
        help="If checked, scores use fixed normalization bounds (stable across filters). "
             "If unchecked, scores are relative to the displayed subset."
    )

    return {
        "strategies": selected_strategies,
        "allocations": selected_allocations,
        "horizons": selected_horizons_int,
        "model_types": selected_types,
        "sharpe_weight": sharpe_weight,
        "ic_weight": ic_weight,
        "maxdd_weight": maxdd_weight,
        "use_fixed_bounds": use_fixed_bounds,
    }


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe."""
    mask = (
        df["strategy"].isin(filters["strategies"]) &
        df["allocation"].isin(filters["allocations"]) &
        df["horizon"].isin(filters["horizons"]) &
        df["model_type"].isin(filters["model_types"])
    )
    filtered_df = df[mask].copy()

    # Recompute scores with new weights
    if len(filtered_df) > 0:
        filtered_df = compute_score(
            filtered_df,
            sharpe_weight=filters["sharpe_weight"],
            ic_weight=filters["ic_weight"],
            maxdd_weight=filters["maxdd_weight"],
            use_fixed_bounds=filters.get("use_fixed_bounds", True),
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

    # Create label column
    type_abbrev = {"Final": "F", "Fair Ensemble": "FE", "WF Ensemble": "WF"}
    display_df["label"] = display_df.apply(
        lambda row: f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M-{type_abbrev.get(row['model_type'], row['model_type'][:2])}",
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


def render_pivot_table(df: pd.DataFrame, metric: str = "sharpe"):
    """Render pivot table with averages."""
    if len(df) == 0:
        return

    st.markdown(f"### Pivot Table: {metric.upper()} by Strategy/Allocation x Horizon")

    # Create pivot
    pivot = df.pivot_table(
        values=metric,
        index=["strategy", "allocation"],
        columns="horizon",
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

    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by sharpe
    sorted_df = df.sort_values("sharpe", ascending=True)

    # Create labels
    type_abbrev = {"Final": "F", "Fair Ensemble": "FE", "WF Ensemble": "WF"}
    labels = sorted_df.apply(
        lambda row: f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M-{type_abbrev.get(row['model_type'], '?')}",
        axis=1
    )

    colors = ["#d32f2f" if s < 0 else "#388e3c" for s in sorted_df["sharpe"]]

    # Plot Sharpe as ratio (not percentage)
    ax.barh(range(len(sorted_df)), sorted_df["sharpe"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=0, color="black", linewidth=0.8)

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

    # Merge on key
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

    # Get top N models by Sharpe
    top_models = df.nlargest(top_n, "sharpe").copy()

    # Filter models that have monthly returns data
    top_models = top_models[top_models["monthly_returns"].str.len() > 0]

    if len(top_models) == 0:
        st.info("No monthly returns data available for top models.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Model colors (solid lines)
    model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    type_abbrev = {"Final": "F", "Fair Ensemble": "FE", "WF Ensemble": "WF"}

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

        label = f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M-{type_abbrev.get(row['model_type'], '?')}"
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
    ax.set_title(f"Top {top_n} Models vs Benchmarks: Cumulative Returns", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

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

    # Get top 3 models by selected metric
    # For maxdd, lower is better (less negative), so we use nsmallest with ascending=False
    if metric_col == "maxdd":
        top_models = df.nsmallest(3, "maxdd")[["strategy", "allocation", "horizon", "model_type", "sharpe", "maxdd", "total_return"]].copy()
    else:
        top_models = df.nlargest(3, metric_col)[["strategy", "allocation", "horizon", "model_type", "sharpe", "maxdd", "total_return"]].copy()

    type_abbrev = {"Final": "F", "Fair Ensemble": "FE", "WF Ensemble": "WF"}
    top_models["name"] = top_models.apply(
        lambda row: f"{row['strategy']}-{row['allocation'][0]}-{row['horizon']}M-{type_abbrev.get(row['model_type'], '?')}",
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
    ax.set_title(f"Top 3 Models vs Benchmarks ({selected_metric_label})", fontweight="bold")
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Results Table",
        "Visualizations",
        "Pivot Analysis",
        "Benchmarks"
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

        st.markdown("---")

        st.markdown("#### Cumulative Returns (Top 3 Models vs Benchmarks)")
        render_cumulative_returns(filtered_df, benchmarks_df, top_n=3)

        st.markdown("---")

        st.markdown("#### Final vs Fair Ensemble")
        render_scatter_final_vs_ensemble(filtered_df)

    with tab3:
        st.markdown("### Pivot Analysis")

        metric_choice = st.selectbox(
            "Select Metric",
            options=["sharpe", "ic", "maxdd", "total_return"],
            format_func=lambda x: {"sharpe": "Sharpe Ratio", "ic": "Information Coefficient", "maxdd": "Max Drawdown", "total_return": "Total Return"}[x]
        )
        render_pivot_table(filtered_df, metric=metric_choice)

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
