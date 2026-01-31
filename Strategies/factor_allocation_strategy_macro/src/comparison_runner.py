"""
Comparison Runner for Factor Allocation Strategy.

Trains and evaluates all 16 combinations:
- 2 strategies (E2E, Supervised)
- 2 allocation modes (Binary, Multi-factor)
- 4 horizons (1M, 3M, 6M, 12M)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.data_loader import Region
from data.factor_data_loader import FactorDataLoader
from features.feature_engineering import FeatureEngineer
from models.transformer import FactorAllocationTransformer
from models.training_strategies import SupervisedTrainer, TrainingConfig
from main_strategy import FactorAllocationStrategy, collate_fn


HORIZONS = [1, 3, 6, 12]
FACTOR_COLUMNS = ["cyclical", "defensive", "value", "growth", "quality", "momentum"]


@dataclass
class ComparisonConfig:
    """Configuration for comparison runner."""
    horizons: List[int] = None
    batch_size: int = 32
    epochs_phase1: int = 20
    epochs_phase2: int = 15
    epochs_phase3: int = 15

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = HORIZONS


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(
    factor_loader: FactorDataLoader,
    factor_data: pd.DataFrame,
    horizons: List[int] = None,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, np.ndarray]]:
    """
    Prepare multi-horizon targets and cumulative returns.

    :param factor_loader: Factor data loader
    :param factor_data: Raw factor returns
    :param horizons: List of horizons (default: [1, 3, 6, 12])

    :return targets: Dict mapping horizon -> target DataFrame
    :return cumulative_returns: Dict mapping horizon -> cumulative returns array
    """
    if horizons is None:
        horizons = HORIZONS

    targets = factor_loader.create_multi_horizon_targets(factor_data, horizons=horizons)

    cumulative_returns = {}
    for h in horizons:
        if h == 1:
            cumulative_returns[h] = factor_data[FACTOR_COLUMNS].values
        else:
            cumulative_returns[h] = factor_loader.get_cumulative_factor_returns_for_horizon(
                factor_data, h
            )

    return targets, cumulative_returns


def train_e2e_model(
    horizon: int,
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    cumulative_returns: np.ndarray,
    indicators: List[str],
    config: Dict,
    output_type: str = "binary",
    seed: int = 42,
) -> Dict:
    """
    Train End-to-End model for a specific horizon and allocation mode.

    :param horizon: Prediction horizon (1, 3, 6, or 12)
    :param macro_data: Macro features
    :param factor_data: Factor returns
    :param market_data: Market context
    :param target_data: Training targets
    :param cumulative_returns: Cumulative returns for Phase 3
    :param indicators: FRED-MD indicators
    :param config: Model configuration
    :param output_type: 'binary' or 'allocation'
    :param seed: Random seed

    :return metrics: Evaluation metrics
    """
    set_seed(seed)

    # Add horizon to config for checkpoint metadata
    config_with_horizon = {**config, "horizon_months": horizon}

    strat = FactorAllocationStrategy(
        region=Region.US,
        use_fred_md=True,
        fred_md_indicators=indicators,
        config=config_with_horizon,
        verbose=False,
    )
    strat.create_model()

    train_ds, val_ds = strat.prepare_data(macro_data, factor_data, market_data, target_data)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 3-phase training
    strat.train_phase1(train_loader, val_loader, verbose=False)
    strat.train_phase2(train_loader, val_loader, verbose=False)
    strat.train_phase3(train_loader, val_loader, cumulative_returns, verbose=False)

    # Backtest and evaluate
    results = strat.backtest(
        macro_data, factor_data, market_data, target_data,
        output_type=output_type, verbose=False
    )

    return strat.evaluate(results, verbose=False)


def train_supervised_model(
    horizon: int,
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    indicators: List[str],
    feature_engineer: FeatureEngineer,
    config: Dict,
    output_type: str = "binary",
    seed: int = 1000,
) -> Dict:
    """
    Train Supervised model for a specific horizon and allocation mode.

    :param horizon: Prediction horizon
    :param macro_data: Macro features
    :param factor_data: Factor returns
    :param market_data: Market context
    :param target_data: Training targets
    :param indicators: FRED-MD indicators
    :param feature_engineer: Feature engineering instance
    :param config: Model configuration
    :param output_type: 'binary' or 'allocation'
    :param seed: Random seed

    :return metrics: Evaluation metrics
    """
    set_seed(seed)

    model = FactorAllocationTransformer(
        num_indicators=feature_engineer.get_num_indicators(),
        num_factors=config['num_factors'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['sequence_length'],
    )

    trainer = SupervisedTrainer(
        model=model,
        config=TrainingConfig(
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            weight_decay=config['weight_decay'],
            epochs_phase3=config['epochs_phase3'],
            horizon_months=horizon,
        ),
        device=torch.device('cpu'),
        verbose=False,
    )

    # Prepare data
    strat_tmp = FactorAllocationStrategy(
        region=Region.US,
        use_fred_md=True,
        fred_md_indicators=indicators,
        config=config,
        verbose=False,
    )
    strat_tmp.create_model()
    train_ds, val_ds = strat_tmp.prepare_data(macro_data, factor_data, market_data, target_data)

    train_dates = [target_data.iloc[i]['timestamp'] for i in range(len(train_ds))]
    opt_weights = trainer.compute_targets(factor_data, train_dates, horizon_months=horizon)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Train
    trainer.train(train_loader, val_loader, opt_weights, train_dates, verbose=False)

    # Create strategy with trained model
    strat_sup = FactorAllocationStrategy(
        region=Region.US,
        use_fred_md=True,
        fred_md_indicators=indicators,
        config=config,
        verbose=False,
    )
    strat_sup.create_model()
    strat_sup.model = model

    # Backtest and evaluate
    results = strat_sup.backtest(
        macro_data, factor_data, market_data, target_data,
        output_type=output_type, verbose=False
    )

    return strat_sup.evaluate(results, verbose=False)


def run_all_combinations(
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    targets: Dict[int, pd.DataFrame],
    cumulative_returns: Dict[int, np.ndarray],
    indicators: List[str],
    feature_engineer: FeatureEngineer,
    config: Dict,
    horizons: List[int] = None,
    verbose: bool = True,
) -> Dict[Tuple[str, str, int], Dict]:
    """
    Run all 16 combinations and return results.

    :param macro_data: Macro features
    :param factor_data: Factor returns
    :param market_data: Market context
    :param targets: Dict of horizon -> target DataFrame
    :param cumulative_returns: Dict of horizon -> cumulative returns
    :param indicators: FRED-MD indicators
    :param feature_engineer: Feature engineering instance
    :param config: Model configuration
    :param horizons: List of horizons (default: [1, 3, 6, 12])
    :param verbose: Print progress

    :return results: Dict mapping (strategy, allocation, horizon) -> metrics
    """
    if horizons is None:
        horizons = HORIZONS

    results = {}
    total = len(horizons) * 4  # 4 combos per horizon
    current = 0

    # E2E Binary
    for h in horizons:
        current += 1
        if verbose:
            print(f"[{current}/{total}] E2E Binary {h}M...")
        results[("E2E", "Binary", h)] = train_e2e_model(
            horizon=h,
            macro_data=macro_data,
            factor_data=factor_data,
            market_data=market_data,
            target_data=targets[h],
            cumulative_returns=cumulative_returns[h],
            indicators=indicators,
            config=config,
            output_type="binary",
            seed=42 + h,
        )

    # E2E Multi
    for h in horizons:
        current += 1
        if verbose:
            print(f"[{current}/{total}] E2E Multi {h}M...")
        results[("E2E", "Multi", h)] = train_e2e_model(
            horizon=h,
            macro_data=macro_data,
            factor_data=factor_data,
            market_data=market_data,
            target_data=targets[h],
            cumulative_returns=cumulative_returns[h],
            indicators=indicators,
            config=config,
            output_type="allocation",
            seed=100 + h,
        )

    # Sup Binary
    for h in horizons:
        current += 1
        if verbose:
            print(f"[{current}/{total}] Sup Binary {h}M...")
        results[("Sup", "Binary", h)] = train_supervised_model(
            horizon=h,
            macro_data=macro_data,
            factor_data=factor_data,
            market_data=market_data,
            target_data=targets[h],
            indicators=indicators,
            feature_engineer=feature_engineer,
            config=config,
            output_type="binary",
            seed=1000 + h,
        )

    # Sup Multi
    for h in horizons:
        current += 1
        if verbose:
            print(f"[{current}/{total}] Sup Multi {h}M...")
        results[("Sup", "Multi", h)] = train_supervised_model(
            horizon=h,
            macro_data=macro_data,
            factor_data=factor_data,
            market_data=market_data,
            target_data=targets[h],
            indicators=indicators,
            feature_engineer=feature_engineer,
            config=config,
            output_type="allocation",
            seed=2000 + h,
        )

    if verbose:
        print(f"\nCompleted: {len(results)} combinations")

    return results


def format_results_table(
    results: Dict[Tuple[str, str, int], Dict],
    horizons: List[int] = None,
) -> str:
    """
    Format results as a readable table string.

    :param results: Dict of results from run_all_combinations
    :param horizons: List of horizons

    :return table: Formatted table string
    """
    if horizons is None:
        horizons = HORIZONS

    def fmt(v):
        return f"{v:+.4f}" if v is not None and not np.isnan(v) else "N/A"

    def fmt_pct(v):
        return f"{v:.1%}" if v is not None and not np.isnan(v) else "N/A"

    lines = []
    lines.append("=" * 100)
    lines.append("                                UNIFIED RESULTS: 16 COMBINATIONS")
    lines.append("=" * 100)
    lines.append(f"{'Strategy':<10} {'Allocation':<12} {'Horizon':<10} {'Sharpe':>10} {'IC':>10} {'Max DD':>10} {'Accuracy':>10}")
    lines.append("-" * 100)

    for strat in ["E2E", "Sup"]:
        for alloc in ["Binary", "Multi"]:
            for h in horizons:
                m = results[(strat, alloc, h)]
                sharpe = m.get('sharpe_ratio', np.nan)
                ic = m.get('information_coefficient', np.nan)
                maxdd = m.get('max_drawdown', np.nan)

                if alloc == "Binary":
                    acc = m.get('accuracy', np.nan)
                    acc_str = fmt_pct(acc)
                else:
                    acc_str = "-"

                lines.append(
                    f"{strat:<10} {alloc:<12} {h}M{'':<8} "
                    f"{fmt(sharpe):>10} {fmt(ic):>10} {fmt(maxdd):>10} {acc_str:>10}"
                )

    lines.append("=" * 100)

    # Best combination
    best_key = max(
        results.keys(),
        key=lambda k: results[k].get('sharpe_ratio', -999)
        if not np.isnan(results[k].get('sharpe_ratio', np.nan)) else -999
    )
    best_sharpe = results[best_key]['sharpe_ratio']
    lines.append(f"\nBEST: {best_key[0]} + {best_key[1]} @ {best_key[2]}M -> Sharpe={best_sharpe:+.4f}")

    return "\n".join(lines)


def compute_summary_stats(
    results: Dict[Tuple[str, str, int], Dict],
    horizons: List[int] = None,
) -> Dict:
    """
    Compute summary statistics by dimension.

    :param results: Dict of results
    :param horizons: List of horizons

    :return summary: Dict with summary stats
    """
    if horizons is None:
        horizons = HORIZONS

    summary = {}

    # By strategy
    e2e_sharpes = [results[k]['sharpe_ratio'] for k in results if k[0] == "E2E"]
    sup_sharpes = [results[k]['sharpe_ratio'] for k in results if k[0] == "Sup"]
    summary['strategy'] = {
        'E2E_avg': np.mean(e2e_sharpes),
        'Sup_avg': np.mean(sup_sharpes),
    }

    # By allocation
    binary_sharpes = [results[k]['sharpe_ratio'] for k in results if k[1] == "Binary"]
    multi_sharpes = [results[k]['sharpe_ratio'] for k in results if k[1] == "Multi"]
    summary['allocation'] = {
        'Binary_avg': np.mean(binary_sharpes),
        'Multi_avg': np.mean(multi_sharpes),
    }

    # By horizon
    summary['horizon'] = {}
    for h in horizons:
        h_sharpes = [results[k]['sharpe_ratio'] for k in results if k[2] == h]
        summary['horizon'][f'{h}M'] = np.mean(h_sharpes)

    return summary


def rank_models(
    results: Dict[Tuple[str, str, int], Dict],
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Rank models using composite score from multiple metrics.

    Score = w_sharpe × Sharpe_norm + w_ic × IC_norm + w_dd × (1 - |MaxDD|_norm)

    Higher score = better model.

    :param results: Dict of results from run_all_combinations
    :param weights: Dict with keys 'sharpe', 'ic', 'maxdd' (default: equal weights)

    :return ranking: DataFrame with scores and ranks
    """
    if weights is None:
        weights = {'sharpe': 0.4, 'ic': 0.3, 'maxdd': 0.3}

    # Extract metrics
    data = []
    for key, metrics in results.items():
        data.append({
            'strategy': key[0],
            'allocation': key[1],
            'horizon': key[2],
            'sharpe': metrics.get('sharpe_ratio', np.nan),
            'ic': metrics.get('information_coefficient', np.nan),
            'maxdd': metrics.get('max_drawdown', np.nan),
        })

    df = pd.DataFrame(data)

    # Normalize each metric to [0, 1]
    # Sharpe: higher is better
    sharpe_min, sharpe_max = df['sharpe'].min(), df['sharpe'].max()
    df['sharpe_norm'] = (df['sharpe'] - sharpe_min) / (sharpe_max - sharpe_min + 1e-8)

    # IC: higher is better
    ic_min, ic_max = df['ic'].min(), df['ic'].max()
    df['ic_norm'] = (df['ic'] - ic_min) / (ic_max - ic_min + 1e-8)

    # MaxDD: closer to 0 is better (less negative)
    # MaxDD is negative, so we want max(maxdd) which is closest to 0
    maxdd_min, maxdd_max = df['maxdd'].min(), df['maxdd'].max()
    df['maxdd_norm'] = (df['maxdd'] - maxdd_min) / (maxdd_max - maxdd_min + 1e-8)

    # Compute composite score
    df['score'] = (
        weights['sharpe'] * df['sharpe_norm'] +
        weights['ic'] * df['ic_norm'] +
        weights['maxdd'] * df['maxdd_norm']
    )

    # Rank (1 = best)
    df['rank'] = df['score'].rank(ascending=False).astype(int)

    # Sort by rank
    df = df.sort_values('rank')

    return df[['rank', 'strategy', 'allocation', 'horizon', 'sharpe', 'ic', 'maxdd', 'score']]


def find_best_model(
    results: Dict[Tuple[str, str, int], Dict],
    weights: Dict[str, float] = None,
) -> Tuple[Tuple[str, str, int], Dict, float]:
    """
    Find the best model based on composite score.

    :param results: Dict of results
    :param weights: Metric weights (default: sharpe=0.4, ic=0.3, maxdd=0.3)

    :return best_key: (strategy, allocation, horizon) tuple
    :return best_metrics: Metrics dict for best model
    :return best_score: Composite score
    """
    ranking = rank_models(results, weights)
    best_row = ranking.iloc[0]

    best_key = (best_row['strategy'], best_row['allocation'], best_row['horizon'])
    best_metrics = results[best_key]
    best_score = best_row['score']

    return best_key, best_metrics, best_score


def format_ranking_table(
    results: Dict[Tuple[str, str, int], Dict],
    weights: Dict[str, float] = None,
    top_n: int = 5,
) -> str:
    """
    Format ranking as a readable table string.

    :param results: Dict of results
    :param weights: Metric weights
    :param top_n: Number of top models to show

    :return table: Formatted table string
    """
    if weights is None:
        weights = {'sharpe': 0.4, 'ic': 0.3, 'maxdd': 0.3}

    ranking = rank_models(results, weights)

    lines = []
    lines.append("=" * 95)
    lines.append(f"MODEL RANKING (weights: Sharpe={weights['sharpe']}, IC={weights['ic']}, MaxDD={weights['maxdd']})")
    lines.append("=" * 95)
    lines.append(f"{'Rank':<6} {'Strategy':<10} {'Alloc':<8} {'Horizon':<8} {'Sharpe':>10} {'IC':>10} {'MaxDD':>10} {'Score':>10}")
    lines.append("-" * 95)

    for _, row in ranking.head(top_n).iterrows():
        lines.append(
            f"{int(row['rank']):<6} {row['strategy']:<10} {row['allocation']:<8} "
            f"{row['horizon']}M{'':<6} {row['sharpe']:>+10.4f} {row['ic']:>+10.4f} "
            f"{row['maxdd']:>+10.4f} {row['score']:>10.4f}"
        )

    lines.append("=" * 95)

    # Best model summary
    best = ranking.iloc[0]
    lines.append(f"\nBEST MODEL: {best['strategy']} + {best['allocation']} @ {best['horizon']}M")
    lines.append(f"  Sharpe={best['sharpe']:+.4f}, IC={best['ic']:+.4f}, MaxDD={best['maxdd']:+.4f}")
    lines.append(f"  Composite Score={best['score']:.4f}")

    return "\n".join(lines)
