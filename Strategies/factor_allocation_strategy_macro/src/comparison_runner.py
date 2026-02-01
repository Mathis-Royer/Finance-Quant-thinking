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
from features.feature_selection import IndicatorSelector, SelectionConfig
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
    use_feature_selection: bool = False,
    n_features: int = 30,
    selection_method: str = "mutual_info",
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
    :param use_feature_selection: Apply indicator selection before training
    :param n_features: Number of indicators to keep (if use_feature_selection=True)
    :param selection_method: Selection method ('mutual_info', 'correlation', 'variance')

    :return results: Dict mapping (strategy, allocation, horizon) -> metrics
    """
    if horizons is None:
        horizons = HORIZONS

    # Apply feature selection if requested
    if use_feature_selection:
        if verbose:
            print(f"Applying feature selection: {selection_method}, n_features={n_features}")

        # Use 1M target for selection (most granular)
        target_for_selection = targets[min(horizons)]

        selector_config = SelectionConfig(
            method=selection_method,
            n_features=n_features,
        )
        selector = IndicatorSelector(selector_config)
        selector.fit(macro_data, target_for_selection, factor_data)

        # Filter macro data
        macro_data = selector.transform(macro_data)

        # Get selected indicator names
        selected_names = set(selector.get_selected_indicators())

        # Filter original indicator objects (not just names)
        # indicators can be list of objects with .name or list of strings
        if indicators and hasattr(indicators[0], 'name'):
            indicators = [ind for ind in indicators if ind.name in selected_names]
        else:
            indicators = [ind for ind in indicators if ind in selected_names]

        if verbose:
            print(f"  Selected {len(indicators)} indicators")
            if indicators and hasattr(indicators[0], 'name'):
                print(f"  Top 5: {[ind.name for ind in indicators[:5]]}")
            else:
                print(f"  Top 5: {indicators[:5]}")

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


def compute_composite_score(
    df: pd.DataFrame,
    sharpe_col: str = 'sharpe',
    ic_col: str = 'ic',
    maxdd_col: str = 'maxdd',
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Compute composite score for a DataFrame with Sharpe, IC, and MaxDD columns.

    Score = w_sharpe × Sharpe_norm + w_ic × IC_norm + w_dd × (1 - |MaxDD|_norm)

    Higher score = better model.

    :param df (pd.DataFrame): DataFrame with metrics columns
    :param sharpe_col (str): Column name for Sharpe ratio
    :param ic_col (str): Column name for IC
    :param maxdd_col (str): Column name for MaxDD
    :param weights (Dict): Weights for each metric (default: sharpe=0.4, ic=0.3, maxdd=0.3)

    :return df (pd.DataFrame): DataFrame with 'score' and 'rank' columns added
    """
    if weights is None:
        weights = {'sharpe': 0.4, 'ic': 0.3, 'maxdd': 0.3}

    df = df.copy()

    # Normalize each metric to [0, 1]
    # Sharpe: higher is better
    sharpe_min, sharpe_max = df[sharpe_col].min(), df[sharpe_col].max()
    sharpe_norm = (df[sharpe_col] - sharpe_min) / (sharpe_max - sharpe_min + 1e-8)

    # IC: higher is better
    ic_min, ic_max = df[ic_col].min(), df[ic_col].max()
    ic_norm = (df[ic_col] - ic_min) / (ic_max - ic_min + 1e-8)

    # MaxDD: closer to 0 is better (less negative)
    maxdd_min, maxdd_max = df[maxdd_col].min(), df[maxdd_col].max()
    maxdd_norm = (df[maxdd_col] - maxdd_min) / (maxdd_max - maxdd_min + 1e-8)

    # Compute composite score
    df['score'] = (
        weights['sharpe'] * sharpe_norm +
        weights['ic'] * ic_norm +
        weights['maxdd'] * maxdd_norm
    )

    # Rank (1 = best)
    df['rank'] = df['score'].rank(ascending=False).astype(int)

    return df


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


# =============================================================================
# WALK-FORWARD HYPERPARAMETER TUNING
# =============================================================================

from utils.hyperparameter_tuning import (
    WalkForwardTuner,
    HyperparameterSpace,
    TuningConfig,
    TuningResult,
)
from utils.walk_forward import WalkForwardValidator, WalkForwardWindow


def create_model_factory(
    feature_engineer: FeatureEngineer,
    base_config: Dict,
) -> callable:
    """
    Create a model factory function for the tuner.

    :param feature_engineer (FeatureEngineer): Feature engineer for indicator count
    :param base_config (Dict): Base configuration

    :return factory (callable): Function that creates model from params
    """
    def factory(params: Dict) -> FactorAllocationTransformer:
        config = {**base_config, **params}
        return FactorAllocationTransformer(
            num_indicators=feature_engineer.get_num_indicators(),
            num_factors=config.get('num_factors', 6),
            d_model=config.get('d_model', 32),
            num_heads=config.get('num_heads', 1),
            num_layers=config.get('num_layers', 1),
            d_ff=config.get('d_ff', 64),
            dropout=config.get('dropout', 0.6),
            max_seq_len=config.get('sequence_length', 12),
        )

    return factory


def create_trainer_factory(strategy: str = "supervised") -> callable:
    """
    Create a trainer factory function.

    :param strategy (str): 'supervised' or 'e2e'

    :return factory (callable): Function that creates trainer from model and params
    """
    def factory(model, params: Dict):
        config = TrainingConfig(
            learning_rate=params.get('learning_rate', 0.001),
            batch_size=params.get('batch_size', 32),
            weight_decay=params.get('weight_decay', 0.02),
            epochs_phase1=params.get('epochs_phase1', 20),
            epochs_phase2=params.get('epochs_phase2', 15),
            epochs_phase3=params.get('epochs_phase3', 15),
            horizon_months=params.get('horizon_months', 1),
        )

        if strategy == "supervised":
            return _SupervisedTrainerWrapper(model, config)
        else:
            return _E2ETrainerWrapper(model, config)

    return factory


class _SupervisedTrainerWrapper:
    """Wrapper to provide consistent training interface for tuner."""

    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        self.trainer = SupervisedTrainer(
            model=model,
            config=config,
            device=torch.device('cpu'),
            verbose=False,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray,
        verbose: bool = False,
    ):
        """Train the model with supervised strategy."""
        # Compute optimal weights from factor returns
        n_samples = len(factor_returns)
        dates = pd.date_range(start='2000-01-01', periods=n_samples, freq='MS')
        factor_df = pd.DataFrame(
            factor_returns,
            columns=FACTOR_COLUMNS,
        )
        factor_df['timestamp'] = dates

        target_dates = list(dates[:len(train_loader.dataset)])
        opt_weights = self.trainer.compute_targets(
            factor_df, target_dates, horizon_months=self.config.horizon_months
        )

        self.trainer.train(
            train_loader, val_loader, opt_weights, target_dates, verbose=verbose
        )


class _E2ETrainerWrapper:
    """Wrapper for E2E trainer with consistent interface."""

    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        from models.training_strategies import EndToEndTrainer
        self.trainer = EndToEndTrainer(
            model=model,
            config=config,
            device=torch.device('cpu'),
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        factor_returns: np.ndarray,
        verbose: bool = False,
    ):
        """Train with E2E strategy (3 phases)."""
        # Suppress output during tuning
        import sys
        from io import StringIO

        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = StringIO()

        try:
            self.trainer.train_phase1(train_loader, val_loader)
            self.trainer.train_phase2(train_loader, val_loader)
            self.trainer.train_phase3(train_loader, val_loader, factor_returns)
        finally:
            if not verbose:
                sys.stdout = old_stdout


def create_data_loader_factory(
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    indicators: List,
    config: Dict,
) -> callable:
    """
    Create a factory that generates data loaders for each walk-forward window.

    :param macro_data (pd.DataFrame): Macro features
    :param factor_data (pd.DataFrame): Factor returns
    :param market_data (pd.DataFrame): Market context
    :param target_data (pd.DataFrame): Target data
    :param indicators (List): FRED-MD indicators
    :param config (Dict): Configuration

    :return factory (callable): Function(window) -> dict of loaders
    """
    def factory(window: WalkForwardWindow) -> Dict:
        """Create train/val/test loaders for a specific window."""
        validator = WalkForwardValidator()

        # Split target data
        train_targets, val_targets, test_targets = validator.split_data(
            target_data, window, "timestamp"
        )

        if len(train_targets) == 0 or len(val_targets) == 0:
            raise ValueError(f"Insufficient data for window {window}")

        # Create strategy for data preparation
        strat = FactorAllocationStrategy(
            region=Region.US,
            use_fred_md=True,
            fred_md_indicators=indicators,
            config=config,
            verbose=False,
        )
        strat.create_model()

        # Prepare datasets for each split
        train_ds, _ = strat.prepare_data(
            macro_data, factor_data, market_data, train_targets
        )
        val_ds, _ = strat.prepare_data(
            macro_data, factor_data, market_data, val_targets
        )
        test_ds, _ = strat.prepare_data(
            macro_data, factor_data, market_data, test_targets
        )

        batch_size = config.get('batch_size', 32)
        g = torch.Generator()
        g.manual_seed(42)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            generator=g,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Get factor returns for each split
        train_start = pd.Timestamp(window.train_start)
        val_end = pd.Timestamp(window.val_end)
        test_end = pd.Timestamp(window.test_end)

        factor_data_ts = factor_data.copy()
        factor_data_ts['timestamp'] = pd.to_datetime(factor_data_ts['timestamp'])

        train_mask = (factor_data_ts['timestamp'] >= train_start) & (factor_data_ts['timestamp'] <= val_end)
        test_mask = (factor_data_ts['timestamp'] > val_end) & (factor_data_ts['timestamp'] <= test_end)

        factor_returns_train = factor_data_ts.loc[train_mask, FACTOR_COLUMNS].values
        factor_returns_test = factor_data_ts.loc[test_mask, FACTOR_COLUMNS].values

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'factor_returns_train': factor_returns_train,
            'factor_returns_test': factor_returns_test,
        }

    return factory


def run_with_tuning(
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    indicators: List,
    feature_engineer: FeatureEngineer,
    config: Dict,
    strategy: str = "supervised",
    n_trials: int = 15,
    optimization_metric: str = "sharpe",
    verbose: bool = True,
) -> Tuple[Dict[str, any], List[TuningResult]]:
    """
    Run walk-forward hyperparameter tuning.

    For each walk-forward window:
    1. Tune hyperparameters on train/val split
    2. Retrain with best params on full train
    3. Evaluate on test set

    :param macro_data (pd.DataFrame): Macro features
    :param factor_data (pd.DataFrame): Factor returns
    :param market_data (pd.DataFrame): Market context
    :param target_data (pd.DataFrame): Target data
    :param indicators (List): FRED-MD indicators
    :param feature_engineer (FeatureEngineer): Feature engineer
    :param config (Dict): Base configuration
    :param strategy (str): 'supervised' or 'e2e'
    :param n_trials (int): Number of trials per window
    :param optimization_metric (str): 'sharpe', 'ic', or 'combined'
    :param verbose (bool): Print progress

    :return summary (Dict): Aggregated results and best params
    :return results (List[TuningResult]): Per-window results
    """
    if verbose:
        print("=" * 70)
        print("WALK-FORWARD HYPERPARAMETER TUNING")
        print(f"Strategy: {strategy}, Trials: {n_trials}, Metric: {optimization_metric}")
        print("=" * 70)

    # Create walk-forward windows
    validator = WalkForwardValidator()
    windows = validator.create_expanding_windows(
        data_start="2000-01-01",
        data_end="2024-12-31",
        initial_train_years=14,
        val_years=3,
        step_years=1,
    )

    if verbose:
        print(f"\nCreated {len(windows)} walk-forward windows:")
        for i, w in enumerate(windows):
            print(f"  Window {i+1}: Train {w.train_start}-{w.train_end}, "
                  f"Val {w.val_start}-{w.val_end}, Test {w.test_start}-{w.test_end}")

    # Create factories
    model_factory = create_model_factory(feature_engineer, config)
    trainer_factory = create_trainer_factory(strategy)
    data_factory = create_data_loader_factory(
        macro_data, factor_data, market_data, target_data, indicators, config
    )

    # Configure tuning
    space = HyperparameterSpace(
        learning_rate=(5e-5, 5e-3, True),
        dropout=(0.4, 0.7),
        weight_decay=(5e-4, 5e-2, True),
        batch_size=[16, 32],
        tune_architecture=False,
        tune_epochs=True,
        epochs_phase1=(10, 25),
        epochs_phase2=(8, 20),
        epochs_phase3=(8, 20),
    )

    tuning_config = TuningConfig(
        n_trials=n_trials,
        optimization_metric=optimization_metric,
        early_stopping_patience=5,
        verbose=verbose,
        use_pruning=True,
    )

    # Create tuner
    tuner = WalkForwardTuner(
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        space=space,
        config=tuning_config,
        device=torch.device('cpu'),
    )

    # Run tuning
    results = tuner.tune_all_windows(windows, data_factory, config)

    # Generate summary
    summary = tuner.aggregate_results()
    summary['best_params'] = tuner.get_best_overall_params()

    if verbose:
        print("\n" + tuner.generate_report())

    return summary, results


def format_tuning_comparison(
    tuned_results: Dict,
    baseline_results: Dict[Tuple[str, str, int], Dict],
    strategy: str = "Sup",
    allocation: str = "Multi",
    horizon: int = 3,
) -> str:
    """
    Format comparison between tuned and baseline results.

    :param tuned_results (Dict): Tuning summary with test metrics
    :param baseline_results (Dict): Results from run_all_combinations
    :param strategy (str): Strategy to compare
    :param allocation (str): Allocation to compare
    :param horizon (int): Horizon to compare

    :return comparison (str): Formatted comparison string
    """
    baseline_key = (strategy, allocation, horizon)
    baseline = baseline_results.get(baseline_key, {})

    lines = []
    lines.append("=" * 60)
    lines.append("TUNING VS BASELINE COMPARISON")
    lines.append("=" * 60)
    lines.append(f"Configuration: {strategy} + {allocation} @ {horizon}M")
    lines.append("")

    lines.append(f"{'Metric':<20} {'Baseline':>15} {'Tuned':>15} {'Delta':>15}")
    lines.append("-" * 60)

    # Sharpe
    baseline_sharpe = baseline.get('sharpe_ratio', 0)
    tuned_sharpe = tuned_results.get('avg_test_sharpe', 0)
    delta_sharpe = tuned_sharpe - baseline_sharpe
    lines.append(f"{'Sharpe':<20} {baseline_sharpe:>+15.4f} {tuned_sharpe:>+15.4f} {delta_sharpe:>+15.4f}")

    # IC
    baseline_ic = baseline.get('information_coefficient', 0)
    tuned_ic = tuned_results.get('avg_test_ic', 0)
    delta_ic = tuned_ic - baseline_ic
    lines.append(f"{'IC':<20} {baseline_ic:>+15.4f} {tuned_ic:>+15.4f} {delta_ic:>+15.4f}")

    lines.append("=" * 60)

    # Best tuned params
    lines.append("\nBest Tuned Parameters:")
    best_params = tuned_results.get('best_params', {})
    for k, v in best_params.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# =============================================================================
# WALK-FORWARD ANALYSIS FOR 16 COMBINATIONS
# =============================================================================

@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_id: int
    test_year: str
    test_start: str
    test_end: str
    train_years: str
    sharpe: float
    ic: float
    maxdd: float
    accuracy: float = None
    monthly_returns: List[float] = None  # Monthly portfolio returns
    monthly_dates: List[str] = None  # Corresponding dates
    model: any = None  # Trained model for this window (optional)


def run_combination_walk_forward(
    strategy: str,
    allocation: str,
    horizon: int,
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    cumulative_returns: np.ndarray,
    indicators: List,
    feature_engineer: FeatureEngineer,
    config: Dict,
    verbose: bool = True,
    holdout_years: int = 0,
    save_models: bool = False,
) -> List[WindowResult]:
    """
    Run a single combination with walk-forward validation.

    Returns per-window metrics for year-by-year analysis.

    :param strategy (str): 'E2E' or 'Sup'
    :param allocation (str): 'Binary' or 'Multi'
    :param horizon (int): Prediction horizon (1, 3, 6, 12)
    :param macro_data (pd.DataFrame): Macro features
    :param factor_data (pd.DataFrame): Factor returns
    :param market_data (pd.DataFrame): Market context
    :param target_data (pd.DataFrame): Target data for this horizon
    :param cumulative_returns (np.ndarray): Cumulative returns for this horizon
    :param indicators (List): FRED-MD indicators
    :param feature_engineer (FeatureEngineer): Feature engineer
    :param config (Dict): Model configuration
    :param verbose (bool): Print progress
    :param holdout_years (int): Years to reserve at end (no model sees these)
    :param save_models (bool): Whether to save trained models in results

    :return results (List[WindowResult]): Per-window metrics (with models if save_models=True)
    """
    from utils.walk_forward import WalkForwardValidator
    from utils.metrics import PerformanceMetrics
    import copy

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"WALK-FORWARD: {strategy} + {allocation} @ {horizon}M")
        if holdout_years > 0:
            print(f"HOLDOUT: Last {holdout_years} years reserved (2023-2024)")
        print(f"{'=' * 60}")

    # Create walk-forward windows (respecting holdout)
    validator = WalkForwardValidator()
    windows = validator.create_expanding_windows(
        data_start="2000-01-01",
        data_end="2024-12-31",
        initial_train_years=14,
        val_years=3,
        step_years=1,
        holdout_years=holdout_years,
    )

    if verbose:
        print(f"Created {len(windows)} walk-forward windows")

    output_type = "binary" if allocation == "Binary" else "allocation"
    window_results = []

    for i, window in enumerate(windows):
        if verbose:
            print(f"\n  Window {i+1}/{len(windows)}: Test {window.test_start[:4]}")

        # Split data for this window
        train_targets, val_targets, test_targets = validator.split_data(
            target_data, window, "timestamp"
        )

        if len(train_targets) < 10 or len(test_targets) < 3:
            if verbose:
                print(f"    Skipping: insufficient data")
            continue

        # Combine train + val for training
        combined_targets = pd.concat([train_targets, val_targets], ignore_index=True)

        # Prepare factor data with timestamps
        factor_ts = factor_data.copy()
        factor_ts['timestamp'] = pd.to_datetime(factor_ts['timestamp'])

        try:
            # Include horizon in seed to ensure different models per horizon
            # Without this, same window + same seed = identical initialization
            set_seed(42 + i * 100 + horizon)
            trained_model = None  # Will be set if save_models=True

            if strategy == "E2E":
                # Train E2E model
                strat = FactorAllocationStrategy(
                    region=Region.US,
                    use_fred_md=True,
                    fred_md_indicators=indicators,
                    config={**config, "horizon_months": horizon},
                    verbose=False,
                )
                strat.create_model()

                train_ds, val_ds = strat.prepare_data(
                    macro_data, factor_data, market_data, combined_targets
                )

                g = torch.Generator()
                g.manual_seed(42 + i * 100 + horizon)

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

                # Get cumulative returns for training period
                train_start = pd.Timestamp(window.train_start)
                val_end = pd.Timestamp(window.val_end)
                train_mask = (factor_ts['timestamp'] >= train_start) & (factor_ts['timestamp'] <= val_end)
                train_cum_returns = factor_ts.loc[train_mask, FACTOR_COLUMNS].values

                # 3-phase training
                strat.train_phase1(train_loader, val_loader, verbose=False)
                strat.train_phase2(train_loader, val_loader, verbose=False)
                strat.train_phase3(train_loader, val_loader, train_cum_returns, verbose=False)

                # Backtest on test period only
                test_results = strat.backtest(
                    macro_data, factor_data, market_data, test_targets,
                    output_type=output_type, verbose=False
                )
                metrics = strat.evaluate(test_results, verbose=False)

                # Save model if requested
                trained_model = copy.deepcopy(strat.model) if save_models else None

            else:  # Supervised
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

                training_config = TrainingConfig(
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                    weight_decay=config['weight_decay'],
                    epochs_phase1=config['epochs_phase1'],
                    epochs_phase2=config['epochs_phase2'],
                    epochs_phase3=config['epochs_phase3'],
                    horizon_months=horizon,
                )

                trainer = SupervisedTrainer(
                    model=model,
                    config=training_config,
                    device=torch.device('cpu'),
                    verbose=False,
                )

                # Prepare data
                strat_sup = FactorAllocationStrategy(
                    region=Region.US,
                    use_fred_md=True,
                    fred_md_indicators=indicators,
                    config=config,
                    verbose=False,
                )
                strat_sup.create_model()

                train_ds, val_ds = strat_sup.prepare_data(
                    macro_data, factor_data, market_data, combined_targets
                )

                g = torch.Generator()
                g.manual_seed(42 + i * 100 + horizon)

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

                # Compute targets
                combined_dates = combined_targets['timestamp'].tolist()
                opt_weights = trainer.compute_targets(factor_data, combined_dates, horizon_months=horizon)

                # Train
                trainer.train(train_loader, val_loader, opt_weights, combined_dates, verbose=False)

                # Backtest on test period
                strat_sup.model = model
                test_results = strat_sup.backtest(
                    macro_data, factor_data, market_data, test_targets,
                    output_type=output_type, verbose=False
                )
                metrics = strat_sup.evaluate(test_results, verbose=False)

                # Save model if requested
                trained_model = copy.deepcopy(model) if save_models else None

            # Extract monthly returns from backtest results
            monthly_rets = test_results.get('portfolio_returns', [])
            monthly_dates_raw = test_results.get('timestamps', [])

            # Convert to lists for storage
            if hasattr(monthly_rets, 'tolist'):
                monthly_rets = monthly_rets.tolist()
            monthly_dates_str = [str(d)[:10] for d in monthly_dates_raw]

            # Store result
            window_results.append(WindowResult(
                window_id=i + 1,
                test_year=window.test_start[:4],
                test_start=window.test_start,
                test_end=window.test_end,
                train_years=f"{window.train_start[:4]}-{window.val_end[:4]}",
                sharpe=metrics.get('sharpe_ratio', 0),
                ic=metrics.get('information_coefficient', 0),
                maxdd=metrics.get('max_drawdown', 0),
                accuracy=metrics.get('accuracy', None),
                monthly_returns=monthly_rets,
                monthly_dates=monthly_dates_str,
                model=trained_model,
            ))

            if verbose:
                print(f"    Sharpe={metrics.get('sharpe_ratio', 0):+.4f}, "
                      f"IC={metrics.get('information_coefficient', 0):+.4f}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Completed {len(window_results)} windows")
        if window_results:
            avg_sharpe = np.mean([r.sharpe for r in window_results])
            avg_ic = np.mean([r.ic for r in window_results])
            print(f"Avg Sharpe: {avg_sharpe:+.4f}, Avg IC: {avg_ic:+.4f}")

    return window_results


def format_walk_forward_table(results: List[WindowResult]) -> str:
    """
    Format walk-forward results as a table.

    :param results (List[WindowResult]): Per-window results

    :return table (str): Formatted table
    """
    if not results:
        return "No results to display"

    lines = []
    lines.append("=" * 80)
    lines.append("WALK-FORWARD YEAR-BY-YEAR RESULTS")
    lines.append("=" * 80)
    lines.append(f"{'Window':<8} {'Test Year':<12} {'Train Period':<15} {'Sharpe':>10} {'IC':>10} {'MaxDD':>10}")
    lines.append("-" * 80)

    for r in results:
        lines.append(f"{r.window_id:<8} {r.test_year:<12} {r.train_years:<15} "
                     f"{r.sharpe:>+10.4f} {r.ic:>+10.4f} {r.maxdd:>+10.4f}")

    lines.append("-" * 80)

    # Summary
    avg_sharpe = np.mean([r.sharpe for r in results])
    std_sharpe = np.std([r.sharpe for r in results])
    avg_ic = np.mean([r.ic for r in results])
    pct_positive = sum(1 for r in results if r.sharpe > 0) / len(results) * 100

    lines.append(f"\nSUMMARY:")
    lines.append(f"  Avg Sharpe: {avg_sharpe:+.4f} (±{std_sharpe:.4f})")
    lines.append(f"  Avg IC: {avg_ic:+.4f}")
    lines.append(f"  Positive Sharpe: {pct_positive:.0f}% of windows")

    best_idx = np.argmax([r.sharpe for r in results])
    worst_idx = np.argmin([r.sharpe for r in results])
    lines.append(f"  Best year: {results[best_idx].test_year} (Sharpe={results[best_idx].sharpe:+.4f})")
    lines.append(f"  Worst year: {results[worst_idx].test_year} (Sharpe={results[worst_idx].sharpe:+.4f})")

    lines.append("=" * 80)

    return "\n".join(lines)


# =============================================================================
# FINAL MODEL & HOLDOUT EVALUATION
# =============================================================================

@dataclass
class HoldoutResult:
    """Result from holdout evaluation."""
    model_type: str  # 'final' or 'ensemble'
    sharpe: float
    ic: float
    maxdd: float
    total_return: float
    accuracy: float = None
    monthly_returns: List[float] = None
    monthly_dates: List[str] = None


def train_final_model(
    strategy: str,
    allocation: str,
    horizon: int,
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    cumulative_returns: np.ndarray,
    indicators: List,
    feature_engineer: FeatureEngineer,
    config: Dict,
    holdout_years: int = 2,
    holdout_start_date: str = None,
    verbose: bool = True,
):
    """
    Train final model on all data EXCEPT holdout period.

    :param strategy (str): 'E2E' or 'Sup'
    :param allocation (str): 'Binary' or 'Multi'
    :param horizon (int): Prediction horizon
    :param macro_data (pd.DataFrame): Macro features
    :param factor_data (pd.DataFrame): Factor returns
    :param market_data (pd.DataFrame): Market context
    :param target_data (pd.DataFrame): Target data
    :param cumulative_returns (np.ndarray): Cumulative returns
    :param indicators (List): FRED-MD indicators
    :param feature_engineer (FeatureEngineer): Feature engineer
    :param config (Dict): Model configuration
    :param holdout_years (int): Years to exclude from training (default: 2, ignored if holdout_start_date set)
    :param holdout_start_date (str): Fixed holdout start date (e.g., "2022-01-01") for comparable periods
    :param verbose (bool): Print progress

    :return model: Trained model
    :return strategy_obj: Strategy object for backtesting
    """
    import copy

    # Determine cutoff date
    target_data = target_data.copy()
    target_data['timestamp'] = pd.to_datetime(target_data['timestamp'])
    data_end_date = target_data['timestamp'].max()

    # Use fixed holdout_start_date if provided, else calculate from holdout_years
    if holdout_start_date:
        cutoff_date = holdout_start_date
        holdout_start_year = int(holdout_start_date[:4])
    else:
        data_end_year = data_end_date.year
        holdout_start_year = data_end_year - holdout_years + 1
        cutoff_date = f"{holdout_start_year}-01-01"

    # Filter target data to exclude holdout
    train_targets = target_data[target_data['timestamp'] < cutoff_date].copy()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"TRAINING FINAL MODEL: {strategy} + {allocation} @ {horizon}M")
        print(f"Training period: 2000-01-01 to {holdout_start_year - 1}-12-31")
        print(f"Holdout period: {holdout_start_year}-01-01 to {data_end_date.date()}")
        print(f"{'=' * 60}")

    set_seed(999)  # Different seed for final model

    # Filter factor data for training period
    factor_ts = factor_data.copy()
    factor_ts['timestamp'] = pd.to_datetime(factor_ts['timestamp'])
    train_factor_mask = factor_ts['timestamp'] < cutoff_date
    train_cum_returns = factor_ts.loc[train_factor_mask, FACTOR_COLUMNS].values

    if strategy == "E2E":
        strat = FactorAllocationStrategy(
            region=Region.US,
            use_fred_md=True,
            fred_md_indicators=indicators,
            config={**config, "horizon_months": horizon},
            verbose=False,
        )
        strat.create_model()

        train_ds, val_ds = strat.prepare_data(
            macro_data, factor_data, market_data, train_targets
        )

        g = torch.Generator()
        g.manual_seed(999)

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
        if verbose:
            print("  Phase 1: Binary classification...")
        strat.train_phase1(train_loader, val_loader, verbose=False)
        if verbose:
            print("  Phase 2: Regression...")
        strat.train_phase2(train_loader, val_loader, verbose=False)
        if verbose:
            print("  Phase 3: Sharpe optimization...")
        strat.train_phase3(train_loader, val_loader, train_cum_returns, verbose=False)

        if verbose:
            print("  Training complete.")

        return copy.deepcopy(strat.model), strat

    else:  # Supervised
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

        training_config = TrainingConfig(
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            weight_decay=config['weight_decay'],
            epochs_phase1=config['epochs_phase1'],
            epochs_phase2=config['epochs_phase2'],
            epochs_phase3=config['epochs_phase3'],
            horizon_months=horizon,
        )

        trainer = SupervisedTrainer(
            model=model,
            config=training_config,
            device=torch.device('cpu'),
            verbose=False,
        )

        strat_sup = FactorAllocationStrategy(
            region=Region.US,
            use_fred_md=True,
            fred_md_indicators=indicators,
            config=config,
            verbose=False,
        )
        strat_sup.create_model()

        train_ds, val_ds = strat_sup.prepare_data(
            macro_data, factor_data, market_data, train_targets
        )

        g = torch.Generator()
        g.manual_seed(999)

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

        # Compute targets
        train_dates = train_targets['timestamp'].tolist()
        opt_weights = trainer.compute_targets(factor_data, train_dates, horizon_months=horizon)

        if verbose:
            print("  Training supervised model...")
        trainer.train(train_loader, val_loader, opt_weights, train_dates, verbose=False)

        if verbose:
            print("  Training complete.")

        strat_sup.model = model
        return copy.deepcopy(model), strat_sup


def evaluate_on_holdout(
    model,
    strategy_obj,
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    holdout_years: int = 2,
    holdout_start_date: str = None,
    output_type: str = "binary",
    model_type: str = "final",
    verbose: bool = True,
) -> HoldoutResult:
    """
    Evaluate a model on the holdout period.

    :param model: Trained model
    :param strategy_obj: Strategy object for backtesting
    :param macro_data (pd.DataFrame): Macro features
    :param factor_data (pd.DataFrame): Factor returns
    :param market_data (pd.DataFrame): Market context
    :param target_data (pd.DataFrame): Target data
    :param holdout_years (int): Years in holdout (default: 2, ignored if holdout_start_date set)
    :param holdout_start_date (str): Fixed holdout start date (e.g., "2022-01-01") for comparable periods
    :param output_type (str): 'binary' or 'allocation'
    :param model_type (str): 'final' or 'ensemble' (for labeling)
    :param verbose (bool): Print progress

    :return result (HoldoutResult): Holdout evaluation metrics
    """
    # Determine holdout period
    target_data = target_data.copy()
    target_data['timestamp'] = pd.to_datetime(target_data['timestamp'])
    data_end_date = target_data['timestamp'].max()

    # Use fixed holdout_start_date if provided, else calculate from holdout_years
    if holdout_start_date:
        holdout_start = holdout_start_date
        holdout_start_year = int(holdout_start_date[:4])
    else:
        data_end_year = data_end_date.year
        holdout_start_year = data_end_year - holdout_years + 1
        holdout_start = f"{holdout_start_year}-01-01"

    holdout_end = str(data_end_date.date())

    # Filter target data to holdout only
    holdout_targets = target_data[
        (target_data['timestamp'] >= holdout_start) &
        (target_data['timestamp'] <= holdout_end)
    ].copy()

    if verbose:
        print(f"\n  Evaluating {model_type} model on holdout ({holdout_start_year}-2024)...")
        print(f"  Holdout samples: {len(holdout_targets)}")

    # Set model in strategy
    strategy_obj.model = model

    # Backtest on holdout
    results = strategy_obj.backtest(
        macro_data, factor_data, market_data, holdout_targets,
        output_type=output_type, verbose=False
    )
    metrics = strategy_obj.evaluate(results, verbose=False)

    # Extract monthly returns
    monthly_rets = results.get('portfolio_returns', [])
    monthly_dates_raw = results.get('timestamps', [])

    if hasattr(monthly_rets, 'tolist'):
        monthly_rets = monthly_rets.tolist()
    monthly_dates_str = [str(d)[:10] for d in monthly_dates_raw]

    # Calculate total return
    if monthly_rets:
        total_return = (np.prod(1 + np.array(monthly_rets)) - 1) * 100
    else:
        total_return = 0.0

    if verbose:
        print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):+.4f}")
        print(f"  IC: {metrics.get('information_coefficient', 0):+.4f}")
        print(f"  Total Return: {total_return:+.2f}%")

    return HoldoutResult(
        model_type=model_type,
        sharpe=metrics.get('sharpe_ratio', 0),
        ic=metrics.get('information_coefficient', 0),
        maxdd=metrics.get('max_drawdown', 0),
        total_return=total_return,
        accuracy=metrics.get('accuracy', None),
        monthly_returns=monthly_rets,
        monthly_dates=monthly_dates_str,
    )


def ensemble_predict(
    models: List,
    strategy_obj,
    macro_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    target_data: pd.DataFrame,
    holdout_years: int = 2,
    holdout_start_date: str = None,
    output_type: str = "binary",
    verbose: bool = True,
) -> HoldoutResult:
    """
    Ensemble prediction from N walk-forward models on holdout period.

    Averages predictions from all models, then computes portfolio returns.

    :param models (List): List of trained models from walk-forward
    :param strategy_obj: Strategy object for backtesting
    :param macro_data (pd.DataFrame): Macro features
    :param factor_data (pd.DataFrame): Factor returns
    :param market_data (pd.DataFrame): Market context
    :param target_data (pd.DataFrame): Target data
    :param holdout_years (int): Years in holdout (default: 2, ignored if holdout_start_date set)
    :param holdout_start_date (str): Fixed holdout start date (e.g., "2022-01-01") for comparable periods
    :param output_type (str): 'binary' or 'allocation'
    :param verbose (bool): Print progress

    :return result (HoldoutResult): Ensemble holdout evaluation metrics
    """
    import copy

    # Determine holdout period
    target_data = target_data.copy()
    target_data['timestamp'] = pd.to_datetime(target_data['timestamp'])
    data_end_date = target_data['timestamp'].max()

    # Use fixed holdout_start_date if provided, else calculate from holdout_years
    if holdout_start_date:
        holdout_start = holdout_start_date
        holdout_start_year = int(holdout_start_date[:4])
    else:
        data_end_year = data_end_date.year
        holdout_start_year = data_end_year - holdout_years + 1
        holdout_start = f"{holdout_start_year}-01-01"

    holdout_end = str(data_end_date.date())

    # Filter target data to holdout only
    holdout_targets = target_data[
        (target_data['timestamp'] >= holdout_start) &
        (target_data['timestamp'] <= holdout_end)
    ].copy()

    if verbose:
        print(f"\n  Ensemble evaluation ({len(models)} models) on holdout ({holdout_start_year}-2024)...")
        print(f"  Holdout samples: {len(holdout_targets)}")

    # Collect predictions/weights from each model
    all_weights = []  # For allocation mode: actual 6-factor weights
    all_predictions = []  # For binary mode: scalar predictions

    for i, model in enumerate(models):
        # Set model in strategy
        strat_copy = copy.deepcopy(strategy_obj)
        strat_copy.model = model

        # Get predictions
        results = strat_copy.backtest(
            macro_data, factor_data, market_data, holdout_targets,
            output_type=output_type, verbose=False
        )

        if output_type == "allocation":
            # For allocation mode, extract the 6-factor weights
            weights = results.get('weights', None)
            if weights is not None and len(weights) > 0:
                all_weights.append(weights)
        else:
            # For binary mode, extract predictions and convert to weights
            predictions = results.get('predictions', None)
            if predictions is not None and len(predictions) > 0:
                all_predictions.append(predictions)

    # Handle allocation mode
    if output_type == "allocation":
        if not all_weights:
            if verbose:
                print("  No weights available from ensemble models")
            return HoldoutResult(
                model_type="ensemble",
                sharpe=0.0,
                ic=0.0,
                maxdd=0.0,
                total_return=0.0,
            )

        # Average weights across models
        avg_weights = np.mean(all_weights, axis=0)

    # Handle binary mode
    else:
        if not all_predictions:
            if verbose:
                print("  No predictions available from ensemble models")
            return HoldoutResult(
                model_type="ensemble",
                sharpe=0.0,
                ic=0.0,
                maxdd=0.0,
                total_return=0.0,
            )

        # Average predictions across models
        avg_predictions = np.mean(all_predictions, axis=0)

    # Compute returns using averaged weights/predictions
    factor_ts = factor_data.copy()
    factor_ts['timestamp'] = pd.to_datetime(factor_ts['timestamp'])

    holdout_dates = holdout_targets['timestamp'].tolist()
    monthly_returns = []
    monthly_dates = []
    actuals = holdout_targets['target'].values

    for i, date in enumerate(holdout_dates):
        # Get factor returns for this month
        next_month = date + pd.DateOffset(months=1)
        mask = (factor_ts['timestamp'] >= date) & (factor_ts['timestamp'] < next_month)
        month_returns = factor_ts.loc[mask, FACTOR_COLUMNS].values

        if len(month_returns) > 0:
            factor_ret = month_returns[0]  # Monthly returns

            if output_type == "allocation":
                if i >= len(avg_weights):
                    break
                weights = avg_weights[i]
                # Normalize weights (should already sum to 1 from softmax)
                weights = weights / (weights.sum() + 1e-8)
            else:
                if i >= len(avg_predictions):
                    break
                pred = avg_predictions[i]
                # Binary: use prediction probability as weights for cyclical/defensive
                # This allows different horizons to produce different results
                # because the magnitude of pred matters, not just direction
                weights = np.array([pred, 1 - pred, 0.0, 0.0, 0.0, 0.0])

            portfolio_ret = np.dot(weights, factor_ret)
            monthly_returns.append(portfolio_ret)
            monthly_dates.append(str(date)[:10])

    # Compute metrics
    returns_arr = np.array(monthly_returns)
    if len(returns_arr) > 1 and returns_arr.std() > 1e-8:
        sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

    # Max drawdown
    if len(returns_arr) > 0:
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        maxdd = drawdown.min() if len(drawdown) > 0 else 0.0
    else:
        maxdd = 0.0

    # Total return
    total_return = (np.prod(1 + returns_arr) - 1) * 100 if len(returns_arr) > 0 else 0.0

    # IC (correlation between predictions and actuals)
    if output_type == "allocation" and len(avg_weights) > 0:
        # For allocation, compute IC using cyclical ratio vs actual
        predicted_cyclical = [w[0] / (w[0] + w[1] + 1e-8) for w in avg_weights[:len(actuals)]]
        if len(predicted_cyclical) > 1 and len(actuals) > 1:
            ic = np.corrcoef(predicted_cyclical, actuals[:len(predicted_cyclical)])[0, 1]
            if np.isnan(ic):
                ic = 0.0
        else:
            ic = 0.0
    elif output_type == "binary" and len(avg_predictions) > 0:
        if len(avg_predictions) > 1 and len(actuals) > 1:
            ic = np.corrcoef(avg_predictions[:len(actuals)], actuals[:len(avg_predictions)])[0, 1]
            if np.isnan(ic):
                ic = 0.0
        else:
            ic = 0.0
    else:
        ic = 0.0

    if verbose:
        print(f"  Ensemble Sharpe: {sharpe:+.4f}")
        print(f"  Ensemble IC: {ic:+.4f}")
        print(f"  Ensemble Total Return: {total_return:+.2f}%")

    return HoldoutResult(
        model_type="ensemble",
        sharpe=sharpe,
        ic=ic,
        maxdd=maxdd,
        total_return=total_return,
        monthly_returns=monthly_returns,
        monthly_dates=monthly_dates,
    )


def compare_final_vs_ensemble(
    final_result: HoldoutResult,
    ensemble_result: HoldoutResult,
) -> str:
    """
    Format comparison between final model and ensemble on holdout.

    :param final_result (HoldoutResult): Final model holdout results
    :param ensemble_result (HoldoutResult): Ensemble holdout results

    :return comparison (str): Formatted comparison string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("HOLDOUT COMPARISON: FINAL MODEL vs ENSEMBLE")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Metric':<20} {'Final Model':>15} {'Ensemble':>15} {'Delta':>15}")
    lines.append("-" * 70)

    # Sharpe
    delta_sharpe = final_result.sharpe - ensemble_result.sharpe
    lines.append(f"{'Sharpe':<20} {final_result.sharpe:>+15.4f} {ensemble_result.sharpe:>+15.4f} {delta_sharpe:>+15.4f}")

    # Total Return
    delta_return = final_result.total_return - ensemble_result.total_return
    lines.append(f"{'Total Return (%)':<20} {final_result.total_return:>+15.2f} {ensemble_result.total_return:>+15.2f} {delta_return:>+15.2f}")

    # Max Drawdown
    delta_dd = final_result.maxdd - ensemble_result.maxdd
    lines.append(f"{'Max Drawdown':<20} {final_result.maxdd:>+15.4f} {ensemble_result.maxdd:>+15.4f} {delta_dd:>+15.4f}")

    lines.append("-" * 70)

    # Verdict
    if final_result.sharpe > ensemble_result.sharpe:
        verdict = "FINAL MODEL outperforms ensemble"
    elif ensemble_result.sharpe > final_result.sharpe:
        verdict = "ENSEMBLE outperforms final model"
    else:
        verdict = "Performance is similar"

    lines.append(f"\nVERDICT: {verdict}")
    lines.append("=" * 70)

    return "\n".join(lines)
