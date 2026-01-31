"""Model implementations for factor allocation strategy."""

from .embeddings import MacroTokenEmbedding
from .transformer import (
    FactorAllocationTransformer,
    SharpeRatioLoss,
    BaselineRegularization,
    calibrate_turnover_penalty,
)
from .execution_gate import ExecutionGate
from .pretraining import (
    EmbeddingPretrainer,
    pretrain_embeddings,
    transfer_pretrained_embeddings,
)
from .training_strategies import (
    TrainingConfig,
    SupervisedTrainer,
    EndToEndTrainer,
    compute_optimal_weights,
    compute_rolling_optimal_weights,
)

__all__ = [
    "MacroTokenEmbedding",
    "FactorAllocationTransformer",
    "SharpeRatioLoss",
    "BaselineRegularization",
    "calibrate_turnover_penalty",
    "ExecutionGate",
    "EmbeddingPretrainer",
    "pretrain_embeddings",
    "transfer_pretrained_embeddings",
    "TrainingConfig",
    "SupervisedTrainer",
    "EndToEndTrainer",
    "compute_optimal_weights",
    "compute_rolling_optimal_weights",
]
