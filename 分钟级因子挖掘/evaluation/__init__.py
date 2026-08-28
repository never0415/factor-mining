"""Walk-forward evaluation for domain-constrained GP."""

from min_gp.evaluation.incremental import (
    IncrementalFitness,
    DEFAULT_COST_BPS,
    WalkForwardConfig,
    evaluate_incremental_fitness,
    trailing_signal_mean,
    cross_section_residual,
    cross_section_residual_multi,
    mean_rank_correlation,
    max_pool_correlation,
)
from min_gp.evaluation.archive import FactorArchive, factor_correlation
from min_gp.evaluation.neutralize import BatchedNeutralizer, neutralize_factor
from min_gp.evaluation.batched_incremental import BatchedIncrementalEvaluator

__all__ = [
    "IncrementalFitness",
    "DEFAULT_COST_BPS",
    "WalkForwardConfig",
    "evaluate_incremental_fitness",
    "trailing_signal_mean",
    "cross_section_residual",
    "cross_section_residual_multi",
    "mean_rank_correlation",
    "max_pool_correlation",
    "FactorArchive",
    "factor_correlation",
    "BatchedNeutralizer",
    "neutralize_factor",
    "BatchedIncrementalEvaluator",
]
