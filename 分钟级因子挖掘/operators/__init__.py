"""Domain operator collections."""

import os
from dataclasses import replace
from pathlib import Path

from min_gp.dsl.calibration_store import load_calibration_file
from min_gp.dsl.registry import OperatorRegistry
from min_gp.dsl.types import SemanticType
from min_gp.operators.cross_section import register_cross_section_operators
from min_gp.operators.event import register_event_operators
from min_gp.operators.spectral import register_spectral_operators
from min_gp.operators.temporal import register_temporal_operators
from min_gp.operators.daily import register_daily_operators
from min_gp.operators.distribution import register_distribution_operators
from min_gp.operators.path import register_path_operators
from min_gp.operators.regression import register_regression_operators
from min_gp.operators.handbook import register_handbook_operators
from min_gp.operators.intraday import register_intraday_operators
from min_gp.operators.seed_tree import register_seed_tree_operators
from min_gp.operators.handbook_parts import register_handbook_part_operators


def _complete_complexity_metadata(registry):
    """Supply conservative axis exponents where an operator omitted them."""
    minute_types = {
        kind for kind in SemanticType
        if kind.value.startswith("minute_")
    } | {SemanticType.SPECTRUM}
    daily_types = {
        kind for kind in SemanticType
        if kind.value.startswith("daily_")
    } | {
        SemanticType.MARKET_DAILY_PRICE, SemanticType.PAIR_SIMILARITY,
        SemanticType.OLS_STATISTICS,
    }
    for spec in registry.specs():
        if spec.complexity or spec.cost == 0:
            continue
        kinds = set(spec.input_types) | {spec.output_type}
        complexity = {}
        if kinds & (minute_types | daily_types):
            complexity.update(I=1.0, D=1.0)
        if kinds & minute_types:
            complexity["M"] = 1.0
        if SemanticType.PAIR_SIMILARITY in kinds:
            complexity["I"] = 2.0
        if SemanticType.OLS_STATISTICS in kinds:
            complexity["P"] = 1.0
        if spec.name in {
            "fft_power", "sort_cumulative_difference_signal",
            "sort_cumulative_difference_volume", "masked_daily_median",
        }:
            complexity["logM"] = 1.0
        if spec.name in {"cross_section_rank", "peer_return_spread"}:
            complexity["logI"] = 1.0
        if spec.name == "peer_return_spread":
            complexity.update(I=2.0, logI=1.0)
        registry.replace(replace(
            spec, complexity=complexity,
            memory_complexity={
                key: value for key, value in complexity.items()
                if not key.startswith("log")
            },
        ))


def build_operator_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    register_spectral_operators(registry)
    register_event_operators(registry)
    register_cross_section_operators(registry)
    register_temporal_operators(registry)
    register_daily_operators(registry)
    register_distribution_operators(registry)
    register_path_operators(registry)
    register_regression_operators(registry)
    register_intraday_operators(registry)
    register_handbook_operators(registry)
    register_seed_tree_operators(registry)
    register_handbook_part_operators(registry)
    _complete_complexity_metadata(registry)
    configured = os.environ.get("MIN_GP_OPERATOR_CALIBRATION")
    calibration_paths = (
        [Path(configured).expanduser()] if configured else
        sorted((
            Path(__file__).resolve().parent.parent / "calibrations"
        ).glob("*.json"))
    )
    for calibration_path in calibration_paths:
        if not calibration_path.is_file():
            continue
        # Stale/removed operator records are ignored here so a source update
        # cannot make the whole package unimportable. The calibration CLI and
        # its tests use strict loading to surface such records explicitly.
        load_calibration_file(registry, calibration_path, strict=False)
    return registry


__all__ = ["build_operator_registry"]
