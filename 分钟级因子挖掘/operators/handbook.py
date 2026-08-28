"""Registered core operators for every factor-reproduction handbook family.

The functions deliberately import templates lazily.  ``factors`` already use
the lower-level operator package, so importing them at module load time would
create a package cycle.  Each wrapper is the exact, tested template
implementation exposed through typed operator metadata and parameter domains.
"""

from min_gp.dsl import (
    CostCalibration, OperatorRegistry, OperatorSpec, SemanticType,
)
from min_gp.operators.temporal import SMOOTH_WINDOWS


CORE_PREFIX = "handbook_"
SMOOTH = SMOOTH_WINDOWS[1:]


def _smooth_history(params):
    return params.get("smooth_window", 1) - 1


def _measured(seconds, **extra_shape):
    return CostCalibration(
        reference_shape={"I": 150, "D": 120, "M": 240, **extra_shape},
        seconds=seconds,
        device="GPU (model not recorded)",
        source="2026-08-20 user benchmark",
    )


def complete_tide(close, volume, neighborhood, exclude_edges, smooth_window):
    from min_gp.factors.handbook import CompleteTideTemplate
    return CompleteTideTemplate(
        neighborhood, exclude_edges, smooth_window
    ).evaluate(close, volume)


def climb_mountain(open_, high, low, close, window, smooth_window):
    from min_gp.factors.handbook import ClimbMountainTemplate
    return ClimbMountainTemplate(window, smooth_window).evaluate(
        open_, high, low, close
    )


def hidden_flower(close, volume, lags, smooth_window, align_component_directions):
    from min_gp.factors.handbook import HiddenFlowerTemplate
    return HiddenFlowerTemplate(
        lags, smooth_window, align_component_directions
    ).evaluate(close, volume)


def long_short_battle(
    high, low, close, volume, return_window, smooth_window
):
    from min_gp.factors.handbook import LongShortBattleTemplate
    return LongShortBattleTemplate(return_window, smooth_window).evaluate(
        high, low, close, volume
    )


def equal_treatment(
    open_, close, volume, response_window, exclude_edges, smooth_window
):
    from min_gp.factors.handbook import EqualTreatmentTemplate
    return EqualTreatmentTemplate(
        response_window, exclude_edges, smooth_window
    ).evaluate(open_, close, volume)


def dark_flow(
    open_, high, low, volume, bins, lookback, multiple, smooth_window
):
    from min_gp.factors.handbook import DarkFlowTemplate
    return DarkFlowTemplate(
        bins, lookback, multiple, smooth_window
    ).evaluate(open_, high, low, volume)


def raw_panic(daily_close, market_close, smooth_window):
    from min_gp.factors.handbook import RawPanicTemplate
    return RawPanicTemplate(smooth_window, False).evaluate(
        daily_close, market_close
    )


def raw_panic_intraday(daily_close, market_close, minute_close, smooth_window):
    from min_gp.factors.handbook import RawPanicTemplate
    return RawPanicTemplate(smooth_window, True).evaluate(
        daily_close, market_close, minute_close
    )


def rushing_forward(amount_share, volume_share, trend_mask, smooth_window):
    from min_gp.factors.handbook import RushingForwardTemplate
    return RushingForwardTemplate(smooth_window).evaluate(
        amount_share, volume_share, trend_mask
    )


def water_boat(high_amount, low_amount, float_market_cap):
    from min_gp.factors.handbook import WaterBoatTemplate
    return WaterBoatTemplate().evaluate(
        high_amount, low_amount, float_market_cap
    )


def cooperation_effect(
    volume_share, price_state, daily_return, pair_similarity,
    peer_count, smooth_window,
):
    from min_gp.factors.handbook import CooperationEffectTemplate
    return CooperationEffectTemplate(peer_count, smooth_window).evaluate(
        volume_share, price_state, daily_return, pair_similarity
    )


def register_handbook_operators(registry: OperatorRegistry) -> None:
    t = SemanticType
    raw = t.DAILY_RAW_FACTOR
    registry.register(OperatorSpec(
        "handbook_complete_tide", (t.MINUTE_CLOSE, t.MINUTE_VOLUME), raw,
        complete_tide, cost=8, parameter_domains={
            "neighborhood": (5, 9, 15), "exclude_edges": (0, 15, 30),
            "smooth_window": SMOOTH,
        },
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1, "M": 1},
        calibration=_measured(0.004),
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_climb_mountain",
        (t.MINUTE_OPEN, t.MINUTE_HIGH, t.MINUTE_LOW, t.MINUTE_CLOSE), raw,
        climb_mountain, cost=10, parameter_domains={
            "window": (3, 5, 10), "smooth_window": SMOOTH,
        },
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1, "M": 1},
        calibration=_measured(0.048),
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_hidden_flower", (t.MINUTE_CLOSE, t.MINUTE_VOLUME), raw,
        hidden_flower, cost=14, parameter_domains={
            "lags": (3, 5, 10), "smooth_window": SMOOTH,
            "align_component_directions": (False, True),
        },
        needs_full_cross_section=True,
        needs_history=True, history_days=_smooth_history,
        # Batched/looped OLS dominates the measured runtime and scales with
        # stock-days; the smaller peer-correlation term is quadratic in I and
        # will receive its own profile when Hidden Flower is decomposed.
        complexity={"I": 1, "D": 1, "M": 1, "P": 2},
        calibration=_measured(29.296, P=7),
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_long_short_battle",
        (t.MINUTE_HIGH, t.MINUTE_LOW, t.MINUTE_CLOSE, t.MINUTE_VOLUME), raw,
        long_short_battle, cost=12, parameter_domains={
            "return_window": (3, 5, 10), "smooth_window": SMOOTH,
        },
        needs_full_cross_section=True,
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1, "M": 1, "logM": 1},
        calibration=_measured(0.058),
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_equal_treatment",
        (t.MINUTE_OPEN, t.MINUTE_CLOSE, t.MINUTE_VOLUME), raw,
        equal_treatment, cost=11, parameter_domains={
            "response_window": (3, 5, 10),
            "exclude_edges": (0, 15, 30), "smooth_window": SMOOTH,
        },
        needs_history=True, history_days=_smooth_history,
        intraday_lookahead_minutes=lambda p: p["response_window"] - 1,
        complexity={"I": 1, "D": 1, "M": 1},
        calibration=_measured(0.045),
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_dark_flow",
        (t.MINUTE_OPEN, t.MINUTE_HIGH, t.MINUTE_LOW, t.MINUTE_VOLUME), raw,
        dark_flow, cost=11, parameter_domains={
            "bins": (24, 48), "lookback": (3, 5, 10),
            "multiple": (0.5, 1.0, 1.5), "smooth_window": SMOOTH,
        },
        needs_full_cross_section=True,
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1, "M": 1},
        calibration=_measured(0.014),
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_raw_panic", (t.DAILY_PRICE, t.MARKET_DAILY_PRICE), raw,
        raw_panic, cost=5,
        parameter_domains={"smooth_window": SMOOTH},
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1},
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_raw_panic_intraday",
        (t.DAILY_PRICE, t.MARKET_DAILY_PRICE, t.MINUTE_CLOSE), raw,
        raw_panic_intraday, cost=7,
        parameter_domains={"smooth_window": SMOOTH},
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1, "M": 1},
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_rushing_forward",
        (t.MINUTE_AMOUNT_SHARE, t.MINUTE_VOLUME_SHARE, t.MINUTE_MASK), raw,
        rushing_forward, cost=5,
        parameter_domains={"smooth_window": SMOOTH},
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1, "M": 1},
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_water_boat",
        (t.MINUTE_HIGH_AMOUNT, t.MINUTE_LOW_AMOUNT, t.DAILY_FLOAT_MARKET_CAP),
        raw, water_boat, cost=3,
        complexity={"I": 1, "D": 1},
        factor_wrapper=True,
    ))
    registry.register(OperatorSpec(
        "handbook_cooperation_effect",
        (t.MINUTE_VOLUME_SHARE, t.MINUTE_PRICE_STATE, t.DAILY_RETURN,
         t.PAIR_SIMILARITY), raw, cooperation_effect, cost=18,
        parameter_domains={
            "peer_count": (10, 20, 30, 50), "smooth_window": SMOOTH,
        },
        needs_full_cross_section=True,
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 2, "D": 1, "M": 1},
        factor_wrapper=True,
    ))
