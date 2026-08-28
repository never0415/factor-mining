"""Semantic value and execution types for the strongly typed GP grammar."""

from enum import Enum, IntFlag


class ExecutionScope(IntFlag):
    """Derived execution constraints; independent requirements combine by OR."""

    LOCAL = 0
    FULL_CROSS_SECTION = 1
    HISTORY = 2


class SemanticType(str, Enum):
    # Economically distinct leaves stay distinct even when their tensor shapes
    # are identical.  This prevents a GP from feeding ``high`` into a slot
    # whose definition specifically requires ``open`` merely because both are
    # minute-price cubes.
    MINUTE_OPEN = "minute_open"
    MINUTE_HIGH = "minute_high"
    MINUTE_LOW = "minute_low"
    MINUTE_CLOSE = "minute_close"
    MINUTE_PRICE = "minute_price"
    MINUTE_RETURN = "minute_return"
    MINUTE_VOLUME = "minute_volume"
    MINUTE_AMOUNT = "minute_amount"
    MINUTE_AMOUNT_SHARE = "minute_amount_share"
    MINUTE_VOLUME_SHARE = "minute_volume_share"
    MINUTE_PRICE_STATE = "minute_price_state"
    MINUTE_HIGH_AMOUNT = "minute_high_amount"
    MINUTE_LOW_AMOUNT = "minute_low_amount"
    MINUTE_SIGNAL = "minute_signal"
    MINUTE_MASK = "minute_mask"
    MINUTE_ACTIVITY = "minute_activity"
    # Generic minute tensors used by the migrated expression-tree seeds.
    LEGACY_MINUTE = "legacy_minute"
    # (I, M, D): one daily history for each stock/minute-of-day pair.
    SAME_MINUTE_HISTORY = "same_minute_history"
    # One-dimensional (M,) session mask, distinct from an event cube.
    SESSION_MASK = "session_mask"
    # An ordered minute position is not interchangeable with an unordered
    # boolean event mask.  Tide/path operators use this to preserve m < p < n.
    MINUTE_INDEX = "minute_index"
    MINUTE_LEFT_INDEX = "minute_left_index"
    MINUTE_RIGHT_INDEX = "minute_right_index"
    SPECTRUM = "spectrum"
    DAILY_RAW_FACTOR = "daily_raw_factor"
    DAILY_PATH_SPEED = "daily_path_speed"
    DAILY_ACTIVITY = "daily_activity"
    DAILY_FACTOR = "daily_factor"
    LEGACY_DAILY = "legacy_daily"
    DAILY_PRICE = "daily_price"
    MARKET_DAILY_PRICE = "market_daily_price"
    DAILY_RETURN = "daily_return"
    DAILY_FLOAT_MARKET_CAP = "daily_float_market_cap"
    PAIR_SIMILARITY = "pair_similarity"
    OLS_STATISTICS = "ols_statistics"
    SCALAR = "scalar"
    WINDOW_PARAM = "window_param"
    BAND_PARAM = "band_param"
    THRESHOLD_PARAM = "threshold_param"


# ``LEGACY_*`` are migration-era *generic numeric* types, not separate data
# layouts.  Precise semantic tensors may flow into a generic seed operator,
# while the reverse is intentionally forbidden: an arbitrary legacy signal is
# not automatically a valid close, volume, mask, or ordered minute index.
_GENERIC_MINUTE_MEMBERS = frozenset({
    SemanticType.MINUTE_OPEN,
    SemanticType.MINUTE_HIGH,
    SemanticType.MINUTE_LOW,
    SemanticType.MINUTE_CLOSE,
    SemanticType.MINUTE_PRICE,
    SemanticType.MINUTE_RETURN,
    SemanticType.MINUTE_VOLUME,
    SemanticType.MINUTE_AMOUNT,
    SemanticType.MINUTE_AMOUNT_SHARE,
    SemanticType.MINUTE_VOLUME_SHARE,
    SemanticType.MINUTE_PRICE_STATE,
    SemanticType.MINUTE_HIGH_AMOUNT,
    SemanticType.MINUTE_LOW_AMOUNT,
    SemanticType.MINUTE_SIGNAL,
    SemanticType.MINUTE_ACTIVITY,
})

_GENERIC_DAILY_MEMBERS = frozenset({
    SemanticType.DAILY_RAW_FACTOR,
    SemanticType.DAILY_PATH_SPEED,
    SemanticType.DAILY_ACTIVITY,
    SemanticType.DAILY_FACTOR,
    SemanticType.DAILY_PRICE,
    SemanticType.MARKET_DAILY_PRICE,
    SemanticType.DAILY_RETURN,
    SemanticType.DAILY_FLOAT_MARKET_CAP,
})


def is_assignable(actual: SemanticType, expected: SemanticType) -> bool:
    """Whether an ``actual`` value can safely fill an ``expected`` slot.

    This is deliberately directional.  The generic migrated seed arithmetic
    accepts economically precise numeric tensors, but specialised operators
    retain their strict input contracts.
    """
    if actual == expected:
        return True
    if expected == SemanticType.LEGACY_MINUTE:
        return actual in _GENERIC_MINUTE_MEMBERS
    if expected == SemanticType.LEGACY_DAILY:
        return actual in _GENERIC_DAILY_MEMBERS
    return False


def types_compatible(left: SemanticType, right: SemanticType) -> bool:
    """Whether at least one safe, directional substitution exists."""
    return is_assignable(left, right) or is_assignable(right, left)
