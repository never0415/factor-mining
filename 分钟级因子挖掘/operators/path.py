"""Intraday path and sort/cumulative-difference operators."""

import torch
import torch.nn.functional as F

from min_gp.dsl import (
    CostCalibration, OperatorRegistry, OperatorSpec, SemanticType,
)


def _gtx1070(seconds, peak_bytes, **parameter_values):
    return CostCalibration(
        reference_shape={"I": 150, "D": 120, "M": 240},
        seconds=seconds,
        peak_bytes=peak_bytes,
        device="NVIDIA GeForce GTX 1070",
        source="local median of 7 runs, 2026-08-20",
        parameter_values=parameter_values,
    )


_TIDE_CALIBRATIONS = {
    "rolling_volume_sum": _gtx1070(
        0.001408000011, 56_160_768, neighborhood=9
    ),
    "locate_peak": _gtx1070(0.001316500013, 38_880_256, exclude_edges=15),
    "locate_left_valley": _gtx1070(
        0.001533200033, 38_882_304, exclude_edges=15
    ),
    "locate_right_valley": _gtx1070(
        0.001530700014, 38_882_304, exclude_edges=15
    ),
    "left_path_return_speed": _gtx1070(0.000606900023, 649_728),
    "right_path_return_speed": _gtx1070(0.000606500020, 649_728),
    "left_activity_value": _gtx1070(0.000204799988, 379_392),
    "right_activity_value": _gtx1070(0.000210300030, 379_392),
    "strong_path_by_activity": _gtx1070(0.000277599960, 253_440),
    "weak_path_by_activity": _gtx1070(0.000290399999, 271_872),
}


def tide_speed_components(close, volume, neighborhood=9, exclude_edges=15):
    """Return full, strong and weak tide speeds for each stock-day."""
    shape = close.shape
    rows, minutes = close.reshape(-1, shape[-1]).float(), shape[-1]
    vol = volume.reshape(-1, minutes).float()
    kernel = torch.ones(1, 1, neighborhood, device=vol.device)
    sums = F.conv1d(vol.nan_to_num().unsqueeze(1), kernel, padding=neighborhood // 2)[:, 0]
    valid = torch.isfinite(vol)
    sums = torch.where(valid, sums, torch.full_like(sums, float("nan")))
    eligible = valid.clone()
    eligible[:, :exclude_edges] = False
    eligible[:, minutes - exclude_edges:] = False
    score = torch.where(eligible, sums, torch.full_like(sums, float("-inf")))
    p = score.argmax(1)
    idx = torch.arange(minutes, device=vol.device).view(1, -1)
    left = eligible & (idx < p[:, None])
    right = eligible & (idx > p[:, None])
    left_score = torch.where(left, sums, torch.full_like(sums, float("inf")))
    right_score = torch.where(right, sums, torch.full_like(sums, float("inf")))
    m, n = left_score.argmin(1), right_score.argmin(1)
    row = torch.arange(rows.shape[0], device=vol.device)
    good = left.any(1) & right.any(1) & torch.isfinite(score[row, p])
    cm, cp, cn = rows[row, m], rows[row, p], rows[row, n]
    vm, vn = sums[row, m], sums[row, n]
    up = (cp / cm.clamp(min=1e-12) - 1) / (p - m).clamp(min=1)
    down = (cn / cp.clamp(min=1e-12) - 1) / (n - p).clamp(min=1)
    full = (cn / cm.clamp(min=1e-12) - 1) / (n - m).clamp(min=1)
    strong = torch.where(vm < vn, up, down)
    weak = torch.where(vm < vn, down, up)
    nan = torch.full_like(full, float("nan"))
    values = [torch.where(good, value, nan).reshape(shape[:2]) for value in (full, strong, weak)]
    return tuple(values)


def sort_cumulative_difference(x, ordering):
    """B(x,q): difference between ascending/descending cumulative paths."""
    valid = torch.isfinite(x) & torch.isfinite(ordering)
    asc = torch.argsort(torch.where(valid, ordering, torch.full_like(ordering, float("inf"))), dim=-1)
    desc = torch.argsort(torch.where(valid, ordering, torch.full_like(ordering, float("-inf"))), dim=-1, descending=True)
    xa = torch.gather(torch.where(valid, x, torch.zeros_like(x)), -1, asc)
    xd = torch.gather(torch.where(valid, x, torch.zeros_like(x)), -1, desc)
    value = (xa.cumsum(-1) - xd.cumsum(-1)).sum(-1)
    return torch.where(valid.sum(-1) >= 2, value, torch.full_like(value, float("nan")))


def rolling_volume_sum(volume, neighborhood=9):
    """Symmetric rolling volume used to locate the tide's activity extrema."""
    shape = volume.shape
    rows = volume.reshape(-1, shape[-1]).float()
    kernel = torch.ones(
        1, 1, neighborhood, device=rows.device, dtype=rows.dtype
    )
    result = F.conv1d(
        rows.nan_to_num().unsqueeze(1), kernel,
        padding=neighborhood // 2,
    )[:, 0]
    result = torch.where(
        torch.isfinite(rows), result, torch.full_like(result, float("nan"))
    )
    return result.reshape(shape)


def rolling_volume_std(volume, neighborhood=9):
    """Alternative local-activity measure with the same typed interface."""
    shape = volume.shape
    rows = volume.reshape(-1, shape[-1]).float()
    valid = torch.isfinite(rows).float()
    clean = rows.nan_to_num()
    kernel = torch.ones(
        1, 1, neighborhood, device=rows.device, dtype=rows.dtype
    )
    padding = neighborhood // 2
    count = F.conv1d(valid.unsqueeze(1), kernel, padding=padding)[:, 0]
    total = F.conv1d(clean.unsqueeze(1), kernel, padding=padding)[:, 0]
    total2 = F.conv1d(clean.square().unsqueeze(1), kernel, padding=padding)[:, 0]
    variance = (total2 / count.clamp(min=1) - (
        total / count.clamp(min=1)
    ).square()).clamp(min=0)
    result = torch.sqrt(variance)
    result = torch.where(
        torch.isfinite(rows) & (count >= 2), result,
        torch.full_like(result, float("nan")),
    )
    return result.reshape(shape)


def _eligible(activity, exclude_edges):
    valid = torch.isfinite(activity)
    if exclude_edges:
        valid = valid.clone()
        valid[..., :exclude_edges] = False
        valid[..., activity.shape[-1] - exclude_edges:] = False
    return valid


def _locate_global(activity, exclude_edges, largest):
    valid = _eligible(activity, exclude_edges)
    fill = float("-inf") if largest else float("inf")
    score = torch.where(valid, activity.float(), torch.full_like(activity.float(), fill))
    index = score.argmax(-1) if largest else score.argmin(-1)
    return torch.where(valid.any(-1), index, torch.full_like(index, -1))


def locate_peak(activity, exclude_edges=15):
    return _locate_global(activity, exclude_edges, True)


def locate_trough(activity, exclude_edges=15):
    return _locate_global(activity, exclude_edges, False)


def _locate_side(activity, pivot, exclude_edges, side, largest):
    valid = _eligible(activity, exclude_edges)
    minute = torch.arange(activity.shape[-1], device=activity.device)
    view = (1,) * (activity.ndim - 1) + (activity.shape[-1],)
    minute = minute.view(view)
    if side == "left":
        valid = valid & (minute < pivot.unsqueeze(-1))
    elif side == "right":
        valid = valid & (minute > pivot.unsqueeze(-1))
    else:
        raise ValueError(f"unknown side {side!r}")
    fill = float("-inf") if largest else float("inf")
    score = torch.where(valid, activity.float(), torch.full_like(activity.float(), fill))
    index = score.argmax(-1) if largest else score.argmin(-1)
    return torch.where(valid.any(-1), index, torch.full_like(index, -1))


def locate_left_valley(activity, pivot, exclude_edges=15):
    return _locate_side(activity, pivot, exclude_edges, "left", False)


def locate_left_peak(activity, pivot, exclude_edges=15):
    return _locate_side(activity, pivot, exclude_edges, "left", True)


def locate_right_valley(activity, pivot, exclude_edges=15):
    return _locate_side(activity, pivot, exclude_edges, "right", False)


def locate_right_peak(activity, pivot, exclude_edges=15):
    return _locate_side(activity, pivot, exclude_edges, "right", True)


def _indexed_value(series, index):
    safe = index.clamp(min=0, max=series.shape[-1] - 1)
    value = series.gather(-1, safe.unsqueeze(-1)).squeeze(-1).float()
    valid = (index >= 0) & torch.isfinite(value)
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def left_activity_value(activity, index):
    return _indexed_value(activity, index)


def right_activity_value(activity, index):
    return _indexed_value(activity, index)


def left_log_activity_value(activity, index):
    return torch.log1p(left_activity_value(activity, index).clamp(min=0))


def right_log_activity_value(activity, index):
    return torch.log1p(right_activity_value(activity, index).clamp(min=0))


def _return_speed(close, start, end, logarithmic=False):
    first = _indexed_value(close.float(), start)
    last = _indexed_value(close.float(), end)
    duration = end - start
    ratio = last / first.clamp(min=1e-12)
    value = torch.log(ratio.clamp(min=1e-12)) if logarithmic else ratio - 1
    value = value / duration.clamp(min=1)
    valid = (
        (duration > 0) & torch.isfinite(first) & torch.isfinite(last)
        & (first > 0) & (last > 0)
    )
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def left_path_return_speed(close, left, pivot):
    return _return_speed(close, left, pivot, False)


def left_path_log_speed(close, left, pivot):
    return _return_speed(close, left, pivot, True)


def right_path_return_speed(close, pivot, right):
    return _return_speed(close, pivot, right, False)


def right_path_log_speed(close, pivot, right):
    return _return_speed(close, pivot, right, True)


def full_path_return_speed(close, left, right):
    return _return_speed(close, left, right, False)


def _select_path(left_speed, right_speed, left_activity, right_activity, strong):
    valid = (
        torch.isfinite(left_speed) & torch.isfinite(right_speed)
        & torch.isfinite(left_activity) & torch.isfinite(right_activity)
    )
    left_is_strong = left_activity < right_activity
    choose_left = left_is_strong if strong else ~left_is_strong
    value = torch.where(choose_left, left_speed, right_speed)
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def strong_path_by_activity(left_speed, right_speed, left_activity, right_activity):
    return _select_path(
        left_speed, right_speed, left_activity, right_activity, True
    )


def weak_path_by_activity(left_speed, right_speed, left_activity, right_activity):
    return _select_path(
        left_speed, right_speed, left_activity, right_activity, False
    )


def average_path_speed(left_speed, right_speed, left_activity, right_activity):
    valid = (
        torch.isfinite(left_speed) & torch.isfinite(right_speed)
        & torch.isfinite(left_activity) & torch.isfinite(right_activity)
    )
    value = 0.5 * (left_speed + right_speed)
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def register_path_operators(registry: OperatorRegistry):
    t = SemanticType
    neighborhoods = (5, 9, 15)
    edges = (0, 15, 30)
    registry.register(OperatorSpec(
        "rolling_volume_sum", (t.MINUTE_VOLUME,), t.MINUTE_ACTIVITY,
        rolling_volume_sum, cost=2,
        parameter_domains={"neighborhood": neighborhoods},
        complexity={"I": 1, "D": 1, "M": 1},
        memory_complexity={"I": 1, "D": 1, "M": 1},
        calibration=_TIDE_CALIBRATIONS["rolling_volume_sum"],
    ))
    registry.register(OperatorSpec(
        "rolling_volume_std", (t.MINUTE_VOLUME,), t.MINUTE_ACTIVITY,
        rolling_volume_std, cost=3,
        parameter_domains={"neighborhood": neighborhoods},
        complexity={"I": 1, "D": 1, "M": 1},
    ))
    for name, implementation in (
        ("locate_peak", locate_peak), ("locate_trough", locate_trough),
    ):
        registry.register(OperatorSpec(
            name, (t.MINUTE_ACTIVITY,), t.MINUTE_INDEX, implementation,
            parameter_domains={"exclude_edges": edges},
            complexity={"I": 1, "D": 1, "M": 1},
            memory_complexity={"I": 1, "D": 1, "M": 1},
            calibration=_TIDE_CALIBRATIONS.get(name),
        ))
    for name, implementation, output in (
        ("locate_left_valley", locate_left_valley, t.MINUTE_LEFT_INDEX),
        ("locate_left_peak", locate_left_peak, t.MINUTE_LEFT_INDEX),
        ("locate_right_valley", locate_right_valley, t.MINUTE_RIGHT_INDEX),
        ("locate_right_peak", locate_right_peak, t.MINUTE_RIGHT_INDEX),
    ):
        registry.register(OperatorSpec(
            name, (t.MINUTE_ACTIVITY, t.MINUTE_INDEX), output,
            implementation, parameter_domains={"exclude_edges": edges},
            complexity={"I": 1, "D": 1, "M": 1},
            memory_complexity={"I": 1, "D": 1, "M": 1},
            calibration=_TIDE_CALIBRATIONS.get(name),
        ))
    for name, implementation, index_type in (
        ("left_activity_value", left_activity_value, t.MINUTE_LEFT_INDEX),
        ("left_log_activity_value", left_log_activity_value, t.MINUTE_LEFT_INDEX),
        ("right_activity_value", right_activity_value, t.MINUTE_RIGHT_INDEX),
        ("right_log_activity_value", right_log_activity_value, t.MINUTE_RIGHT_INDEX),
    ):
        registry.register(OperatorSpec(
            name, (t.MINUTE_ACTIVITY, index_type), t.DAILY_ACTIVITY,
            implementation, complexity={"I": 1, "D": 1},
            memory_complexity={"I": 1, "D": 1},
            calibration=_TIDE_CALIBRATIONS.get(name),
        ))
    for name, implementation, inputs in (
        ("left_path_return_speed", left_path_return_speed,
         (t.MINUTE_CLOSE, t.MINUTE_LEFT_INDEX, t.MINUTE_INDEX)),
        ("left_path_log_speed", left_path_log_speed,
         (t.MINUTE_CLOSE, t.MINUTE_LEFT_INDEX, t.MINUTE_INDEX)),
        ("right_path_return_speed", right_path_return_speed,
         (t.MINUTE_CLOSE, t.MINUTE_INDEX, t.MINUTE_RIGHT_INDEX)),
        ("right_path_log_speed", right_path_log_speed,
         (t.MINUTE_CLOSE, t.MINUTE_INDEX, t.MINUTE_RIGHT_INDEX)),
        ("full_path_return_speed", full_path_return_speed,
         (t.MINUTE_CLOSE, t.MINUTE_LEFT_INDEX, t.MINUTE_RIGHT_INDEX)),
    ):
        registry.register(OperatorSpec(
            name, inputs, t.DAILY_PATH_SPEED, implementation,
            complexity={"I": 1, "D": 1},
            memory_complexity={"I": 1, "D": 1},
            calibration=_TIDE_CALIBRATIONS.get(name),
        ))
    selector_inputs = (
        t.DAILY_PATH_SPEED, t.DAILY_PATH_SPEED,
        t.DAILY_ACTIVITY, t.DAILY_ACTIVITY,
    )
    for name, implementation in (
        ("strong_path_by_activity", strong_path_by_activity),
        ("weak_path_by_activity", weak_path_by_activity),
        ("average_path_speed", average_path_speed),
    ):
        registry.register(OperatorSpec(
            name, selector_inputs, t.DAILY_RAW_FACTOR, implementation,
            complexity={"I": 1, "D": 1},
            memory_complexity={"I": 1, "D": 1},
            calibration=_TIDE_CALIBRATIONS.get(name),
        ))
    registry.register(OperatorSpec(
        "sort_cumulative_difference_volume",
        (SemanticType.MINUTE_VOLUME, SemanticType.MINUTE_RETURN),
        SemanticType.DAILY_RAW_FACTOR, sort_cumulative_difference, cost=4,
    ))
