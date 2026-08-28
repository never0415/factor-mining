"""Closed daily-factor algebra used by the second-layer typed GP."""

import torch

from min_gp.dsl import OperatorRegistry, OperatorSpec, SemanticType


def daily_identity(x):
    return x.float()


def daily_neg(x):
    return -x.float()


def daily_abs(x):
    return torch.abs(x.float())


def daily_add(a, b):
    return a.float() + b.float()


def daily_sub(a, b):
    return a.float() - b.float()


def daily_mul(a, b):
    return a.float() * b.float()


def daily_div(a, b):
    a, b = a.float(), b.float()
    good = torch.isfinite(a) & torch.isfinite(b) & (b.abs() > 1e-8)
    value = a / torch.where(good, b, torch.ones_like(b))
    return torch.where(good, value, torch.full_like(value, float("nan")))


def register_daily_operators(registry: OperatorRegistry) -> None:
    raw = SemanticType.DAILY_RAW_FACTOR
    daily = SemanticType.DAILY_FACTOR
    registry.register(OperatorSpec("daily_identity", (raw,), daily, daily_identity, cost=0))
    registry.register(OperatorSpec("daily_neg", (raw,), raw, daily_neg))
    registry.register(OperatorSpec("daily_abs", (raw,), raw, daily_abs))
    registry.register(OperatorSpec("daily_add", (raw, raw), raw, daily_add))
    registry.register(OperatorSpec("daily_sub", (raw, raw), raw, daily_sub))
    registry.register(OperatorSpec("daily_mul", (raw, raw), raw, daily_mul, cost=2))
    registry.register(OperatorSpec("daily_div", (raw, raw), raw, daily_div, cost=2))

