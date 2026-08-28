"""Distribution transforms, entropy and liquidity elasticity."""

import torch

from min_gp.dsl import OperatorRegistry, OperatorSpec, SemanticType


BOXCOX_LAMBDAS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def boxcox_grid_mle(volume, lambdas=BOXCOX_LAMBDAS):
    """Per stock-day Box-Cox transform with lambda selected by grid MLE."""
    x = volume.float()
    positive = torch.isfinite(x) & (x > 0)
    safe = torch.where(positive, x, torch.ones_like(x))
    logx = torch.log(safe)
    candidates, likelihood = [], []
    n = positive.sum(-1).clamp(min=1)
    for value in lambdas:
        transformed = logx if value == 0 else (safe.pow(value) - 1) / value
        transformed = torch.where(positive, transformed, torch.full_like(x, float("nan")))
        clean = transformed.nan_to_num()
        mean = clean.sum(-1) / n
        var = (((clean - mean.unsqueeze(-1)) ** 2) * positive).sum(-1) / n
        ll = (value - 1) * logx.sum(-1) - .5 * n * torch.log(var.clamp(min=1e-12))
        candidates.append(transformed)
        likelihood.append(ll)
    choice = torch.stack(likelihood).argmax(0)
    stack = torch.stack(candidates)
    selected = torch.gather(stack, 0, choice.unsqueeze(0).unsqueeze(-1).expand(1, *choice.shape, x.shape[-1]))[0]
    return selected


def relative_volume_entropy(volume, bins=48):
    """Entropy of a stock's minute volume share relative to the market."""
    valid = torch.isfinite(volume)
    market = torch.where(valid, volume.float(), 0.).sum(0, keepdim=True)
    share = volume.float() / market.clamp(min=1e-12)
    minutes = share.shape[-1]
    if minutes % bins:
        raise ValueError("minute count must divide evenly into bins")
    block = share.reshape(*share.shape[:2], bins, minutes // bins)
    mass = block.nan_to_num().sum(-1)
    p = mass / mass.sum(-1, keepdim=True).clamp(min=1e-12)
    entropy = -(torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p))).sum(-1)
    return torch.where(valid.any(-1), entropy, torch.full_like(entropy, float("nan")))


def register_distribution_operators(registry: OperatorRegistry):
    registry.register(OperatorSpec(
        "boxcox_grid_mle", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_VOLUME, boxcox_grid_mle, cost=5,
    ))
    registry.register(OperatorSpec(
        "relative_volume_entropy", (SemanticType.MINUTE_VOLUME,),
        SemanticType.DAILY_RAW_FACTOR, relative_volume_entropy, cost=5,
        parameter_domains={"bins": (24, 48)},
        needs_full_cross_section=True,
        complexity={"I": 1, "D": 1, "M": 1},
    ))
