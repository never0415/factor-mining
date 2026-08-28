"""Batched intraday regression and conditional-covariance operators."""

import torch

from min_gp.dsl import (
    CostCalibration, OperatorRegistry, OperatorSpec, SemanticType,
)


def conditional_covariance(x, y, mask, min_count=3):
    valid = torch.isfinite(x) & torch.isfinite(y) & mask.bool()
    count = valid.sum(-1)
    xv, yv = torch.where(valid, x, 0.), torch.where(valid, y, 0.)
    mx = xv.sum(-1) / count.clamp(min=1)
    my = yv.sum(-1) / count.clamp(min=1)
    cov = (((xv - mx.unsqueeze(-1)) * (yv - my.unsqueeze(-1))) * valid).sum(-1)
    cov = cov / count.clamp(min=1)
    return torch.where(count >= min_count, cov, torch.full_like(cov, float("nan")))


def minute_ols_statistics(minute_ret, delta_volume, lags=5):
    """Daily OLS r ~ intercept + dV[t..t-lags]; returns t-stats and F."""
    shape = minute_ret.shape
    y = minute_ret.reshape(-1, shape[-1]).float()
    dv = delta_volume.reshape(-1, shape[-1]).float()
    rows, minutes = y.shape
    outputs_t = torch.full((rows, lags + 2), float("nan"), device=y.device)
    outputs_f = torch.full((rows,), float("nan"), device=y.device)
    for row in range(rows):
        start = lags
        target = y[row, start:]
        columns = [torch.ones_like(target)] + [dv[row, start-k:minutes-k] for k in range(lags + 1)]
        design = torch.stack(columns, 1)
        valid = torch.isfinite(target) & torch.isfinite(design).all(1)
        n, p = int(valid.sum()), design.shape[1]
        if n <= p + 1:
            continue
        X, yy = design[valid], target[valid]
        beta = torch.linalg.lstsq(X, yy[:, None]).solution[:, 0]
        residual = yy - X @ beta
        sse = (residual * residual).sum()
        sigma2 = sse / (n - p)
        inv = torch.linalg.pinv(X.T @ X)
        se = torch.sqrt(torch.diag(inv) * sigma2).clamp(min=1e-12)
        outputs_t[row] = beta / se
        centered = yy - yy.mean()
        ssr = (centered * centered).sum() - sse
        outputs_f[row] = (ssr / (p - 1)) / sigma2.clamp(min=1e-12)
    return outputs_t.reshape(*shape[:2], lags + 2), outputs_f.reshape(shape[:2])


def register_regression_operators(registry: OperatorRegistry):
    registry.register(OperatorSpec(
        "conditional_covariance",
        (SemanticType.MINUTE_SIGNAL, SemanticType.MINUTE_SIGNAL, SemanticType.MINUTE_MASK),
        SemanticType.DAILY_RAW_FACTOR, conditional_covariance, cost=4,
        parameter_domains={"min_count": (3, 5, 10)},
        complexity={"I": 1, "D": 1, "M": 1},
        memory_complexity={"I": 1, "D": 1, "M": 1},
        calibration=CostCalibration(
            reference_shape={"I": 150, "D": 120, "M": 240},
            seconds=0.003114900028,
            peak_bytes=91_009_024,
            device="NVIDIA GeForce GTX 1070",
            source="local median of 7 runs, 2026-08-20",
            parameter_values={"min_count": 3},
        ),
    ))
