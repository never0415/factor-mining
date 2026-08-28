"""Fitness: daily cross-sectional Spearman IC (double-argsort ranks)."""
import numpy as np
import torch
from min_gp.numeric.ranking import cross_section_rank as _cs_rank
from min_gp.numeric.preprocessing import remove_outliers


def daily_spearman_ic(factor, fwd_ret, min_n=30, outlier_mad=5.0):
    """(I,D) factor vs (I,D) fwd_ret → (D,) daily IC series. NaN pairs dropped per day."""
    # Mandatory PIT preprocessing: each date is winsorized independently.
    # Future returns are labels and must not be altered here.
    factor = remove_outliers(factor, n_mad=outlier_mad, dim=0)
    rf = _cs_rank(factor)
    ry = _cs_rank(fwd_ret)
    v = ~(torch.isnan(rf) | torch.isnan(ry))
    n = v.sum(0).clamp(min=1.0)
    rf_, ry_ = torch.nan_to_num(rf), torch.nan_to_num(ry)
    mx = (rf_ * v).sum(0) / n
    my = (ry_ * v).sum(0) / n
    cov = ((rf_ - mx) * (ry_ - my) * v).sum(0) / n
    sx = torch.sqrt(((rf_ - mx) ** 2 * v).sum(0) / n).clamp(min=1e-12)
    sy = torch.sqrt(((ry_ - my) ** 2 * v).sum(0) / n).clamp(min=1e-12)
    ic = cov / (sx * sy)
    return torch.where(n >= min_n, ic, torch.full_like(ic, float("nan")))


def industry_neutral(factor, ind_ids, valid=None):
    """Daily cross-sectional industry-neutralization (OLS residual on industry dummies)."""
    factor = factor.float()
    I, D = factor.shape
    dev = factor.device
    if valid is None:
        valid = ~torch.isnan(factor)
    v = valid.float()
    K = int(ind_ids.max().item()) + 1
    X = torch.zeros(I, K, device=dev)
    X[torch.arange(I, device=dev), ind_ids] = 1.0
    Xv = X.unsqueeze(2) * v.unsqueeze(1)
    XTX = torch.einsum("ikd,ild->kld", Xv, Xv)
    XTy = torch.einsum("ikd,id->kd", Xv, torch.nan_to_num(factor))
    ridge = torch.eye(K, device=dev) * 1e-6
    beta = torch.linalg.solve(XTX.permute(2, 0, 1) + ridge, XTy.permute(1, 0))
    fit = torch.einsum("dk,ik->id", beta, X)
    resid = factor - fit
    return torch.where(valid, resid, torch.full_like(resid, float("nan")))


def market_cap_neutral(factor, mcap, valid=None):
    """Daily cross-sectional market-cap neutralization (OLS residual on log(mcap))."""
    factor = factor.float()
    if valid is None:
        valid = ~torch.isnan(factor) & ~torch.isnan(mcap)
    I, D = factor.shape
    dev = factor.device
    X = torch.log(mcap.clamp(min=1.0))
    X = torch.where(valid, X, torch.zeros_like(X))
    ones = torch.ones(I, D, device=dev)
    X = torch.stack([ones, X.nan_to_num()], dim=2)
    Xv = X * valid.unsqueeze(2)
    XTX = torch.einsum("idk,idl->dkl", Xv, Xv)
    XTy = torch.einsum("idk,id->dk", Xv, torch.nan_to_num(factor))
    ridge = torch.eye(2, device=dev) * 1e-6
    beta = torch.linalg.solve(XTX + ridge, XTy.unsqueeze(2)).squeeze(2)
    fit = torch.einsum("dk,idk->id", beta, X)
    resid = factor - fit
    return torch.where(valid, resid, torch.full_like(resid, float("nan")))


def factor_health(f, missing_rate=None, min_unique=10, zero_frac=0.5, corr_thr=0.3):
    """因子体检: 识别退化因子与流动性伪信号 (valid 阶段过滤用).

    连续量当事件掩码 (如 day_istd(close)) → 大部分值恒 0 + 少量来自分钟缺失
    股票 → 与流动性相关 → test 段 IC 是伪信号. 三个检查:
      1. low_unique_frac: 每天截面 unique 值数 < min_unique 的天数占比 (退化)
      2. zero_frac:      恰好 0 的值占比 (连续量掩码 → 恒 0)
      3. miss_corr_med:  与分钟缺失率的截面 Spearman 相关中位数 (流动性伪信号)
    missing_rate: (I,D) 分钟缺失率, 可由 tens 张量算 (None 跳过检查 3).
    Returns (ok, report). ok=False → 疑似伪信号/退化, 应剔除.
    """
    import numpy as _np
    I, D = f.shape
    report = {}
    # 1. 退化: 每天截面 unique 值数 (GPU 循环, 采样每 5 天提速)
    low_uniq = 0
    n_days = 0
    for d in range(0, D, 5):
        col = f[:, d]
        col = col[~torch.isnan(col)]
        if col.numel() > 0:
            n_days += 1
            if col.unique().numel() < min_unique:
                low_uniq += 1
    report["low_unique_frac"] = float(low_uniq / max(n_days, 1))
    # 2. 0 值占比
    v = f[~torch.isnan(f)]
    report["zero_frac"] = float((v == 0).float().mean().item()) if v.numel() else 1.0
    # 3. 与分钟缺失率的截面相关 (Spearman, 采样每 5 天)
    report["miss_corr_med"] = 0.0
    if missing_rate is not None:
        corrs = []
        for d in range(0, D, 5):
            a, b = f[:, d], missing_rate[:, d]
            m = ~(torch.isnan(a) | torch.isnan(b))
            if m.sum() >= 30:
                ra, rb = _cs_rank(a[m]), _cs_rank(b[m])
                corrs.append(torch.corrcoef(torch.stack([ra, rb]))[0, 1].item())
        report["miss_corr_med"] = float(_np.median(corrs)) if corrs else 0.0
    ok = (report["low_unique_frac"] < 0.3 and report["zero_frac"] < zero_frac
          and abs(report["miss_corr_med"]) < corr_thr)
    return ok, report


def summarize_ic(ic):
    """(D,) → dict(ic_mean, icir, n_days)."""
    ic = ic[~torch.isnan(ic)]
    if ic.numel() == 0:
        return dict(ic_mean=float("nan"), icir=float("nan"), n_days=0)
    m = ic.mean().item()
    s = ic.std().item()
    return dict(ic_mean=m, icir=m / s if s > 0 else float("nan"), n_days=ic.numel())


# ──────────────────────────────────────────────
# Multi-objective fitness (NSGA-II objectives from Huatai 2026)
# ──────────────────────────────────────────────

def _ic_direction(ic):
    """Fix a factor's tradable direction from its in-sample mean IC."""
    mean_ic = ic.mean()
    direction = 1.0 if mean_ic.item() >= 0 else -1.0
    return mean_ic, direction


def multi_fitness(factor, fwd_ret, k=20, direction=None, return_details=False):
    """Compute direction-symmetric objectives for multi-objective GP.

    The raw factor direction is fixed from its in-sample mean IC. Stability
    and top-k quality are evaluated after orienting the factor, so a stable
    negative-IC factor is treated symmetrically with a positive one.
    """
    ic = daily_spearman_ic(factor, fwd_ret)
    ic = ic[~torch.isnan(ic)]
    if ic.numel() < 50:
        invalid = (-1e9, -1e9, -1e9, -1e9)
        return (invalid, (float("nan"), 0)) if return_details else invalid
    raw_mean, inferred_direction = _ic_direction(ic)
    direction = inferred_direction if direction is None else int(direction)
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or +1, got {direction}")
    aligned_ic = ic * direction
    obj1 = float(aligned_ic.mean().item())
    obj2 = float((aligned_ic > 0).float().mean().item())
    obj3 = ndcg_k(factor, fwd_ret, k, direction=direction) if k > 0 else 0.0
    obj4 = -factor_turnover(factor)
    result = (obj1, obj2, obj3, obj4)
    details = (float(raw_mean.item()), int(direction))
    return (result, details) if return_details else result


def factor_turnover(factor):
    """Estimate daily rank turnover: mean absolute rank change."""
    I, D = factor.shape
    if D < 2:
        return 1.0
    rank = _cs_rank(factor)
    valid = ~torch.isnan(rank)
    turnover_vals = []
    for d in range(1, D):
        v = valid[:, d-1] & valid[:, d]
        if v.sum() < 30:
            continue
        to = (rank[v, d-1] - rank[v, d]).abs().mean().item()
        turnover_vals.append(to)
    return float(np.mean(turnover_vals)) if turnover_vals else 1.0


def ndcg_k(factor, fwd_ret, k=20, direction=1.0):
    """NDCG@k for top-k stocks after applying a fixed factor direction."""
    I, D = factor.shape
    ndcg_vals = []
    for d in range(D):
        f_col = factor[:, d]
        r_col = fwd_ret[:, d]
        valid = ~(torch.isnan(f_col) | torch.isnan(r_col))
        n_valid = valid.sum().item()
        if n_valid < k:
            continue
        rel = _cs_rank(r_col)
        neg_inf_f = torch.full_like(f_col, float("-inf"))
        neg_inf_r = torch.full_like(rel, float("-inf"))
        _, idx = torch.topk(torch.where(valid, f_col * direction, neg_inf_f), k)
        _, ideal_idx = torch.topk(torch.where(valid, rel, neg_inf_r), k)
        i = torch.arange(k, device=factor.device, dtype=torch.float32)
        dcg = (rel[idx] / torch.log2(i + 2.0)).sum()
        idcg = (rel[ideal_idx] / torch.log2(i + 2.0)).sum()
        ndcg = (dcg / idcg).item() if idcg > 1e-8 else 0.0
        ndcg_vals.append(ndcg)
    return float(np.mean(ndcg_vals)) if ndcg_vals else 0.0


# ──────────────────────────────────────────────
# Batch evaluation: N factors at once, chunked for memory safety
# ──────────────────────────────────────────────

_CHUNK = 128  # chunk size for IC+turnover to avoid OOM at N≥2000


def batch_fitness(factors, fwd_ret, k=20, _chunk=_CHUNK, return_details=False):
    """Evaluate direction-symmetric objectives for N factors in chunks."""
    results = [(-1e9, -1e9, -1e9, -1e9)] * len(factors)
    details = [(float("nan"), 0)] * len(factors)
    valid_idx = []
    valid_tensors = []

    for i, f in enumerate(factors):
        if f is not None and not torch.isnan(f).all() and f.ndim == 2:
            valid_idx.append(i)
            valid_tensors.append(f)

    if not valid_tensors:
        return (results, details) if return_details else results

    N = len(valid_tensors)
    I, D = valid_tensors[0].shape
    dev = valid_tensors[0].device
    rank_y = _cs_rank(fwd_ret)  # (I, D) — shared, computed once

    # chunked: IC + turnover per chunk; NDCG batched per-day
    ic_np = [None] * N
    raw_ic_means = [float("nan")] * N
    directions = [0] * N
    ndcg_acc = torch.zeros(N, device=dev, dtype=torch.float64)
    ndcg_count = torch.zeros(N, device=dev, dtype=torch.float64)
    to_vals_cpu = [1.0] * N
    discount = (1.0 / torch.log2(torch.arange(k, device=dev, dtype=torch.float32) + 2.0)).view(1, k)

    for ci in range(0, N, _chunk):
        ce = min(ci + _chunk, N)
        chunk_t = valid_tensors[ci:ce]
        C = ce - ci
        st = torch.stack(chunk_t)                                           # (C, I, D)
        st = remove_outliers(st, n_mad=5.0, dim=1)
        # Rank each factor independently across stocks. Never flatten C and I:
        # factors in the same evaluation chunk must not affect one another.
        rank_f = _cs_rank(st, dim=1)                                        # (C, I, D)

        # ── IC ──
        v_i = ~torch.isnan(rank_f) & ~torch.isnan(rank_y).unsqueeze(0)     # (C, I, D)
        n_raw = v_i.sum(1)                                                   # (C, D)
        n_i = n_raw.clamp(min=1.0)
        rf_ = torch.nan_to_num(rank_f)
        ry_ = torch.nan_to_num(rank_y).unsqueeze(0)
        mx_i = (rf_ * v_i).sum(1) / n_i                                      # (C, D)
        my_i = (ry_ * v_i).sum(1) / n_i
        cov_i = ((rf_ - mx_i.unsqueeze(1)) * (ry_ - my_i.unsqueeze(1)) * v_i).sum(1) / n_i
        sx_i = torch.sqrt(((rf_ - mx_i.unsqueeze(1)) ** 2 * v_i).sum(1) / n_i).clamp(min=1e-12)
        sy_i = torch.sqrt(((ry_ - my_i.unsqueeze(1)) ** 2 * v_i).sum(1) / n_i).clamp(min=1e-12)
        ic_chunk = cov_i / (sx_i * sy_i)                                     # (C, D)
        ic_chunk = torch.where(n_raw >= 30, ic_chunk,
                               torch.full_like(ic_chunk, float("nan")))
        n_ok = (~torch.isnan(ic_chunk)).sum(1)
        dir_chunk = torch.ones(C, device=dev, dtype=st.dtype)
        for i_local in range(C):
            n_abs = ci + i_local
            if n_ok[i_local] >= 50:
                ic_valid = ic_chunk[i_local, ~torch.isnan(ic_chunk[i_local])]
                raw_mean, direction = _ic_direction(ic_valid)
                ic_np[n_abs] = ic_valid.cpu().numpy()
                raw_ic_means[n_abs] = float(raw_mean.item())
                directions[n_abs] = int(direction)
                dir_chunk[i_local] = direction

        # ── NDCG (per-day) ──
        for d in range(D):
            f_col = st[:, :, d]                                             # (C, I)
            r_col = fwd_ret[:, d]                                            # (I,)
            valid = ~torch.isnan(f_col) & ~torch.isnan(r_col).unsqueeze(0)   # (C, I)
            n_valid = valid.sum(1)                                           # (C,)
            ok = n_valid >= k
            if not ok.any():
                continue
            rel = _cs_rank(r_col)                                            # (I,)
            neg_inf_f = torch.full_like(f_col, float("-inf"))
            neg_inf_r = torch.full((C, I), float("-inf"), device=dev, dtype=rel.dtype)
            fv = torch.where(valid, f_col * dir_chunk.unsqueeze(1), neg_inf_f)
            _, idx = torch.topk(fv, k, dim=1)                               # (C, k)
            rel_c = rel.expand(C, -1)
            _, ideal_idx = torch.topk(torch.where(valid, rel_c, neg_inf_r), k, dim=1)
            dcg = (rel_c.gather(1, idx).clamp(min=0) * discount).sum(1)
            idcg = (rel_c.gather(1, ideal_idx).clamp(min=0) * discount).sum(1)
            ndcg_ok = ok & (idcg > 1e-8)
            ndcg_acc[ci:ce] += torch.where(ndcg_ok, dcg / idcg,
                                           torch.zeros(C, device=dev, dtype=torch.float64))
            ndcg_count[ci:ce] += ndcg_ok.to(torch.float64)

        # ── Turnover ──
        diff = (rank_f[:, :, 1:] - rank_f[:, :, :-1]).abs()                # (C, I, D-1)
        v_t = ~torch.isnan(diff)
        n_t = v_t.sum(1).clamp(min=1.0)
        chunk_to = torch.nan_to_num(diff).sum(1) / n_t                      # (C, D-1)
        for i_local in range(C):
            to_vals_cpu[ci + i_local] = float(chunk_to[i_local].nanmean().clamp(0, 1).cpu().item())

        del st, rank_f, rf_, ry_, v_i, mx_i, my_i, cov_i, sx_i, sy_i, ic_chunk, diff, chunk_to

    # ── assemble ──
    ndcg_vals = (ndcg_acc / ndcg_count.clamp(min=1)).cpu().tolist()
    for i, n in enumerate(valid_idx):
        if ic_np[i] is None:
            results[n] = (-1e9, -1e9, -1e9, -1e9)
        else:
            ic = ic_np[i]
            direction = directions[i]
            aligned_ic = ic * direction
            results[n] = (float(aligned_ic.mean()), float((aligned_ic > 0).mean()),
                          ndcg_vals[i], -to_vals_cpu[i])
            details[n] = (raw_ic_means[i], direction)

    return (results, details) if return_details else results
