"""
min_gp expression system: AST + tag type system + torch operators.

Tags: A3=(I,D,M) minute world | B2=(I*M,D) same-minute | D2=(I,D) daily | M1=(M,) masks | SCALAR
Every node eval(ctx) -> (tensor, tag). Type check happens at eval time bottom-up;
illegal combos raise TypeTagError (GP catches → fitness=-inf).

Registry: OP[name] = dict(arity=(min,max), infer=fn(tags)->tag, apply=fn(args, ctx)->tensor)
"""
import torch
import torch.nn.functional as F
import numpy as np

from min_gp.numeric.ranking import cross_section_rank

A3, B2, D2, M1, SCALAR = "A3", "B2", "D2", "M1", "SCALAR"

MIN_ROLL = 1       # min valid days in a rolling window for ts_* (count>=1)
K_SLOTS = 256      # interval-stats slot cap: 每日事件数可达 M=241, 槽必须 ≥ M+1


class TypeTagError(Exception):
    pass


# ──────────────────────────────────────────────
# AST nodes
# ──────────────────────────────────────────────

class Node:
    __slots__ = ("tag",)

    def eval(self, ctx):
        raise NotImplementedError


class Leaf(Node):
    """Named leaf (3D tensor, (M,) mask, or scalar constant from ctx)."""

    def __init__(self, name):
        self.name = name
        self.tag = None

    def eval(self, ctx):
        if self.name not in ctx.leaves:
            raise TypeTagError(f"unknown leaf: {self.name}")
        v, tag = ctx.leaves[self.name]
        self.tag = tag
        return v, tag

    def __str__(self):
        return self.name


class Const(Node):
    def __init__(self, v, domain=None):
        self.v = float(v)
        self.tag = SCALAR
        self.domain = domain  # None='generic', 'window'=roll window, 'q'=quantile

    def eval(self, ctx):
        return torch.tensor(self.v, dtype=ctx.fp, device=ctx.device), SCALAR

    def __str__(self):
        if self.v == int(self.v) and abs(self.v) < 1e6:
            return str(int(self.v))
        return f"{self.v:.6g}"


class Op(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = [a if isinstance(a, Node) else Const(a) for a in args]
        self.tag = None

    def eval(self, ctx):
        reg = OP.get(self.name)
        if reg is None:
            raise TypeTagError(f"unknown op: {self.name}")
        vals = []
        for a in self.args:
            t, tag = a.eval(ctx)
            vals.append((t, tag))
        out_tag = reg["infer"]([t for _, t in vals])
        out = reg["apply"]([v for v, _ in vals], ctx)
        self.tag = out_tag
        return out, out_tag

    def __str__(self):
        inner = ", ".join(str(a) for a in self.args)
        return f"{self.name}({inner})"


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _as_float(t):
    if t.dtype.is_floating_point:
        return t
    return t.to(torch.bfloat16)   # match default fp precision


def _to_3d(t, tag, shape):
    if tag == A3:
        return t
    if tag == B2:
        return t.reshape(shape[0], shape[2], shape[1]).permute(0, 2, 1)
    raise TypeTagError("expected A3/B2 for to_A")


def _nan_fill(shape, device, dtype):
    return torch.full(shape, float("nan"), dtype=dtype, device=device)


def _roll_conv(x2d, w, mode="sum", min_count=MIN_ROLL):
    """NaN-aware rolling over dim=1 (past w days, right-aligned). x2d: (R, D)."""
    x2d = _as_float(x2d)
    is_b2 = x2d.dim() == 3
    b2_shape = x2d.shape if is_b2 else None
    if is_b2:
        I, M, D = x2d.shape
        x2d = x2d.reshape(I * M, D)
    R, D = x2d.shape
    if w <= 1:
        return x2d.clone()
    xc = torch.nan_to_num(x2d)
    valid = (~torch.isnan(x2d)).to(x2d.dtype)
    k = torch.ones(1, 1, w, device=x2d.device, dtype=x2d.dtype)
    if mode in ("sum", "mean"):
        xp = F.pad(xc.unsqueeze(1), (w - 1, 0))
        vp = F.pad(valid.unsqueeze(1), (w - 1, 0))
        cnt = F.conv1d(vp, k)[:, 0, :]
        out = F.conv1d(xp, k)[:, 0, :]
        if mode == "mean":
            out = out / cnt.clamp(min=1.0)
    elif mode == "max":
        xm = torch.nan_to_num(x2d, nan=-float("inf"))
        xp = F.pad(xm.unsqueeze(1), (w - 1, 0), value=-float("inf"))
        vp = F.pad(valid.unsqueeze(1), (w - 1, 0))
        cnt = F.conv1d(vp, k)[:, 0, :]
        out = F.max_pool1d(xp, w, 1)[:, 0, :]
        out = torch.where(torch.isinf(out) & (out < 0), torch.tensor(float("nan"), device=x2d.device, dtype=x2d.dtype), out)
    else:
        raise ValueError(mode)
    out = torch.where(cnt >= min_count, out, torch.tensor(float("nan"), device=x2d.device, dtype=x2d.dtype))
    if is_b2:
        out = out.reshape(I, M, D)
    return out


def _roll_pair(x2d, y2d, w):
    """Rolling cov/corr/beta of two series, NaN-aware, one conv pass.
    D2: (I,D) → (I,D).  B2: (I,M,D) → (I,M,D) — flattens to (I*M,D) internally."""
    x2d, y2d = _as_float(x2d), _as_float(y2d)
    if x2d.dtype != y2d.dtype:
        # 混合 dtype (bf16 算子输出 + float32 算子输出, 如 time_barycenter vs day_ikurt):
        # conv1d 输入与 kernel 须同型 → 统一到提升类型 (bf16+float32 → float32)
        common = torch.promote_types(x2d.dtype, y2d.dtype)
        x2d, y2d = x2d.to(common), y2d.to(common)
    is_b2 = x2d.dim() == 3
    b2_shape = x2d.shape if is_b2 else None
    if is_b2:
        I, M, D = x2d.shape
        x2d = x2d.reshape(I * M, D)
        y2d = y2d.reshape(I * M, D)
    R, D = x2d.shape
    valid = (torch.isfinite(x2d) & torch.isfinite(y2d)).to(x2d.dtype)
    z = torch.stack([x2d, y2d, x2d * y2d, x2d ** 2, y2d ** 2], 1)          # (R,5,D)
    zc = torch.nan_to_num(z)
    k = torch.ones(5, 1, w, device=x2d.device, dtype=x2d.dtype)
    zp = F.pad(zc, (w - 1, 0))
    vp = F.pad(valid.unsqueeze(1), (w - 1, 0))
    s = F.conv1d(zp, k, groups=5)                       # (R,5,D) depthwise: 5 independent rolling sums
    cnt_all = F.conv1d(vp, k)                           # (R,5,D) (identical 5 channels)
    cnt = cnt_all[:, 0].clamp(min=1.0)
    ok = cnt_all[:, 0] >= MIN_ROLL
    sx, sy, sxy, sx2, sy2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]   # each (R,D)
    mx, my = sx / cnt, sy / cnt
    cov = sxy / cnt - mx * my
    vx = (sx2 / cnt - mx ** 2).clamp(min=0)
    vy = (sy2 / cnt - my ** 2).clamp(min=0)
    nan = torch.tensor(float("nan"), device=x2d.device, dtype=x2d.dtype)
    cov = torch.where(ok, cov, nan); vx = torch.where(ok, vx, nan); vy = torch.where(ok, vy, nan)
    if is_b2:
        cov = cov.reshape(I, M, D); vx = vx.reshape(I, M, D); vy = vy.reshape(I, M, D)
    return (cov, vx, vy)


def _nan_reduce(x, fn):
    """NaN-aware day reduction over dim=2. _day_reduce handles B2→A3 conversion upstream."""
    vb = ~torch.isnan(x)
    valid = vb.to(x.dtype)
    xc = torch.nan_to_num(x)
    if fn == "sum":
        return (xc * valid).sum(2)
    if fn == "mean":
        return (xc * valid).sum(2) / valid.sum(2).clamp(min=1.0)
    if fn == "std":
        n = valid.sum(2).clamp(min=1.0)
        m = (xc * valid).sum(2) / n
        var = ((xc - m.unsqueeze(2)) ** 2 * valid).sum(2) / n
        return torch.sqrt(var.clamp(min=0))
    if fn == "min":
        inf = torch.full_like(xc, float("inf"))
        r = torch.where(vb, xc, inf).min(2).values
        return torch.where(vb.any(2), r, torch.full_like(r, float("nan")))
    if fn == "max":
        ninf = torch.full_like(xc, float("-inf"))
        r = torch.where(vb, xc, ninf).max(2).values
        return torch.where(vb.any(2), r, torch.full_like(r, float("nan")))
    raise ValueError(fn)


def _day_reduce(x3d, fn, ctx=None):
    x = _as_float(x3d)
    # to_B flattens A3→(I*M,D) as 2D.  Restore to (I, M, D) for correct minute-wise reduction.
    if x.dim() == 2 and ctx is not None and x.shape[0] % ctx.NM == 0:
        I = x.shape[0] // ctx.NM
        x = x.reshape(I, ctx.NM, -1).permute(0, 2, 1)  # → (I, D, M) = A3
    return _nan_reduce(x, fn)


def _day_moment(x3d, order):
    """Central moment of given order over minutes (nan-aware), (I,D,M)->(I,D)."""
    x = _as_float(x3d)
    valid = (~torch.isnan(x)).to(x.dtype)
    cnt = valid.sum(2).clamp(min=1.0)
    m = (torch.nan_to_num(x) * valid).sum(2) / cnt
    d = torch.nan_to_num(x) - m.unsqueeze(2)
    mom = ((d * valid.unsqueeze(2)) ** order).sum(2) / cnt
    return mom


def _day_skew(x3d):
    x3d = _as_float(x3d)
    valid = (~torch.isnan(x3d)).to(x3d.dtype)
    cnt = valid.sum(2).clamp(min=2.0)
    m = (torch.nan_to_num(x3d) * valid).sum(2) / cnt
    d = torch.nan_to_num(x3d) - m.unsqueeze(2)
    m2 = (d ** 2 * valid).sum(2) / cnt
    m3 = (d ** 3 * valid).sum(2) / cnt
    s = torch.sqrt(m2.clamp(min=0))
    return torch.where(s > 1e-12, m3 / (s ** 3), torch.zeros_like(m3))


def _day_kurt(x3d):
    x3d = _as_float(x3d)
    valid = (~torch.isnan(x3d)).to(x3d.dtype)
    cnt = valid.sum(2).clamp(min=2.0)
    m = (torch.nan_to_num(x3d) * valid).sum(2) / cnt
    d = torch.nan_to_num(x3d) - m.unsqueeze(2)
    m2 = (d ** 2 * valid).sum(2) / cnt
    m4 = (d ** 4 * valid).sum(2) / cnt
    v = m2.clamp(min=0)
    return torch.where(v > 1e-12, m4 / (v ** 2) - 3.0, torch.zeros_like(m4))


def _interval_stats(mask3d, stat, out_dtype=None):
    """Adjacent-event interval stats over minutes, (I,D,M) bool -> (I,D).
    高效实现: 事件段起点 → 位置排序 → 相邻事件间隔 → 分布统计.
    避免 3D scatter (原逐日循环 4y 数据 41s → 现 <1s).
    每个 (i,d) 独立: 间隔 = 同日相邻事件起点位置差 (研报: 同日前后两个同类状态)."""
    if out_dtype is None:
        out_dtype = mask3d.dtype if mask3d.dtype.is_floating_point else torch.float32
    m = mask3d.bool()
    I, D, M = m.shape
    dev = m.device
    dt = out_dtype
    start = m & ~_shift_m(m, -1)                        # event starts (每段第一分钟)
    m2 = start.reshape(I * D, M)                          # (I*D, M)
    k = m2.sum(1)                                         # (I*D,) 事件数
    maxk = int(k.max().item()) if k.numel() else 0
    if maxk < 2:
        return torch.zeros(I, D, device=dev, dtype=dt)
    pos = torch.arange(M, device=dev).unsqueeze(0)        # (1, M)
    # 分块处理行, 限制峰值: 每块 ROWS 行, 中间量 ROWS×M×几个
    ROWS = 20000
    out = torch.full((I * D,), float("nan"), device=dev, dtype=dt)
    for r0 in range(0, I * D, ROWS):
        r1 = min(r0 + ROWS, I * D)
        m_b = m2[r0:r1]                                   # (R, M) bool
        ev_pos = torch.where(m_b, pos.expand(r1 - r0, M), torch.zeros(r1 - r0, M, device=dev, dtype=pos.dtype))
        sorted_pos, _ = torch.sort(ev_pos, dim=1)         # (R, M)
        tail = sorted_pos[:, M - maxk:]                   # (R, maxk)
        kb = k[r0:r1]
        col_ok = torch.arange(maxk, device=dev).unsqueeze(0) >= (maxk - kb).unsqueeze(1)
        valid = col_ok & (tail > 0)
        intv = torch.diff(tail, dim=1)                    # (R, maxk-1)
        v = valid[:, :-1] & valid[:, 1:]
        n = v.sum(1).clamp(min=1.0)
        if stat == "std":
            mean = (intv * v).sum(1) / n
            var = ((intv - mean.unsqueeze(1)) ** 2 * v).sum(1) / n
            ob = torch.sqrt(var.clamp(min=0))
        elif stat == "skew":
            mean = (intv * v).sum(1) / n
            m2v = ((intv - mean.unsqueeze(1)) ** 2 * v).sum(1) / n
            m3 = ((intv - mean.unsqueeze(1)) ** 3 * v).sum(1) / n
            s = torch.sqrt(m2v.clamp(min=0))
            ob = torch.where(s > 1e-12, m3 / (s ** 3), torch.zeros_like(m3))
        elif stat == "kurt":
            mean = (intv * v).sum(1) / n
            m2v = ((intv - mean.unsqueeze(1)) ** 2 * v).sum(1) / n
            m4 = ((intv - mean.unsqueeze(1)) ** 4 * v).sum(1) / n
            vv = m2v.clamp(min=0)
            ob = torch.where(vv > 1e-12, m4 / (vv ** 2) - 3.0, torch.zeros_like(m4))
        else:  # mean
            ob = (intv * v).sum(1) / n
        out[r0:r1] = torch.where(kb >= 2, ob, torch.full_like(ob, float("nan")))
    return out.reshape(I, D)


def _shift_m(x, d):
    """Shift along minute axis (dim=2). d=-1: prev minute. bool-safe, zero fill."""
    y = torch.zeros_like(x)
    if d < 0:
        y[:, :, :d] = x[:, :, -d:]
    else:
        y[:, :, d:] = x[:, :, :-d]
    return y


# Moved to `min_gp.numeric.ranking` so the evaluation layer no longer imports
# the legacy expression runtime. Re-exported here to keep the old backtest and
# index-enhancement CLIs working unchanged.
_cs_rank = cross_section_rank


# ──────────────────────────────────────────────
# tag inference
# ──────────────────────────────────────────────

def _t_binary(tags):
    a, b = tags
    if a == SCALAR:
        return b
    if b == SCALAR:
        return a
    if a != b:
        raise TypeTagError(f"tag mismatch: {a} op {b}")
    return a


def _t_bool_binary(tags):
    a, b = tags
    if a == SCALAR:
        return b
    if b == SCALAR:
        return a
    if a != b:
        raise TypeTagError(f"tag mismatch: {a} cmp {b}")
    return a


def _t_unary(tags):
    return tags[0]


def _t_intra(tags):
    if tags[0] != A3:
        raise TypeTagError(f"intra_* needs A3, got {tags[0]}")
    return A3


def _t_day(tags):
    t = tags[0]
    if t not in (A3, B2):
        raise TypeTagError(f"day_* needs A3|B2, got {t}")
    for tg in tags[1:]:
        if tg == SCALAR:
            continue   # day_quantile(x, q) 等带 Const 参数
        if tg != t:
            raise TypeTagError(f"day_* tag mismatch: {t} vs {tg} (A3/D2 混合 → 广播错位)")
    return D2


def _t_ts(tags):
    t = tags[0]
    if t not in (B2, D2):
        raise TypeTagError(f"ts_* needs B2|D2, got {t} (hint: day_agg first or to_B)")
    return t


def _t_ts_pair(tags):
    for t in tags[:2]:
        if t not in (B2, D2):
            raise TypeTagError(f"ts_corr needs B2|D2, got {t}")
    if tags[0] != tags[1]:
        raise TypeTagError(f"ts_corr tag mismatch: {tags[0]} vs {tags[1]}")
    return tags[0]


def _t_cs(tags):
    t = tags[0]
    if t == B2:
        raise TypeTagError("cs_* forbidden on B2 (rows are (i,m) mixtures)")
    if t not in (A3, D2):
        raise TypeTagError(f"cs_* needs A3|D2, got {t}")
    for tg in tags[1:]:
        if tg != t:
            raise TypeTagError(f"cs_* tag mismatch: {t} vs {tg} (A3/D2 混合 → 广播错位)")
    return t


def _t_to_b(tags):
    if tags[0] != A3:
        raise TypeTagError(f"to_B needs A3, got {tags[0]}")
    return B2


def _t_to_a(tags):
    if tags[0] != B2:
        raise TypeTagError(f"to_A needs B2, got {tags[0]}")
    return A3


def _t_quantile(tags):
    if tags[0] != D2:
        raise TypeTagError(f"ts_quantile needs D2, got {tags[0]}")
    return D2


def _t_istats(tags):
    if tags[0] != A3:
        raise TypeTagError(f"day_interval_* needs A3, got {tags[0]}")
    return D2


def _t_mask(tags):
    a, b = tags
    if a == M1:
        if b != A3:
            raise TypeTagError(f"mask can only combine with A3, got {b}")
        return A3
    if b == M1:
        if a != A3:
            raise TypeTagError(f"mask can only combine with A3, got {a}")
        return A3
    return _t_binary(tags)


# ──────────────────────────────────────────────
# operator applications
# ──────────────────────────────────────────────

def _apply_binary(fn):
    def apply(args, ctx):
        a, b = args
        if a.ndim < b.ndim:
            a, b = b, a
        return fn(_as_float(a), _as_float(b))
    return apply


def _apply_unary(fn):
    return lambda args, ctx: fn(_as_float(args[0]))


def _apply_cmp(fn):
    return lambda args, ctx: fn(_as_float(args[0]), _as_float(args[1]))


def _a_intra_shift(args, ctx):
    x = args[0]
    d = int(args[1].item())
    return _shift_m(x, -d)      # d>0: take future (x[t+d] at t); d<0: take past


def _a_intra_mean(args, ctx):
    x = _as_float(args[0])   # bool A3 (is_* 掩码叶子) → bf16 0/1: 防 conv1d cudnn Bool 崩溃
    w = int(args[1].item())
    I, D, M = x.shape
    xc = torch.nan_to_num(x).reshape(I * D, 1, M)
    valid = (~torch.isnan(x)).to(x.dtype).reshape(I * D, 1, M)
    k = torch.ones(1, 1, w, device=x.device, dtype=x.dtype)
    # 左填充 (w-1, 0): 窗口 [t-w+1, t] 只用过去分钟 (右填充会含未来 → 未来函数)
    xp = F.pad(xc, (w - 1, 0))
    vp = F.pad(valid, (w - 1, 0))
    cnt = F.conv1d(xp, k, padding=0)[:, :, :M]
    vcnt = F.conv1d(vp, k)[:, :, :M].clamp(min=1.0)
    return (cnt / vcnt).reshape(I, D, M)


def _a_intra_std(args, ctx):
    x = _as_float(args[0])   # bool A3 (is_* 掩码叶子) → bf16 0/1: 防 conv1d cudnn Bool 崩溃
    w = int(args[1].item())
    I, D, M = x.shape
    m = _a_intra_mean(args, ctx)
    xc = torch.nan_to_num(x).reshape(I * D, 1, M)
    valid = (~torch.isnan(x)).to(x.dtype).reshape(I * D, 1, M)
    k = torch.ones(1, 1, w, device=x.device, dtype=x.dtype)
    # 左填充 (w-1, 0): 只用过去窗口
    x2p = F.pad(xc ** 2, (w - 1, 0))
    vp = F.pad(valid, (w - 1, 0))
    s2 = F.conv1d(x2p, k)[:, :, :M]
    vcnt = F.conv1d(vp, k)[:, :, :M].clamp(min=1.0)
    return torch.sqrt((s2 / vcnt - m.reshape(I * D, 1, M) ** 2).clamp(min=0)).reshape(I, D, M)


def _a_mask_agg(args, ctx):
    """Masked aggregation over minutes → DAILY (I,D) sequence (no rolling).
    Composition: wrap with ts_mean/ts_sum for N-day smoothing — keeps 先除后均
    semantics possible (研报: 每日比值→20日均值).
    op: 0=count 1=sum 2=weighted-mean(sum/sum) 3=std 4=skew 5=kurt 6=min 7=max 8=median.
    No-mask days → NaN (rank-safe). x may be a scalar. mask=1 → all minutes (全真).
    args: [x(A3|SCALAR), mask(A3|SCALAR), op(SCALAR)] → (I,D)"""
    x = args[0]
    mask_arg = args[1]
    op = int(args[2].item())
    I, D, M = _as_float(mask_arg if not hasattr(mask_arg, "ndim") or mask_arg.ndim == 0 else mask_arg).shape if (hasattr(mask_arg, "ndim") and mask_arg.ndim == 3) else (0, 0, 0)
    if hasattr(mask_arg, "ndim") and mask_arg.ndim == 3:
        mask = _as_float(mask_arg)
        I, D, M = mask.shape
        m = mask > 0.5
    else:
        # scalar mask → all minutes true (全日掩码)
        I, D, M = _as_float(x).shape
        m = torch.ones(I, D, M, dtype=torch.bool, device=_as_float(x).device)
    dev = _as_float(x).device
    dt = _as_float(x).dtype if hasattr(x, "shape") and x.ndim == 3 else torch.float32
    mf = m.to(dt)
    if hasattr(x, "shape") and x.ndim == 3:
        xv = torch.where(m, _as_float(x), torch.full_like(_as_float(x), float("nan")))
    else:
        xv = torch.where(m, torch.ones(I, D, M, dtype=dt, device=dev) * float(x.item()),
                         torch.full((I, D, M), float("nan"), device=dev, dtype=dt))
    if op == 0:      # count
        daily = mf.sum(2)
    elif op == 1:    # sum
        daily = torch.nan_to_num(xv).sum(2)
    elif op == 2:    # weighted mean (sum/sum — VWAP 语义, 与手写 day_sum(x·m)/day_sum(m) 一致)
        daily = torch.nan_to_num(xv).sum(2) / mf.sum(2).clamp(min=1.0)
    elif op == 3:    # std
        n = mf.sum(2).clamp(min=1.0)
        mm = torch.nan_to_num(xv).sum(2) / n
        var = ((xv - mm.unsqueeze(2)).square().nan_to_num() * mf).sum(2) / n
        daily = torch.sqrt(var.clamp(min=0))
    elif op in (4, 5):  # skew / kurt (population moments on masked minutes)
        n = mf.sum(2).clamp(min=1.0)
        mm = torch.nan_to_num(xv).sum(2) / n
        d = (xv - mm.unsqueeze(2)).nan_to_num() * mf
        sd = torch.sqrt((d.square().sum(2) / n).clamp(min=1e-12))
        if op == 4:
            daily = (d ** 3).sum(2) / n / sd ** 3
        else:
            daily = (d ** 4).sum(2) / n / sd ** 4
    elif op == 6:    # min
        daily = torch.where(m, xv, torch.full_like(xv, float("inf"))).min(2).values
    elif op == 7:    # max
        daily = torch.where(m, xv, torch.full_like(xv, float("-inf"))).max(2).values
    else:            # 8 median — GPU 批量: 掩码位置排前, 取第 n//2 个 (原逐日逐股循环 53 万次迭代挂起)
        xm = torch.where(m, xv, torch.full_like(xv, float("inf")))   # 掩码位置保留, 非掩码 inf
        sm, _ = torch.sort(xm, dim=2)                                 # (I,D,M) 升序, 掩码值在前
        n = m.float().sum(2).clamp(min=1.0).long()                    # (I,D) 掩码计数
        idx = (n - 1) // 2                                            # 中位索引 (0-based)
        med = sm.gather(2, idx.clamp(max=240).unsqueeze(2)).squeeze(2)  # (I,D)
        daily = torch.where(n > 0, med, torch.full_like(med, float("nan")))
    return torch.where(m.any(2), daily, torch.full_like(daily, float("nan")))


def _a_mask_ratio(args, ctx):
    """Ratio of masked aggregates over N days: Σ_N Σ_m x·A / (Σ_N Σ_m x·B + 1).
    A/B are masks. +1 smoothing on denominator prevents 0-div explosion.
    args: [x(A3), maskA(A3), maskB(A3), N(SCALAR)] → (I,D)"""
    x = _as_float(args[0])
    ma = _as_float(args[1]) > 0.5
    mb = _as_float(args[2]) > 0.5
    N = int(args[3].item())
    I, D, M = x.shape
    dev = x.device
    xc = torch.nan_to_num(x)
    sa = (xc * ma.to(x.dtype)).sum(2)          # daily A-weighted sum
    sb = (xc * mb.to(x.dtype)).sum(2)          # daily B-weighted sum
    # rolling N-day sums — conv1d 向量化 (原 for d 973 次切片求和)
    # 左侧 pad (N-1, 0): conv1d 输出 j = Σ x[j-N+1..j] (过去窗口, 无前视)
    import torch.nn.functional as F
    k = torch.ones(1, 1, N, device=dev, dtype=sa.dtype)
    sa_p = F.pad(sa.unsqueeze(1), (N - 1, 0))
    sb_p = F.pad(sb.unsqueeze(1), (N - 1, 0))
    ps_all = F.conv1d(sa_p, k)[:, 0, :]    # (I, D) 滚动 N 日 A 和
    rs_all = F.conv1d(sb_p, k)[:, 0, :]    # (I, D) 滚动 N 日 B 和
    out = torch.full((I, D), float("nan"), device=dev, dtype=x.dtype)
    out[:, N - 1:] = ps_all[:, N - 1:] / (rs_all[:, N - 1:] + 1.0)
    return out


def _a_dist_to_event(args, ctx):
    """距离字段: 每分钟到同日内**下一个**事件时点的分钟距离 (A3).
    事件时点上 = 到下一事件的间隔; 最后一个事件及之后无事件 → NaN.
    args: [mask(A3)] → (I,D,M)"""
    x = _as_float(args[0])
    mask = x > 0.5
    dt = x.dtype
    I, D, M = mask.shape
    dev = mask.device
    # 向量化反向扫描: 事件位置 pos (非事件 inf), 反向 cummin 得"从 t 起的最近事件位置"
    pos = torch.where(mask, torch.arange(M, device=dev, dtype=dt).view(1, 1, M),
                      torch.full((I, D, M), float("inf"), device=dev, dtype=dt))
    rev = torch.cummin(pos.flip(2), dim=2).values.flip(2)   # (I,D,M)
    rev_shift = torch.full((I, D, M), float("nan"), device=dev, dtype=dt)
    rev_shift[:, :, :-1] = torch.where(torch.isfinite(rev[:, :, 1:]), rev[:, :, 1:],
                                       torch.full_like(rev[:, :, 1:], float("nan")))
    t_idx = torch.arange(M, device=dev, dtype=dt).view(1, 1, M)
    out = rev_shift - t_idx                                  # 到下一个事件的分钟距离
    return out


def _a_mask_stat(args, ctx):
    """Joint stat of (x, y) on masked minutes over an N-day window (merged samples).
    研报口径: 过去 N 天所有掩码时点合并为一个样本集, 算 corr/slope/cov.
    op: 0=corr 1=slope 2=cov.
    args: [x(A3), y(A3), mask(A3), N(SCALAR), op(SCALAR)] → (I,D)"""
    x = _as_float(args[0])
    y = _as_float(args[1])
    mask = _as_float(args[2]) > 0.5
    N = int(args[3].item())
    op = int(args[4].item())
    I, D, M = x.shape
    dev = x.device
    out = torch.full((I, D), float("nan"), device=dev, dtype=x.dtype)
    xc = torch.nan_to_num(x)
    yc = torch.nan_to_num(y)
    mf = mask.to(x.dtype)
    # precompute daily sums for rolling window
    sx = (xc * mf).sum(2); sy = (yc * mf).sum(2)
    sxy = (xc * yc * mf).sum(2); sx2 = (xc * xc * mf).sum(2); sy2 = (yc * yc * mf).sum(2)
    n = mf.sum(2)
    # rolling N-day sums — conv1d 向量化 (原 for d 973 次切片求和)
    # 左侧 pad (N-1, 0): conv1d 输出 j = Σ x[j-N+1..j] (过去窗口, 无前视)
    import torch.nn.functional as F
    k = torch.ones(1, 1, N, device=dev, dtype=x.dtype)
    def _roll(t):
        return F.conv1d(F.pad(t.unsqueeze(1), (N - 1, 0)), k)[:, 0, :]   # (I, D)
    ns = _roll(n)
    SX = _roll(sx); SY = _roll(sy); SXY = _roll(sxy); SX2 = _roll(sx2); SY2 = _roll(sy2)
    ok = ns >= 2
    ns_c = ns.clamp(min=1)
    mx = SX / ns_c
    my = SY / ns_c
    cov = SXY / ns_c - mx * my
    vx = (SX2 / ns_c - mx ** 2).clamp(min=0)
    vy = (SY2 / ns_c - my ** 2).clamp(min=0)
    if op == 0:
        stat = cov / torch.sqrt((vx * vy).clamp(min=1e-12))
    elif op == 1:
        stat = cov / vx.clamp(min=1e-12)
    else:
        stat = cov
    out = torch.full((I, D), float("nan"), device=dev, dtype=x.dtype)
    out[:, N - 1:] = torch.where(ok[:, N - 1:], stat[:, N - 1:],
                                 torch.full_like(stat[:, N - 1:], float("nan")))
    return out


def _a_ts_ema(args, ctx):
    """Exponential moving average over days: y[t] = α·x[t] + (1−α)·y[t−1], α=2/(w+1).
    NaN-aware: skip NaN days (carry previous value), start from first valid day."""
    x = args[0]                                   # (I, D) D2
    w = int(args[1].item())
    a = 2.0 / (w + 1)
    I, D = x.shape
    dev = x.device
    out = torch.full_like(x, float("nan"))
    # vectorized recurrence via cumsum trick: y[t] = Σ α(1−α)^(t−k) x[k] / Σ α(1−α)^(t−k)
    # over valid days k≤t (NaN days contribute 0 to numerator & denominator)
    xc = torch.nan_to_num(x)
    v = (~torch.isnan(x)).to(x.dtype)
    t_idx = torch.arange(D, device=dev, dtype=x.dtype)
    decay = torch.pow(1 - a, t_idx)               # (D,)
    # numerator: Σ_k α(1−α)^(t−k) x[k] = α(1−α)^t Σ_k x[k](1−α)^(−k)
    # cumsum of xc·(1−a)^(−k) then multiply by α(1−a)^t
    inv_decay = torch.pow(1 - a, -t_idx)
    num = torch.cumsum(xc * inv_decay.unsqueeze(0), dim=1) * (a * decay.unsqueeze(0))
    den = torch.cumsum(v * inv_decay.unsqueeze(0), dim=1) * (a * decay.unsqueeze(0))
    out = num / den.clamp(min=1e-6)
    out = torch.where(den > 1e-6, out, torch.full_like(out, float("nan")))
    return out


def _a_ts_mean(args, ctx):
    x = args[0]
    w = int(args[1].item())
    return _roll_conv(x, w, "mean")


def _a_ts_sum(args, ctx):
    x = args[0]
    w = int(args[1].item())
    return _roll_conv(x, w, "sum")


def _a_ts_std(args, ctx):
    x = args[0]
    w = int(args[1].item())
    m = _roll_conv(x, w, "mean")
    m2 = _roll_conv(x ** 2, w, "mean")
    return torch.sqrt((m2 - m ** 2).clamp(min=0))


def _a_ts_min(args, ctx):
    return _roll_conv(args[0], int(args[1].item()), "max") * -1.0


def _a_ts_max(args, ctx):
    return _roll_conv(args[0], int(args[1].item()), "max")


def _a_ts_delay(args, ctx):
    x = args[0]
    d = int(args[1].item())
    y = torch.full_like(x, float("nan"))
    if d > 0:
        y[:, d:] = x[:, :-d]
    else:
        y[:, :d] = x[:, -d:]
    return y


def _a_ts_delta(args, ctx):
    x = args[0]
    d = int(args[1].item())
    return x - _a_ts_delay(args, ctx)


def _a_ts_corr(args, ctx):
    w = int(args[2].item())
    cov, vx, vy = _roll_pair(args[0], args[1], w)
    return cov / torch.sqrt((vx * vy).clamp(min=1e-6))


def _a_ts_cov(args, ctx):
    cov, _, _ = _roll_pair(args[0], args[1], int(args[2].item()))
    return cov


def _a_ts_zscore(args, ctx):
    x = args[0]
    w = int(args[1].item())
    return (x - _a_ts_mean(args, ctx)) / _a_ts_std(args, ctx).clamp(min=1e-6)


def _a_ts_rank(args, ctx):
    x = _as_float(args[0])   # bool → float (gt/lt 输出是 bool, 类型系统标记为 D2)
    w = int(args[1].item())
    lo = _a_ts_min(args, ctx)
    hi = _a_ts_max(args, ctx)
    return (x - lo) / (hi - lo).clamp(min=1e-6)


def _a_interval_stat(args, ctx):
    """研报27/33 间隔统计因子: 同日前后两个同类状态时点间间隔分钟数的分布统计.
    研报定义: 对过去 20 日每天分别计算间隔分布统计, 再取 20 日均值
    (同日 = 只统计当天内相邻峰/岭的间隔, 不跨天合并).
    args: [mask(A3), N(SCALAR), stat(SCALAR: 1=mean 2=std 3=kurt 4=skew)] → (I,D)
    GPU 批量: _interval_stats 一次算全 (I,D) 日频统计, 再 ts_mean(N)."""
    mask = _as_float(args[0])
    N = int(args[1].item())
    stat = int(args[2].item())
    I, D, M = mask.shape
    dev = mask.device
    stat_map = {1: "mean", 2: "std", 3: "kurt", 4: "skew"}
    key = stat_map.get(stat)
    if key is None:
        raise ValueError(f"interval_stat: unknown stat={stat} (1=mean 2=std 3=kurt 4=skew)")
    daily = _interval_stats(mask, key, mask.dtype)          # (I, D) 每日间隔统计, GPU 批量
    # N日均值 (研报: 过去 N 日同日间隔统计的均值)
    from min_gp.expr import _a_ts_mean
    out = torch.full((I, D), float("nan"), device=dev, dtype=mask.dtype)
    out[:, N-1:] = _a_ts_mean([daily, torch.tensor(float(N), device=dev, dtype=mask.dtype)], ctx)[:, N-1:]
    return out


def _quantile_bf16(t, q, dim):
    """nanquantile with bf16→fp32→bf16 (torch.nanquantile rejects bf16)."""
    if t.dtype == torch.bfloat16:
        t32 = t.float(); del t
        out = torch.nanquantile(t32, q, dim=dim).to(torch.bfloat16)
        del t32; return out
    return torch.nanquantile(t, q, dim=dim)


def _a_roll_cut(args, ctx):
    """通用分位切割: 过去 N 天全部分钟合并, 按 y 排序, 取最高 λ 的 x 均值 − 最低 λ 的 x 均值.
    研报30 分钟理想振幅因子 = roll_cut(amp, close, 10, 0.25). 无泄漏 (每 d 独立窗口).
    批量矩阵实现: unfold 窗口化(分块) + nanquantile 阈值 + 掩码均值.
    args: [x(A3), y(A3), N(SCALAR), λ(SCALAR)] → (I,D)"""
    x, y = _as_float(args[0]), _as_float(args[1])
    N = int(args[2].item())
    lam = float(args[3].item())
    I, D, M = x.shape
    dev = x.device
    out = torch.full((I, D), float("nan"), device=dev, dtype=x.dtype)
    CHUNK = 8  # 天分块控制内存 (unfold 每块 8 天; 2778 只 test 池时 16 仍 OOM)
    for d0 in range(N - 1, D, CHUNK):
        d1 = min(d0 + CHUNK, D)
        Dp = d1 - d0
        # 窗口化: (I, Dp, N*M)
        def _win(t):
            return t[:, d0 - N + 1:d1, :].permute(0, 2, 1).unfold(2, N, 1).permute(0, 2, 1, 3).reshape(I, Dp, N * M)
        aw, pw = _win(x), _win(y)
        valid = torch.isfinite(aw) & torch.isfinite(pw)
        n_ok = valid.sum(2)
        hi_thr = _quantile_bf16(pw, 1.0 - lam, dim=2)
        lo_thr = _quantile_bf16(pw, lam, dim=2)
        ac, pc = torch.nan_to_num(aw), torch.nan_to_num(pw)
        top = (pc >= hi_thr.unsqueeze(2)) & valid
        bot = (pc <= lo_thr.unsqueeze(2)) & valid
        hi = (ac * top).sum(2) / top.sum(2).clamp(min=1.0)
        lo = (ac * bot).sum(2) / bot.sum(2).clamp(min=1.0)
        out[:, d0:d1] = torch.where(n_ok >= 30, hi - lo, torch.full_like(hi, float("nan")))
    return out


def _a_ts_quantile(args, ctx):
    x = args[0]                                  # (I, D) D2 only
    q = float(args[1].item())
    w = int(args[2].item())
    if w <= 1:
        return x.clone()
    win = x.unfold(1, w, 1)                      # (I, D-w+1, w)
    sv = win.sort(-1).values
    valid = (~torch.isnan(sv)).sum(-1).clamp(min=1)
    pos = (q * (valid - 1)).long().clamp(min=0, max=w - 1)
    out = torch.full((x.shape[0], x.shape[1]), float("nan"), device=x.device, dtype=x.dtype)
    out[:, w - 1:] = torch.gather(sv, 2, pos.unsqueeze(2)).squeeze(2)
    return out


def _a_day_interval(args, ctx):
    return _interval_stats(args[0], "std")


def _a_day_iskew(args, ctx):
    return _interval_stats(args[0], "skew")


def _a_day_ikurt(args, ctx):
    return _interval_stats(args[0], "kurt")


def _a_cs_rank(args, ctx):
    return _cs_rank(args[0], dim=0)


def _day_quantile(x3d, q):
    """Intraday q-quantile over minutes (nan-aware), (I,D,M) → (I,D)."""
    x = _as_float(x3d)
    valid = ~torch.isnan(x)
    n = valid.sum(2).clamp(min=1)
    pos = (q * (n - 1)).long().clamp(min=0, max=x3d.shape[2] - 1)
    xc = torch.nan_to_num(x)
    sv = xc.sort(2).values                          # NaN sorts last
    return torch.gather(sv, 2, pos.unsqueeze(2)).squeeze(2)


def _a_day_median(args, ctx):
    return _day_quantile(args[0], 0.5)


def _a_day_quantile(args, ctx):
    return _day_quantile(args[0], float(args[1].item()))


def _a_cs_resid(args, ctx):
    """Daily cross-sectional OLS residual of y on x: resid = y − (α + βx)."""
    y, x = _as_float(args[0]), _as_float(args[1])
    vb = ~(torch.isnan(y) | torch.isnan(x))
    valid = vb.to(y.dtype)
    yc, xc = torch.nan_to_num(y), torch.nan_to_num(x)
    n = valid.sum(0, keepdim=True).clamp(min=2.0)
    mx = (xc * valid).sum(0, keepdim=True) / n
    my = (yc * valid).sum(0, keepdim=True) / n
    cov = ((xc - mx) * (yc - my) * valid).sum(0, keepdim=True) / n
    vx = ((xc - mx) ** 2 * valid).sum(0, keepdim=True) / n
    beta = cov / vx.clamp(min=1e-12)
    alpha = my - beta * mx
    resid = y - (alpha + beta * x)
    return torch.where(vb, resid, torch.full_like(resid, float("nan")))


def _a_time_barycenter(args, ctx):
    """Volume-weighted time barycenter: Σ(t·x)/Σ(x) over minutes, (I,D,M)→(I,D)."""
    x = _as_float(args[0])
    valid = (~torch.isnan(x)).to(x.dtype)
    xc = torch.nan_to_num(x)
    t = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).view(1, 1, -1)
    num = (t * xc * valid).sum(2)
    den = (xc * valid).sum(2).abs().clamp(min=1e-6)
    return num / den


def _a_day_corr(args, ctx):
    """Intraday Pearson corr of x and y over minutes, (I,D,M)×2 → (I,D)."""
    x, y = _as_float(args[0]), _as_float(args[1])
    vb = torch.isfinite(x) & torch.isfinite(y)
    valid = vb.to(x.dtype)
    xc, yc = torch.nan_to_num(x), torch.nan_to_num(y)
    n = valid.sum(2).clamp(min=2.0)
    mx = (xc * valid).sum(2) / n
    my = (yc * valid).sum(2) / n
    cov = ((xc - mx.unsqueeze(2)) * (yc - my.unsqueeze(2)) * valid).sum(2) / n
    vx = ((xc - mx.unsqueeze(2)) ** 2 * valid).sum(2) / n
    vy = ((yc - my.unsqueeze(2)) ** 2 * valid).sum(2) / n
    return cov / torch.sqrt((vx * vy).clamp(min=1e-6))


def _a_bcast(args, ctx):
    """D2 → A3: broadcast daily value over the minute axis (threshold comparison)."""
    return args[0].unsqueeze(2).expand(-1, -1, ctx.NM)


def _a_cs_zscore(args, ctx):
    x = _as_float(args[0])
    # inf 防御: inf 混入 mean → 整列 NaN (ts_corr 方差0窗口输出巨大值/前级 pow 爆炸)
    xf = torch.where(torch.isfinite(x), x, torch.full_like(x, float("nan")))
    valid = (~torch.isnan(xf)).to(xf.dtype)
    xc = torch.nan_to_num(xf)
    n = valid.sum(0, keepdim=True).clamp(min=1.0)
    m = (xc * valid).sum(0, keepdim=True) / n
    s = torch.sqrt(((xc - m) ** 2 * valid).sum(0, keepdim=True) / n).clamp(min=1e-6)
    return torch.where(torch.isfinite(x), (x - m) / s, torch.full_like(x, float("nan")))


def _a_to_b(args, ctx):
    return args[0].permute(0, 2, 1).reshape(-1, args[0].shape[1])


def _a_to_a(args, ctx):
    R, D = args[0].shape
    I = ctx.I
    return args[0].reshape(I, ctx.NM, D).permute(0, 2, 1)


def _a_day_last(args, ctx):
    x = _as_float(args[0])   # bool A3 → bf16: 否则 sum 输出 int64, pool_mask 处 nan→int64 溢出
    valid = ~torch.isnan(x)
    last = valid.cumsum(2) == valid.sum(2, keepdim=True)
    return torch.where(last, x, torch.tensor(0.0, device=x.device, dtype=x.dtype)).sum(2)


def _a_day_first(args, ctx):
    x = _as_float(args[0])   # bool A3 → bf16: 否则 sum 输出 int64, pool_mask 处 nan→int64 溢出
    valid = ~torch.isnan(x)
    first = valid.cumsum(2) == 1
    return torch.where(first, x, torch.tensor(0.0, device=x.device, dtype=x.dtype)).sum(2)


def _a_day_ratio(args, ctx):
    return _a_day_last(args, ctx) / _a_day_first(args, ctx).clamp(min=1e-6)


def _a_if(args, ctx):
    cond, a, b = args
    return torch.where(cond.bool(), _as_float(a), _as_float(b))


def _a_gt(args, ctx):
    a = _as_float(args[0])
    return (a > _as_float(args[1])).to(a.dtype)


def _a_ge(args, ctx):
    a = _as_float(args[0])
    return (a >= _as_float(args[1])).to(a.dtype)


def _a_lt(args, ctx):
    a = _as_float(args[0])
    return (a < _as_float(args[1])).to(a.dtype)


def _a_le(args, ctx):
    a = _as_float(args[0])
    return (a <= _as_float(args[1])).to(a.dtype)


def _a_and(args, ctx):
    return args[0].bool() & args[1].bool()


def _a_or(args, ctx):
    return args[0].bool() | args[1].bool()


def _a_not(args, ctx):
    return ~args[0].bool()


def _a_to_float(args, ctx):
    return _as_float(args[0])


def _a_log(args, ctx):
    x = _as_float(args[0])
    return torch.log(torch.clamp(x, min=1e-12))


def _a_log1p(args, ctx):
    return torch.log1p(torch.clamp(_as_float(args[0]), min=0))


def _a_sqrt(args, ctx):
    return torch.sqrt(torch.clamp(_as_float(args[0]), min=0))


def _a_pow(args, ctx):
    a, b = _as_float(args[0]), _as_float(args[1])
    r = torch.pow(a, b)
    # 数值爆炸防护: GP 生成 pow(open, tp) 指数是价格 → open^tp 可达 1e60 (float32 inf),
    # inf 会污染下游 (ts_corr 窗口/cs_zscore 整列) → 爆炸值转 NaN
    return torch.where(torch.isfinite(r), r, torch.full_like(r, float("nan")))


def _a_div(args, ctx):
    a, b = _as_float(args[0]), _as_float(args[1])
    # sign-preserving 0 除保护: |b|<1e-6 → ±1e-6 (保持 b 符号; b=0 → +1e-6)
    # 不用 clamp(min=1e-12): 负分母会被抬成正的 → 符号翻转 (如 div(ret, x) 全变巨大正值)
    eps = 1e-6
    sgn = torch.sign(b)
    bs = torch.where(sgn == 0, torch.ones_like(sgn), sgn) * eps
    bs = torch.where(b.abs() >= eps, b, bs)
    return a / bs


def _a_clip(args, ctx):
    x = _as_float(args[0])
    lo, hi = float(args[1].item()), float(args[2].item())
    if x.dtype == torch.bfloat16:
        x32 = x.float(); del x
        out = x32.clamp(lo, hi).to(torch.bfloat16)
        del x32; return out
    return x.clamp(lo, hi)


def _a_regime(args, ctx):
    """regime(field, N, k, method) → (I,D,M) bool 分钟状态分类.

    method: 0=above(>μ+kσ)  1=below(≤μ+kσ)  2=isolated_peak  3=clustered_ridge  4=valley(=below)
    与预计算 is_peak/is_ridge 的区别: field/N/k 可由 GP 自由组合, 不限于固定叶子.
    """
    from min_gp.data import _roll_2d_stats
    x = _as_float(args[0])
    N = int(args[1].item())
    k = float(args[2].item())
    method = int(args[3].item())

    # 过去 N 日同时点 μ/σ (用前一日数据, 排除当天 — 无前视)
    x_noday = torch.full_like(x, float("nan"))
    x_noday[:, 1:, :] = x[:, :-1, :]
    vm, vs = _roll_2d_stats(x_noday, N)  # (I, M, D)

    above = x > vm + k * vs
    below = x <= vm + k * vs

    if method == 0:
        return above
    elif method == 1 or method == 4:
        return below
    elif method == 2:     # isolated peak: above + prev below + next below
        return above & _shift(below, -1, False) & _shift(below, 1, False)
    elif method == 3:     # clustered ridge: above + (prev above | next above)
        return above & (_shift(above, -1, False) | _shift(above, 1, False))
    else:
        raise ValueError(f"regime: unknown method={method}")


def _shift(x, d, fill):
    """Shift along minute axis. d<0=prev, d>0=next. fill used at boundaries."""
    out = torch.full_like(x, fill)
    if d < 0:
        out[:, :, :d] = x[:, :, -d:]
    else:
        out[:, :, d:] = x[:, :, :-d]
    return out


def _t_regime(tags):
    """regime(A3, SCALAR, SCALAR, SCALAR) → A3"""
    if tags[0] not in (A3, B2):
        raise TypeTagError(f"regime needs A3 field, got {tags[0]}")
    for i in (1, 2, 3):
        if tags[i] != SCALAR:
            raise TypeTagError(f"regime arg {i} must be SCALAR, got {tags[i]}")
    return A3
# registry
# ──────────────────────────────────────────────

def _reg(name, arity, infer, apply):
    OP[name] = dict(arity=arity, infer=infer, apply=apply)


OP = {}
for _n, _f in [("add", lambda a, b: a + b), ("sub", lambda a, b: a - b),
               ("mul", lambda a, b: a * b)]:
    _reg(_n, 2, _t_binary, _apply_binary(_f))
_reg("pow", 2, _t_binary, _a_pow)
_reg("div", 2, _t_binary, _a_div)
_reg("log", 1, _t_unary, _a_log)
_reg("log1p", 1, _t_unary, _a_log1p)
_reg("sqrt", 1, _t_unary, _a_sqrt)
for _n in ["abs", "sign", "neg"]:
    _f = {"abs": lambda a: torch.abs(a), "sign": lambda a: torch.sign(a),
          "neg": lambda a: -a}[_n]
    _reg(_n, 1, _t_unary, _apply_unary(_f))
_reg("clip", 3, _t_unary, _a_clip)
_reg("if", 3, _t_binary, _a_if)
for _n, _f in [("gt", _a_gt), ("ge", _a_ge), ("lt", _a_lt), ("le", _a_le)]:
    _reg(_n, 2, _t_bool_binary, _f)
_reg("and_", 2, _t_bool_binary, _a_and)
_reg("or_", 2, _t_bool_binary, _a_or)
_reg("not_", 1, _t_unary, _a_not)
_reg("f", 1, _t_unary, _a_to_float)

_reg("intra_mean", 2, _t_intra, _a_intra_mean)
_reg("intra_std", 2, _t_intra, _a_intra_std)
_reg("intra_shift", 2, _t_intra, _a_intra_shift)
_reg("intra_cumsum", 1, _t_intra,
     lambda a, c: torch.cumsum(torch.nan_to_num(_as_float(a[0])), dim=2))
_reg("intra_cummax", 1, _t_intra,
     lambda a, c: torch.cummax(torch.nan_to_num(_as_float(a[0])), dim=2).values)

for _n, _f in [("day_sum", "sum"), ("day_mean", "mean"), ("day_std", "std"),
               ("day_min", "min"), ("day_max", "max")]:
    _reg(_n, 1, _t_day, lambda a, c, _f=_f: _day_reduce(a[0], _f, c))
_reg("day_last", 1, _t_day, _a_day_last)
_reg("day_first", 1, _t_day, _a_day_first)
_reg("day_ratio", 1, _t_day, _a_day_ratio)
_reg("day_skew", 1, _t_day, lambda a, c: _day_skew(a[0]))
_reg("day_kurt", 1, _t_day, lambda a, c: _day_kurt(a[0]))

for _n, _f in [("ts_mean", _a_ts_mean), ("ts_sum", _a_ts_sum), ("ts_std", _a_ts_std),
               ("ts_min", _a_ts_min), ("ts_max", _a_ts_max), ("ts_delay", _a_ts_delay),
               ("ts_delta", _a_ts_delta), ("ts_zscore", _a_ts_zscore), ("ts_rank", _a_ts_rank),
               ("ts_ema", _a_ts_ema)]:
    _reg(_n, 2, _t_ts, _f)
_reg("ts_corr", 3, _t_ts_pair, _a_ts_corr)
_reg("ts_cov", 3, _t_ts_pair, _a_ts_cov)

def _t_mask_agg(tags):
    # x 必须是 A3 (形状来源); mask 必须 A3 — SCALAR x 广播 (I,D,M) 全量会死锁/超时
    if tags[0] != A3 or tags[1] not in (A3, SCALAR):
        raise TypeTagError(f"mask_agg needs (A3, A3|SCALAR), got {tags[:2]}")
    return D2
_reg("mask_agg", 3, _t_mask_agg, _a_mask_agg)

def _t_mask3(tags):
    for tg in tags:
        if tg == SCALAR:
            continue
        if tg != A3:
            raise TypeTagError(f"needs all-A3 args, got {tags}")
    return D2
_reg("mask_ratio", 4, _t_mask3, _a_mask_ratio)
_reg("mask_stat", 5, _t_mask3, _a_mask_stat)
_reg("interval_stat", 3, _t_mask3, _a_interval_stat)
_reg("regime", 4, _t_regime, _a_regime)
def _t_dist(tags):
    if tags[0] != A3:
        raise TypeTagError(f"dist_to_event needs A3, got {tags[0]}")
    return A3
_reg("dist_to_event", 1, _t_dist, _a_dist_to_event)
_reg("ts_quantile", 3, _t_quantile, _a_ts_quantile)

_reg("cs_rank", 1, _t_cs, _a_cs_rank)
_reg("cs_zscore", 1, _t_cs, _a_cs_zscore)
_reg("cs_resid", 2, _t_cs, _a_cs_resid)
_reg("time_barycenter", 1, _t_day, _a_time_barycenter)
_reg("day_corr", 2, _t_day, _a_day_corr)
_reg("day_median", 1, _t_day, _a_day_median)
_reg("day_quantile", 2, _t_day, _a_day_quantile)
_reg("roll_cut", 4, _t_day, _a_roll_cut)
def _t_bcast(tags):
    if tags[0] != D2:
        raise TypeTagError(f"bcast needs D2, got {tags[0]}")
    return A3


_reg("bcast", 1, _t_bcast, _a_bcast)
_reg("to_B", 1, _t_to_b, _a_to_b)
_reg("to_A", 1, _t_to_a, _a_to_a)
_reg("day_istd", 1, _t_istats, _a_day_interval)
_reg("day_iskew", 1, _t_istats, _a_day_iskew)
_reg("day_ikurt", 1, _t_istats, _a_day_ikurt)

# mask ops: mul/add/sub with M1 broadcast into A3
_reg("mask_mul", 2, _t_mask, lambda a, c: _as_float(a[0]) * _as_float(a[1]))


# ──────────────────────────────────────────────
# execution context
# ──────────────────────────────────────────────

class Ctx:
    """Binds leaf tensors (3D → A3, masks → M1) for expression evaluation."""

    def __init__(self, tens, masks, meta, device="cuda", fp=torch.float32):
        self.leaves = {}
        # aliases: 缩写 → 完整名 (种子表达式用 v/l/h/c/o, 数据层用 volume/low/high/close/open)
        _ALIAS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        for k, v in tens.items():
            self.leaves[k] = (v.to(device), A3)
            if k in _ALIAS.values():
                abbr = next((a for a, full in _ALIAS.items() if full == k), None)
                if abbr:
                    self.leaves[abbr] = (v.to(device), A3)
        for k, v in masks.items():
            self.leaves[k] = (v.to(device), M1)
        self.I = meta["I"]
        self.NM = meta["NM"]
        self.fp = fp
        self.device = device

    def eval(self, node):
        t, tag = node.eval(self)
        if tag == B2:
            t = t.reshape(self.I, self.NM, -1).permute(0, 2, 1)   # back to A3
            tag = A3
        if tag not in (D2, A3):
            raise TypeTagError(f"factor root must be D2|A3, got {tag}")
        if tag == A3:
            raise TypeTagError("factor root is A3 — must day_agg to D2 first")
        return t


# ──────────────────────────────────────────────
# parser (safe: ast-based, whitelist dispatch)
# ──────────────────────────────────────────────

import ast as _ast


def parse(s):
    """Parse 'op(arg, arg, ...)' / leaf / number strings into a Node tree."""

    def build(n):
        if isinstance(n, _ast.Expression):
            return build(n.body)
        if isinstance(n, _ast.Constant):
            return Const(n.value)
        if isinstance(n, _ast.Name):
            return Leaf(n.id)
        if isinstance(n, _ast.Call):
            fn = n.func.id if isinstance(n.func, _ast.Name) else None
            if fn not in OP:
                raise TypeTagError(f"unknown op: {fn}")
            args = [build(a) for a in n.args]
            if len(args) != OP[fn]["arity"]:
                raise TypeTagError(f"bad arity for {fn}: {len(args)} != {OP[fn]['arity']}")
            return Op(fn, args)
        if isinstance(n, _ast.BinOp):
            import operator as _op
            opmap = {_ast.Add: "add", _ast.Sub: "sub", _ast.Mult: "mul", _ast.Div: "div", _ast.Pow: "pow"}
            if type(n.op) not in opmap:
                raise TypeTagError(f"unsupported binop: {type(n.op).__name__}")
            return Op(opmap[type(n.op)], [build(n.left), build(n.right)])
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, _ast.USub):
            return Op("neg", [build(n.operand)])
        raise TypeTagError(f"unsupported syntax: {type(n).__name__}")

    return build(_ast.parse(s, mode="eval"))
