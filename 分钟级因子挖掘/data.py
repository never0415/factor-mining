"""
min_gp data layer: build (I, D, M) minute tensors + leaf features from stock_min.parquet.

Grid: M = 241 minutes/day
  morning  09:30-11:30  hm 570..690  → idx 0..120   (121 bars, incl. auction bar)
  afternoon 13:00-14:59  hm 780..899  → idx 121..240 (120 bars)
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from min_gp.label import tensor_fwd_ret, tensor_open_d

NM = 241
S27_W = 20          # 20-day same-minute rolling stats
MIN_COUNT = 5       # min valid days for rolling stats

_COLS = ["trade_date", "open", "high", "low", "close", "volume", "instrument", "datetime"]


def hm_to_midx(hm):
    return np.where(hm < 780, hm - 570, hm - 780 + 121)


def build_slice(parquet_path, start, end, instruments=None, device="cuda", fp=torch.bfloat16,
                extend_days=0):
    """Load a time slice → dict of (I, D, M) tensors + (I, D) daily tensors + (M,) masks.

    extend_days: 向前延展 N 个交易日 (warmup), 使滚动掩码 (S27/S33 20d) 与 ts_*
    滚动算子在本 slice 起点有真实历史 (按年分块回测时年初不缺失).
    meta['warmup']: 实际延展的交易日数 (数据不足时为 0 或 < extend_days).

    Uses pyarrow directly (no pandas) to avoid 30GB+ RAM spike from to_pandas().
    """
    load_start, warmup = start, 0
    if extend_days > 0:
        # 只查 start 前若干日历日窗口: 全历史查询 (< start) 会逐年累积读 1~8 年
        # trade_date 列 → 2019:54s→2026:卡死。
        # 窗口必须随 extend_days 伸缩: 90 日历日 ≈ 62 交易日, 够 extend_days=45,
        # 但再大的请求会被 min(extend_days, len(prev)) 静默截断 —— 60 日 ts_std
        # 需要 59 天历史加平滑尾巴, 截断后每段开头若干周是用不满窗口算的。
        # 下限 90 保证既有 extend_days=45 的调用方查询不变。
        from pandas import Timestamp as _Ts
        lookback = max(90, int(extend_days * 1.7) + 30)
        win0 = (_Ts(start) - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
        prev_filters = [("trade_date", ">=", win0), ("trade_date", "<", start)]
        if instruments is not None:
            prev_filters.append(("instrument", "in", instruments))
        t_prev = pq.read_table(
            parquet_path, columns=["trade_date"], filters=prev_filters)
        # The minute file repeats a date for every stock/bar. Deduplicate before
        # sorting so a warmup lookup does not retain millions of Python strings.
        prev = sorted(set(t_prev.column("trade_date").to_pylist()))
        if prev:
            warmup = min(extend_days, len(prev))
            load_start = prev[-warmup]
    filters = [("trade_date", ">=", load_start), ("trade_date", "<=", end)]
    if instruments is not None:
        filters.append(("instrument", "in", instruments))
    table = pq.read_table(parquet_path, columns=_COLS, filters=filters)
    if table.num_rows == 0:
        raise ValueError("empty slice")

    import pyarrow.compute as pc

    inst_col = table.column("instrument")
    date_col = table.column("trade_date")
    dt_col = table.column("datetime")

    insts = sorted(pc.unique(inst_col).to_pylist())
    dates = sorted(pc.unique(date_col).to_pylist())

    inst_arr = pa.array(insts)
    date_arr = pa.array(dates)

    i_arr = pc.index_in(inst_col, inst_arr).to_numpy().astype(np.int64)
    d_arr = pc.index_in(date_col, date_arr).to_numpy().astype(np.int64)
    hm = (pc.hour(dt_col).to_numpy().astype(np.int64) * 60
          + pc.minute(dt_col).to_numpy().astype(np.int64))
    m_arr = hm_to_midx(hm).astype(np.int64)

    I, D = len(insts), len(dates)
    _t = __import__("time").time
    t0 = _t()
    i_t = torch.as_tensor(i_arr, device=device)
    d_t = torch.as_tensor(d_arr, device=device)
    m_t = torch.as_tensor(m_arr, device=device)
    flat_idx = (i_t * (D * NM) + d_t * NM + m_t).long()
    del i_arr, d_arr, m_arr, i_t, d_t, m_t

    tens = {}
    for col in ("open", "high", "low", "close", "volume"):
        t = torch.full((I, D, NM), float("nan"), dtype=fp, device=device)
        t.view(-1)[flat_idx] = torch.as_tensor(table.column(col).to_numpy(), dtype=fp, device=device)
        tens[col] = t
    del flat_idx, table
    print(f"    [build] scatter {(_t()-t0)*1000:.0f}ms", flush=True)

    # ── derived leaves ──
    O, H, L, C, V = (tens[c] for c in ("open", "high", "low", "close", "volume"))
    tens["tp"] = (O + H + L + C) / 4.0
    tens["amp"] = torch.where(L > 0, H / L - 1.0, torch.full_like(H, float("nan")))  # 分钟振幅 = 最高价/最低价-1 (研报30定义; low=0 数据异常→NaN)
    tens["ret"] = C / O - 1.0

    # ── S27 volume regime leaves (20d same-minute μ/σ on log1p volume, excluding today) ──
    # log1p 是必需的: volume 右偏, raw μ−σ<0 占 37.7% → 温和/喷发划分失效 (σ>μ).
    # log 将分布对称化 (标准做法), 使 μ−σ>0 恒成立
    LV = torch.log1p(V)
    lv_prev = torch.full_like(LV, float("nan"))
    lv_prev[:, 1:, :] = LV[:, :-1, :]           # exclude today
    vm, vs = _roll_2d_stats(lv_prev, S27_W)
    E = LV > vm + vs                            # eruptive
    Ml = LV <= vm + vs                          # mild = 非喷发 (研报: 低于1σ即温和, 含中间区)
    p1m = _shift_m(Ml, -1, fill=False)          # prev minute mild
    n1m = _shift_m(Ml, +1, fill=False)          # next minute mild
    p1e = _shift_m(E, -1, fill=False)
    n1e = _shift_m(E, +1, fill=False)
    tens["is_peak"] = E & p1m & n1m
    tens["is_ridge"] = E & (p1e | n1e)
    tens["is_valley"] = Ml
    del LV, lv_prev, vm, vs, E, Ml, p1m, n1m, p1e, n1e
    torch.cuda.empty_cache()   # 释放 S27 中间量缓存碎片 (build_slice 峰值 16.9GB 主因)

    # ── S33 price-jump regime leaves (AMP-based, 20d same-minute μ/σ, dual-dim) ──
    AMP = tens["amp"]
    amp_prev = torch.full_like(AMP, float("nan"))
    amp_prev[:, 1:, :] = AMP[:, :-1, :]
    am_vm, am_vs = _roll_2d_stats(amp_prev, S27_W)
    is_jump = AMP > am_vm + am_vs
    is_amp_valley = ~is_jump  # 价谷 = 非跳跃时点 (研报33: 振幅 ≤ μ+1σ 全部, 非仅低端)
    p1j = _shift_m(is_jump, -1, fill=False)
    n1j = _shift_m(is_jump, +1, fill=False)
    # gap: prev/next minute price ranges [L,H] disjoint (valid pairs only)
    pL, nL = _shift_m(L, -1), _shift_m(L, +1)
    pH, nH = _shift_m(H, -1), _shift_m(H, +1)
    ok_pair = ~(torch.isnan(pL) | torch.isnan(nL) | torch.isnan(pH) | torch.isnan(nH))
    overlap = (pL <= nH) & (nL <= pH)
    has_gap = is_jump & ok_pair & ~overlap
    # 研报33: 价峰 = 非局域情绪高涨(非双高: 低迷/适中) + 无缺口;
    #        价岭 = 非局域情绪低迷(非双低: 高涨/适中) + 有缺口
    tens["is_jump"] = is_jump
    tens["is_amp_valley"] = is_amp_valley
    tens["is_jump_peak"] = is_jump & ~(p1j & n1j) & ~has_gap
    tens["is_jump_ridge"] = is_jump & (p1j | n1j) & has_gap
    tens["has_gap"] = has_gap
    del amp_prev, am_vm, am_vs, is_jump, is_amp_valley, p1j, n1j, pL, nL, pH, nH, ok_pair, overlap, has_gap
    torch.cuda.empty_cache()   # 释放 S33 中间量缓存碎片

    # ── minute masks (M,) ──
    masks = _build_masks()
    # ── daily tensors ──
    close_d = _day_last(C)
    fwd_ret = tensor_fwd_ret(close_d, close_d, period=1)   # t+1 close-to-close
    # (close 入场 = 收盘前几分钟执行; open 入场口径见 min_gp.label.LABEL)

    meta = dict(I=I, D=D, NM=NM, instruments=insts, dates=dates,
                start=start, end=end, device=str(device), dtype=str(fp), warmup=warmup)
    return tens, masks, fwd_ret, meta


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _roll_2d_stats(x3d, w):
    """20d same-minute rolling mean/std of 3D tensor (excludes current day via caller).
    NaN-aware: uses valid-day count, min count → NaN.
    Chunked over minute axis to bound peak memory (4y train = 10GB+ tensors)."""
    I, D, M = x3d.shape
    dev = x3d.device
    mean = torch.full((I, M, D), float("nan"), device=dev, dtype=x3d.dtype)
    std = torch.full((I, M, D), float("nan"), device=dev, dtype=x3d.dtype)
    k = torch.ones(1, 1, w, device=dev, dtype=x3d.dtype)
    CH = 32  # minutes per chunk → conv input (I*CH, 1, D) ≈ 2.1GB for 4y
    for m0 in range(0, M, CH):
        m1 = min(m0 + CH, M)
        x = x3d[:, :, m0:m1].permute(0, 2, 1).reshape(-1, 1, D)   # (I*CH, 1, D)
        mask = (~torch.isnan(x)).to(x3d.dtype)
        xc = torch.nan_to_num(x)
        xp = F.pad(xc, (w - 1, 0))
        mp = F.pad(mask, (w - 1, 0))
        sx = F.conv1d(xp, k)[:, 0, :]
        ct = F.conv1d(mp, k)[:, 0, :]
        ok = ct >= MIN_COUNT
        cc = ct.clamp(min=1)
        mm = torch.where(ok, sx / cc, torch.tensor(float("nan"), device=dev, dtype=x3d.dtype))
        sx2 = F.conv1d(F.pad(xc ** 2, (w - 1, 0)), k)[:, 0, :]
        var = (sx2 / cc - mm ** 2).clamp(min=0)
        sd = torch.where(ok, torch.sqrt(var), torch.tensor(float("nan"), device=dev, dtype=x3d.dtype))
        mean[:, m0:m1, :] = mm.reshape(I, m1 - m0, D)
        std[:, m0:m1, :] = sd.reshape(I, m1 - m0, D)
    return (mean.permute(0, 2, 1), std.permute(0, 2, 1))


def _shift_m(x, d, fill=False):
    """Shift along minute axis (dim=2). d=-1: prev minute, d=+1: next minute.
    Edge positions are filled with `fill` (False for bool masks): minute 0 has no
    prev minute → cannot be a peak (needs both neighbors mild); last minute likewise.
    NOTE: never fill bool tensors with NaN — torch.full_like casts NaN → True."""
    y = torch.full_like(x, fill)
    if d < 0:
        y[:, :, :d] = x[:, :, -d:]
    else:
        y[:, :, d:] = x[:, :, :-d]
    return y


def _day_last(x):
    """Value at last valid minute of each day (I, D, M) → (I, D).
    Memory-safe: reversed argmax finds last valid index without (I,D,M) cumsum."""
    valid = ~torch.isnan(x)
    # 反转后 argmax = 最后一个 True 的位置; 无有效 → idx=0 (值被 where 置 0)
    idx = x.shape[2] - 1 - valid.flip(2).to(x.dtype).argmax(2)   # (I, D)
    return torch.where(valid.any(2), x.gather(2, idx.unsqueeze(2)).squeeze(2),
                       torch.tensor(0.0, device=x.device, dtype=x.dtype))


def _build_masks():
    """(M,) masks: am/pm + 18 multi-scale + w_time. Returns dict name → 1D tensor."""
    hm = np.concatenate([np.arange(570, 691), np.arange(780, 900)])
    m = hm_to_midx(hm)
    def mask(h0, h1):
        return torch.tensor((hm >= h0) & (hm <= h1), dtype=torch.bfloat16)
    out = {
        "mask_am": mask(571, 690),      # 09:31-11:30
        "mask_pm": mask(781, 899),      # 13:01-14:59
        "mask_open_5m": mask(571, 575), "mask_open_15m": mask(571, 585),
        "mask_open_30m": mask(571, 600), "mask_open_60m": mask(571, 630),
        "mask_open_90m": mask(571, 660), "mask_open_120m": mask(571, 690),
        "mask_close_5m": mask(895, 899), "mask_close_15m": mask(885, 899),
        "mask_close_30m": mask(870, 899), "mask_close_60m": mask(840, 899),
        "mask_close_90m": mask(810, 899), "mask_close_120m": mask(781, 899),
        "mask_mid_30m": mask(601, 630), "mask_mid_60m": mask(601, 660),
        "mask_afternoon_30m": mask(781, 810), "mask_afternoon_60m": mask(781, 840),
        "mask_lunch_30m": mask(661, 690), "mask_lunch_60m": mask(631, 690),
    }
    w_time = torch.exp(-torch.arange(NM, dtype=torch.bfloat16) / 60.0)
    out["w_time"] = w_time
    return out


# ──────────────────────────────────────────────
# universe pool (CSI 500/1000 monthly component weights, point-in-time)
# ──────────────────────────────────────────────

def _component_code(csv_code):
    """000006.XSHE → sz000006; 600004.XSHG → sh600004."""
    num, exch = csv_code.split(".")
    return ("sh" if exch == "XSHG" else "sz") + num


def load_pit_codes(pit_pq, start=None, end=None):
    """Load unique normalized instruments from a daily PIT membership parquet."""
    filters = []
    if start:
        filters.append(("trade_date", ">=", start))
    if end:
        filters.append(("trade_date", "<=", end))
    table = pq.read_table(
        pit_pq, columns=["instrument"], filters=filters or None)
    return sorted(table.column("instrument").unique().to_pylist())


def load_pit_dates(pit_pq, start=None, end=None):
    """Load sorted trading dates from a daily PIT membership parquet."""
    filters = []
    if start:
        filters.append(("trade_date", ">=", start))
    if end:
        filters.append(("trade_date", "<=", end))
    table = pq.read_table(
        pit_pq, columns=["trade_date"], filters=filters or None)
    return sorted(table.column("trade_date").unique().to_pylist())


def load_pit_pool(pit_pq, start=None, end=None):
    """Return daily PIT members in the same shape as component-pool snapshots.

    The local PIT file contains membership but no index weights, so equal
    benchmark weights are used. This keeps backtests/index enhancement
    point-in-time correct while making the missing vendor weights explicit.
    """
    filters = []
    if start:
        filters.append(("trade_date", ">=", start))
    if end:
        filters.append(("trade_date", "<=", end))
    df = pq.read_table(
        pit_pq, columns=["trade_date", "instrument"],
        filters=filters or None).to_pandas()
    result = []
    for date, group in df.groupby("trade_date", sort=True):
        codes = sorted(group["instrument"].dropna().unique().tolist())
        if not codes:
            continue
        weights = np.full(len(codes), 100.0 / len(codes), dtype=np.float32)
        result.append((date, codes, weights))
    return result


def load_index_codes(index_source):
    """Load all unique CSI 500 codes from PIT parquet or snapshot CSVs.

    Each CSV: updateDate, code (sh.600004 / szse.000006), code_name.
    Returns sorted list of normalized codes (sh600004, sz000006).
    """
    if os.path.isfile(index_source) and str(index_source).lower().endswith(".parquet"):
        return load_pit_codes(index_source)
    import glob as _glob
    codes = set()
    for f in sorted(_glob.glob(os.path.join(index_source, "*.csv"))):
        df = pd.read_csv(f, encoding="gbk")
        for c in df.code:
            c = c.lower().replace(".", "").replace("szse", "sz")
            codes.add(c)
    return sorted(codes)


def build_pit_mask(index_dir, dates, insts, device="cuda"):
    """Build PIT (point-in-time) CSI 500 membership mask from weekly snapshots.

    For each date d ∈ dates, use the most recent snapshot ≤ d.
    Returns (I, D) bool tensor — True where stock was in CSI 500 on that date.
    No weights — just membership.
    """
    import glob as _glob
    # Load all snapshots: {snapshot_date: set(codes)}
    snaps = {}
    for f in sorted(_glob.glob(os.path.join(index_dir, "*.csv"))):
        snap_date = os.path.basename(f)[:10]  # "2015-01-05"
        df = pd.read_csv(f, encoding="gbk")
        codes = set()
        for c in df.code:
            c = c.lower().replace(".", "").replace("szse", "sz")
            codes.add(c)
        snaps[snap_date] = codes
    sdates = sorted(snaps.keys())

    inst_idx = {s: i for i, s in enumerate(insts)}
    I, D = len(insts), len(dates)
    mask = torch.zeros(I, D, dtype=torch.bool, device=device)
    pi = 0
    for j, d in enumerate(dates):
        while pi + 1 < len(sdates) and sdates[pi + 1] <= d:
            pi += 1
        for code in snaps[sdates[pi]]:
            i = inst_idx.get(code)
            if i is not None:
                mask[i, j] = True
    return mask


def load_pit_daily_mask(pit_pq, dates, insts, device="cuda"):
    """读预构建日频 PIT mask (research/zz500_pit_daily.parquet) → (I,D) bool.

    与回测 (bt_common) 同源同口径: 每天成分 ≈500 (2081 天里 98.9% 恰好 500).
    dates/insts 必须与 build_slice 的 meta 对齐; 不在快照内的 (日期,股票) 保持 False.
    """
    import pyarrow.parquet as pq
    if not dates or not insts:
        return torch.zeros(len(insts), len(dates), dtype=torch.bool, device=device)
    filters = [
        ("trade_date", ">=", min(dates)),
        ("trade_date", "<=", max(dates)),
        ("instrument", "in", list(insts)),
    ]
    df = pq.read_table(
        pit_pq, columns=["trade_date", "instrument"], filters=filters
    ).to_pandas()
    inst_idx = {s: i for i, s in enumerate(insts)}
    d_idx = {d: j for j, d in enumerate(dates)}
    i_arr = df["instrument"].map(inst_idx)
    j_arr = df["trade_date"].map(d_idx)
    ok = i_arr.notna() & j_arr.notna()
    mask = torch.zeros(len(insts), len(dates), dtype=torch.bool, device=device)
    if ok.any():
        mask[i_arr[ok].to_numpy().astype(np.int64), j_arr[ok].to_numpy().astype(np.int64)] = True
    return mask


def load_component_pool(csv_path, top_n=None):
    """Monthly CSI component CSV: dedup + convert → list of (date, codes, weights).

    Args:
        csv_path: path to zz500_component.csv or zz1000_component.csv.
        top_n: optional, take only top-N by weight per date (e.g. 500 for CSI 500).
    Returns: list of (date: str, codes: list[str], weights: np.ndarray) sorted by date.
    """
    df = pd.read_csv(csv_path)
    # dedup (data vendor duplicates per code-date)
    df = df.drop_duplicates(["code", "date"])
    dates = sorted(df.date.unique())
    result = []
    for d in dates:
        sub = df[df.date == d].sort_values("weight", ascending=False)
        if top_n is not None and len(sub) > top_n:
            sub = sub.head(top_n)
        codes = [_component_code(c) for c in sub.code]
        w = sub.weight.values.astype(np.float32)
        result.append((d, codes, w))
    return result


def build_pool_mask(pool_data, dates, insts, device="cuda"):
    """(mask: (I,D) bool, weights: (I,D) float32) — point-in-time membership + weight.

    Most recent pool snapshot ≤ date is used. Stocks not in the pool get weight=0,
    mask=False. Weights are raw CSI component weights (annualized rebalance,
    sum-to-100 per snapshot).
    """
    I, D = len(insts), len(dates)
    inst_idx = {s: i for i, s in enumerate(insts)}
    mask = torch.zeros(I, D, dtype=torch.bool, device=device)
    w_mat = torch.zeros(I, D, dtype=torch.float32, device=device)
    pi = 0
    pdates = [p[0] for p in pool_data]
    for j, d in enumerate(dates):
        while pi + 1 < len(pdates) and pdates[pi + 1] <= d:
            pi += 1
        _, codes, weights = pool_data[pi]
        for code, w in zip(codes, weights):
            i = inst_idx.get(code)
            if i is not None:
                mask[i, j] = True
                w_mat[i, j] = float(w)
    return mask, w_mat


# ── legacy: weekly snapshot dir (backward compat) ──

def load_pool(index_dir):
    """Weekly CSI 500 snapshots: {date: set(codes)}. codes normalized sh.600004 → sh600004."""
    if not os.path.isdir(index_dir):
        return {}
    pools = {}
    for f in sorted(os.listdir(index_dir)):
        if not f.endswith(".csv"):
            continue
        d = f[:-4]
        df = pd.read_csv(os.path.join(index_dir, f), encoding="gbk")
        pools[d] = set(df["code"].str.replace(".", "").tolist())
    return pools


def pool_mask(pools, dates, insts):
    """(I, D) bool: point-in-time CSI 500 membership (most recent snapshot ≤ date).

    Returns (mask, None) — zero-weight sentinel; use build_pool_mask for weighted pools.
    """
    if not pools:
        I, D = len(insts), len(dates)
        return torch.ones(I, D, dtype=torch.bool), None
    pdates = sorted(pools)
    inst_idx = {s: i for i, s in enumerate(insts)}
    mask = torch.zeros(len(insts), len(dates), dtype=torch.bool)
    pi = 0
    for j, d in enumerate(dates):
        while pi + 1 < len(pdates) and pdates[pi + 1] <= d:
            pi += 1
        for code in pools[pdates[pi]]:
            i = inst_idx.get(code)
            if i is not None:
                mask[i, j] = True
    return mask, None


# ──────────────────────────────────────────────
# industry map (latest snapshot per instrument)
# ──────────────────────────────────────────────

def load_industry(parquet_path):
    """Return latest available industry bucket per normalized instrument."""
    df = pd.read_parquet(parquet_path)
    if "instrument" not in df.columns and "symbol" in df.columns:
        symbol = df["symbol"].astype(str).str.zfill(6)
        df["instrument"] = np.where(
            symbol.str.startswith(("5", "6", "9")), "sh", "sz") + symbol
    date_col = next(
        (c for c in ("trade_date", "available_date", "start_date") if c in df.columns),
        None)
    if date_col:
        df = df.sort_values(date_col)
    df = df.groupby("instrument").last()
    if "sw_level1" in df.columns:
        sector = df["sw_level1"].astype(str)
    elif "industry" in df.columns:
        sector = df["industry"].astype(str).str[0]
    else:
        raise ValueError(f"industry column not found in {parquet_path}")
    ids, uniq = pd.factorize(sector)
    return dict(zip(df.index, ids.tolist()))


def industry_ids(industry_map, insts):
    """(I,) int64 industry ids for the slice's instrument order; unknown → max+1 bucket."""
    n = len(insts)
    ids = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(insts):
        ids[i] = industry_map.get(s, -1)
    ids = np.where(ids < 0, ids.max() + 1, ids) if (ids < 0).any() else ids
    return torch.as_tensor(ids, dtype=torch.int64)


# ──────────────────────────────────────────────
# market cap (daily per-instrument, from valuation CSV cache)
# ──────────────────────────────────────────────

def load_market_cap(parquet_path):
    """Return daily float market cap series by instrument."""
    df = pd.read_parquet(parquet_path)
    date_col = "day" if "day" in df.columns else "trade_date"
    if "market_cap" in df.columns:
        value_col = "market_cap"
    elif "ln_float_market_cap" in df.columns:
        value_col = "ln_float_market_cap"
        df[value_col] = np.exp(df[value_col])
    else:
        raise ValueError(f"market-cap column not found in {parquet_path}")
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    out = {}
    for inst, g in df.groupby("instrument"):
        s = g.set_index(date_col)[value_col].sort_index()
        out[inst] = s
    return out


def build_mcap_tensor(mcap_dict, dates, insts, device="cuda"):
    """(I, D) float32 market-cap tensor aligned with slice instruments/dates.
    NaN where stock not in valuation data or suspended.
    """
    I, D = len(insts), len(dates)
    t = torch.full((I, D), float("nan"), dtype=torch.float32, device=device)
    for i, inst in enumerate(insts):
        s = mcap_dict.get(inst)
        if s is None:
            continue
        for j, d in enumerate(dates):
            if d in s.index:
                t[i, j] = s.loc[d]
    return t
