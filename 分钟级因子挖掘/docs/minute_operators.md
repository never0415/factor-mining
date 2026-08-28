# Minute-Frequency Operators — Complete Taxonomy

All operators assume input tensors in either Layout A `(I, D, M)` or Layout B `(I×M, D)`.

> 注：本文是**设计分类法超集**（102 个概念算子）。min_gp/expr.py 实际注册 70 个（`OP` 表），
> 未注册算子的功能大多由原生算子覆盖：`mask_agg`/`mask_stat`（聚合与统计）、`interval_stat`（事件间隔分布）、
> `regime`（峰岭谷状态机）、`time_barycenter`、`roll_cut`。具体注册清单以 min_gp.md §3.2 为准。

## 1. Cross-Day Operators *(Layout B: `(I×M, D)` → `(I×M, D)`)*

Same minute-of-day, sliding over dates. Core building block for "S27 style" factors.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 1 | `ts_mean(x, w)` | (I×M, D) → (I×M, D) | Rolling mean, `w` prior days |
| 2 | `ts_std(x, w)` | (I×M, D) → (I×M, D) | Rolling standard deviation |
| 3 | `ts_var(x, w)` | (I×M, D) → (I×M, D) | Rolling variance |
| 4 | `ts_min(x, w)` | (I×M, D) → (I×M, D) | Rolling minimum |
| 5 | `ts_max(x, w)` | (I×M, D) → (I×M, D) | Rolling maximum |
| 6 | `ts_median(x, w)` | (I×M, D) → (I×M, D) | Rolling median (approximate via quantile) |
| 7 | `ts_skew(x, w)` | (I×M, D) → (I×M, D) | Rolling skewness |
| 8 | `ts_kurt(x, w)` | (I×M, D) → (I×M, D) | Rolling kurtosis |
| 9 | `ts_sum(x, w)` | (I×M, D) → (I×M, D) | Rolling sum |
| 10 | `ts_delay(x, d)` | (I×M, D) → (I×M, D) | x `d` days ago |
| 11 | `ts_delta(x, d)` | (I×M, D) → (I×M, D) | x − x `d` days ago |
| 12 | `ts_pct_change(x, d)` | (I×M, D) → (I×M, D) | (x / x_days_ago) − 1 |
| 13 | `ts_corr(x, y, w)` | (I×M, D)×2 → (I×M, D) | Rolling Pearson correlation |
| 14 | `ts_cov(x, y, w)` | (I×M, D)×2 → (I×M, D) | Rolling covariance |
| 15 | `ts_beta(x, y, w)` | (I×M, D)×2 → (I×M, D) | Rolling regression β of x on y |
| 16 | `ts_resid(x, y, w)` | (I×M, D)×2 → (I×M, D) | Rolling regression residual |
| 17 | `ts_rank(x, w)` | (I×M, D) → (I×M, D) | Percentile rank within window |
| 18 | `ts_zscore(x, w)` | (I×M, D) → (I×M, D) | (x − mean) / std within window |
| 19 | `ts_quantile(x, q, w)` | (I×M, D) → (I×M, D) | q-th quantile within window |
| 20 | `ts_argmin(x, w)` | (I×M, D) → (I×M, D) | Days since minimum in window |
| 21 | `ts_argmax(x, w)` | (I×M, D) → (I×M, D) | Days since maximum in window |
| 22 | `ts_ewm_mean(x, halflife)` | (I×M, D) → (I×M, D) | Exponential weighted moving average |
| 23 | `ts_ewm_std(x, halflife)` | (I×M, D) → (I×M, D) | Exponential weighted std |
| 24 | `ts_diff(x, order, w)` | (I×M, D) → (I×M, D) | `order`-th difference, rolled |
| 25 | `ts_above(x, threshold, w)` | (I×M, D) → (I×M, D) | Fraction of days x > threshold in window |
| 26 | `ts_consecutive(x, w)` | (I×M, D) → (I×M, D) | Max consecutive days x > 0 in window |

## 2. Intra-Day Operators *(Layout A: `(I, D, M)` → `(I, D, M)`)*

Sliding over minutes within a single trading day.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 27 | `intra_mean(x, w)` | (I, D, M) → (I, D, M) | Rolling mean, `w` prior minutes |
| 28 | `intra_std(x, w)` | (I, D, M) → (I, D, M) | Rolling std |
| 29 | `intra_min(x, w)` | (I, D, M) → (I, D, M) | Rolling min |
| 30 | `intra_max(x, w)` | (I, D, M) → (I, D, M) | Rolling max |
| 31 | `intra_delta(x, d)` | (I, D, M) → (I, D, M) | x − x `d` minutes ago |
| 32 | `intra_corr(x, y, w)` | (I, D, M)×2 → (I, D, M) | Rolling correlation (e.g. price×volume) |
| 33 | `intra_cumsum(x)` | (I, D, M) → (I, D, M) | Cumulative sum from open |
| 34 | `intra_cummax(x)` | (I, D, M) → (I, D, M) | Running high from open |
| 35 | `intra_cummin(x)` | (I, D, M) → (I, D, M) | Running low from open |
| 36 | `intra_ewm_mean(x, halflife)` | (I, D, M) → (I, D, M) | Intra-day exponential weighted mean |
| 37 | `intra_vwap(x, v)` | (I, D, M)×2 → (I, D, M) | Cumulative VWAP from open = cumsum(x·v)/cumsum(v) |
| 38 | `intra_turnover(x)` | (I, D, M) → (I, D, M) | Cumulative fraction of day's total volume |
| 39 | `intra_rank(x, w)` | (I, D, M) → (I, D, M) | Percentile rank within intra-day window |

## 3. Daily Aggregation *(Layout A → `(I, D)`)*

Collapse minute dimension. All operators in this group change output shape to 2D.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 40 | `day_sum(x)` | (I, D, M) → (I, D) | Sum over 240 minutes |
| 41 | `day_mean(x)` | (I, D, M) → (I, D) | Mean |
| 42 | `day_std(x)` | (I, D, M) → (I, D) | Standard deviation |
| 43 | `day_var(x)` | (I, D, M) → (I, D) | Variance |
| 44 | `day_max(x)` | (I, D, M) → (I, D) | Maximum |
| 45 | `day_min(x)` | (I, D, M) → (I, D) | Minimum |
| 46 | `day_skew(x)` | (I, D, M) → (I, D) | Skewness |
| 47 | `day_kurt(x)` | (I, D, M) → (I, D) | Kurtosis |
| 48 | `day_median(x)` | (I, D, M) → (I, D) | Median |
| 49 | `day_ratio(x)` | (I, D, M) → (I, D) | Last minute / first minute |
| 50 | `day_first(x)` | (I, D, M) → (I, D) | First valid minute value |
| 51 | `day_last(x)` | (I, D, M) → (I, D) | Last valid minute value |
| 52 | `day_weighted_mean(x, w)` | (I, D, M) → (I, D) | Σ(x·w) / Σ(w) using time weight vector |
| 53 | `day_weighted_std(x, w)` | (I, D, M) → (I, D) | Time-weighted std |
| 54 | `day_fraction_above(x, t)` | (I, D, M) → (I, D) | Fraction of minutes with x > t |
| 55 | `day_fraction_below(x, t)` | (I, D, M) → (I, D) | Fraction of minutes with x < t |
| 56 | `day_range(x)` | (I, D, M) → (I, D) | max − min |
| 57 | `day_argmax(x)` | (I, D, M) → (I, D) | Minute index (0–239) of maximum |
| 58 | `day_argmin(x)` | (I, D, M) → (I, D) | Minute index of minimum |

## 4. Time-Period Operators

Operate on a masked subset of the minute axis using time-of-day filters.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 59 | `period_sum(x, mask)` | (I, D, M) → (I, D) | Sum over masked minutes only |
| 60 | `period_mean(x, mask)` | (I, D, M) → (I, D) | Mean over masked minutes |
| 61 | `period_std(x, mask)` | (I, D, M) → (I, D) | Std over masked minutes |
| 62 | `period_max(x, mask)` | (I, D, M) → (I, D) | Max over masked minutes |
| 63 | `period_min(x, mask)` | (I, D, M) → (I, D) | Min over masked minutes |
| 64 | `period_ratio(x, m1, m2)` | (I, D, M) → (I, D) | `period_mean(x, m1) / period_mean(x, m2)` |

Built-in masks (18 total, all `(M,)` broadcast):

| Session | 5m | 15m | 30m | 60m | 90m | 120m |
|---------|-----|------|------|------|------|-------|
| Open | `mask_open_5m` | `mask_open_15m` | `mask_open_30m` | `mask_open_60m` | `mask_open_90m` | `mask_am` |
| Close | `mask_close_5m` | `mask_close_15m` | `mask_close_30m` | `mask_close_60m` | `mask_close_90m` | `mask_pm` |
| Mid | — | — | `mask_mid_30m` | `mask_mid_60m` | — | — |
| Lunch | — | — | `mask_lunch_30m` | `mask_lunch_60m` | — | — |
| Afternoon | — | — | `mask_afternoon_30m` | `mask_afternoon_60m` | — | — |

Example: `time_congestion(V, mask_open_5m)` → barycenter of volume in first 5 minutes. `time_congestion(V, mask_open_60m)` → same but over first hour. GP discovers the optimal window automatically.

## 5. Cross-Sectional Operators

Operate across instruments (dim=0). Can be applied to any shape: 3D `(I, D, M)`, 2D `(I, D)`, or masked subsets.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 65 | `cs_rank(x)` | any → same shape | Cross-sectional rank (0–1) |
| 66 | `cs_zscore(x)` | any → same shape | Cross-sectional z-score |
| 67 | `cs_quantile(x, q)` | any → same shape | q-th quantile threshold |
| 68 | `cs_percentile(x)` | any → same shape | Equivalent to cs_rank |
| 69 | `cs_demean(x)` | any → same shape | x − cross-sectional mean |

## 6. Time-Axis Operators *(中信建投 市场微观结构 研报)*

Operate on the minute index as a positional coordinate. Core innovation from CITIC Securities (2025): use time barycenter to capture intraday distribution shifts.

### Time Barycenter

The barycenter of a variable along the minute axis within a single day:

$$G = \frac{\sum_{t=1}^{M} t \cdot x_t}{\sum_{t=1}^{M} x_t}$$

Where `t` is the minute index (1–240) and `x` is the variable. The result `G` is a scalar per day, representing the "center of gravity" — a low `G` means the variable concentrates early in the day (e.g., opening volume surge), a high `G` means late concentration.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 70 | `time_barycenter(x, session)` | (I, D, M) → (I, D) | Σ(t·x) / Σx within `am` or `pm`, t reset to 1..120 per session |
| 71 | `time_barycenter_up(x, ret, session)` | (I, D, M)×2 → (I, D) | Barycenter for minutes where ret > 0 |
| 72 | `time_barycenter_down(x, ret, session)` | (I, D, M)×2 → (I, D) | Barycenter for minutes where ret < 0 |
| 73 | `time_barycenter_deviation(x, ret, session)` | (I, D, M)×2 → (I, D) | Residual of down ~ up barycenter regression |
| 74 | `time_congestion(x, mask)` | (I, D, M) → (I, D) | Barycenter within a specific time mask (e.g., last 30 min of a session) |

### Time-Segmented Distribution

Split the day into N periods (each ~30 min), compute statistics per period, then aggregate or compare.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 76 | `segment_mean(x, n_segments)` | (I, D, M) → (I, D, S) | Mean per segment → (I, D, S) where S=segments |
| 77 | `segment_std(x, n_segments)` | (I, D, M) → (I, D, S) | Std per segment |
| 78 | `segment_skew(x, n_segments)` | (I, D, M) → (I, D, S) | Skew per segment (CSKEW-style) |
| 79 | `segment_diff(x, n_segments, s1, s2)` | (I, D, M) → (I, D) | Difference between segment s1 and s2 (e.g., open vs close) |
| 80 | `segment_ratio(x, n_segments, s1, s2)` | (I, D, M) → (I, D) | Ratio of segment s1 / s2 |
| 81 | `segment_first(x)` | (I, D, M) → (I, D) | Exclude first N minutes, compute stat on remainder (CSKEW style) |
| 82 | `segment_exclude(x, n_segments, exclude_s)` | (I, D, M) → (I, D) | Aggregate over all segments except excluded ones |

### Event Interval Operators

Given a boolean mask on the minute axis, compute statistics of the gaps between consecutive True events. Captures clustering vs dispersion patterns.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 83 | `event_interval_mean(mask)` | (I, D, M) → (I, D) | Mean gap between consecutive True events |
| 84 | `event_interval_std(mask)` | (I, D, M) → (I, D) | Std of gaps |
| 85 | `event_interval_skew(mask)` | (I, D, M) → (I, D) | Skewness of gap distribution |
| 86 | `event_interval_kurt(mask)` | (I, D, M) → (I, D) | Kurtosis of gap distribution |
| 87 | `event_count(mask)` | (I, D, M) → (I, D) | Number of True events per day (=day_sum on bool) |
| 88 | `event_interval_min(mask)` | (I, D, M) → (I, D) | Minimum gap |
| 89 | `event_interval_max(mask)` | (I, D, M) → (I, D) | Maximum gap |

**torch implementation**:
```python
# Per (i,d): find indices where mask=True, compute diffs, aggregate
pos = mask.nonzero(as_tuple=True)          # positions of True
gaps = pos[1:] - pos[:-1]                  # intervals
```

Requires segmented operation per (instrument, date) — best done as a custom CUDA kernel or via `torch.where` + scattered indexing.

### Examples from the Paper

**CSKEW (corrected return skewness)**: `segment_skew(ret, 8)` excluding first 30 min → rank over 20 days → factor. IC=-0.053, IR=3.3.

**STC (short-term trading congestion)**: `time_barycenter(V, mask_close_30min)` + `day_sum(V, mask_last_3min) / day_sum(V)` → rank sum. IC=-0.052.

**TGD (time barycenter deviation)**: `time_barycenter_deviation(V, ret)` → residual of down-barycenter regressed on up-barycenter. Captures asymmetric timing.

## 7. Element-Wise Operators

Dimension-agnostic, zero overhead. Work in any layout.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 70 | `+`, `-`, `*`, `/` | binary | Arithmetic |
| 71 | `log(x)` | unary | Natural log |
| 72 | `log1p(x)` | unary | log(1 + x), safe for zero |
| 73 | `sqrt(x)` | unary | Square root |
| 74 | `abs(x)` | unary | Absolute value |
| 75 | `sign(x)` | unary | Sign (−1, 0, +1) |
| 76 | `x^y` | binary | Power |
| 77 | `exp(x)` | unary | Exponential |
| 78 | `inv(x)` | unary | 1 / x |
| 79 | `sq(x)` | unary | x² |
| 80 | `clip(x, lo, hi)` | unary | Clamp to [lo, hi] |
| 81 | `if(cond, a, b)` | ternary | Conditional selection |
| 82 | `>(x, t)`, `<(x, t)`, `==(x, t)` | binary | Comparisons → boolean |
| 83 | `and(a, b)`, `or(a, b)`, `not(a)` | boolean | Logical |

## 8. Pattern / Event Operators

Specialized operators for detecting market microstructure patterns.

| # | Operator | Signature | Description |
|---|----------|-----------|-------------|
| 90 | `jump_detect(x, w)` | (I, D, M) → (I, D, M) | Is x > ts_mean(x,w) + N×ts_std(x,w) at this minute? → boolean |
| 91 | `gap_detect(H, L)` | (I, D, M)×2 → (I, D, M) | Gap between consecutive minutes: prev_H < next_L or prev_L > next_H (price ranges don't overlap) |
| 92 | `local_sentiment(mask)` | (I, D, M) → (I, D, M) | For each True in mask: -1 if prev&next are False (isolated low), +1 if prev&next are True (clustered high), 0 otherwise |
| 93 | `prev_state(x)` | T → T | x shifted right by 1 on dim=2 (previous minute) |
| 94 | `next_state(x)` | T → T | x shifted left by 1 on dim=2 (next minute) |
| 95 | `volume_climax(V, w)` | (I, D, M) → (I, D, M) | Volume spike: V > N×intra_mean(V, w) |
| 96 | `reversal_detect(O, C)` | (I, D, M)×2 → (I, D, M) | Direction reversal from previous minute |
| 97 | `breakout_detect(x, w)` | (I, D, M) → (I, D, M) | x > ts_max(x, w) over prior w minutes |
| 98 | `engulfing_detect(O, C)` | (I, D, M)×2 → (I, D, M) | Current bar engulfs previous bar |

### State Classification: Price Peak/Ridge/Valley (开源证券 2026)

The paper defines three price states using TWO dimensions — a richer taxonomy than the volume-based version:

**Dimension 1: Local sentiment** (amplitude of neighbors)
```
is_jump = AMP > ts_mean(AMP, 20) + ts_std(AMP, 20)
sentiment = local_sentiment(is_jump)
  → -1: isolated low (prev=False, next=False) → 局域情绪低迷
  →  0: mixed (one True, one False)
  → +1: clustered high (prev=True, next=True) → 局域情绪高涨
```

**Dimension 2: Jump result** (price gap)
```
has_gap = gap_detect(H, L)   # price ranges don't overlap between adjacent minutes
```

**Combined states**:
```
is_price_peak  = is_jump & (sentiment != +1) & (has_gap == False)   # 价峰
is_price_ridge = is_jump & (sentiment != -1) & (has_gap == True)    # 价岭
is_price_valley = AMP < ts_mean(AMP, 20) - ts_std(AMP, 20)         # 价谷
```

Paper IR comparison (price-based vs volume-based):
| Factor | Volume IR | Price IR |
|--------|-----------|----------|
| Ridge interval skew | 0.22 | **2.44** |
| Peak count ratio | 0.31 | **2.10** |
| Valley VWAP ratio | 0.43 | **2.38** |
| Jump volume corr | 0.22 | **2.98** |

## 9. Summary

| Category | Count | Key for GP |
|----------|-------|------------|
| Cross-day rolling | 26 | ts_mean, ts_std, ts_corr are the workhorses |
| Intra-day rolling | 13 | intra_vwap, intra_cumsum most useful |
| Daily aggregation | 19 | day_sum, day_mean, day_std cover 80% of use |
| Time-period | 6 | period_ratio(open/close) is S27-like |
| Time-axis (CITIC 2025) | 13 | time_barycenter, segment_skew, TGD |
| Cross-sectional | 5 | cs_rank for normalization |
| Element-wise | 14 | universal glue |
| Pattern/event | 6 | jump_detect, gap_detect |
| **Total** | **102** | |

**Coverage**: the S27 paper's 9 factors use only operators {ts_mean, ts_std, `+`, `-`, `*`, `/`, `>`, day_sum, day_mean} — 8 out of 102. The GP has 12× the operator vocabulary to explore.
