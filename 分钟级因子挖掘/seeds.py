"""Seed factors from 开源证券 S27/S33/S28, expressed in min_gp expression strings."""

AMT = "mul(tp,v)"          # turnover ≈ tp × volume (no AMT field in stock_min.parquet)


def _vwap(mask):
    return f"div(day_sum(mul(mul(tp,v),{mask})), day_sum(mul(v,{mask})))"


def _vwap_ratio(mask_a, mask_b):
    return f"ts_mean(div({_vwap(mask_a)},{_vwap(mask_b)}),20)"


SEEDS = {}

# ═══════════════ S27 volume regime (m_* leaves) — 21 factors ═══════════════
# 1-3. minute counts
SEEDS["s27_peak_count"] = "ts_sum(day_sum(f(is_peak)),20)"
SEEDS["s27_ridge_count"] = "ts_sum(day_sum(f(is_ridge)),20)"
SEEDS["s27_valley_count"] = "ts_sum(day_sum(f(is_valley)),20)"
# 4. ridge minute return (20d sum of ridge-minute returns)
SEEDS["s27_ridge_return"] = "ts_sum(day_sum(mul(ret,f(is_ridge))),20)"
# 5-6. relative VWAP (new definitions below at §5)
# 7-8. VWAP percentile vs day range (new definitions below at §5)
# 9-14. interval stats (std/skew/kurt of adjacent event gaps, 20d smoothed)
SEEDS["s27_peak_istd"] = "ts_mean(day_istd(f(is_peak)),20)"
SEEDS["s27_peak_iskew"] = "ts_mean(day_iskew(f(is_peak)),20)"
SEEDS["s27_peak_ikurt"] = "ts_mean(day_ikurt(f(is_peak)),20)"
SEEDS["s27_ridge_istd"] = "ts_mean(day_istd(f(is_ridge)),20)"
SEEDS["s27_ridge_iskew"] = "ts_mean(day_iskew(f(is_ridge)),20)"
SEEDS["s27_ridge_ikurt"] = "ts_mean(day_ikurt(f(is_ridge)),20)"
# 17. peak/ridge turnover ratio (20d sums, 分母 +1 平滑防 0 除爆炸)
SEEDS["s27_peak_ridge_turnover"] = (f"div(ts_sum(day_sum(mul({AMT},f(is_peak))),20),"
                                    f"add(ts_sum(day_sum(mul({AMT},f(is_ridge))),20),1))")
# 18. follow ratio: next-minute turnover / eruptive turnover (20d sums)
_ER = "or_(f(is_peak),f(is_ridge))"
SEEDS["s27_follow_ratio"] = (f"div(ts_sum(day_sum(mul(intra_shift({AMT},1),{_ER})),20),"
                             f"add(ts_sum(day_sum(mul({AMT},{_ER})),20),1))")
# 19. sensitivity: regression slope of next-minute turnover on turnover, at eruptive minutes
def _me(mask, x):
    return f"div(day_sum(mul({x},{mask})),add(day_sum({mask}),1))"

_E1 = _me(_ER, AMT)
_E2 = _me(_ER, f"intra_shift({AMT},1)")
_EXY = _me(_ER, f"mul({AMT},intra_shift({AMT},1))")
_EX2 = _me(_ER, f"mul({AMT},{AMT})")
_EY2 = _me(_ER, f"mul(intra_shift({AMT},1),intra_shift({AMT},1))")
_COV = f"sub({_EXY},mul({_E1},{_E2}))"
_VX = f"sub({_EX2},mul({_E1},{_E1}))"
_VY = f"sub({_EY2},mul({_E2},{_E2}))"
SEEDS["s27_sensitivity"] = f"div({_COV},{_VX})"
# 20. correlation
SEEDS["s27_corr"] = f"div({_COV},sqrt(mul({_VX},{_VY})))"
# 21. same-minute peak/ridge count correlation (20d, per minute-of-day)
SEEDS["s27_same_minute_corr"] = "day_mean(ts_corr(ts_mean(to_B(f(is_peak)),20), ts_mean(to_B(f(is_ridge)),20), 20))"

# ═══════════════ S33 price-jump regime (j_* leaves) — 17 factors ═══════════════
SEEDS["s33_peak_count"] = "ts_sum(day_sum(f(is_jump_peak)),20)"
SEEDS["s33_ridge_count"] = "ts_sum(day_sum(f(is_jump_ridge)),20)"
SEEDS["s33_ridge_return"] = "ts_sum(day_sum(mul(ret,f(is_jump_ridge))),20)"
SEEDS["s33_valley_vwap"] = _vwap_ratio("f(is_amp_valley)", "1")
SEEDS["s33_valley_vwap_pct"] = f"div(sub({_vwap('f(is_amp_valley)')},day_min(l)),sub(day_max(h),day_min(l)))"
SEEDS["s33_peak_istd"] = "ts_mean(day_istd(f(is_jump_peak)),20)"
SEEDS["s33_peak_iskew"] = "ts_mean(day_iskew(f(is_jump_peak)),20)"
SEEDS["s33_peak_ikurt"] = "ts_mean(day_ikurt(f(is_jump_peak)),20)"
SEEDS["s33_ridge_istd"] = "ts_mean(day_istd(f(is_jump_ridge)),20)"
SEEDS["s33_ridge_iskew"] = "ts_mean(day_iskew(f(is_jump_ridge)),20)"
SEEDS["s33_ridge_ikurt"] = "ts_mean(day_ikurt(f(is_jump_ridge)),20)"
SEEDS["s33_valley_ridge_price"] = (
    f"ts_mean(div(mask_agg(mul(mul(tp,v),1), is_amp_valley, 2), "
    f"mask_agg(mul(mul(tp,v),1), is_jump_ridge, 2)),20)")
SEEDS["s33_peak_ridge_turnover"] = "mask_ratio(mul(volume,close), is_jump_peak, is_jump_ridge, 20)"
_JER = "or_(f(is_jump_peak),f(is_jump_ridge))"
SEEDS["s33_follow_ratio"] = (f"div(ts_sum(day_sum(mul(intra_shift({AMT},1),{_JER})),20),"
                             f"add(ts_sum(day_sum(mul({AMT},{_JER})),20),1))")
_J1 = _me(_JER, AMT)
_J2 = _me(_JER, f"intra_shift({AMT},1)")
_JXY = _me(_JER, f"mul({AMT},intra_shift({AMT},1))")
_JX2 = _me(_JER, f"mul({AMT},{AMT})")
_JY2 = _me(_JER, f"mul(intra_shift({AMT},1),intra_shift({AMT},1))")
_JCOV = f"sub({_JXY},mul({_J1},{_J2}))"
_JVX = f"sub({_JX2},mul({_J1},{_J1}))"
_JVY = f"sub({_JY2},mul({_J2},{_J2}))"
SEEDS["s33_sensitivity"] = f"div({_JCOV},{_JVX})"
SEEDS["s33_corr"] = f"div({_JCOV},sqrt(mul({_JVX},{_JVY})))"
SEEDS["s33_same_minute_corr"] = "day_mean(ts_corr(ts_mean(to_B(f(is_jump_peak)),20), ts_mean(to_B(f(is_jump_ridge)),20), 20))"

# ═══════════════ S28 理想振幅 (daily cutting) ═══════════════
_AMP_D = "sub(div(day_max(h),day_min(l)),1)"
_C_D = "day_last(c)"
_Q75 = "ts_quantile(day_last(c),0.75,20)"
_Q25 = "ts_quantile(day_last(c),0.25,20)"
_V_HIGH = f"div(ts_mean(mul({_AMP_D},ge({_C_D},{_Q75})),20),ts_mean(ge({_C_D},{_Q75}),20))"
_V_LOW = f"div(ts_mean(mul({_AMP_D},le({_C_D},{_Q25})),20),ts_mean(le({_C_D},{_Q25}),20))"
SEEDS["s28_ideal_amp"] = f"sub({_V_HIGH},{_V_LOW})"

# ═══════════════ older series (OHLCV-expressible) ═══════════════
# (17) ERR: extreme-minute return + prev-1min return (S = |ret − day median|)
_ERR_S = "abs(sub(ret,bcast(day_median(ret))))"
_ERR_MASK = f"ge({_ERR_S},bcast(day_max({_ERR_S})))"
SEEDS["s17_err"] = f"add(day_sum(mul(ret,f({_ERR_MASK}))),day_sum(mul(intra_shift(ret,-1),f({_ERR_MASK}))))"

# (19) TGD: down-barycenter residual on up-barycenter, 20d mean
_UP_BARY = "time_barycenter(mul(ret,f(gt(ret,0))))"
_DN_BARY = "time_barycenter(mul(ret,f(lt(ret,0))))"
SEEDS["s19_tgd"] = f"ts_mean(cs_resid({_DN_BARY},{_UP_BARY}),20)"

# (27) 量峰间隔峰度 (研报27 §6): 20日同日前后两个量峰间时间间隔分布的峰度
# RankIC 7.19%, ICIR 4.63, 多空 23.3%  (dist_to_event 字段 + mask_agg 偏/峰度)
SEEDS["s27_peak_interval_kurt"] = "ts_mean(mask_agg(dist_to_event(is_peak), is_peak, 5),20)"

# (33) 价岭间隔偏度 (研报33 §5): 20日同日前后两个价岭间时间间隔分布的偏度
# RankIC -7.62%, ICIR -3.5, 多空 21.69%
SEEDS["s33_ridge_interval_skew"] = "ts_mean(mask_agg(dist_to_event(is_jump_ridge), is_jump_ridge, 4),20)"

# (27) 峰岭成交比 (研报27 §7.2): 20日量峰总成交额 / 量岭总成交额
# 成交额≈volume×close. RankIC 10.28%, ICIR 4.07, 多空 27.13%
SEEDS["s27_peak_ridge_ratio"] = "mask_ratio(mul(volume,close), is_peak, is_ridge, 20)"

# (27) 量谷加权价格分位点 (研报27 §5): 区间=[min(low,昨收),max(high,昨收)],
# (量谷VWAP−min)/(max−min), 20日分位点均值, 反转中性化
# RankIC 6.34%, ICIR 4.32, 多空 20.22%  (min/max 用 |a-b| 公式实现)
_SEED_VALLEY_VWAP = f"div(mask_agg(mul(mul(tp,v),1), is_valley, 1), mask_agg(mul(v,1), is_valley, 1))"
_SEED_PREV_C = "ts_delay(day_last(c),1)"
_SEED_MIN2 = lambda a, b: f"div(sub(add({a},{b}),abs(sub({a},{b}))),2)"
_SEED_MAX2 = lambda a, b: f"div(add(add({a},{b}),abs(sub({a},{b}))),2)"
_SEED_LO = _SEED_MIN2("day_min(l)", _SEED_PREV_C)
_SEED_HI = _SEED_MAX2("day_max(h)", _SEED_PREV_C)
SEEDS["s27_valley_vwap_pct"] = f"ts_mean(div(sub({_SEED_VALLEY_VWAP},{_SEED_LO}),sub({_SEED_HI},{_SEED_LO})),20)"

# (27) 量峰加权价格分位点 (研报27 对应): 同上但用量峰VWAP
_SEED_PEAK_VWAP = f"div(mask_agg(mul(mul(tp,v),1), is_peak, 1), mask_agg(mul(v,1), is_peak, 1))"
SEEDS["s27_peak_vwap_pct"] = f"ts_mean(div(sub({_SEED_PEAK_VWAP},{_SEED_LO}),sub({_SEED_HI},{_SEED_LO})),20)"

# (27) 量谷相对加权价格 (研报27 §4): 量谷VWAP/全日VWAP, 20日均值
# RankIC 8.69%, ICIR 4.44, 多空 25.35%
_SEED_FULL_VWAP = f"div(mask_agg(mul(mul(tp,v),1), 1, 1), mask_agg(mul(v,1), 1, 1))"
SEEDS["s27_valley_vwap"] = f"ts_mean(div({_SEED_VALLEY_VWAP},{_SEED_FULL_VWAP}),20)"

# (27) 量峰相对加权价格: 量峰VWAP/全日VWAP, 20日均值
SEEDS["s27_peak_ridge_price"] = f"ts_mean(div({_SEED_PEAK_VWAP},{_SEED_FULL_VWAP}),20)"
SEEDS["s27_ridge_vwap"] = (
    f"ts_mean(div(div(mask_agg(mul(mul(tp,v),1), is_ridge, 1), mask_agg(mul(v,1), is_ridge, 1)),"
    f"{_SEED_FULL_VWAP}),20)")

# (27) 谷岭加权价格比 (研报27 §7.1): 量谷VWAP/量岭VWAP, 20日均值 (无掩码日→NaN)
# RankIC 6.98%, ICIR 3.56, 多空 15.83%
SEEDS["s27_valley_ridge_price"] = (
    f"ts_mean(div({_SEED_VALLEY_VWAP},"
    f"div(mask_agg(mul(mul(tp,v),1), is_ridge, 1), mask_agg(mul(v,1), is_ridge, 1))),20)")
# (30) 分钟理想振幅因子 (研报表2): 过去N=10天全部分钟合并,
# 按分钟收盘价切割 λ=25%: V_high - V_low (跨天合并, 非日内切割)
# 抽象: roll_cut(x, y, N, λ) 通用分位切割算子
SEEDS["s30_amp_cut"] = "roll_cut(amp, close, 10, 0.25)"

# (4) intraday vs overnight return decomposition (黄金律 style, stock-level)
_SEED_O = "day_last(c)"
SEEDS["s04_intraday_ret"] = "ts_sum(sub(day_ratio(c),1),20)"
SEEDS["s04_overnight_ret"] = f"ts_sum(sub(div({_SEED_O},ts_delay({_SEED_O},1)),1),20)"
SEEDS["s04_io_diff"] = ("ts_sum(sub(day_ratio(c),1),20) - ts_sum(sub(div(day_last(c),ts_delay(day_last(c),1)),1),20)")

# (5) APM-style session returns: am / pm / overnight
SEEDS["s05_am_ret"] = "ts_sum(day_sum(mask_mul(ret,mask_am)),20)"
SEEDS["s05_pm_ret"] = "ts_sum(day_sum(mask_mul(ret,mask_pm)),20)"
SEEDS["s05_ovn_ret"] = f"ts_sum(sub(div(day_last(c),ts_delay(day_last(c),1)),1),20)"
SEEDS["s05_ovp"] = f"add(ts_sum(day_sum(mask_mul(ret,mask_pm)),20),ts_sum(sub(div(day_last(c),ts_delay(day_last(c),1)),1),20))"

# (3) smart money: VWAP of top-20% S minutes / total VWAP, S = |ret|·√V
_SMART_S = "mul(abs(ret),sqrt(v))"
_SMART_Q = "bcast(day_quantile(mul(abs(ret),sqrt(v)),0.8))"
_SMART_MASK = f"ge({_SMART_S},{_SMART_Q})"
SEEDS["s03_smart_vwap"] = (f"div(div(day_sum(mul(mul(tp,v),f({_SMART_MASK}))),day_sum(mul(v,f({_SMART_MASK})))),"
                           f"div(day_sum(mul(tp,v)),day_sum(v)))")

# ═══════════════ (20) GA paper patterns (OHLCV-expressible) ═══════════════
# 日内分钟收益波动 (intraday return volatility)
SEEDS["s20_ret_vol"] = "day_std(ret)"
# 交易情绪不稳定性: 日间时序极差 × 日内收益波动 (N=10)
SEEDS["s20_emotion_instab"] = "sub(ts_max(day_std(ret),10),ts_min(day_std(ret),10))"
# 分钟量价相关性 (intraday corr of ret and volume)
SEEDS["s20_pv_corr"] = "day_corr(ret,v)"
# 标准成交量波动: 日内 std of volume / day-mean volume
SEEDS["s20_std_vol_vol"] = "day_std(div(v,bcast(day_mean(v))))"
# 理想振幅变体: cut by price but object = intraday ret vol (paper: 振幅→分钟收益波动 improves ICIR)
_VH_RV = f"div(ts_mean(mul(day_std(ret),ge({_C_D},{_Q75})),20),ts_mean(ge({_C_D},{_Q75}),20))"
_VL_RV = f"div(ts_mean(mul(day_std(ret),le({_C_D},{_Q25})),20),ts_mean(le({_C_D},{_Q25}),20))"
SEEDS["s20_ideal_retvol"] = f"sub({_VH_RV},{_VL_RV})"
# 主力控盘: corr(amp, ret-vol) & corr(amp, std-vol) 合成版 (single: amp × retvol corr 20d)
SEEDS["s20_control_corr"] = "ts_mean(day_corr(amp,bcast(day_std(ret))),20)"

# ═══════════════ paper direction reference (A-share, for validation) ═══════════════
# sign: + means factor positively predicts next-day return cross-sectionally
PAPER_DIR = {
    "s27_peak_count": +1, "s27_ridge_count": -1, "s27_valley_count": +1,
    "s27_ridge_return": -1, "s27_valley_vwap": +1, "s27_ridge_vwap": -1,
    "s27_peak_vwap_pct": +1, "s27_valley_vwap_pct": +1,
    "s27_peak_istd": -1, "s27_peak_iskew": +1, "s27_peak_ikurt": +1,
    "s27_ridge_istd": +1, "s27_ridge_iskew": -1, "s27_ridge_ikurt": -1,
    "s27_valley_ridge_price": +1, "s27_peak_ridge_price": +1,
    "s27_peak_ridge_turnover": +1, "s27_follow_ratio": -1,
    "s27_sensitivity": -1, "s27_corr": -1, "s27_same_minute_corr": -1,
    "s33_peak_count": +1, "s33_ridge_count": -1, "s33_ridge_return": -1,
    "s33_valley_vwap": +1, "s33_valley_vwap_pct": +1,
    "s33_peak_istd": -1, "s33_peak_iskew": +1, "s33_peak_ikurt": +1,
    "s33_ridge_istd": +1, "s33_ridge_iskew": -1, "s33_ridge_ikurt": -1,
    "s33_valley_ridge_price": +1, "s33_peak_ridge_turnover": +1,
    "s33_follow_ratio": -1, "s33_sensitivity": -1, "s33_corr": -1,
    "s33_same_minute_corr": +1,
    "s28_ideal_amp": -1,   # paper: V_high − V_low negatively predicts (high-price-state amplitude)
    # older series (17)(19)(30)(4)(5)(3) — direction where paper states it
    "s17_err": -1,          # ERR RankIC -7.08%
    "s19_tgd": None,
    "s30_amp_cut": -1,      # 日内振幅切割 rankIC -0.067
    "s04_intraday_ret": None,
    "s04_overnight_ret": +1,   # 隔夜收益正向预测稳定
    "s04_io_diff": None,
    "s05_am_ret": None,
    "s05_pm_ret": -1,          # 分时收益预测性由正转负（下午）
    "s05_ovn_ret": +1,
    "s05_ovp": None,
    "s03_smart_vwap": +1,      # 聪明钱因子正向
}


def all_seeds():
    from min_gp.expr import parse
    return {name: parse(s) for name, s in SEEDS.items()}
