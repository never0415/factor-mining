"""Fine-grained reusable parts for the eight remaining handbook factors."""

import torch

from min_gp.dsl import CostCalibration, OperatorRegistry, OperatorSpec, SemanticType
from min_gp.numeric.ranking import cross_section_rank
from min_gp.operators.event import minute_return
from min_gp.operators.path import sort_cumulative_difference
from min_gp.operators.regression import minute_ols_statistics


def close_minute_return(close, horizon=1):
    return minute_return(close, horizon)


def ols_bundle(ret, delta_volume, lags=5):
    return minute_ols_statistics(ret, delta_volume, lags)


def ols_intercept(bundle):
    return bundle[0][..., 0]


def ols_lag_dispersion(bundle):
    return bundle[0][..., 2:].std(-1, unbiased=False)


def ols_conditional_abs(intercept, bundle):
    f_all = bundle[1]
    valid = torch.isfinite(f_all)
    mean = f_all.nan_to_num().sum(0, keepdim=True) / valid.sum(0, keepdim=True).clamp(min=1)
    return torch.where(f_all < mean, -intercept.abs(), intercept.abs())


def rolling_market_abs_corr(x, window=20):
    instruments, days = x.shape
    out = torch.full_like(x.float(), float("nan"))
    for day in range(window - 1, days):
        block = x[:, day-window+1:day+1].float()
        complete = torch.isfinite(block).all(1)
        if int(complete.sum()) < 2:
            continue
        z = block[complete]
        z = (z-z.mean(1,keepdim=True))/z.std(1,keepdim=True,unbiased=False).clamp(min=1e-8)
        corr = (z @ z.T) / window
        out[complete, day] = (corr.abs().sum(1)-1)/(corr.shape[0]-1)
    return out


def signed_raw(x, sign=-1):
    return float(sign) * x


def rank_daily_factor(x):
    return cross_section_rank(x.float())


def blend_three(a, b, c):
    return (a + b + c) / 3


def window_return(close, window=5):
    out = torch.full_like(close.float(), float("nan"))
    out[..., window:] = close[..., window:] / close[..., :-window].clamp(min=1e-12) - 1
    return out


def prior_range_position(low, high, close, window=5):
    prior_low, prior_high = torch.full_like(close.float(), float("nan")), torch.full_like(close.float(), float("nan"))
    if close.shape[-1] > window:
        prior_low[..., window:] = low[..., :-1].unfold(-1, window, 1).min(-1).values
        prior_high[..., window:] = high[..., :-1].unfold(-1, window, 1).max(-1).values
    return .5 * (close/prior_low.clamp(min=1e-12)-1 + (prior_high-close)/prior_high.clamp(min=1e-12))


def close_amplitude(high, low, close):
    return (high-low)/close.clamp(min=1e-12)


def open_amplitude(open_, high, low):
    return (high-low)/open_.clamp(min=1e-12)


def daily_open_close_return(open_, close):
    def boundary(x, first):
        valid = torch.isfinite(x)
        index = valid.float().argmax(-1) if first else x.shape[-1]-1-valid.flip(-1).float().argmax(-1)
        value = x.gather(-1,index.unsqueeze(-1)).squeeze(-1)
        return torch.where(valid.any(-1),value,torch.full_like(value,float("nan")))
    return boundary(close,False)/boundary(open_,True).clamp(min=1e-12)-1


def raw_abs_difference(a, b):
    return (a-b).abs()


def raw_multiply(a, b):
    return a*b


def inverse_event(mask):
    return ~mask.bool()


def liquidity_elasticity(spike, normal):
    return 1-spike/normal.clamp(min=1e-12)


def raw_panic_weight(daily_close, market_close):
    stock = torch.full_like(daily_close.float(),float("nan")); market = torch.full_like(market_close.float(),float("nan"))
    stock[:,1:] = daily_close[:,1:]/daily_close[:,:-1].clamp(min=1e-12)-1
    market[1:] = market_close[1:]/market_close[:-1].clamp(min=1e-12)-1
    panic = (stock-market.unsqueeze(0)).abs()/(stock.abs()+market.abs().unsqueeze(0)+.1)
    return panic*stock


def intraday_return_volatility(close):
    return minute_return(close,1).std(-1,unbiased=False)


def rushing_imbalance(amount_share, volume_share, trend_mask):
    difference = amount_share.float() - volume_share.float()
    valid = torch.isfinite(difference)
    selected = valid & trend_mask.bool()
    value = torch.where(
        selected, difference, torch.zeros_like(difference)
    ).sum(-1)
    any_valid = valid.any(-1)
    return torch.where(
        any_valid, value, torch.full_like(value, float("nan"))
    )


def amount_spread(high_amount, low_amount):
    return high_amount-low_amount


def cap_scale(spread, cap):
    return spread/cap.clamp(min=1e-12)


def state_peer_volume(volume_share, price_state):
    result = torch.zeros_like(volume_share.float())
    for state in (0,1,2):
        mask = price_state == state
        total = torch.where(mask,volume_share,torch.zeros_like(volume_share)).sum(0)
        result = torch.where(mask,total.unsqueeze(0),result)
    return result


def minute_path_correlation(own_volume, peer_volume):
    own = own_volume-own_volume.mean(-1,keepdim=True)
    peer = peer_volume-peer_volume.mean(-1,keepdim=True)
    return (own*peer).sum(-1)/torch.sqrt((own.square().sum(-1)*peer.square().sum(-1)).clamp(min=1e-12))


def peer_return_spread(daily_return, pair_similarity, peer_count=30):
    instruments, days = daily_return.shape
    out = torch.full_like(daily_return.float(),float("nan"))
    eye = torch.eye(instruments,dtype=torch.bool,device=daily_return.device)
    for day in range(days):
        score = pair_similarity[:,:,day].masked_fill(eye,float("-inf"))
        peers = score.topk(min(peer_count,instruments-1),dim=1).indices
        out[:,day] = daily_return[:,day]-daily_return[:,day][peers].mean(1)
    return out


def register_handbook_part_operators(registry: OperatorRegistry):
    t=SemanticType; raw=t.DAILY_RAW_FACTOR; factor=t.DAILY_FACTOR
    registry.register(OperatorSpec("minute_ols_bundle",(t.MINUTE_RETURN,t.MINUTE_SIGNAL),t.OLS_STATISTICS,ols_bundle,
        parameter_domains={"lags":(3,5,10)},cost=14,complexity={"I":1,"D":1,"M":1,"P":2},
        calibration=CostCalibration(
            reference_shape={"I":150,"D":120,"M":240,"P":7},
            seconds=29.296,peak_bytes=None,device="NVIDIA GeForce GTX 1070",
            source="local benchmark, 2026-08-20",parameter_values={"lags":5},
        )))
    registry.register(OperatorSpec("ols_intercept",(t.OLS_STATISTICS,),raw,ols_intercept))
    registry.register(OperatorSpec("ols_lag_dispersion",(t.OLS_STATISTICS,),raw,ols_lag_dispersion))
    registry.register(OperatorSpec("ols_conditional_abs",(raw,t.OLS_STATISTICS),raw,ols_conditional_abs,
        needs_full_cross_section=True))
    registry.register(OperatorSpec("rolling_market_abs_corr",(raw,),raw,rolling_market_abs_corr,
        parameter_domains={"window":(5,10,20,40)},needs_full_cross_section=True,needs_history=True,
        history_days=lambda p:p["window"]-1,complexity={"I":2,"D":1}))
    registry.register(OperatorSpec("signed_raw",(raw,),raw,signed_raw,parameter_domains={"sign":(-1,1)}))
    registry.register(OperatorSpec("rank_daily_factor",(factor,),factor,rank_daily_factor,needs_full_cross_section=True))
    registry.register(OperatorSpec("blend_three_factors",(factor,factor,factor),factor,blend_three))

    registry.register(OperatorSpec("window_return",(t.MINUTE_CLOSE,),t.MINUTE_RETURN,window_return,
        parameter_domains={"window":(3,5,10)}))
    registry.register(OperatorSpec("prior_range_position",(t.MINUTE_LOW,t.MINUTE_HIGH,t.MINUTE_CLOSE),t.MINUTE_RETURN,
        prior_range_position,parameter_domains={"window":(3,5,10)}))
    registry.register(OperatorSpec("close_amplitude",(t.MINUTE_HIGH,t.MINUTE_LOW,t.MINUTE_CLOSE),t.MINUTE_RETURN,close_amplitude))
    registry.register(OperatorSpec("sort_cumulative_difference_signal",(t.MINUTE_RETURN,t.MINUTE_RETURN),raw,
        sort_cumulative_difference,cost=4))
    registry.register(OperatorSpec("open_amplitude",(t.MINUTE_OPEN,t.MINUTE_HIGH,t.MINUTE_LOW),t.MINUTE_SIGNAL,open_amplitude))
    registry.register(OperatorSpec("daily_open_close_return",(t.MINUTE_OPEN,t.MINUTE_CLOSE),raw,daily_open_close_return))
    registry.register(OperatorSpec("raw_abs_difference",(raw,raw),raw,raw_abs_difference))
    registry.register(OperatorSpec("raw_multiply",(raw,raw),raw,raw_multiply))
    registry.register(OperatorSpec("inverse_event",(t.MINUTE_MASK,),t.MINUTE_MASK,inverse_event))
    registry.register(OperatorSpec("liquidity_elasticity",(raw,raw),raw,liquidity_elasticity))

    registry.register(OperatorSpec("raw_panic_weight",(t.DAILY_PRICE,t.MARKET_DAILY_PRICE),raw,raw_panic_weight,
        needs_history=True,history_days=1))
    registry.register(OperatorSpec("intraday_return_volatility",(t.MINUTE_CLOSE,),raw,intraday_return_volatility))
    registry.register(OperatorSpec("rushing_imbalance",(t.MINUTE_AMOUNT_SHARE,t.MINUTE_VOLUME_SHARE,t.MINUTE_MASK),raw,rushing_imbalance))
    registry.register(OperatorSpec("amount_spread",(t.MINUTE_HIGH_AMOUNT,t.MINUTE_LOW_AMOUNT),t.DAILY_ACTIVITY,amount_spread))
    registry.register(OperatorSpec("cap_scale",(t.DAILY_ACTIVITY,t.DAILY_FLOAT_MARKET_CAP),raw,cap_scale))
    registry.register(OperatorSpec("state_peer_volume",(t.MINUTE_VOLUME_SHARE,t.MINUTE_PRICE_STATE),t.MINUTE_VOLUME_SHARE,
        state_peer_volume,needs_full_cross_section=True))
    registry.register(OperatorSpec("minute_path_correlation",(t.MINUTE_VOLUME_SHARE,t.MINUTE_VOLUME_SHARE),raw,minute_path_correlation))
    registry.register(OperatorSpec("peer_return_spread",(t.DAILY_RETURN,t.PAIR_SIMILARITY),raw,peer_return_spread,
        parameter_domains={"peer_count":(10,20,30,50)},needs_full_cross_section=True))
