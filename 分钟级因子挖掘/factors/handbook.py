"""Reproducible handbook factor anchors beyond the original three families.

Definitions marked incomplete in the source handbook expose the missing value
as a required input instead of silently inventing it.
"""

from dataclasses import dataclass

import torch

from min_gp.operators.cross_section import cross_section_distance, cross_section_rank
from min_gp.operators.distribution import boxcox_grid_mle, relative_volume_entropy
from min_gp.operators.event import (
    forward_window_std, intraday_sigma_event, masked_daily_mean,
    minute_delta_volume, minute_return, relative_volume_event,
)
from min_gp.operators.path import sort_cumulative_difference, tide_speed_components
from min_gp.operators.regression import conditional_covariance, minute_ols_statistics
from min_gp.operators.temporal import equal_blend, mean_std_blend, smooth_daily, rolling_daily_std


class IncompleteDefinitionError(ValueError):
    pass


def _first(x):
    valid = torch.isfinite(x)
    index = valid.float().argmax(-1)
    value = x.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid.any(-1), value, torch.full_like(value, float("nan")))


def _last(x):
    valid = torch.isfinite(x)
    index = x.shape[-1] - 1 - valid.flip(-1).float().argmax(-1)
    value = x.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid.any(-1), value, torch.full_like(value, float("nan")))


def _daily_return(open_, close):
    return _last(close) / _first(open_).clamp(min=1e-12) - 1


@dataclass(frozen=True)
class CompleteTideTemplate:
    neighborhood: int = 9
    exclude_edges: int = 15
    smooth_window: int = 20
    direction: int = -1
    required_fields = ("close", "volume")

    def evaluate(self, close, volume):
        _, strong, weak = tide_speed_components(
            close, volume, self.neighborhood, self.exclude_edges
        )
        return equal_blend(
            smooth_daily(strong, "mean", self.smooth_window),
            rolling_daily_std(weak, self.smooth_window),
        )


@dataclass(frozen=True)
class ClimbMountainTemplate:
    window: int = 5
    smooth_window: int = 20
    direction: int = 1
    required_fields = ("open", "high", "low", "close")

    def evaluate(self, open_, high, low, close):
        prices = torch.stack((open_, high, low, close), dim=-1).float()
        out = torch.full_like(close.float(), float("nan"))
        if close.shape[-1] >= self.window:
            win = prices.unfold(2, self.window, 1)
            # unfold gives (..., 4, window): flatten the 20 price observations.
            flat = win.transpose(-1, -2).flatten(-2)
            good = torch.isfinite(flat).all(-1)
            ov = (flat.std(-1, unbiased=False) / flat.mean(-1).clamp(min=1e-12)) ** 2
            out[..., self.window - 1:] = torch.where(good, ov, torch.full_like(ov, float("nan")))
        ret = minute_return(close, 1)
        rvr = ret / out.clamp(min=1e-12)
        valid = torch.isfinite(out)
        count = valid.sum(-1, keepdim=True).clamp(min=1)
        mean = out.nan_to_num().sum(-1, keepdim=True) / count
        var = (((out.nan_to_num() - mean) ** 2) * valid).sum(-1, keepdim=True) / count
        mask = out >= mean + torch.sqrt(var.clamp(min=0))
        raw = conditional_covariance(out, rvr, mask, min_count=3)
        return mean_std_blend(raw, self.smooth_window)


def _rolling_market_abs_corr(x, window=20):
    instruments, days = x.shape
    out = torch.full_like(x.float(), float("nan"))
    for day in range(window - 1, days):
        block = x[:, day-window+1:day+1].float()
        complete = torch.isfinite(block).all(1)
        if int(complete.sum()) < 2:
            continue
        z = block[complete]
        z = (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True, unbiased=False).clamp(min=1e-8)
        corr = (z @ z.T) / window
        value = (corr.abs().sum(1) - 1) / (corr.shape[0] - 1)
        out[complete, day] = value
    return out


@dataclass(frozen=True)
class HiddenFlowerTemplate:
    lags: int = 5
    smooth_window: int = 20
    align_component_directions: bool = True
    direction: int = -1
    required_fields = ("close", "volume")

    def evaluate(self, close, volume):
        ret = minute_return(close, 1)
        delta = minute_delta_volume(volume, 1)
        tstats, f_all = minute_ols_statistics(ret, delta, self.lags)
        intercept = tstats[..., 0]
        morning = tstats[..., 2:].std(-1, unbiased=False)
        f_mean = f_all.nan_to_num().sum(0, keepdim=True) / torch.isfinite(f_all).sum(0, keepdim=True).clamp(min=1)
        noon = torch.where(f_all < f_mean, -intercept.abs(), intercept.abs())
        night = _rolling_market_abs_corr(intercept, self.smooth_window)
        components = [
            smooth_daily(morning, "mean", self.smooth_window),
            smooth_daily(noon, "mean", self.smooth_window),
            -night if self.align_component_directions else night,
        ]
        ranked = [cross_section_rank(value) for value in components]
        return sum(ranked) / len(ranked)


@dataclass(frozen=True)
class LongShortBattleTemplate:
    return_window: int = 5
    smooth_window: int = 20
    direction: int = -1
    required_fields = ("high", "low", "close", "volume")

    def evaluate(self, high, low, close, volume):
        q = torch.full_like(close.float(), float("nan"))
        q[..., self.return_window:] = close[..., self.return_window:] / close[..., :-self.return_window].clamp(min=1e-12) - 1
        b_vr = sort_cumulative_difference(volume.float(), q)
        volume_return = mean_std_blend(cross_section_distance(b_vr, True), self.smooth_window)
        prior_low = torch.full_like(close.float(), float("nan"))
        prior_high = torch.full_like(close.float(), float("nan"))
        if close.shape[-1] > self.return_window:
            low_win = low[..., :-1].unfold(-1, self.return_window, 1)
            high_win = high[..., :-1].unfold(-1, self.return_window, 1)
            prior_low[..., self.return_window:] = low_win.min(-1).values
            prior_high[..., self.return_window:] = high_win.max(-1).values
        position = .5 * (
            close / prior_low.clamp(min=1e-12) - 1
            + (prior_high - close) / prior_high.clamp(min=1e-12)
        )
        b_vp = sort_cumulative_difference(volume.float(), position)
        volume_position = mean_std_blend(cross_section_distance(b_vp), self.smooth_window)
        volume_battle = equal_blend(volume_return, volume_position)
        amplitude = (high - low) / close.clamp(min=1e-12)
        b_amp = sort_cumulative_difference(amplitude, q)
        amp_battle = mean_std_blend(cross_section_distance(b_amp), self.smooth_window)
        return equal_blend(volume_battle, amp_battle)


@dataclass(frozen=True)
class EqualTreatmentTemplate:
    response_window: int = 5
    exclude_edges: int = 15
    smooth_window: int = 20
    direction: int = -1
    required_fields = ("open", "close", "volume")

    def evaluate(self, open_, close, volume):
        transformed = boxcox_grid_mle(volume)
        delta = minute_delta_volume(transformed, 1)
        spike = intraday_sigma_event(delta, 1., "above", self.exclude_edges, 0)
        drop = intraday_sigma_event(delta, 1., "below", self.exclude_edges, 0)
        ret = minute_return(close, 1)
        vol = forward_window_std(ret, self.response_window, 0)
        fair_vol = (masked_daily_mean(vol, spike) - masked_daily_mean(vol, drop)).abs()
        fair_ret = (masked_daily_mean(ret, spike) - masked_daily_mean(ret, drop)).abs()
        day_ret = _daily_return(open_, close)
        a = smooth_daily(day_ret * fair_vol, "mean", self.smooth_window)
        b = smooth_daily(day_ret * fair_ret, "mean", self.smooth_window)
        return equal_blend(a, b)


@dataclass(frozen=True)
class DarkFlowTemplate:
    bins: int = 48
    lookback: int = 5
    multiple: float = 1.0
    smooth_window: int = 20
    direction: int = -1
    required_fields = ("open", "high", "low", "volume")

    def evaluate(self, open_, high, low, volume):
        entropy = relative_volume_entropy(volume, self.bins)
        entropy = mean_std_blend(cross_section_distance(entropy), self.smooth_window)
        spike = relative_volume_event(volume, self.lookback, self.multiple, 0)
        amplitude = (high - low) / open_.clamp(min=1e-12)
        spike_mean = masked_daily_mean(amplitude, spike)
        normal_mean = masked_daily_mean(amplitude, ~spike)
        elasticity = 1 - spike_mean / normal_mean.clamp(min=1e-12)
        elasticity = mean_std_blend(cross_section_distance(elasticity), self.smooth_window)
        return equal_blend(entropy, elasticity)


@dataclass(frozen=True)
class RawPanicTemplate:
    smooth_window: int = 20
    intraday_volatility: bool = False
    direction: int = -1
    required_fields = ("close",)

    def evaluate(self, daily_close, market_close, minute_close=None):
        stock_ret = torch.full_like(daily_close.float(), float("nan"))
        market_ret = torch.full_like(market_close.float(), float("nan"))
        stock_ret[:, 1:] = daily_close[:, 1:] / daily_close[:, :-1].clamp(min=1e-12) - 1
        market_ret[1:] = market_close[1:] / market_close[:-1].clamp(min=1e-12) - 1
        panic = (stock_ret - market_ret.unsqueeze(0)).abs() / (
            stock_ret.abs() + market_ret.abs().unsqueeze(0) + .1
        )
        weighted = panic * stock_ret
        if self.intraday_volatility:
            if minute_close is None:
                raise IncompleteDefinitionError("minute_close is required for volatility panic")
            weighted = weighted * minute_return(minute_close, 1).std(-1, unbiased=False)
        return equal_blend(
            smooth_daily(weighted, "mean", self.smooth_window),
            rolling_daily_std(weighted, self.smooth_window),
        )


@dataclass(frozen=True)
class RushingForwardTemplate:
    smooth_window: int = 20
    direction: int = 1
    required_fields = ("amount", "volume")

    def evaluate(self, amount_share, volume_share, up_volume_down_price_mask):
        """Strict interface: all three ambiguous report definitions are inputs."""
        if amount_share is None or volume_share is None or up_volume_down_price_mask is None:
            raise IncompleteDefinitionError(
                "amount/volume shares and the report-defined trend mask are required"
            )
        difference = amount_share.float() - volume_share.float()
        valid = torch.isfinite(difference)
        selected = valid & up_volume_down_price_mask.bool()
        raw = torch.where(
            selected, difference, torch.zeros_like(difference)
        ).sum(-1)
        raw = torch.where(
            valid.any(-1), raw, torch.full_like(raw, float("nan"))
        )
        return smooth_daily(raw, "mean", self.smooth_window)


@dataclass(frozen=True)
class WaterBoatTemplate:
    direction: int = -1
    required_fields = ("amount", "float_market_cap")

    def evaluate(self, high_amount, low_amount, float_market_cap):
        if high_amount is None or low_amount is None or float_market_cap is None:
            raise IncompleteDefinitionError(
                "report-defined amount partitions and PIT float market cap are required"
            )
        return (high_amount - low_amount) / float_market_cap.clamp(min=1e-12)


@dataclass(frozen=True)
class CooperationEffectTemplate:
    peer_count: int = 30
    smooth_window: int = 20
    direction: int = -1
    required_fields = ("full_market_ohlcv",)

    def evaluate(self, volume_share, price_state, daily_return, pair_similarity):
        """Evaluate once the report-specific share and fallback score are supplied.

        ``pair_similarity`` is (I,I,D), after applying the source's two-level
        zero-sign fallback. Keeping that ambiguous construction outside this
        method prevents an approximate rule from being mistaken for the paper.
        """
        if any(value is None for value in (volume_share, price_state, daily_return, pair_similarity)):
            raise IncompleteDefinitionError(
                "volume-share denominator and fallback-adjusted pair similarity are required"
            )
        instruments, days, minutes = volume_share.shape
        coop = torch.zeros_like(volume_share.float())
        for state in (0, 1, 2):
            mask = price_state == state
            total = torch.where(mask, volume_share, torch.zeros_like(volume_share)).sum(0)
            coop = torch.where(mask, total.unsqueeze(0), coop)
        own = volume_share - volume_share.mean(-1, keepdim=True)
        peer = coop - coop.mean(-1, keepdim=True)
        corr = (own * peer).sum(-1) / torch.sqrt(
            (own.square().sum(-1) * peer.square().sum(-1)).clamp(min=1e-12)
        )
        volume_component = mean_std_blend(corr, self.smooth_window)
        spread = torch.full_like(daily_return.float(), float("nan"))
        eye = torch.eye(instruments, dtype=torch.bool, device=volume_share.device)
        for day in range(days):
            score = pair_similarity[:, :, day].masked_fill(eye, float("-inf"))
            peers = score.topk(min(self.peer_count, instruments - 1), dim=1).indices
            peer_ret = daily_return[:, day][peers]
            spread[:, day] = daily_return[:, day] - peer_ret.mean(1)
        spread_component = mean_std_blend(spread, self.smooth_window)
        return equal_blend(volume_component, spread_component)


HANDBOOK_FACTORS = {
    "complete_tide": CompleteTideTemplate,
    "climb_mountain": ClimbMountainTemplate,
    "hidden_flower": HiddenFlowerTemplate,
    "long_short_battle": LongShortBattleTemplate,
    "equal_treatment": EqualTreatmentTemplate,
    "dark_flow": DarkFlowTemplate,
    "raw_panic": RawPanicTemplate,
    "rushing_forward": RushingForwardTemplate,
    "water_boat": WaterBoatTemplate,
    "cooperation_effect": CooperationEffectTemplate,
}

LOCAL_MINUTE_FACTORS = (
    "complete_tide", "climb_mountain", "hidden_flower",
    "long_short_battle", "equal_treatment", "dark_flow",
)


def evaluate_local_minute_factor(name, minute_tensors, **template_params):
    """Uniform dispatcher for factors strictly supported by local OHLCV."""
    if name not in LOCAL_MINUTE_FACTORS:
        raise ValueError(f"{name!r} is not a strict local-minute factor")
    template = HANDBOOK_FACTORS[name](**template_params)
    arguments = {
        "complete_tide": ("close", "volume"),
        "climb_mountain": ("open", "high", "low", "close"),
        "hidden_flower": ("close", "volume"),
        "long_short_battle": ("high", "low", "close", "volume"),
        "equal_treatment": ("open", "close", "volume"),
        "dark_flow": ("open", "high", "low", "volume"),
    }[name]
    missing = set(arguments) - set(minute_tensors)
    if missing:
        raise ValueError(f"{name} requires missing fields {sorted(missing)}")
    return template.evaluate(*(minute_tensors[field] for field in arguments))
