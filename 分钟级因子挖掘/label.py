"""
统一收益率标签规格 (声明式, 挖掘/回测共用).

标签口径:  fwd[t] = exit[t+period] / entry[t+entry_off] - 1
  - 信号在 t 日收盘后确定 (day_last 等因子依赖当日收盘)
  - entry:  入场价列 (close=收盘, open=09:30 竞价开盘)
  - entry_off: 入场价相对信号日 t 的偏移
      close 入场 (收盘前几分钟执行): entry_off=0, 成交价≈close (滑点 ~0.05%)
      open 入场 (次日开盘竞价):      entry_off=1, 集合竞价价精确可复制
  - exit:   出场价列; exit[t+period] = t+period 日出场价
  - 重叠窗口: 逐日 t 构造, 相邻标签共享 period-1 天收益
    (IC 均值无偏 + 样本多; 自相关致 ICIR 虚高 → 显著性报告用 Newey-West 修正)

改标签口径只改 LABEL 一行, 挖掘 (engine/eval_test) 与回测
(bt_common/backtest_portfolio) 全部跟随, 不散落硬编码.

常用口径:
  close-to-close: LABEL = LabelSpec()  (默认)
      fwd[t] = close[t+p]/close[t]-1   (t 收盘前执行, 资金无缝换仓)
  open-to-open:   entry="open", exit="open", entry_off=1
      fwd[t] = open[t+p]/open[t+1]-1   (开盘竞价精确成交, 资金闲置 1 天)
  open-to-close:  entry="open", entry_off=1
      fwd[t] = close[t+p]/open[t+1]-1  (t+1 开盘买入, t+p 收盘卖出)
"""
from dataclasses import dataclass
from datetime import date

import torch


@dataclass(frozen=True)
class LabelSpec:
    entry: str = "close"     # 入场价列 (tens 键 / daily 列)
    exit: str = "close"      # 出场价列
    entry_off: int = 0       # 入场价相对信号日 t 的偏移 (close=0, open=+1)
    period: int = 5          # 持有期 (交易日)
    overlap: bool = True     # 逐日重叠窗口


LABEL = LabelSpec()          # close-to-close (收盘前几分钟执行, 资金无缝)


def tensor_open_d(tens):
    """日开盘 (I, D) = 每日第一个有效分钟 open (09:30 竞价开盘; 该分钟缺失顺延).
    兼容 tens dict (值=张量) 或 leaves dict (值=(张量, tag)); 全天无效 → 0.0
    (与 _day_last 停牌语义一致, 后续 pool_mask 过滤因子, 不进 IC)."""
    v = tens["open"]
    x = v[0] if isinstance(v, tuple) else v
    valid = ~torch.isnan(x)
    idx = valid.to(x.dtype).argmax(2)   # 第一个 True 的位置; 无有效 → 0
    return torch.where(valid.any(2), x.gather(2, idx.unsqueeze(2)).squeeze(2),
                       torch.tensor(0.0, device=x.device, dtype=x.dtype))


def tensor_fwd_ret(entry_d, exit_d, period=LABEL.period, entry_off=LABEL.entry_off):
    """GPU (I, D) 重叠标签: fwd[t] = exit[t+period]/entry[t+entry_off]-1.
    t 有效范围 0..D-period-1 (要求 period >= entry_off); 停牌/缺失 → NaN/±inf,
    由调用方 pool_mask + IC valid 对过滤."""
    D = entry_d.shape[1]
    fwd = torch.full_like(entry_d, float("nan"))
    fwd[:, : D - period] = (exit_d[:, period:]
                            / entry_d[:, entry_off : D - period + entry_off] - 1.0)
    return fwd


def week_end_mask(dates, device=None) -> torch.Tensor:
    """Mark the last available trading day in each ISO calendar week."""
    parsed = [date.fromisoformat(str(value)[:10]) for value in dates]
    mask = []
    for index, value in enumerate(parsed):
        current = value.isocalendar()[:2]
        following = (
            parsed[index + 1].isocalendar()[:2]
            if index + 1 < len(parsed) else None
        )
        mask.append(following != current)
    return torch.tensor(mask, dtype=torch.bool, device=device)


def tensor_weekly_fwd_ret(close_d, dates):
    """Close-to-close return from one calendar-week end to the next.

    Values outside rebalance dates, and the final rebalance date without a
    subsequent exit, remain NaN.  Holiday-shortened weeks therefore rebalance
    on their actual last trading day rather than on a hard-coded weekday.
    """
    if close_d.ndim != 2 or close_d.shape[1] != len(dates):
        raise ValueError("close_d columns must align one-for-one with dates")
    mask = week_end_mask(dates, close_d.device)
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    result = torch.full_like(close_d, float("nan"))
    if indices.numel() < 2:
        return result
    entry, exit_ = indices[:-1], indices[1:]
    result[:, entry] = close_d[:, exit_] / close_d[:, entry] - 1.0
    return result


def tensor_rebalance_fwd_ret(close_d, dates, rule="week_end", period=1):
    """Build labels for the selected portfolio rebalance convention."""
    if rule == "week_end":
        return tensor_weekly_fwd_ret(close_d, dates)
    if rule == "daily":
        return tensor_fwd_ret(close_d, close_d, period=period)
    raise ValueError(f"unknown rebalance rule: {rule}")


def frame_fwd_ret(daily, horizon=LABEL.period, entry=LABEL.entry, exit=LABEL.exit,
                  entry_off=LABEL.entry_off, clip=0.11):
    """pandas 版 (回测 IC 标签, 按 instrument 分组):
    fwd_ret[t] = exit[t+horizon]/entry[t+entry_off]-1. daily 需含 entry/exit 列."""
    d = daily.sort_values(["instrument", "trade_date"]).copy()
    en = d.groupby("instrument")[entry].shift(-entry_off)
    ex = d.groupby("instrument")[exit].shift(-horizon)
    d["fwd_ret"] = ex / en - 1.0
    d["fwd_ret"] = d["fwd_ret"].clip(-clip, clip)
    return d


def frame_entry_ret(daily, entry="open", exit="close", clip=0.11):
    """入场日收益 (回测分层复利, open 入场用): entry_ret[t] = close[t+1]/open[t+1]-1
    (t 收盘信号 → t+1 开盘买入 → 当日收盘). 其余持有日仍 T+1 close-to-close."""
    d = daily.sort_values(["instrument", "trade_date"]).copy()
    en = d.groupby("instrument")[entry].shift(-1)
    ex = d.groupby("instrument")[exit].shift(-1)
    d["entry_ret"] = ex / en - 1.0
    d["entry_ret"] = d["entry_ret"].clip(-clip, clip)
    return d


def frame_exit_ret(daily, entry="open", exit="close", clip=0.11):
    """出场日收益 (回测分层复利, open-to-open 用): exit_ret[t] = open[t+1]/close[t]-1
    (持仓最后一天 t 收盘 → t+1 开盘卖出, 隔夜收益)."""
    d = daily.sort_values(["instrument", "trade_date"]).copy()
    en = d[exit]                                          # close[t]
    ex = d.groupby("instrument")[entry].shift(-1)         # open[t+1]
    d["exit_ret"] = ex / en - 1.0
    d["exit_ret"] = d["exit_ret"].clip(-clip, clip)
    return d
