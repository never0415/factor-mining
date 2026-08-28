# 事件型因子 GP

当前事件岛覆盖两个仅依赖本地分钟数据的手册因子：

- `moderate_risk`：适度冒险；
- `wait_rescue`：待著而救。

## 运行

```powershell
python -m min_gp.event_gp --family moderate_risk --start 2018-01-02 --end 2024-12-31 --pop 60 --gens 8
python -m min_gp.event_gp --family wait_rescue --start 2018-01-02 --end 2024-12-31 --pop 60 --gens 8
```

两个因子的研报方向均为负向，`--direction paper` 会固定使用 `-1`。使用
`--direction discovery` 时，每个走样本外折叠只在训练段确定方向。

## 适度冒险模板

```text
delta(volume)
 -> intraday mean + sigma×std spike
 -> [t, ..., t+N-1] return std / event-minute return
 -> event mean
 -> cross-sectional mean distance
 -> rolling mean + rolling std
 -> equal blend
```

可进化插槽：sigma阈值、事件响应窗口、日频窗口、开收盘边缘剔除分钟数、
标准差ddof，以及距离化之前是否截面标准化。

手册摘录没有给出确切的开收盘剔除长度，也没有确认ddof，因此默认锚点使用
`exclude_edges=0, ddof=0`，并将二者明确保留为实验参数。没有核对原PDF前，
包含这些参数选择的结果应称为“适度冒险变体”，不能宣称逐点复现。

## 待著而救模板

```text
volume after 09:45
 -> daily top 10 minutes
 -> chronological suppression when gap < 5 minutes
 -> sum(volume[t+1:t+5]) / volume[t]
 -> event mean
 -> rolling mean + rolling std
```

靠近收盘、无法获得完整后续窗口的事件自动排除，不跨日补齐。

可进化插槽：Top-K、开盘排除分钟、最小事件间隔、跟随窗口和日频窗口。

## 数据和评价

数据入口使用连续240分钟流式张量。质量字段不会作为叶子。评价目标与频谱岛
一致：走样本外稳健IC、相对原始锚点的增量IC、扣成本多空收益和复杂度。
