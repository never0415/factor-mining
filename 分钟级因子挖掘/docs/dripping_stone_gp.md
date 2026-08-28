# 滴水穿石强类型 GP

## 入口

```powershell
.\.venv\Scripts\Activate.ps1
python -m min_gp.dripping_gp --start 2018-01-02 --end 2024-12-31 --pop 80 --gens 8 --direction paper
```

默认结果写入 `分钟级因子挖掘/output/dripping_stone_gp.jsonl`。每行包含：

- Pareto 等级；
- 走样本外稳健 Rank IC；
- 对原始滴水穿石残差化后的增量 IC；
- 扣除目标权重换手成本后的多空日收益；
- 复杂度、折叠方向一致率和覆盖率；
- 可复现的结构参数和表达式。

## 原始锚点

原始锚点固定为：

```text
volume
 -> median ± 3×IQR clip
 -> demean
 -> Hann window
 -> rFFT power
 -> period 2–5 minute band power / non-DC total power
```

日频平滑窗口在现有复现手册中没有闭式定义，因此原始锚点不带平滑。
任何带 MA/EMA 的结果都标记为改进变体。

## 可进化插槽

- 成交量变换：raw / log1p / sqrt；
- IQR 倍数：1.5 / 2 / 3 / 4；
- 每日最小分钟覆盖率：90% / 95% / 98% / 100%；
- 去趋势：去均值 / 线性去趋势；
- 窗函数：Hann / Hamming / none；
- 周期频带：2–3、2–4、2–5、3–5、3–8、5–10分钟；
- 时段：全天 / 上午 / 下午；
- 平滑：none / MA / EMA，窗口5/10/20/40日。

频带与平滑作为原子语义插槽参与交叉，避免产生下界高于上界或关闭平滑却
携带无效窗口等结构。

## 数据口径

频谱入口只读取 `trade_date/instrument/datetime/volume`，构建真实连续的240分钟
序列：09:30–11:29和13:00–14:59。它不会使用旧网格中不存在的11:30空位置，
也不会构建S27/S33叶子。多年分钟数据通过 Arrow record batch 流式扫描并直接
scatter 到目标张量，不会先在主机内存中生成完整明细表。

分钟缺失处理规则：

1. 每日覆盖率低于模板阈值时整日因子为NaN；
2. 达到阈值的短缺口只用当日中位数填充；
3. 不跨日填充；
4. 常量成交量导致总非零频率功率为0，整日因子为NaN。

已有 `stage1/ds_raw_*` 可以通过 `load_daily_factor_leaves` 对齐加载。`status`
只用于有效性过滤，不会暴露为GP叶子。

## 方向模式

- `--direction paper`：固定滴水穿石为正向；
- `--direction discovery`：每个走样本外折叠仅用训练段确定方向，验证段不翻转。

正式挖掘建议先使用 `paper` 模式验证是否真正改进原始因子，再单独运行
`discovery` 模式探索反向结构。

## 外部数据

当前频谱岛只依赖本地分钟成交量、日线收盘价和PIT成分股，不需要外部API。
未来若其他因子缺少字段，外部服务凭证必须由环境变量或本机密钥存储提供，
不得写入源码、命令行样例、日志或实验结果。
