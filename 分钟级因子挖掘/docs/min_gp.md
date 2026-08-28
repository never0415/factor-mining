# min_gp — 高频因子遗传规划挖掘系统

> 基于开源证券市场微观结构系列研报（27/30/33），GPU 加速的多目标 GP 因子挖掘

---

## 1. 研究动机

开源证券 S27/S28/S33 研报提出"峰岭谷"因子家族：将分钟级量价数据通过状态判别（峰/岭/谷）→ 日频聚合 → 滚动统计 → 截面 IC 评估，构成完整的因子挖掘管线。状态判别依赖人工定义（20日同时点滚动 μ±σ），因子组合依赖人工枚举。

**目标**：用遗传规划（GP）自动发现状态判别 + 因子组合，GPU 加速，多目标 NSGA-II 避免过拟合。

---

## 2. 数据架构

### 2.1 3D 分钟张量

```
tensor: (instrument × date × minute) × field
shape:  (691 × D × 241) × float32   # D = 交易日数, 按切片区间
```

| 字段 | 说明 |
|---|---|
| open, high, low, close, volume | 原始 OHLCV |
| tp = (O+H+L+C)/4 | 典型价格 |
| amp = H/L − 1 | 分钟振幅（**研报30定义，low=0→NaN 防 inf**） |
| ret = C/O−1 | 分钟收益 |

显存：1年 ~4.7 GiB，2年 ~10 GiB，4年训练切片 ~7.6 GB（chunked 滚动后）。

### 2.2 预计算叶子（37个）

| 类别 | 叶子 | 数量 |
|---|---|---|
| 原始 | open, high, low, close, volume, tp, amp, ret | 8 |
| S27 量能态 | is_peak, is_ridge, is_valley | 3 |
| S33 价跳态 | is_jump, is_amp_valley, is_jump_peak, is_jump_ridge, has_gap | 5 |
| 时段掩码 | mask_am/pm, 18个多尺度掩码 | 20 |
| 时间权重 | w_time | 1 |

**S27 量能态定义**（data.py，研报27 两阶段划分）：
- log1p(volume) 上做 20 日同时点滚动 μ/σ（**排除当天**，无前视）
- 喷发 E = LV > μ+1σ；**温和 = LV ≤ μ+1σ（非喷发，含中间区）**——研报"低于1σ划为温和"即同一阈值 μ+σ
- 峰 = E & 前后1分钟均温和（孤立喷发）；岭 = E & 前后1分钟存在喷发（连续喷发）；谷 = 温和

**S33 价跳态定义**（data.py，研报33 双特征 8 类）：
- 跳跃 = amp > 20日同时点 μ+1σ
- 局域情绪：前后时点振幅高低（前低后低=低迷，前高后高=高涨，混合=适中）
- 跳跃结果：前后分钟价格区间 [low,high] 是否重叠（无重叠=有缺口）
- **价峰 = 非局域情绪高涨（非双高）+ 无缺口**；**价岭 = 非局域情绪低迷（非双低）+ 有缺口**；价谷 = 非跳跃（振幅 ≤ μ+σ，含中间区）

### 2.3 CSI500 成分池

- `stock_research/zz500_component.csv`：月度快照 104 个月 × 500 只 × 权重（**非每日**，PIT 用最近快照）
- 分层等权；**权重仅用于基准线**（中证500 指数日收益）
- 停牌股（fwd NaN）**分层前排除**（因子置 NaN）

### 2.4 市值/行业数据

- `future_research/market_cap.parquet`、`download/industry_map.parquet`
- 用于中性化（`--industry/--mcap`，只降 IC 属正常，为分离风格变量）

---

## 3. 表达式系统

### 3.1 类型系统（tag-based）

```
A3 = (I, D, M)     # 分钟世界
B2 = (I×M, D)      # 分钟世界展平（用于跨日同时点滚动）
D2 = (I, D)        # 日频世界
M1 = (M,)          # 掩码/权重
SCALAR             # 常量（仅 domain 受限）
```

**关键设计**：`to_A`/`to_B` 是 A3↔B2 的唯一桥梁。`day_*` 是 A3→D2 的唯一闸门。标签显式声明，非法组合在生成阶段拒绝（TypeTagError）。

### 3.2 算子库（70 个注册算子，以 expr.py `OP` 表为准）

| 类别 | 算子 | 数量 |
|---|---|---|
| 二元算术 | add, sub, mul, div, pow | 5 |
| 比较 | gt, ge, lt, le | 4 |
| 一元 | log, log1p, sqrt, abs, sign, neg, clip | 7 |
| 条件选择 | if(cond, a, b) — torch.where 三选 | 1 |
| 状态分类 | regime(field, N, k, method) — 原生分钟状态机（0=above 1=below 2=isolated_peak 3=clustered_ridge 4=valley） | 1 |
| 日聚合 | day_sum/mean/std/min/max/last/first/ratio/skew/kurt/median/quantile | 12 |
| 日对 | day_corr | 1 |
| 滚动 | ts_mean/sum/std/min/max/delay/delta/zscore/rank/corr/cov/quantile/**ema** | 13 |
| 截面 | cs_rank, cs_zscore, cs_resid | 3 |
| 间隔统计 | day_istd, day_iskew, day_ikurt | 3 |
| 时间重心 | time_barycenter | 1 |
| 日内 | intra_mean/std/shift/cumsum/cummax | 5 |
| 转换 | to_A, to_B, bcast | 3 |
| 逻辑 | and_, or_, not_, f | 4 |
| 掩码 | mask_mul, mask_agg, mask_ratio, mask_stat | 4 |
| 事件距离 | dist_to_event | 1 |
| 跨天切割 | roll_cut | 1 |
| **间隔统计** | **interval_stat** | 1 |

**研报核心算子**：
- `mask_ratio(x, A, B, N)` = Σ_N x·A / (Σ_N x·B + 1)——峰岭成交比等（分母 +1 防 0 除爆炸）
- `mask_agg(x, mask, op)` = 单掩码日聚合（op: 0=count 1=sum 2=VWAP 3=std 4=skew 5=kurt 6=min 7=max 8=median）
- `roll_cut(x, y, N, λ)` = 跨天合并切割（S30 分钟理想振幅，N 天全部分钟合并按 y 排序切 λ）
- `interval_stat(mask, N, stat)` = 同日相邻事件间隔分布统计（stat: 1=mean 2=std 3=kurt 4=skew），**每日单独算→N日均值**（研报"同日"语义）
- `regime(field, N, k, method)` = 把验证过的峰岭谷判别固化为原生算子（深树会致 depth 预算失效 + OOM），method 覆盖 above/below/孤立峰/连续岭/谷

### 3.3 缓存策略

无表达式缓存（Ctx 无 LRU——已废弃）。`_batch_eval` 内局部 `seen={}` 同代去重；跨 run 用 `all_exprs.txt` 文件级去重。

### 3.4 算子方法论分层（概况）

70 个算子按方法论角色分七层：

| 层 | 算子 | 方法论角色 |
|---|---|---|
| ① 原料层（叶子 37 个） | 8 原始量价 + tp/amp/ret；S27 量能态 3；S33 价跳态 5；20 时段掩码 + w_time | 研报状态机预计算，GP 无需重造状态判别 |
| ② 维度桥接层 | to_A / to_B / bcast | A3⇄B2 唯一桥梁；day_* 是 A3→D2 唯一闸门 |
| ③ 量价统计层 | ts_* 13（同分钟跨日滚动）；intra_* 5；day_* 13 + day_istd/iskew/ikurt | S27 风格骨架：同分钟跨日滚动是核心 |
| ④ 研报语义算子层 | mask_ratio / mask_agg / mask_stat / interval_stat / roll_cut / dist_to_event / time_barycenter | 抽象共性计算骨架，非逐因子手写 |
| ⑤ 状态发现层 | gt/ge/lt/le + 桥接自由组合 | GP 可自动生成 is_valley 等价式：lt(volume, to_A(ts_mean(to_B(volume),20))) |
| ⑥ 截面层 | cs_rank / cs_zscore / cs_resid | 因子归一化 / 风格中性化 |
| ⑦ 胶水层 | 算术 5 + 一元 7 + 逻辑 4 + if | 通用拼接 |

**设计要点**：零常数策略（常量仅 window/q/op/stat 四 domain）；类型系统在生成期拒绝非法组合；掩码类算子参数须掩码语义（连续量当掩码 → 流动性伪信号）；`_tag_of` 静态表须与新算子同步；验证过的模式做成原生算子（regime）而非深树。

---

## 4. GA 引擎

### 4.1 进化路径

```
v1: 单目标 |IC|, tournament selection
v2: NSGA-II 三目标 (|IC|, win-rate, NDCG@20)
v3: NSGA-II 四目标 (+turnover)
v4: 批量评估 (11x 提速)
v5: 论文参数对齐 + 状态发现 (to_A/to_B)
v6: 零常数策略 + 研报算子家族 (mask_*/roll_cut/interval_stat)
```

### 4.2 NSGA-II 四目标（不设手动权重）

| 目标 | 计算方式 | 最大化方向 |
|---|---|---|
| aligned IC | 训练期按 IC 均值确定 direction=±1，再计算定向后 IC 均值 | 越大越好 |
| win-rate | direction × IC > 0 的天数占比 | 越稳定越好 |
| NDCG@20 | 按 direction 定向后的因子前20只排序质量 | 越接近1越好 |
| -turnover | 日频排序变化均值取负 | 换手越低越好 |

正负因子对称挖掘：方向只在训练期确定，并写入输出；valid/test 固定沿用训练方向，不允许重新翻转。进化期不加 IC 硬阈值，阈值只在 valid/test 筛选期使用（aligned IC > 0.05）。

### 4.3 关键机制

- **批量评估**：整代因子 stack 为 (N, I, D) → 批量 rank/IC/NDCG/turnover，_CHUNK=128 防 OOM
- **种子注入**：59 个研报因子占初始种群 30%
- **类型感知变异/交叉**：保持 tag 合法性；Const domain=None（种子常量）与 SCALAR 子树跳过变异
- **精英保留 + 去重**：按 Pareto rank + 四目标 crowding distance
- **锦标赛选择**：默认每次抽取 6 个候选
- **独立交叉/变异概率**：先按 crossover_rate 交叉，再按 mutation_rate 对子代变异

### 4.4 参数（run_mining.sh 默认）

| 参数 | 值 |
|---|---|
| 种群规模 | 2000（quick: 100） |
| 进化代数 | 10（quick: 5） |
| 交叉率 | 0.85 |
| 变异率 | 0.25 |
| 随机种子 | 8（quick: 1） |
| **period** | **以 run_mining.sh 的 `PERIOD=` 为准**（当前 5 = t+5 重叠标签，close-to-close 口径，ICIR 用 Newey-West 修正） |

**零常数策略**：`CONSTS=[]`；`gen_tree(SCALAR)`/`rand_const(domain=None)`→ValueError。常量只允许 4 个 domain：
- `window` = {5, 10, 20, 40}
- `q` = {0.25, 0.5, 0.75}
- `op` = {0..8}（mask_agg/mask_stat 聚合类型）
- `stat` = {1, 2, 3, 4}（interval_stat 统计量）

### 4.5 状态发现

`to_A`/`to_B` 管道使 GA 能自动生成跨日滚动统计状态：

```
lt(volume, to_A(ts_mean(to_B(volume), 20)))
→ "成交量低于 20 日同时点均值" ≡ is_valley 的 GP 版本
```

### 4.6 interval_stat 进池设计

- mask 参数 50% 用叶子掩码（is_peak 等稀疏事件）+ 50% 用阈值型 `gt(x, ts_mean(x, N))`
- **不能用裸 gen_tree(A3)**：全正连续值掩码（如 div(open,tp)）→ `mask.bool()` 几乎全 True → 间隔恒 1 → 因子退化

---

## 5. 数据集划分（run_mining.sh）

```
train: 2018-01-02 ~ 2021-12-31  → GA 进化
valid: 2022-01-02 ~ 2024-12-31  → 筛选因子
test:  2025-01-02 ~ 2026-07-31  → 独立评估（eval_test.py，只看一次）
```

fitness 的 forward return 随 `--period` 自动匹配（当前 t+5 重叠标签，close-to-close：fwd[t]=close[t+5]/close[t]−1），具体值以 run_mining.sh 的 `PERIOD=` 为准。

---

## 6. 回测系统

### 6.1 自定义回测（backtest.py）——信号/收益分离语义

- **信号**：调仓日 d0 用因子截面排序分层（5/10 层），持仓固定 period 天
- **收益**：层 PnL = 持仓股票**每日真实日收益** close[t+1]/close[t]−1 累积（连续 NAV，非事件式）
- **IC 方向**：用 t+period 前视收益定方向（fwd 仅用于定方向，**严禁进 PnL 累乘**——曾致 LS Sharpe 2.4 假象）
- **全局方向 flip**：全样本 IC 符号决定（2018-2019 代表切片判方向），禁逐年 flip
- **停牌股分层前排除**（因子置 NaN，防退化值挤最差层）
- 层等权；CSI500 权重仅用于基准线（红色虚线）
- fee=0.003 双边千三（调仓日换手 × fee）
- **换手报告调仓日实际换手**（turn[:, ::period]，非全天数摊薄）
- LS = L1−L{n}，**两层都有效的日子才参与**（NaN 对齐）
- 图：18×9，NAV 从 1 或累计收益率%，白底线性 y 轴，英文标签

### 6.2 Backtrader 桥接（bt_run.py）

- 因子 → 日频 OHLCV DataFrame → bt.Cerebro；截面排序策略，整数手 + 资金约束 + 百分比佣金

### 6.3 指数增强（index_enhance.py）

- 华泰 LP 优化器：`max w^T r`；个股权重偏离 ≤1%，L1 换手率 ≤δ，满仓，只做多
- z-score × IC 校准预期收益；基准线 + 增强组合双线图

---

## 7. 关键验证结果

### 7.1 研报因子实现核查（本轮完成）

**修复的 bug**：
1. 叶子别名缺失（种子用 v/l/h/c/o，数据层 volume/low/high/close/open）——14 个种子 TypeTagError 全死
2. amp = H−L → H/L−1（研报定义）
3. S30 日内切割 → 跨天合并切割（roll_cut）
4. 间隔统计跨天合并 → 同日每日算→N日均值（interval_stat）
5. 研报33 价谷 = 非跳跃（含中间区），非仅低端
6. 温和 = 非喷发（≤μ+σ）——峰率 0.11%→4.65%，**量峰间隔峰度方向反转**（−0.012→+0.044）
7. 比值类分母 +1 平滑（1e18 爆炸→正常）

**研报方向验证**（59 种子全部可评估）：
- 研报27：8/9 方向正确；研报33：全部方向正确
- 量级匹配研报（差异 = 池子 691 vs 5000+，区间 2018-2023 vs 2013-2025）

### 7.2 回测（峰岭成交比，月频 21d，10 层，千三）

| 指标 | 值 |
|---|---|
| 分段 IC | train +0.045 / valid +0.077 / test +0.092 |
| 全量 IC_mean | +0.041 |
| LS CAGR（10层） | +8.5%（Sharpe 0.68） |
| LS CAGR（5层） | +5.5%（口径换算与研报 27.13% 一致：÷1.4 分组 ÷2.5 池子 ÷1.5 区间） |
| 调仓日真实换手 | ~87%/期 |

### 7.3 平滑对比结论（SMA vs EMA）

- 窗口长度比平滑类型重要：SMA 20-40 最优，EMA 无优势
- **研报的 20 日聚合是统计窗口（稀疏事件样本量），不是时间平滑**——额外 ts_mean(40) 会把月频 IC 从 0.079 砍到 0.050
- t+1 标签 + SMA ≠ t+21 标签：时域平滑稀释截面区分度，标签端聚合才对

---

## 8. 性能基准

| 操作 | 耗时 | 显存 |
|---|---|---|
| 2年切片加载 | 12s | 10 GiB |
| 4年切片加载（chunked 滚动） | ~40s | 7.6 GB |
| 单因子评估 | <1ms（无缓存，直接 eval） | — |
| 批量 100 因子评估 | ~100ms | — |
| **GA 2000×10×8 seeds（4年数据）** | **~92min**（每 seed ~11min，~31 expr/s） | ~20 GiB |

---

## 9. CLI 速查

```bash
# 快速调试
python -m min_gp.engine --pop 100 --gens 5 --seeds 1 --period 1

# 正式挖掘（参数以 run_mining.sh 为准）
python -m min_gp.engine --start 2018-01-02 --end 2021-12-31 \
    --valid-start 2022-01-02 --valid-end 2024-12-31 \
    --pop 2000 --gens 10 --seeds 8 --period 1

# 独立测试集评估
python -m min_gp.eval_test --start 2025-01-02 --end 2026-07-31 --period 1 --exprs min_gp/output/valid_pass.txt

# 回测（5/10层，含基准线）
python -m min_gp.backtest --seed s27_peak_ridge_ratio --full --period 21 --layers 10 \
    --pool-csv stock_research/zz500_component.csv --fee 0.003

# 种子批量 IC 核查
python check_seeds_ic.py
```

---

## 10. 文件索引

| 文件 | 功能 |
|---|---|
| `min_gp/data.py` | 3D 张量构建 + 叶子（峰岭谷状态机）+ 池子/行业/市值加载 |
| `min_gp/expr.py` | 表达式 AST + tag 类型系统 + 69 算子 |
| `min_gp/fitness.py` | Spearman IC + 多目标 + 批量评估（_CHUNK=128） |
| `min_gp/engine.py` | NSGA-II GA + gen_tree + mutate/crossover + 常量域 |
| `min_gp/seeds.py` | 59 个研报因子模板 |
| `min_gp/backtest.py` | 分层回测（信号/收益分离、全局 flip、停牌排除、调仓日换手） |
| `min_gp/bt_run.py` | Backtrader 桥接 |
| `min_gp/index_enhance.py` | 华泰指数增强 LP 优化器 |
| `min_gp/eval_test.py` | 独立测试集评估 |
| `run_mining.sh` | 全流程挖掘脚本（自动 tee 日志到 output/mining_*.log） |
| `check_seeds_ic.py` | 全部种子 train 段月频 IC/ICIR 批量核查 |
