# 提高因子表现：六项改动

> 2026-08-26。起因是 `rushing_forward_strict_full_20260826` 跑出 `passed_count = 0`。
> 本文按需求条目编号，逐条记录改了什么、为什么、实测结果。

---

## 0. 诊断依据

| 现象 | 实测 |
|---|---|
| 门槛通过数 | 0 / 8 |
| 8 个候选的 RankIC（样本/验证/测试中位数） | 0.0289 / 0.0512 / 0.0226 |
| 同一批的 Pearson IC | 0.0094 / 0.0241 / −0.0013 |
| 8 个候选两两秩相关 | 中位数 0.848，最高 0.997 |
| 种群净收益 `net_long_short` | 中位数 0.0001，最大 0.0009 |
| 有效个体 | 34 / 60 |

---

## 1. 门槛：RankIC 与 Pearson 双轨

[`rushing_forward_ic_gate_report.py`](../rushing_forward_ic_gate_report.py)

原判据是 raw Pearson IC ≥ 0.03，而 GP 优化的 `robust_ic` 是 Spearman RankIC。
这类分钟衍生因子分布厚尾，Pearson IC 只有 RankIC 的 1/4 ~ 1/2，
用 RankIC 校准出的阈值去卡 Pearson IC，必然全数否决。

**双轨判据（两条都要满足，默认开启）**：

- **主轨**：方向由样本期 RankIC 符号锁定；样本期与验证期的方向对齐 RankIC 都 ≥ `--threshold`（默认 0.025）。管排序质量。
- **副轨**：样本期与验证期的方向对齐 Pearson IC 都为正。管尾部反转——排序赚钱但极值端反向的因子在这里被拦下。

`--no-pearson-sign` 可退回单轨；`--pearson-floor X` 是可选的第三道数值门槛。
另外补了验证期 ICIR、t 值、胜率（周频调仓 + 一周持有，IC 序列不重叠，普通 t 值即可）。

**实测**：同一批因子 **0/8 → 8/8 通过**，验证期 t 值 2.4–4.4。
副轨在这批上不淘汰任何因子，因为 Pearson IC 只在**测试期**转负（−0.0046 ~ +0.002），
而测试期按设计不参与判定。

---

## 2. 窗口：训练截到 2022-12-31

原 `config.json` 训练区间是 2018-01-02 → 2024-12-31，而门槛把 2022-03 → 2024-12 称作"验证期"——
整段在训练集内。这解释了验证 IC（0.051）反而高于样本 IC（0.029）。

- 窗口全部 CLI 可配，默认 样本 2018-01-02→2022-12-31 / 验证 2023-01-01→2024-12-31 / 测试 2025-01-02→2026-07-31
- `--train-end` 声明搜索实际截止点；重叠窗口标 `in-sample`，输出 `gate_trustworthy`，验证期落在训练区时报告页顶端直接警告
- `--holdout-leaf-dir`：每个窗口按**实际交易日集合**匹配面板（端点不能用来判覆盖——窗口边界常是周末），训练截止后的窗口在独立叶子上重算基因组
- `--train-metadata` 改为可选（导出的 parquet 自带 instrument/trade_date 列）

**实测**：以 2022-12-31 重训后 `gate_trustworthy: true`。
验证 RankIC 约 0.042，与泄漏版本几乎一致——因为搜索空间只有 30 个结构，本来也没多少可过拟合的余地（见第 3 条）。

---

## 3. `rushing_imbalance` 作为 leaf，让自由树在它周围长新结构

[`factors/rushing_skeleton.py`](../factors/rushing_skeleton.py) 把核心写死，只留
`invert_mask`(2) × `cross_section`(3) × `temporal`(5) = **30 种结构**，
`--pop 60 --gens 8` 等于穷举了十几遍（重跑日志里第 8 代 56/60 命中缓存）。

**做法**：`install_anchor_leaves()` 把核心**求值一次，作为日频 leaf 装进 context**，
名字就叫 `rushing_forward` / `rushing_forward_inverted`。
搜索看它的方式与看 `close` 完全一样——任何日频 seed 算子都能消费它，
`random_tree` 需要日频终端时可以直接选它。
另有 `--daily-leaf NAME=PATH` 把任何已导出因子装成 leaf。

配套改了 `gp/typed_tree.py`：`random_tree` 选叶子改用 `registry.accepts` 而非类型相等。
此前精确语义叶子永远进不了通用 `LEGACY_MINUTE` / `LEGACY_DAILY` 槽位。

**为什么不是子树**：我第一版把核心做成子树，那是错的。它的根是 `DAILY_RAW_FACTOR`，
没有任何 seed 算子产出这个类型，所以**变异永远无法重新生成它**——
一旦最后一个含它的个体离开种群，这个核心就永久消失，只能靠交叉从已含它的个体传播。
而且每个含它的候选都要重新塌缩一遍整个分钟立方，是树里最贵的一步。

**实测**：`random_tree` 在 300 次日频树生成中 **207 次**自己长到了这个 leaf（子树版本是 0）。
产出的是新结构而非参数微调，例如：

```
seed_ts_corr_daily(seed_cs_rank(seed_div_daily_scalar_right(
    seed_mul_daily(rushing_forward, rushing_forward), ...)), ...)
```

**硬件**：实测常驻输入 8.61 GB（5 个分钟字段 bf16 + rushing 叶子 fp32），
超过这块 GTX 1070 的 8 GiB，PyTorch 溢出到主存，单候选 5 分钟。
**≥12 GB 显存，或用 `--cpu`**——CPU 实测 1.8–2.4 秒/候选。
`--max-peak-bytes` / `--max-cost-units` / `--max-estimated-seconds` 已接到 CLI。

---

## 4. 新颖性是 Pareto 目标，不是拒绝

[`evaluation/incremental.py`](../evaluation/incremental.py)

你的顾虑是对的，我第一版做成硬拒绝是错的：
**一个平滑变体可以与池内因子相关 0.9，同时在 IC 和换手上都更好**，
硬门槛把它扔掉，是为了守一条多样性规则而丢掉一个更好的因子。

现在 `pool_correlation` 是**第 5 个 Pareto 目标**（`-pool_correlation`）：

- 与池重合的候选不被取消资格，它只是必须在 IC、净收益或复杂度上更好才能占住前沿位置——这正是组合层面真实做的权衡。
- `max_pool_correlation` 仍在，但**默认关闭**，只在刻意挖去相关第二条腿时才设。
- 多元残差 `cross_section_residual_multi` 对整个池做联合残差，而不是只对单个 baseline：
  一个候选可以与池里每个因子的两两相关都低于阈值，却仍接近其中两个的线性组合。
- `IncrementalFitness.invalid()` 的 `pool_correlation` 设为 1.0（最差）。
  否则默认值 0.0 是这一目标上的**最优值**，当所有有效个体都与池有重合时，
  一个被拒绝的候选会因为在新颖性上无人能比而进入前沿。

**实测**：相关度 0.92 但 `robust_ic` 0.05 的变体 → Pareto rank 0（留在前沿）；
相关度 0.05 但 IC 0.02 的弱新颖因子 → 也是 rank 0；invalid → rank 1。
无池时 `pool_correlation` 恒为 0，该目标退化为常数，行为与改动前完全一致。

---

## 5. 多因子合成：相关度聚类挑代表

[`factor_combination_report.py`](../factor_combination_report.py)、
[`materialize_jsonl_factors.py`](../materialize_jsonl_factors.py)

**聚类而非贪心**。贪心回答的是"这个因子是否已被某个已选因子解释"，问错了问题：
两个因子可以各自与在位者都低于阈值，彼此却是近似复制品，贪心会把两个都留下，
组合里这个想法就被重复计了一次。
现在用 1 − |秩相关| 上的**平均连接层次聚类**，在 1 − limit 处切开，
每个簇出一个代表（样本期 RankIC 最高的成员）。

**物化 all76**。run log 记录了每个候选的基因组，但只有 `--export-rank` 子集写成了面板，
`all76.jsonl` 的 76 个 event_skeleton 候选一个都没导出。
`materialize_jsonl_factors.py` 遍历 jsonl 求值并写出标准 parquet + sidecar，
输出可直接喂给组合报告和 IC 门槛；`--skip-existing` 让中断的运行可续跑。
`genome_from_export` 增加了旧格式推断（`all76.jsonl` 的记录没有 `kind` 字段）。

单候选成本实测（GTX 1070，2018-01-02→2026-07-31 的 volume 立方 2.36 GB）：
切片加载 85 s（一次性）、求值 4.6 s、写 parquet 2.0 s，76 个约 10 分钟。
`--chunk-rows` 在 1024→30000 之间只差 7.2 s→4.4 s，因为
`element_budget = max(chunk_rows × 240, 4M)` 已被默认值托底。

**因子池构成**。除 all76 外，`output/experiments` 下已有 101 个导出面板，
横跨 6 个 handbook 家族（climb_mountain 13、complete_tide 16、dark_flow 10、
equal_treatment 10、long_short_battle 12、raw_panic 32、rushing_forward 8+8），
都覆盖 2018-01-02→2024-12-31。合计 **177 个面板、7 个家族**。

all76 内部的结构重复度很高：76 个候选只有 **29 个不同 detector**（最多的一个出现 14 次），
**63/76 共用同一个 primary statistic** `follow_ratio_series(window=5)`，
robust_ic 中位数 0.0415、最大 0.0708。

**成分覆盖必须显式报告**。handbook 家族的导出只到 2024，
而 all76 物化到了 2026。组合按逐格取"存在的成分"平均，
所以在测试期它会**静默变成只由 all76 成分构成的另一个组合**。
报告现在逐窗口统计实际有数据的成分数并列在表里，
不足时在日志里直接警告；跨家族的 holdout 叶子缺失也只跳过该因子而非中断整轮。

---

## 6. 降换手：合成前还是合成后

两种放置不是同一件事，因为 `rank_z` 是非线性的（截面排序），与时间平均不可交换：

- `smooth→combine`：`rank_z(mean_t(x))` —— 先让每个因子与自己的历史平均，**再重新排序**，
  然后由合成抵消分歧。重新排序会恢复一部分被平均压掉的离散度。
- `combine→smooth`：`mean_t(rank_z(x))` —— 先抵消分歧，再对秩做时间平均。

哪个净收益高取决于各成分换手的相关性，是经验问题。报告现在把
**平滑天数 × 放置位置**一起扫描，两条都出。

正确性校验：平滑 1 天时两者数值完全相同（此时平滑是恒等运算），
这是实现无误的必要条件。
