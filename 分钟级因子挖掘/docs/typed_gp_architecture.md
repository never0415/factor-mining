# 强类型分钟因子 GP 架构

## 搜索层级

1. `event_skeleton`：分钟事件因子的算子插槽 GP。基因包括 detector、statistic、aggregator、cross-section、low-frequency 和 combiner，参数只是每个算子的附属基因。
2. `dripping_stone`：频谱结构保持 GP，围绕成交量预处理、去趋势、窗函数、频带和低频化搜索。
3. `daily_tree`：以落盘日频因子为叶子的第二层强类型树 GP，可搜索算术、截面变换、时序变换和因子组合。

注册算子会通过语义签名自动进入兼容插槽；不允许价格进入成交量算子，也不允许频谱充当事件掩码。

## 正负方向

三个正式入口默认使用 `--direction discovery`。每个走样本外折只使用该折的训练段确定方向，最终输出方向在完整训练样本上锁定；hold-out 不会重新翻转方向。需要严格论文方向时显式传 `--direction paper`。

## 运行示例

```powershell
# 算子本身参与进化
uv run python -m min_gp.event_gp --family event_skeleton --search gp --run-id event_v1

# 频谱因子
uv run python -m min_gp.dripping_gp --run-id dripping_v1

# 第二层日频组合；默认读取本地4个已落盘因子
uv run python -m min_gp.daily_gp --neutralize --run-id daily_v1

# 生成一个新手册因子，再通过 --leaf 接入第二层 GP
uv run python -m min_gp.build_handbook_factor --factor complete_tide `
  --start 2018-01-02 --end 2024-12-31 --out F:/factors/complete_tide.parquet
uv run python -m min_gp.daily_gp `
  --leaf complete_tide=F:/factors/complete_tide.parquet:factor --run-id tide_meta_v1

# 从 checkpoint 恢复，必须复用原 run-id
uv run python -m min_gp.daily_gp --neutralize --run-id daily_v1 --resume

# 日频树独立样本外检验
uv run python -m min_gp.daily_holdout `
  --candidates min_gp/output/experiments/daily_tree_daily_v1/daily_gp.jsonl `
  --start 2025-01-02 --end 2026-07-31 --cpu
```

每个实验目录包含 `config.json`、`checkpoint.json`、结果 JSONL，以及发生异常时的 `failures.jsonl`。结果不再默认覆盖上一轮实验。

## 中性化与档案

`daily_gp --neutralize` 在评价层对行业和对数流通市值做每日截面 OLS 残差化，不污染原始因子定义。`FactorArchive` 按截面秩相关系数去重，默认拒绝与已保留因子绝对相关系数不低于 0.85 的候选。

## 研报因子覆盖

本地 OHLCV 可计算的模板位于 `factors/handbook.py`：完整潮汐、勇攀高峰、花隐林间、多空博弈、一视同仁、暗流涌动和原始/波动率惊恐。

激流勇进、水中行舟和协同效应的源文摘录没有展开成交额占比分母、趋势或零符号回退等定义。代码为这些量保留必填接口，缺少时抛出 `IncompleteDefinitionError`，避免把猜测结果标成论文复现。

## 外部数据

安装方式：

```powershell
uv sync --extra data
```

AkShare 用于补充公开 A 股日线和成交额；TqSdk 用于其支持品种的 K 线。TqSdk 凭据仅从 `.env` 或系统环境读取：

```text
TQ_USER=...
TQ_PASS=...
```

不得把真实凭据写入源码或提交到版本库。

## 验证

```powershell
uv run python -m unittest discover -s 分钟级因子挖掘/tests -v
$env:MIN_GP_RUN_INTEGRATION='1'
uv run python -m unittest 分钟级因子挖掘.tests.test_real_data_integration -v
```
