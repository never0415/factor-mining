# 全因子叶子、注册算子与进化插槽

`min_gp.factors.catalog` 是迁移完整性的唯一清单。新增因子但没有提供叶子、
注册算子、可序列化锚点或算子插槽时，`migration_audit()` 和测试会失败。

当前目录包含 72 个锚点：

- 滴水穿石：1 个强类型频谱插槽基因组；
- 适度冒险、待著而救：2 个事件插槽锚点；
- 因子复现手册：10 个强类型核心/截面/低频插槽锚点；
- `seeds.py`：59 个注册表达式树锚点，通过兼容基因组接入统一目录，继续使用
  原表达式运行时已经验证的类型推导、子树变异和交叉。

## 插槽层次

滴水穿石使用：session、transform、clip、detrend、window、spectrum、reducer、
low_frequency。

事件因子使用：detector、statistic、aggregator、cross_section、
low_frequency、combiner。

其余手册因子使用：core、cross_section、low_frequency。`core` 是注册的领域
算子，选择哪个手册结构本身就是基因；算子参数是该插槽内部的参数基因。

旧表达式树的每个 `Op` 节点都是算子插槽，叶子替换、同类型子树替换、算子
重生成和常数域变异沿用原 GA 的成熟实现。

## 数据约束

本地 OHLCV 能直接计算六个手册锚点。raw_panic、rushing_forward、water_boat、
cooperation_effect 的锚点同样已经注册并可进化，但仍严格要求各自的市场序列、
报告定义派生张量或 PIT 市值输入；迁移层不会伪造这些数据。

运行完整性审计：

```powershell
python -m min_gp.audit_factor_migration
```
