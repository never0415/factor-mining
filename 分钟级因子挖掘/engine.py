"""
GA engine for min_gp: seeded population, type-aware mutation/crossover, Spearman IC fitness.

Population init: 39 paper seeds + random individuals (tag-guided top-down generation).
Fitness: |mean(daily cross-sectional Spearman IC)| — sign-free, GP explores both directions.
Invalid individuals (TypeTagError / runtime errors) get fitness = -inf.
"""
import random, time, hashlib, os

# PyTorch 缓存碎片: 必须在 import torch 之前设 (PyTorch 初始化时读此配置)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from min_gp.expr import (A3, B2, D2, M1, SCALAR, OP, Const, Leaf, Node, Op,
                         TypeTagError, parse)
from min_gp.fitness import daily_spearman_ic
from min_gp.seeds import all_seeds, SEEDS
from min_gp.config import output_path

# ── operator pools by output tag ──
_BINARY = ["add", "sub", "mul", "div", "pow"]
_UNARY = ["log", "log1p", "sqrt", "abs", "sign", "neg", "clip"]

POOL = {
    D2: {
        "binary": _BINARY + ["gt", "ge", "lt", "le", "cs_resid"],
        "unary": _UNARY + ["f"],
        "leaf": [],
        "day": ["day_last", "day_first", "day_ratio", "time_barycenter"],
        "daypair": ["day_corr"],
        "ts": ["ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max",
               "ts_delay", "ts_delta", "ts_zscore", "ts_rank",
               "ts_corr", "ts_cov", "ts_ema"],
        "cs": ["cs_rank", "cs_zscore"],
        "istat": ["day_istd", "day_iskew", "day_ikurt"],
        "quantile": ["ts_quantile", "day_quantile"],
        "maskagg": ["mask_agg"],        # mask_agg(x, mask, op)
        "maskratio": ["mask_ratio"],    # mask_ratio(x, A, B, N)
        "maskstat": ["mask_stat"],      # mask_stat(x, y, mask, N, op)
        "dist": ["dist_to_event"],      # dist_to_event(mask) → A3
        "rollcut": ["roll_cut"],        # roll_cut(x, y, N, λ)
        "interval": ["interval_stat"],  # interval_stat(mask, N, stat) — 事件间隔分布统计
    },
    A3: {
        "binary": _BINARY,
        "unary": _UNARY,
        "leaf": ["open", "high", "low", "close", "volume", "tp", "amp", "ret"],
        "intra": ["intra_mean", "intra_std", "intra_shift", "intra_cumsum", "intra_cummax"],
        "maskmul": ["mask_mul"],
        "convert": ["to_A", "to_B"],
    },
    M1: {"leaf": ["is_peak", "is_ridge", "is_valley",
                  "is_jump", "is_amp_valley", "is_jump_peak", "is_jump_ridge", "has_gap",
                  "mask_am", "mask_pm", "mask_open_5m", "mask_open_15m", "mask_open_30m",
                  "mask_open_60m", "mask_open_90m", "mask_open_120m",
                  "mask_close_5m", "mask_close_15m", "mask_close_30m", "mask_close_60m",
                  "mask_close_90m", "mask_close_120m",
                  "mask_mid_30m", "mask_mid_60m", "mask_afternoon_30m", "mask_afternoon_60m",
                  "mask_lunch_30m", "mask_lunch_60m", "w_time"]},
}

_A3_LEAVES = POOL[A3]["leaf"]

WINDOWS = [5, 10, 20, 40]
CONSTS = []   # no arithmetic constants whatsoever
QS = [0.25, 0.5, 0.75]
OPS = [0, 1, 2, 3, 4, 5, 6, 7, 8]   # mask_agg/mask_stat op enum
STATS = [1, 2, 3, 4]   # interval_stat: 1=mean 2=std 3=kurt 4=skew
SIGMAS = [1, 1.5, 2]   # 喷发阈值 σ 倍数: μ + k·σ, k∈{1,1.5,2}
REGIME_METHODS = [0, 1, 2, 3, 4]   # regime: 0=above 1=below 2=isolated_peak 3=clustered_ridge 4=valley
# 独立 domain: 复用 'op' (0..8) 会让 mutate 生成 5..8 → _a_regime ValueError


def rand_const(domain=None):
    """domain=None→error (no generic consts), 'window'→WINDOWS, 'q'→QS, 'op'→OPS,
    'stat'→STATS, 'sigma'→SIGMAS"""
    pool = (WINDOWS if domain == 'window' else QS if domain == 'q'
            else OPS if domain == 'op' else STATS if domain == 'stat'
            else SIGMAS if domain == 'sigma'
            else REGIME_METHODS if domain == 'regime_method' else None)
    if pool is None:
        raise ValueError("no generic constants allowed — use domain='window'/'q'/'op'")
    return Const(random.choice(pool), domain=domain)


def _rand_param(opname):
    """Random window/quantile parameter for rolling ops."""
    if opname.startswith("ts_") or opname.startswith("intra_"):
        return Const(random.choice(WINDOWS), domain='window')
    if opname == "ts_quantile":
        return Const(random.choice(QS), domain='q')
    return None


def gen_tree(tag, depth):
    """Type-aware top-down random generation."""
    if depth <= 0 or tag == M1:
        if tag == A3:
            return Leaf(random.choice(_A3_LEAVES))
        if tag == M1:
            return Leaf(random.choice(POOL[M1]["leaf"]))
        if tag == SCALAR:
            raise ValueError("no generic constants allowed")
        if tag == D2:                      # no D2 leaves — descend
            return gen_tree(D2, depth + 1)
        raise ValueError(tag)

    # B2 only reachable via to_B(A3); SCALAR not allowed — no generic constants
    if tag == B2:
        return Op("to_B", [gen_tree(A3, depth + 1)])
    if tag is None or tag == SCALAR:
        raise ValueError(f"gen_tree: unsupported tag={tag}")

    pool = POOL[tag]
    # pick an op family by output tag
    choices = []
    if tag == D2:
        choices += [("day", random.choice(pool["day"])),
                    ("ts", random.choice(pool["ts"])),
                    ("cs", random.choice(pool["cs"])),
                    ("istat", random.choice(pool["istat"]))]
        if random.random() < 0.15:
            choices.append(("quantile", random.choice(pool["quantile"])))
        if random.random() < 0.10:
            choices.append(("tspair", random.choice(["ts_corr", "ts_cov"])))
        if random.random() < 0.15:
            choices.append(("daypair", random.choice(pool["daypair"])))
        if random.random() < 0.20:
            choices.append(("maskagg", random.choice(pool["maskagg"])))
        if random.random() < 0.15:
            choices.append(("maskratio", random.choice(pool["maskratio"])))
        if random.random() < 0.15:
            choices.append(("maskstat", random.choice(pool["maskstat"])))
        if random.random() < 0.15:
            choices.append(("dist", random.choice(pool["dist"])))
        if random.random() < 0.15:
            choices.append(("rollcut", random.choice(pool["rollcut"])))
        if random.random() < 0.15:
            choices.append(("interval", random.choice(pool["interval"])))
        choices += [("binary", random.choice(pool["binary"])),
                    ("unary", random.choice(pool["unary"]))]
    else:  # A3
        choices += [("intra", random.choice(pool["intra"])),
                    ("binary", random.choice(pool["binary"])),
                    ("unary", random.choice(pool["unary"]))]
        if random.random() < 0.25:
            choices.append(("maskmul", "mask_mul"))
        if random.random() < 0.15:
            choices.append(("convert", random.choice(pool["convert"])))

    fam, opname = random.choice(choices)

    if fam in ("day", "ts", "intra", "istat", "cs", "quantile", "tspair", "convert",
               "maskagg", "maskratio", "maskstat", "dist", "rollcut", "interval"):
        if fam in ("day", "istat"):
            if fam == "istat":
                # day_istd/iskew/ikurt: 事件间隔统计, 参数必须掩码语义 —
                # 连续量 (close/volume) 当掩码 → 整天一个事件段 → 恒 0 +
                # 缺失模式伪信号 (day_istd(close)≡0, test IC 是流动性代理)
                return Op(opname, [gen_mask(depth - 1)])
            return Op(opname, [gen_tree(A3, depth - 1)])
        if fam == "ts":
            if opname in ("ts_corr", "ts_cov"):
                return Op(opname, [gen_tree(D2, depth - 1), gen_tree(D2, depth - 1),
                                   Const(random.choice(WINDOWS), domain='window')])
            return Op(opname, [gen_tree(D2, depth - 1), Const(random.choice(WINDOWS), domain='window')])
        if fam == "tspair":
            return Op(opname, [gen_tree(D2, depth - 1), gen_tree(D2, depth - 1),
                               Const(random.choice(WINDOWS), domain='window')])
        if fam == "intra":
            if opname in ("intra_cumsum", "intra_cummax"):
                return Op(opname, [gen_tree(A3, depth - 1)])
            return Op(opname, [gen_tree(A3, depth - 1), Const(random.choice(WINDOWS), domain='window')])
        if fam == "cs":
            return Op(opname, [gen_tree(tag, depth - 1)])
        if fam == "quantile":
            if opname == "day_quantile":
                return Op(opname, [gen_tree(A3, depth - 1), Const(random.choice(QS), domain='q')])
            return Op(opname, [gen_tree(D2, depth - 1), Const(random.choice(QS), domain='q'),
                               Const(random.choice(WINDOWS), domain='window')])
        if fam == "convert":
            if opname == "to_A":
                inner = Op("to_B", [gen_tree(A3, depth - 1)])
                if random.random() < 0.5:
                    inner = Op(random.choice(["ts_mean", "ts_std"]),
                               [inner, Const(random.choice(WINDOWS), domain='window')])
                return Op("to_A", [inner])
            # to_B: 必须包 ts 算子返回 D2 再 to_A — 裸 B2 会污染 A3 参数位
            inner = Op("to_B", [gen_tree(A3, depth - 1)])
            return Op("to_A", [Op(random.choice(["ts_mean", "ts_std"]),
                                  [inner, Const(random.choice(WINDOWS), domain='window')])])
        if fam == "maskagg":
            # mask_agg(x, mask, op): x 常为 A3 量 (volume×close 等), mask 常为 A3 掩码
            x = gen_tree(A3, depth - 1)
            mask = random.choice(["is_peak", "is_ridge", "is_valley",
                                  "is_jump_peak", "is_jump_ridge", "is_amp_valley"])
            return Op(opname, [x, Leaf(mask), Const(random.choice(OPS), domain='op')])
        if fam == "maskratio":
            x = gen_tree(A3, depth - 1)
            ma = random.choice(["is_peak", "is_ridge", "is_valley", "is_jump_peak", "is_jump_ridge"])
            mb = random.choice(["is_peak", "is_ridge", "is_valley", "is_jump_peak", "is_jump_ridge"])
            return Op(opname, [x, Leaf(ma), Leaf(mb), Const(random.choice(WINDOWS), domain='window')])
        if fam == "maskstat":
            x = gen_tree(A3, depth - 1)
            y = gen_tree(A3, depth - 1)
            mask = random.choice(["is_peak", "is_ridge", "is_valley", "is_jump_peak", "is_jump_ridge"])
            return Op(opname, [x, y, Leaf(mask), Const(random.choice(WINDOWS), domain='window'),
                               Const(random.choice(OPS[:3]), domain='op')])
        if fam == "dist":
            mask = random.choice(["is_peak", "is_ridge", "is_valley", "is_jump_peak", "is_jump_ridge"])
            return Op("dist_to_event", [Leaf(mask)])
        if fam == "rollcut":
            x = gen_tree(A3, depth - 1)
            y = random.choice(["close", "open", "high", "low", "volume", "tp", "amp", "ret"])
            return Op("roll_cut", [x, Leaf(y), Const(random.choice(WINDOWS), domain='window'),
                                   Const(random.choice(QS), domain='q')])
        if fam == "interval":
            # interval_stat(mask, N, stat): 事件间隔分布统计.
            # mask: 稀疏布尔 A3 ( >0 即事件). 两种合法来源:
            #   A (35%). 叶子掩码 (is_peak/is_ridge/… 预计算)
            #   B (65%). regime(field, N, k, method) — 单算子, 深度可控
            Nw = Const(random.choice(WINDOWS), domain='window')
            if random.random() < 0.35:
                mask = Leaf(random.choice(["is_peak", "is_ridge", "is_valley",
                                           "is_jump_peak", "is_jump_ridge", "is_amp_valley"]))
            else:
                x = gen_tree(A3, depth - 2)   # regime 本身占 1 层, interval_stat 外又占 1 层
                k = Const(random.choice(SIGMAS), domain='sigma')
                m = Const(random.randint(0, 4), domain='regime_method')
                mask = Op("regime", [x, Nw, k, m])
            return Op("interval_stat", [mask, Const(random.choice(WINDOWS), domain='window'),
                                        Const(random.choice(STATS), domain='stat')])
    elif fam == "daypair":
        return Op(opname, [gen_tree(A3, depth - 1), gen_tree(A3, depth - 1)])
    elif fam == "binary":
        opname = _BINARY[random.randrange(len(_BINARY))] if opname in _BINARY else opname
        if opname in ("gt", "ge", "lt", "le"):
            a, b = gen_tree(D2 if tag == D2 else A3, depth - 1), gen_tree(D2 if tag == D2 else A3, depth - 1)
            return Op(opname, [a, b])
        return Op(opname, [gen_tree(tag, depth - 1), gen_tree(tag, depth - 1)])
    elif fam == "unary":
        if opname == "clip":
            return Op(opname, [gen_tree(tag, depth - 1),
                               Const(random.choice(WINDOWS), domain='window'),
                               Const(random.choice(WINDOWS), domain='window')])
        return Op(opname, [gen_tree(tag, depth - 1)])
    elif fam == "maskmul":
        return Op("mask_mul", [gen_tree(A3, depth - 1), gen_tree(M1, depth - 1)])
    raise ValueError(fam)


def gen_individual(max_depth=6):
    return gen_tree(D2, random.randint(3, max_depth))


_MASK_LEAVES = ["is_peak", "is_ridge", "is_valley", "is_jump", "is_amp_valley",
                "is_jump_peak", "is_jump_ridge", "has_gap"]


def gen_mask(depth):
    """掩码语义 A3 节点 — 事件统计算子 (day_istd/iskew/ikurt/interval_stat) 的参数位专用.

    来源: is_* 掩码叶子 / regime / 比较 (gt/lt/ge/le → float 0/1) / 逻辑组合.
    禁止裸连续量: 连续量当掩码 (mask3d.bool()) 会产生"整天单事件段 → 恒 0 +
    缺失分钟模式"伪信号 (如 day_istd(close)), test 段 IC 是流动性代理而非 alpha.
    """
    if depth <= 0 or random.random() < 0.45:
        return Leaf(random.choice(_MASK_LEAVES))
    r = random.random()
    if r < 0.45:
        x = gen_tree(A3, max(depth - 2, 0))
        return Op("regime", [x, Const(random.choice(WINDOWS), domain="window"),
                             Const(random.choice(SIGMAS), domain="sigma"),
                             Const(random.randint(0, 4), domain="regime_method")])
    if r < 0.8:
        a = gen_tree(A3, max(depth - 1, 0))
        b = gen_tree(A3, max(depth - 1, 0))
        return Op(random.choice(["gt", "ge", "lt", "le"]), [a, b])
    return Op(random.choice(["and_", "or_"]),
              [gen_mask(max(depth - 1, 0)), gen_mask(max(depth - 1, 0))])


def depth(node):
    if isinstance(node, (Leaf, Const)):
        return 1
    return 1 + max(depth(a) for a in node.args)


def all_nodes(node):
    yield node
    if isinstance(node, Op):
        for a in node.args:
            yield from all_nodes(a)


def subtree(node):
    nodes = [n for n in all_nodes(node) if n.tag is not None]
    if not nodes:
        return node
    return random.choice(nodes)


def mutate(node, max_depth=6):
    """Replace a random subtree with a fresh one (type-aware), or tweak a param."""
    r = random.random()
    if r < 0.6:
        target = subtree(node)
        if isinstance(target, Leaf):
            if target.tag == M1:
                new = Leaf(random.choice(POOL[M1]["leaf"]))
            else:
                new = Leaf(random.choice(_A3_LEAVES))
        elif isinstance(target, Const):
            if target.domain is None:
                return node  # domain-less constants can't be regenerated
            new = rand_const(target.domain)   # preserve domain
        else:
            # regenerate a subtree of the same output tag
            tag = _tag_of(target)
            if tag == SCALAR:
                return node  # can't regenerate — constants are domain-specific
            new = gen_tree(tag, min(4, max_depth))
        candidate = _replace(node, target, new)
        if depth(candidate) <= max_depth and _check_types(candidate):
            return candidate
        return node
    # param tweak: replace a Const within same domain
    consts = [n for n in all_nodes(node) if isinstance(n, Const) and n.domain is not None]
    if consts:
        c = random.choice(consts)
        pool = (WINDOWS if c.domain == 'window' else QS if c.domain == 'q'
                else OPS if c.domain == 'op' else STATS if c.domain == 'stat'
                else SIGMAS if c.domain == 'sigma'
                else REGIME_METHODS if c.domain == 'regime_method' else [])
        if pool:
            candidate = _replace(node, c, Const(random.choice(pool), domain=c.domain))
            if depth(candidate) <= max_depth and _check_types(candidate):
                return candidate
    return node


def _tag_of(node):
    if isinstance(node, Const):
        return SCALAR
    if isinstance(node, Leaf):
        return node.tag if node.tag is not None else A3
    if isinstance(node, Op):
        if node.tag is not None:
            return node.tag
        # static output tag for known operators (avoids running full type inference)
        _STATIC = {"to_B": B2, "to_A": A3,
                   "day_sum": D2, "day_mean": D2, "day_std": D2, "day_min": D2, "day_max": D2,
                   "day_last": D2, "day_first": D2, "day_ratio": D2, "day_skew": D2, "day_kurt": D2,
                   "day_median": D2, "time_barycenter": D2, "day_corr": D2, "day_quantile": D2,
                   "day_istd": D2, "day_iskew": D2, "day_ikurt": D2,
                   "ts_mean": D2, "ts_sum": D2, "ts_std": D2, "ts_min": D2, "ts_max": D2,
                   "ts_delay": D2, "ts_delta": D2, "ts_zscore": D2, "ts_rank": D2,
                   "ts_corr": D2, "ts_cov": D2, "ts_quantile": D2, "ts_ema": D2,
                   "cs_rank": D2, "cs_zscore": D2, "cs_resid": D2,
                   "intra_mean": A3, "intra_std": A3, "intra_shift": A3,
                   "intra_cumsum": A3, "intra_cummax": A3,
                   "mask_mul": A3, "bcast": A3,
                   "mask_agg": D2, "mask_ratio": D2, "mask_stat": D2,
                   "interval_stat": D2, "roll_cut": D2, "dist_to_event": A3, "regime": A3,
                   "add": (node.args[0].tag or D2) if node.args else D2,
                   "sub": (node.args[0].tag or D2) if node.args else D2,
                   "mul": (node.args[0].tag or D2) if node.args else D2,
                   "div": (node.args[0].tag or D2) if node.args else D2,
                   "pow": (node.args[0].tag or D2) if node.args else D2,
                   "log": (node.args[0].tag or D2) if node.args else D2,
                   "log1p": (node.args[0].tag or D2) if node.args else D2,
                   "sqrt": (node.args[0].tag or D2) if node.args else D2,
                   "abs": (node.args[0].tag or D2) if node.args else D2,
                   "sign": (node.args[0].tag or D2) if node.args else D2,
                   "neg": (node.args[0].tag or D2) if node.args else D2,
                   "clip": (node.args[0].tag or D2) if node.args else D2,
                   "gt": D2, "ge": D2, "lt": D2, "le": D2,
                   "f": D2, "not_": M1, "and_": M1, "or_": M1, "if": D2,
                   }
        if node.name in _STATIC:
            t = _STATIC[node.name]
            return t() if callable(t) else t
    return node.tag


def _replace(root, target, new):
    if root is target:
        return new
    if isinstance(root, Op):
        return Op(root.name, [_replace(a, target, new) if a is not target else new
                              for a in root.args])
    return root


def _check_types(node):
    """递归校验表达式类型: 每个 Op 的参数 tag 必须通过注册的 infer 守卫.
    返回 True 合法 / False 非法 (crossover/mutate 后用, 防 eval 期 TypeTagError/RuntimeError)."""
    try:
        return _infer_type(node) is not None
    except Exception:
        return False


def _infer_type(node):
    """完整类型推断 (递归): 返回输出 tag, 非法抛异常.
    Leaf: 与 Ctx 注册一致 — tens 叶子 (含 is_peak 等布尔掩码) 全为 A3;
    masks 时段掩码 (mask_*/w_time) 为 M1. eval 前 tag=None 需静态推断."""
    from min_gp.expr import OP
    if isinstance(node, Const):
        return SCALAR
    if isinstance(node, Leaf):
        if node.tag is not None:
            return node.tag
        # 静态分类: 只有 mask_*/w_time 是 M1 (Ctx 的 masks), 其余 (含 is_peak) 是 A3
        from min_gp.engine import POOL
        return M1 if node.name.startswith("mask_") or node.name == "w_time" else A3
    if isinstance(node, Op):
        # 不信任 node.tag (crossover/mutate 后旧 tag 失效), 总是递归重推断
        tags = [_infer_type(a) for a in node.args]
        reg = OP.get(node.name)
        if reg is None:
            return None
        return reg["infer"](tags)
    return None


_HEAVY = {"interval_stat", "roll_cut", "mask_ratio", "mask_stat", "mask_agg", "dist_to_event"}
# 真正的大中间量算子 (单 eval 峰值 2-3GB, 嵌套过多必 OOM/卡死):
# interval_stat (sort 大张量), roll_cut (unfold 展开)
_HEAVY2 = {"interval_stat", "roll_cut"}


def _heavy_ops(node):
    """统计大中间量算子数量 (嵌套过多 → 峰值/耗时超标)."""
    n = 0
    for nd in all_nodes(node):
        if isinstance(nd, Op) and nd.name in _HEAVY:
            n += 1
    return n


def _heavy2_ops(node):
    """统计超重算子 (interval_stat/roll_cut) 数量 — 超过 1 个就跳过."""
    n = 0
    for nd in all_nodes(node):
        if isinstance(nd, Op) and nd.name in _HEAVY2:
            n += 1
    return n


def crossover(a, b, max_depth=6):
    """Swap random subtrees; retry on depth/type violation.
    完整类型校验: 交换后整树递归推断, 非法丢弃 (防 eval 期 TypeTagError/RuntimeError).
    Const 交换要求同 domain (q/window/stat/op/sigma 值域不同, 换位产生非法值)."""
    for _ in range(20):
        na, nb = subtree(a), subtree(b)
        if isinstance(na, Const) or isinstance(nb, Const):
            # Const 只与同 domain Const 交换 (否则 window→stat 等值域错位)
            if not (isinstance(na, Const) and isinstance(nb, Const) and na.domain == nb.domain):
                continue
        a2, b2 = _replace(a, na, nb), _replace(b, nb, na)
        if depth(a2) <= max_depth and depth(b2) <= max_depth:
            if _check_types(a2) and _check_types(b2):
                return a2, b2
    return a, b


class GA:
    def __init__(self, ctx, fwd_ret, pop_size=2000, gens=10, max_depth=6,
                 seed_ratio=0.3, elite=10, tournament=6,
                 crossover_rate=0.85, mutation_rate=0.25,
                 verbose=True, period=1, pool_mask=None):
        self.ctx = ctx
        # build target forward return matching rebalance period
        if period == 1:
            self.fwd_target = fwd_ret                         # t+1 (daily, close-to-close)
        else:
            from min_gp.label import tensor_fwd_ret
            close_d = ctx.leaves['close'][0][:, :, -1]        # (I, D) daily close
            # 重叠标签: fwd[t] = close[t+p]/close[t]-1 (close 入场 = 收盘前几分钟
            # 执行, 资金无缝; 其他口径见 min_gp.label.LABEL).
            # 挖掘阶段用重叠标签: IC 均值无偏, 样本多 5 倍 → GA 排序更稳
            # (重叠只高估 ICIR/显著性, 不污染均值; 显著性修正交给 eval_test NW 修正)
            self.fwd_target = tensor_fwd_ret(close_d, close_d, period)
        self.pop_size = pop_size
        self.gens = gens
        self.max_depth = max_depth
        self.seed_ratio = seed_ratio
        self.elite = elite
        self.tournament = tournament
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.period = period
        self.pool_mask = pool_mask          # (I,D) bool — True where stock ∈ CSI500 on date
        self.eval_meta = {}                 # expr -> (raw signed train IC, direction)

    def _fitness(self, node):
        from min_gp.fitness import multi_fitness
        # 复杂度保护: interval_stat/roll_cut 嵌套 ≥2 → 单表达式峰值/耗时超标, 直接跳过
        if _heavy2_ops(node) > 1:
            return (-1e9, -1e9, -1e9, -1e9)
        try:
            factor = self.ctx.eval(node)
            if self.pool_mask is not None:
                factor = torch.where(self.pool_mask, factor, torch.tensor(float('nan'), device=factor.device, dtype=factor.dtype))
            fit, details = multi_fitness(factor, self.fwd_target, return_details=True)
            self.eval_meta[str(node)] = details
            return fit
        except TypeTagError:
            return (-1e9, -1e9, -1e9, -1e9)
        except Exception:
            return (-1e9, -1e9, -1e9, -1e9)

    def _init_pop(self):
        seeds = list(all_seeds().values())
        n_seed = min(len(seeds), int(self.pop_size * self.seed_ratio))
        pop = seeds[:n_seed]
        while len(pop) < self.pop_size:
            pop.append(gen_individual(self.max_depth))
        return pop

    def _select(self, scored, k):
        """Binary tournament on Pareto rank, tie-break by crowding distance."""
        best = None
        for _ in range(k):
            cand = random.choice(scored)
            if best is None or _dominates(cand, best):
                best = cand
        return best[1]

    def _batch_eval(self, pop, eval_chunk=64):
        """Evaluate all individuals: factor eval (deduped) → batch fitness.
        eval_chunk: 分批 eval + 分批 fitness, 防止 factors 列表全量持有张量 OOM.
        每 chunk 间 empty_cache 释放 PyTorch 缓存碎片 (batch_fitness OOM 根因).
        失败因子记录到 min_gp/output/eval_failures.log (按错误类型分组, 方便定位)."""
        from min_gp.fitness import batch_fitness
        import torch as _t
        import os
        seen = {}      # str(expr) → tensor, same-gen dedup only (bounded: 防 OOM)
        self.eval_meta = {}
        results = [(-1e9, -1e9, -1e9, -1e9)] * len(pop)
        errs = 0
        err_types = {}
        fail_log = {}
        for ci in range(0, len(pop), eval_chunk):
            chunk = pop[ci:ci + eval_chunk]
            factors = []
            for n in chunk:
                # 复杂度保护: interval_stat/roll_cut 嵌套 ≥2 → 跳过 (防单表达式峰值/耗时超标)
                if _heavy2_ops(n) > 1:
                    errs += 1
                    factors.append(None)
                    continue
                try:
                    key = str(n)
                    if key in seen:
                        factors.append(seen[key])
                    else:
                        t = self.ctx.eval(n)
                        if self.pool_mask is not None:
                            t = torch.where(self.pool_mask, t, torch.tensor(float('nan'), device=t.device, dtype=t.dtype))
                        if len(seen) < 512:
                            seen[key] = t
                        factors.append(t)
                except Exception as e:
                    errs += 1
                    en = type(e).__name__
                    err_types[en] = err_types.get(en, 0) + 1
                    if en != "TypeTagError":   # 类型错误是 GP 探索噪声, 不打印
                        print(f"  [{en}] {str(e)[:100]}  <-  {str(n)[:200]}", flush=True)
                    if en not in fail_log:
                        fail_log[en] = []
                    if len(fail_log[en]) < 20:
                        fail_log[en].append(f"{en}: {str(e)[:80]}  <-  {str(n)[:160]}")
                    factors.append(None)
                # 每个表达式后清理缓存 — 6µs, 几乎免费
                _t.cuda.empty_cache()
            _t.cuda.empty_cache()   # 释放 eval 累积的缓存碎片
            try:
                chunk_results, chunk_details = batch_fitness(
                    factors, self.fwd_target, return_details=True)
                results[ci:ci + eval_chunk] = chunk_results
                for n, details in zip(chunk, chunk_details):
                    if details[1] in (-1, 1):
                        self.eval_meta[str(n)] = details
            except Exception as e:
                print(f"  [batch_eval] batch_fitness crashed: {e}", flush=True)
                # 保持 -1e9 (该 chunk 全部失败)
            del factors, chunk
            _t.cuda.empty_cache()
        if errs:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(err_types.items(), key=lambda x: -x[1])[:3])
            print(f"  [batch_eval] {errs}/{len(pop)} eval failures ({summary})", flush=True)
            # 追加失败因子日志 (追加不覆盖, 跨代累积)
            try:
                with open(output_path("eval_failures.log"), "a", encoding="utf-8") as f:
                    f.write(f"\n=== {errs}/{len(pop)} failures {err_types} ===\n")
                    for en, lines in fail_log.items():
                        for ln in lines:
                            f.write(ln + "\n")
            except Exception:
                pass
        return results

    def run(self, all_exprs=None):
        if all_exprs is None:
            all_exprs = set()
        t0 = time.time()
        pop = self._init_pop()
        fits = self._batch_eval(pop)
        scored = [(fits[i], pop[i]) for i in range(len(pop))]
        fits_all = {str(n): fits[i] for i, n in enumerate(pop)}
        scored_display = non_dominated_sort(scored)
        if self.verbose:
            fronts = {}
            for (rank, _), _ in scored_display:
                fronts[rank] = fronts.get(rank, 0) + 1
            print(f"[nsga2] init pop={len(pop)} fronts={fronts} ({time.time()-t0:.0f}s)", flush=True)

        for gen in range(self.gens):
            # re-sort after fitness re-evaluation from previous gen
            scored = non_dominated_sort(scored)
            scored.sort(key=lambda x: (x[0][0], -x[0][1]))  # rank asc, crowd desc
            if self.verbose:
                pareto = [s for s in scored if s[0][0] == 0]
                n_front0 = len(pareto)
                # Use original fitness from scored (before non_dominated_sort overwrites)
                avg_objs = [np.mean([fits_all.get(str(n), (0,0,0,0))[i] for (_, _), n in pareto])
                           for i in range(4)] if pareto else [0, 0, 0, 0]
                print(f"[gen {gen}] front0={n_front0} pareto_avg=(|IC|={avg_objs[0]:.4f}, "
                      f"win={avg_objs[1]:.3f}, ndcg={avg_objs[2]:.4f}, to={-avg_objs[3]:.3f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)

            # elites: best by Pareto rank + crowding
            nxt = []
            for obj, n in scored:
                if len(nxt) >= self.elite:
                    break
                k = str(n)
                if not any(str(x) == k for x in nxt):
                    nxt.append(n)

            # Offspring: crossover and mutation are independent probabilities.
            # This lets crossover_rate=0.85 and mutation_rate=0.25 both take
            # effect without requiring the two probabilities to sum to one.
            while len(nxt) < self.pop_size:
                p1 = self._select(scored, self.tournament)
                child = p1
                if random.random() < self.crossover_rate:
                    p2 = self._select(scored, self.tournament)
                    child, _ = crossover(p1, p2, self.max_depth)
                if random.random() < self.mutation_rate:
                    child = mutate(child, self.max_depth)
                nxt.append(child)
            pop = nxt
            fits = self._batch_eval(pop)
            scored = [(fits[i], pop[i]) for i in range(len(pop))]
            fits_all = {str(n): fits[i] for i, n in enumerate(pop)}

            # checkpoint: accumulate unique expressions, rank-sorted by |IC|
            ns = non_dominated_sort(scored)
            ns.sort(key=lambda x: x[0][0])
            lines = []
            cur_rank, batch = None, []
            for (r, _), n_node in ns:
                expr = fmt(n_node)
                if expr in all_exprs:
                    continue
                raw_ic, direction = self.eval_meta.get(str(n_node), (float("nan"), 0))
                if direction not in (-1, 1):
                    continue
                all_exprs.add(expr)
                ic_val = fits_all.get(str(n_node), (0,))[0]
                if r != cur_rank:
                    if batch:
                        batch.sort(key=lambda x: x[0], reverse=True)
                        for _, raw, direct, e in batch:
                            lines.append(
                                f"rank={cur_rank} trainIC={raw:.6f} direction={direct:+d}  {e}")
                    cur_rank, batch = r, []
                batch.append((ic_val, raw_ic, direction, expr))
            if batch:
                batch.sort(key=lambda x: x[0], reverse=True)
                for _, raw, direct, e in batch:
                    lines.append(
                        f"rank={cur_rank} trainIC={raw:.6f} direction={direct:+d}  {e}")
            if lines:
                with open(output_path("all_exprs.txt"), "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")

        scored = non_dominated_sort(scored)
        return scored


# ═══════════════════════════════════════════════════════
# NSGA-II: non-dominated sorting + crowding distance
# ═══════════════════════════════════════════════════════

def _dominates(a, b):
    """a Pareto-dominates b? a=(rank, crowd), b=(rank, crowd)."""
    ra, ca = a[0]
    rb, cb = b[0]
    if ra < rb:
        return True
    if ra == rb and ca > cb:
        return True
    return False


def non_dominated_sort(scored, crowd_k=None):
    """Assign (Pareto_rank, crowding_distance) to each individual.

    scored: list of ((obj1, obj2, obj3), node). All objectives maximized.
    Returns sorted list of ((rank, crowd), node).
    """
    n = len(scored)
    ranks = [0] * n
    dominated_by = [[] for _ in range(n)]
    dominate_count = [0] * n

    # 1. Pareto ranking
    for i in range(n):
        for j in range(i + 1, n):
            oi, oj = scored[i][0], scored[j][0]
            if all(oi[k] >= oj[k] for k in range(len(oi))) and any(oi[k] > oj[k] for k in range(len(oi))):
                dominated_by[i].append(j)
                dominate_count[j] += 1
            elif all(oj[k] >= oi[k] for k in range(len(oj))) and any(oj[k] > oi[k] for k in range(len(oj))):
                dominated_by[j].append(i)
                dominate_count[i] += 1

    front = [i for i in range(n) if dominate_count[i] == 0]
    rank = 0
    while front:
        for i in front:
            ranks[i] = rank
        next_front = []
        for i in front:
            for j in dominated_by[i]:
                dominate_count[j] -= 1
                if dominate_count[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1

    # 2. Crowding distance per rank. By default every objective that
    # participates in Pareto dominance also participates in diversity.
    if crowd_k is None:
        crowd_k = len(scored[0][0]) if scored else 0
    crowds = [0.0] * n
    for r in range(rank):
        front_i = [i for i in range(n) if ranks[i] == r]
        if len(front_i) <= 2:
            for i in front_i:
                crowds[i] = float("inf")
            continue
        for k in range(crowd_k):
            front_i.sort(key=lambda i: scored[i][0][k])
            f_min, f_max = scored[front_i[0]][0][k], scored[front_i[-1]][0][k]
            if f_max == f_min:
                continue
            crowds[front_i[0]] = float("inf")
            crowds[front_i[-1]] = float("inf")
            for idx in range(1, len(front_i) - 1):
                crowds[front_i[idx]] += (scored[front_i[idx + 1]][0][k] -
                                         scored[front_i[idx - 1]][0][k]) / (f_max - f_min)

    return [((ranks[i], crowds[i]), scored[i][1]) for i in range(n)]


def fmt(node):
    return str(node)


def parse_result_record(line):
    """Parse a metadata/expression output line, including legacy records.

    New records use one final double-space separator before the expression.
    Legacy records are accepted for their expression, but return direction=0
    because an absolute IC cannot safely reconstruct the training direction.
    """
    line = line.strip()
    if not line or "  " not in line:
        return "", float("nan"), 0
    header, expr = line.rsplit("  ", 1)
    raw_ic, direction = float("nan"), 0
    for token in header.split():
        try:
            if token.startswith("trainIC="):
                raw_ic = float(token.split("=", 1)[1])
            elif token.startswith("direction="):
                direction = int(token.split("=", 1)[1])
        except ValueError:
            pass
    return expr.strip(), raw_ic, direction


def main():
    import argparse
    from min_gp.config import MINUTE_PARQUET, ZZ500_PIT_PARQUET, require_path
    from min_gp.data import (build_slice, load_index_codes, load_pit_codes,
                             load_pit_daily_mask)
    from min_gp.expr import Ctx

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--valid-start", default="2022-01-02")
    ap.add_argument("--valid-end", default="2024-12-31")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of random seeds (paper recommends 8-10); runs in parallel on GPU")
    ap.add_argument("--period", type=int, default=1,
                    help="rebalance period: fitness uses t+N forward return (1=daily, 5=weekly)")
    ap.add_argument("--parquet", default=str(MINUTE_PARQUET),
                    help="minute OHLCV parquet")
    ap.add_argument("--pit", "--pool-csv", dest="pit", default=str(ZZ500_PIT_PARQUET),
                    help="daily point-in-time CSI 500 membership parquet")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    parquet_path = require_path(args.parquet, "minute parquet")
    pit_path = require_path(args.pit, "CSI500 PIT parquet")

    # ── stock universe: all stocks ever in CSI 500 ──
    zz500_codes = load_index_codes(pit_path)
    print(f"[universe] {len(zz500_codes)} stocks from {pit_path}", flush=True)

    device = "cpu" if args.cpu else "cuda"

    # ── 只加载训练期间进过 CSI500 的股票 ──
    active_codes = load_pit_codes(pit_path, args.start, args.end)
    print(f"[active] {len(active_codes)}/{len(zz500_codes)} stocks in CSI500 during {args.start}..{args.end}", flush=True)

    tens, masks, fwd_ret, meta = build_slice(parquet_path,
                                             args.start, args.end, device=device,
                                             instruments=active_codes)
    print(f"[data] train {meta['I']} x {meta['D']} x {meta['NM']}", flush=True)
    # ── CSI 500 PIT mask: 每天截面 = 当天成分 ∩ 有数据 (停牌剔除), 与回测同源 (pit_daily) ──
    close_d = tens['close'][:, :, -1]
    pm = load_pit_daily_mask(pit_path, meta['dates'],
                             meta['instruments'], device=device)
    pm = pm & ~torch.isnan(close_d)
    print(f"[mask] CSI500 PIT+daily: {pm.sum().item()/1e6:.1f}M valid (stock,date) pairs", flush=True)
    ctx = Ctx(tens, masks, meta, device=device)

    # ── multi-seed parallel GA ──
    all_nodes = []   # collect raw nodes (dedup across seeds)
    # pre-load cross-run unique expressions for dedup (append never truncates)
    all_exprs = set()
    all_exprs_file = output_path("all_exprs.txt")
    if all_exprs_file.exists():
        with open(all_exprs_file, encoding="utf-8") as f:
            for line in f:
                expr, _, direction = parse_result_record(line)
                # Legacy records contain only |IC|, so their direction cannot
                # be recovered safely. Let a new run evaluate and rewrite them.
                if expr and direction in (-1, 1):
                    all_exprs.add(expr)
    seen = set()
    t_ga = time.time()
    for seed_i in range(args.seeds):
        random.seed(seed_i)
        np.random.seed(seed_i)
        torch.manual_seed(seed_i)
        ga = GA(ctx, fwd_ret, pop_size=args.pop, gens=args.gens, verbose=args.seeds <= 2, period=args.period, pool_mask=pm)
        scored_i = ga.run(all_exprs)
        for (_, _), node in scored_i:
            k = str(node)
            if k not in seen:
                seen.add(k)
                all_nodes.append(node)
        if args.seeds > 2:
            print(f"[seed {seed_i+1}/{args.seeds}] done, unique={len(all_nodes)} "
                  f"({time.time()-t_ga:.0f}s total)", flush=True)

    # ── merge: re-evaluate fitness, then re-rank ──
    tmp_ga = GA(ctx, fwd_ret, period=args.period, pool_mask=pm)   # for fitness eval
    all_scored = [(tmp_ga._fitness(n), n) for n in all_nodes]
    scored = non_dominated_sort(all_scored)
    if args.seeds > 1:
        print(f"\n[merged] {len(all_nodes)} unique individuals from {args.seeds} seeds → "
              f"pareto={sum(1 for (r,_),_ in scored if r==0)} front0", flush=True)
    # scored: [((rank, crowd), node), ...] — from final non_dominated_sort
    pareto = [(rank, node) for (rank, crowd), node in scored if rank == 0]
    print(f"\n=== Pareto front ({len(pareto)} individuals) === (total scored: {len(scored)})")
    for i, ((rank, crowd), node) in enumerate(scored[:5]):
        obj = tmp_ga._fitness(node)
        raw_ic, direction = tmp_ga.eval_meta.get(str(node), (float("nan"), 0))
        print(f"  [{i}] rank={rank} crowd={crowd:.2f} trainIC={raw_ic:+.4f} "
              f"dir={direction:+d} |IC|={obj[0]:.4f} win={obj[1]:.3f} "
              f"ndcg={obj[2]:.4f}  {fmt(node)[:100]}")

    # ── save train-passed factors with metrics ──
    train_pass_file = output_path("train_pass.txt")
    with open(train_pass_file, "a", encoding="utf-8") as f:
        for (rank, crowd), node in scored:
            obj = tmp_ga._fitness(node)
            raw_ic, direction = tmp_ga.eval_meta.get(str(node), (float("nan"), 0))
            f.write(
                f"rank={rank} trainIC={raw_ic:.6f} direction={direction:+d} "
                f"alignedIC={obj[0]:.6f} win={obj[1]:.4f} "
                f"ndcg={obj[2]:.6f} to={-obj[3]:.6f}  {fmt(node)}\n")
    print(f"[train] {len(scored)} factors → min_gp/train_pass.txt")

    # free train GPU memory before loading validation data
    del tens, masks, fwd_ret, meta, ctx, ga, tmp_ga, all_nodes, all_scored
    torch.cuda.empty_cache()

    # ── validation: read all_exprs.txt, dedup, evaluate with direction check ──
    passed = []
    if args.valid_start:
        # Load the direction fixed on train. Never infer/flip it on valid.
        train_direction = {}
        train_file = output_path("train_pass.txt")
        if train_file.exists():
            with open(train_file, encoding="utf-8") as f:
                for line in f:
                    expr, _, direction = parse_result_record(line)
                    if expr and direction in (-1, 1):
                        train_direction[expr] = direction

        pareto_exprs = []
        seen_expr = set()
        if all_exprs_file.exists():
            with open(all_exprs_file, encoding="utf-8") as f:
                for line in f:
                    expr, _, direction = parse_result_record(line)
                    if not expr or direction not in (-1, 1):
                        continue
                    train_direction[expr] = direction
                    if expr not in seen_expr:
                        seen_expr.add(expr)
                        pareto_exprs.append(expr)
        print(f"[valid] {len(pareto_exprs)} unique expressions from all_exprs.txt")

        tens_v, masks_v, fwd_v, meta_v = build_slice(
            parquet_path,
            args.valid_start, args.valid_end, device=device,
            instruments=zz500_codes)
        if args.period > 1:
            close_d = tens_v['close'][:, :, -1]
            # 与 train 段同口径: 重叠标签 (IC 均值无偏, 样本多; 显著性用 Newey-West 修正)
            fwd_v = torch.full_like(close_d, float('nan'))
            fwd_v[:, :-args.period] = close_d[:, args.period:] / close_d[:, :-args.period] - 1.0
        pm_v = load_pit_daily_mask(pit_path, meta_v['dates'],
                                   meta_v['instruments'], device=device)
        pm_v = pm_v & ~torch.isnan(tens_v['close'][:, :, -1])   # 停牌剔除, 与 train 同口径
        ctx_v = Ctx(tens_v, masks_v, meta_v, device=device)
        # 因子体检用: 每分钟缺失率 (I,D) — 连续量当掩码的因子与之相关 → 伪信号
        miss_v = torch.isnan(tens_v['close']).float().mean(dim=2)
        print(f"\n=== validation {args.valid_start}..{args.valid_end} ===\n")
        # 批量评估 (分批 + 复杂度保护 + empty_cache), 防单表达式 GPU 挂起
        tmp_ga_v = GA(ctx_v, fwd_v, period=args.period, pool_mask=pm_v)
        vfits = tmp_ga_v._batch_eval([parse(e) for e in pareto_exprs], eval_chunk=64)
        for i, (expr_str, fit) in enumerate(zip(pareto_exprs, vfits)):
            if fit[0] <= -1e8:   # eval 失败/跳过
                continue
            direction = train_direction[expr_str]
            fv = ctx_v.eval(parse(expr_str))
            fv = torch.where(pm_v, fv, torch.full_like(fv, float("nan")))
            ic_series = daily_spearman_ic(fv, fwd_v)
            from min_gp.fitness import summarize_ic, factor_health
            s = summarize_ic(ic_series)
            raw_vIC = s['ic_mean'] if s['ic_mean'] is not None else float('nan')
            aligned_vIC = direction * raw_vIC
            raw_vICIR = s['icir'] or 0
            aligned_vICIR = direction * raw_vICIR
            # 因子体检: 退化/流动性伪信号过滤 (day_istd(close) 类 → 恒0/缺失模式)
            ok_health, hrep = factor_health(fv, miss_v)
            print(f"{i:4d} {'-':>8s} {'-':>7s} {'-':>7s} "
                  f"{raw_vIC:8.4f} {aligned_vIC:8.4f} dir={direction:+d} "
                  f"{aligned_vICIR:7.2f}  {'OK ' if ok_health else 'DROP'} "
                  f"uniq={hrep['low_unique_frac']:.2f} zero={hrep['zero_frac']:.2f} "
                  f"misscorr={hrep['miss_corr_med']:+.2f}  {expr_str[:70]}")
            # Direction is fixed on train; valid only confirms persistence.
            if aligned_vIC > 0.05 and ok_health and s['n_days'] >= 60:
                passed.append((aligned_vIC, raw_vIC, aligned_vICIR, direction, expr_str))
        # save valid-passed factors
        if passed:
            passed.sort(key=lambda x: x[0], reverse=True)
            with open(output_path("valid_pass.txt"), "a", encoding="utf-8") as f:
                for aligned_vIC, raw_vIC, icir_val, direction, expr in passed:
                    f.write(
                        f"vIC={raw_vIC:.6f} direction={direction:+d} "
                        f"alignedVIC={aligned_vIC:.6f} vICIR={icir_val:.4f}  {expr}\n")
            print(f"[valid] {len(passed)}/{len(pareto_exprs)} factors passed → min_gp/valid_pass.txt")
        else:
            print("[valid] no factors passed (aligned IC<=0.05 or health/n_days failed)")


if __name__ == "__main__":
    main()
