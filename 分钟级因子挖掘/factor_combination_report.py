"""Correlation-aware factor combination, turnover sweep and neutralisation.

A single minute-derived factor on CSI 500 tops out near a weekly rank IC of
0.03, and at 30 bps the eight rank-0 exports of the 20260826 run each cleared
about one basis point of net long-short return per rebalance. Raising that
number is not a search problem: it is a portfolio problem. Combining factors
whose ranks disagree raises IC roughly with the square root of the count,
while smoothing the composite trades a little IC for the turnover that the
cost actually charges.

The report answers three questions on one grid:

  1. Which of the supplied candidates are genuinely distinct? Selection runs
     on the in-sample window alone, so the validation and test columns stay
     honest out-of-sample readings of a decision already made.
  2. How much does combining them add over the best single factor?
  3. What smoothing, and what neutralisation, leaves the most net return?

Every number is reported per window, and the window a candidate was searched
in is labelled in-sample so nothing here reads as evidence it is not.
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation import BatchedNeutralizer, mean_rank_correlation
from min_gp.evaluation.incremental import (
    net_long_short_return, trailing_signal_mean,
)
from min_gp.factor_export import load_factor_parquet
from min_gp.factors.catalog import genome_from_export
from min_gp.factors.seed_tree import SeedTreeGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.report_candidates import _pearson
from min_gp.operators import build_operator_registry
from min_gp.spectral_data import load_daily_close_tensor, load_daily_exposures


DEFAULT_WINDOWS = (
    ("select", "样本期", "2018-01-02", "2022-12-31"),
    ("valid", "验证期", "2023-01-01", "2024-12-31"),
    ("test", "测试期", "2025-01-02", "2026-07-31"),
)

WEEKS_PER_YEAR = 52.0


def _discover(directories, explicit):
    """Every exported factor panel in the supplied directories.

    Matching on the extension rather than an exported-by-the-GP name prefix
    keeps materialised run logs, which carry their own prefix, in scope.
    """
    paths = [Path(value) for value in explicit]
    for directory in directories:
        paths.extend(sorted(Path(directory).glob("*.parquet")))
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise SystemExit("no factor parquet supplied")
    return unique


def _short_name(path: Path) -> str:
    return f"{path.parent.parent.name}/{path.stem}"


def _load_leaf_context(directory: Path, device):
    """Every leaf a hold-out archive publishes, plus its own grid."""
    context = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                context[key] = torch.as_tensor(archive[key], device=device)
    payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return (
        context,
        [str(v) for v in payload["instruments"]],
        [str(v) for v in payload["dates"]],
    )


def _splice(target, values, source_instruments, source_dates,
            instruments, dates):
    """Write a hold-out panel into the cells of the full grid it covers.

    The exported parquet stops at the training cut, so on its own a factor is
    all-NaN over validation and test and the report would have nothing
    out-of-sample to say. Re-evaluating the genome on the hold-out leaves and
    splicing by name fills exactly the dates the export never held.
    """
    row_index = {name: i for i, name in enumerate(source_instruments)}
    col_index = {day: d for d, day in enumerate(source_dates)}
    rows = np.array([row_index.get(str(name), -1) for name in instruments])
    cols = np.array([col_index.get(str(day), -1) for day in dates])
    present_rows = torch.as_tensor(rows >= 0, device=target.device)
    present_cols = torch.as_tensor(cols >= 0, device=target.device)
    gathered = values[
        torch.as_tensor(np.where(rows >= 0, rows, 0), device=target.device)
    ][:, torch.as_tensor(np.where(cols >= 0, cols, 0), device=target.device)]
    usable = present_rows.unsqueeze(1) & present_cols.unsqueeze(0)
    usable &= torch.isfinite(gathered)
    return torch.where(usable, gathered, target)


def _evaluate_genome(genome, context, registry, chunk_rows=4096):
    if isinstance(genome, SeedTreeGenome):
        return genome.evaluate(context, registry, chunk_rows)
    return genome.evaluate(context, chunk_rows=chunk_rows)


def _window_mask(dates, begin, finish, device):
    array = np.asarray(dates)
    return torch.as_tensor((array >= begin) & (array <= finish), device=device)


def _scope(tensor, mask):
    return torch.where(
        mask.unsqueeze(0), tensor, torch.full_like(tensor, float("nan"))
    )


def _series_stats(series: np.ndarray) -> dict:
    finite = series[np.isfinite(series)]
    if finite.size < 2:
        return dict(mean=float("nan"), icir=float("nan"), t_stat=float("nan"),
                    win_rate=float("nan"), weeks=int(finite.size))
    mean, sd = float(finite.mean()), float(finite.std(ddof=1))
    icir = mean / sd if sd > 0 else float("nan")
    return dict(
        mean=mean, icir=icir,
        t_stat=icir * float(np.sqrt(finite.size)) if np.isfinite(icir) else float("nan"),
        win_rate=float((finite > 0).mean()), weeks=int(finite.size),
    )


def evaluate(factor, fwd, mask, direction, cost_bps, quantile, min_cross_section):
    """Rank IC statistics and the net long-short book inside one window."""
    scoped_factor, scoped_fwd = _scope(factor, mask), _scope(fwd, mask)
    valid = torch.isfinite(scoped_factor) & torch.isfinite(scoped_fwd)
    rank_ic = _pearson(
        cross_section_rank(scoped_factor), cross_section_rank(scoped_fwd), valid
    ).float().cpu().numpy() * direction
    ic = _pearson(scoped_factor, scoped_fwd, valid).float().cpu().numpy() * direction
    rank_stats = _series_stats(rank_ic)
    net, turnover = net_long_short_return(
        scoped_factor, scoped_fwd, direction, quantile, cost_bps, min_cross_section,
    )
    # Cost is linear in turnover, so the gross book separates cleanly from the
    # fee. Without it a losing line cannot be read: a negative net return means
    # something very different when the gross leg is positive and eaten by
    # turnover than when the extreme quantiles themselves are moving the wrong
    # way, and only the second is a reason to stop trading the factor.
    cost = turnover * cost_bps * 1e-4
    gross = net + cost if np.isfinite(net) else float("nan")
    return {
        "rank_ic": rank_stats["mean"],
        "rank_icir": rank_stats["icir"],
        "rank_t_stat": rank_stats["t_stat"],
        "rank_win_rate": rank_stats["win_rate"],
        "weeks": rank_stats["weeks"],
        "ic": _series_stats(ic)["mean"],
        "gross_per_rebalance": gross,
        "gross_annual": gross * WEEKS_PER_YEAR if np.isfinite(gross) else float("nan"),
        "cost_per_rebalance": cost,
        "net_per_rebalance": net,
        # Weekly rebalance, so the per-rebalance figure annualises by 52.
        "net_annual": net * WEEKS_PER_YEAR if np.isfinite(net) else float("nan"),
        "turnover": turnover,
    }


def rank_z(factor):
    """Direction-free cross-sectional standardisation of a factor's ranks.

    Components must enter a composite on one scale. Ranks put every factor on
    [0, 1] regardless of its units, and the daily z-score removes the residual
    dependence on how many names were rankable that day.
    """
    ranked = cross_section_rank(factor.float())
    valid = torch.isfinite(ranked)
    weight = valid.to(torch.float32)
    count = weight.sum(0, keepdim=True).clamp(min=1.0)
    clean = torch.nan_to_num(ranked)
    mean = (clean * weight).sum(0, keepdim=True) / count
    variance = (((clean - mean) ** 2) * weight).sum(0, keepdim=True) / count
    z = (ranked - mean) / variance.sqrt().clamp(min=1e-8)
    return torch.where(valid, z, torch.full_like(z, float("nan")))


def combine(components, weights):
    """Weighted mean of standardised components over whatever is present.

    Averaging only the components that exist on a given stock-day keeps a
    composite alive where one input is missing, instead of propagating that
    factor's coverage hole to the whole book.
    """
    stacked = torch.stack(components)
    valid = torch.isfinite(stacked)
    w = torch.as_tensor(
        weights, device=stacked.device, dtype=stacked.dtype
    ).view(-1, 1, 1) * valid
    total = w.sum(0)
    combined = (torch.nan_to_num(stacked) * w).sum(0) / total.clamp(min=1e-12)
    return torch.where(total > 0, combined, torch.full_like(combined, float("nan")))


def cluster_selection(correlation, scores, limit):
    """Group factors by correlation, then keep the best member of each group.

    Greedy admission answers "is this factor explained by one already kept",
    which is not the same question. Two factors can each sit below the cut
    against the incumbent and still be near-copies of each other, so greedy
    keeps both and the composite double-counts one idea. Average-linkage
    clustering on 1 - |rank correlation|, cut at 1 - limit, asks instead which
    factors form one idea, and every idea then contributes exactly one
    representative -- the member with the strongest in-sample rank IC.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    size = len(scores)
    if size == 1:
        return [0], {0: 1}
    matrix = np.abs(np.asarray(correlation, dtype=float))
    matrix = np.nan_to_num(matrix, nan=0.0)
    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    # Rank correlation is symmetric by construction; squareform rejects the
    # float noise that einsum-free accumulation can leave behind.
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 2.0)
    labels = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=1.0 - limit, criterion="distance",
    )
    kept = []
    for cluster in sorted(set(labels)):
        members = [i for i in range(size) if labels[i] == cluster]
        kept.append(max(members, key=lambda i: (
            scores[i] if np.isfinite(scores[i]) else -np.inf
        )))
    kept.sort(key=lambda i: -(scores[i] if np.isfinite(scores[i]) else -np.inf))
    return kept, {i: int(labels[i]) for i in range(size)}


def _number(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _percent(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metric_cells(row, keys):
    return "".join(f"<td>{_number(row.get(key))}</td>" for key in keys)


def _build_html(report, path):
    windows = report["window_ranges"]
    provenance = report["window_provenance"]
    cards = "".join(
        f"<div class='metric'>{label}<b>{windows[key][0]}～{windows[key][1]}</b>"
        f"<span class='tag {'insample' if provenance[key] == 'in-sample' else 'oos'}'>"
        f"{'样本内' if provenance[key] == 'in-sample' else '样本外'}</span></div>"
        for key, label in (("select", "样本期"), ("valid", "验证期"), ("test", "测试期"))
        if key in windows
    )

    corr = report["correlation"]
    names = [entry["short"] for entry in report["factors"]]
    header = "".join(f"<th>{name}</th>" for name in names)
    corr_rows = "".join(
        "<tr><td>{}</td>{}</tr>".format(
            names[i],
            "".join(
                "<td class='{}'>{}</td>".format(
                    "hot" if i != j and abs(corr[i][j] or 0) >= report["max_correlation"] else "",
                    _number(corr[i][j], 2),
                )
                for j in range(len(names))
            ),
        )
        for i in range(len(names))
    )

    factor_rows = "".join(
        "<tr class='{}'><td>{}</td><td>{:+d}</td>{}<td>{}</td></tr>".format(
            "pass" if entry["selected"] else "fail",
            f"{entry['short']}<span class='cluster'>#{entry['cluster']}</span>",
            entry["direction"],
            "".join(
                f"<td>{_number(entry['windows'][key]['rank_ic'])}</td>"
                for key in ("select", "valid", "test") if key in entry["windows"]
            ),
            "入选" if entry["selected"] else f"剔除 (|ρ|={_number(entry['max_correlation'], 2)})",
        )
        for entry in report["factors"]
    )

    keys = ("rank_ic", "rank_icir", "rank_t_stat", "turnover",
            "gross_annual", "net_annual")
    head = ("<th>组合</th><th>窗口</th><th>成分数</th><th>RankIC</th><th>ICIR</th>"
            "<th>t值</th><th>换手</th><th>年化毛收益</th><th>年化净收益</th>")
    composite_rows = "".join(
        "<tr class='{}'><td>{}</td><td>{}</td><td>{}</td>{}</tr>".format(
            "pass" if entry["name"].startswith("composite") else "",
            entry["name"], entry["window"], entry["components"],
            _metric_cells(entry, keys),
        )
        for entry in report["composite_table"]
    )

    sweep_rows = "".join(
        "<tr class='{}'><td>{}</td><td>{}</td><td>{}</td>{}</tr>".format(
            "pass" if entry["best"] else "",
            entry["smoothing"], entry["placement"], entry["window"],
            _metric_cells(entry, keys),
        )
        for entry in report["smoothing_table"]
    )

    neutral_rows = "".join(
        "<tr><td>{}</td><td>{}</td>{}</tr>".format(
            entry["variant"], entry["window"], _metric_cells(entry, keys)
        )
        for entry in report["neutralization_table"]
    )

    quantile_rows = "".join(
        "<tr><td>{:.0%}</td><td>{}</td>{}</tr>".format(
            entry["quantile"], entry["window"], _metric_cells(entry, keys)
        )
        for entry in report["quantile_table"]
    )

    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>因子组合与换手/中性化扫描</title><style>
body{{margin:0;background:#f4f6f9;color:#172033;font:14px/1.55 system-ui,'Microsoft YaHei'}}
main{{max-width:1500px;margin:auto;padding:28px}}
.card{{background:white;border:1px solid #dfe4ec;border-radius:12px;padding:18px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.metric{{border:1px solid #dfe4ec;border-radius:9px;padding:12px}}.metric b{{display:block;font-size:19px}}
.tag{{display:inline-block;margin-top:6px;padding:1px 8px;border-radius:20px;font-size:12px}}
.tag.oos{{background:#e6f4ee;color:#1c6b4a}}.tag.insample{{background:#fdecec;color:#a02c2c}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}
th,td{{padding:8px 10px;border-bottom:1px solid #e4e8ef;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
tr.pass{{background:#edf8f4}}tr.fail{{color:#71798a}}td.hot{{background:#fdecec;color:#a02c2c}}
.cluster{{display:inline-block;margin-left:8px;padding:0 7px;border-radius:20px;background:#eef1f6;color:#5b6474;font-size:12px}}
.note{{color:#687187}}h2{{margin:0 0 10px;font-size:17px}}
</style></head><body><main>
<h1>因子组合 · 换手 · 中性化</h1>
<p class='note'>入选完全在样本期内决定，验证期与测试期仅作样本外读数。RankIC 为周度截面 Spearman；净收益已扣 {report['cost_bps']:.0f} bps 双边成本，单次为每次调仓，年化按 52 周折算。</p>
<section class='card'><div class='grid'>{cards}
<div class='metric'>相关度上限<b>{report['max_correlation']:.2f}</b><span class='tag oos'>样本期内测</span></div></div></section>

<section class='card'><h2>候选因子（{report['selected_count']} / {len(report['factors'])} 入选，#n 为相关度簇号）</h2>
<div class='scroll'><table><thead><tr><th>因子</th><th>方向</th><th>样本RankIC</th><th>验证RankIC</th><th>测试RankIC</th><th>结论</th></tr></thead>
<tbody>{factor_rows}</tbody></table></div></section>

<section class='card'><h2>样本期两两秩相关（红格 ≥ {report['max_correlation']:.2f}）</h2>
<div class='scroll'><table><thead><tr><th></th>{header}</tr></thead><tbody>{corr_rows}</tbody></table></div></section>

<section class='card'><h2>最优单因子 vs 组合</h2>
<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{composite_rows}</tbody></table></div></section>

<section class='card'><h2>平滑天数 × 放置位置扫描（降换手）</h2>
<p class='note'>两种放置不是同一件事：<code>smooth→combine</code> 先让每个因子与自己的历史平均，再由合成抵消分歧；<code>combine→smooth</code> 先抵消分歧再平均残余。前者通常换手更低，后者保留更多日内信号。绿行为该窗口下年化净收益最高的组合。</p>
<div class='scroll'><table><thead><tr><th>平滑天数</th><th>放置</th><th>窗口</th><th>RankIC</th><th>ICIR</th><th>t值</th><th>换手</th><th>年化毛收益</th><th>年化净收益</th></tr></thead>
<tbody>{sweep_rows}</tbody></table></div></section>

<section class='card'><h2>多空腿宽度扫描（尾部诊断）</h2>
<p class='note'>RankIC 为正而窄腿亏损，说明收益信息不在极值端：腿放宽后毛收益回升即为尾部反转，若各腿宽一致亏损则是信号本身失效。</p>
<div class='scroll'><table><thead><tr><th>单边腿宽</th><th>窗口</th><th>RankIC</th><th>ICIR</th><th>t值</th><th>换手</th><th>年化毛收益</th><th>年化净收益</th></tr></thead>
<tbody>{quantile_rows}</tbody></table></div></section>

<section class='card'><h2>中性化前后</h2>
<div class='scroll'><table><thead><tr><th>口径</th><th>窗口</th><th>RankIC</th><th>ICIR</th><th>t值</th><th>换手</th><th>年化毛收益</th><th>年化净收益</th></tr></thead>
<tbody>{neutral_rows}</tbody></table></div></section>
</main></body></html>""", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--factor-dir", action="append", default=[],
                        help="directory of exported candidate_*.parquet")
    parser.add_argument("--factor-parquet", action="append", default=[])
    parser.add_argument("--daily-parquet", required=True)
    parser.add_argument("--pit", required=True)
    parser.add_argument("--exposures", help="industry/size exposures for neutralisation")
    parser.add_argument(
        "--holdout-leaf-dir",
        help="leaf archive covering dates after the training cut; factor "
             "genomes are re-evaluated there so the report has an "
             "out-of-sample column",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-end", help="last date the search saw")
    for key, label, begin, finish in DEFAULT_WINDOWS:
        parser.add_argument(f"--{key}-start", default=begin, help=f"{label}起始")
        parser.add_argument(f"--{key}-end", default=finish, help=f"{label}结束")
    parser.add_argument("--max-correlation", type=float, default=0.6)
    parser.add_argument("--smoothing", default="1,3,5,10,20",
                        help="trailing signal-average days to sweep")
    parser.add_argument("--base-smoothing", type=int, default=5,
                        help="smoothing used for selection and the composite tables")
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument(
        "--quantile-sweep", default="0.05,0.1,0.2,0.33,0.5",
        help="long-short leg sizes to sweep; separates a tail reversal from a "
             "signal that has simply stopped working",
    )
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")

    windows = tuple(
        (key, label, getattr(args, f"{key}_start"), getattr(args, f"{key}_end"))
        for key, label, _b, _f in DEFAULT_WINDOWS
    )
    span_start = min(begin for _k, _l, begin, _f in windows)
    span_end = max(finish for _k, _l, _b, finish in windows)

    paths = _discover(args.factor_dir, args.factor_parquet)
    instruments = load_pit_codes(args.pit, span_start, span_end)
    dates = load_pit_dates(args.pit, span_start, span_end)
    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device)
    pool &= torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    masks = {
        key: _window_mask(dates, begin, finish, device)
        for key, _label, begin, finish in windows
    }
    print(f"[combine] grid I={len(instruments)} D={len(dates)}; {len(paths)} factors", flush=True)

    holdout = None
    if args.holdout_leaf_dir:
        registry = build_operator_registry()
        holdout_context, holdout_instruments, holdout_dates = _load_leaf_context(
            Path(args.holdout_leaf_dir), device
        )
        holdout = (registry, holdout_context, holdout_instruments, holdout_dates)
        print(f"[combine] hold-out leaves {holdout_dates[0]}..{holdout_dates[-1]} "
              f"I={len(holdout_instruments)}", flush=True)

    missing = torch.full((len(instruments), len(dates)), float("nan"), device=device)
    raw = []
    for path in paths:
        values = load_factor_parquet(path, instruments, dates, device=device)
        sidecar = path.with_suffix(path.suffix + ".json")
        if holdout is not None and sidecar.exists():
            registry, holdout_context, holdout_instruments, holdout_dates = holdout
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            genome = genome_from_export(payload["genome"], registry)
            # A hold-out archive publishes one family's leaves. Pools drawn from
            # several families are the normal case, so a genome the archive
            # cannot feed is left at its exported extent rather than crashing
            # the run; its empty windows then show up as a component count.
            unmet = sorted(set(genome.required_fields) - set(holdout_context))
            if unmet:
                print(f"[combine] {_short_name(path)}: hold-out lacks {unmet}; "
                      "left at its exported range", flush=True)
            else:
                extended = _evaluate_genome(genome, holdout_context, registry).float()
                values = _splice(values, extended, holdout_instruments,
                                 holdout_dates, instruments, dates)
        raw.append(torch.where(pool, values, missing))
    # Smoothing runs after the splice so the trailing mean is continuous across
    # the join rather than restarting at the hold-out boundary. The unsmoothed
    # panel is kept because the sweep below must measure a smoothing window in
    # absolute terms; applying it on top of the base would silently compound
    # the two and mislabel every row of that table.
    factors = [trailing_signal_mean(value, args.base_smoothing) for value in raw]

    # Direction and selection are settled in the sample window only.
    directions = []
    for path, value in zip(paths, factors):
        scoped = _scope(value, masks["select"])
        scoped_fwd = _scope(fwd, masks["select"])
        valid = torch.isfinite(scoped) & torch.isfinite(scoped_fwd)
        series = _pearson(
            cross_section_rank(scoped), cross_section_rank(scoped_fwd), valid
        )
        series = series[torch.isfinite(series)]
        direction = 1 if (series.numel() == 0 or series.mean().item() >= 0) else -1
        directions.append(direction)

    select_ic = []
    entries = []
    for path, value, direction in zip(paths, factors, directions):
        per_window = {
            key: evaluate(value, fwd, masks[key], direction, args.cost_bps,
                          args.quantile, args.min_cross_section)
            for key, _label, _b, _f in windows
        }
        select_ic.append(per_window["select"]["rank_ic"])
        entries.append({
            "path": str(path), "short": _short_name(path),
            "direction": direction, "windows": per_window,
        })
        print(f"[combine] {_short_name(path)} dir={direction:+d} "
              f"selectRankIC={per_window['select']['rank_ic']:+.4f}", flush=True)

    aligned = [
        rank_z(value) * direction for value, direction in zip(factors, directions)
    ]
    # Ranking is day-local, so scoping a ranked panel and ranking a scoped one
    # agree. Ranking once per factor rather than once per pair turns 5700
    # rankings into 76 at this pool size.
    select_mask = masks["select"]
    ranked = [_scope(cross_section_rank(value), select_mask) for value in aligned]
    correlation = [[1.0] * len(aligned) for _ in aligned]
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            value = mean_rank_correlation(
                ranked[i], ranked[j], args.min_cross_section, pre_ranked=True
            )
            correlation[i][j] = correlation[j][i] = value
        if len(aligned) > 20 and (i + 1) % 20 == 0:
            print(f"[combine] correlations {i + 1}/{len(aligned)}", flush=True)
    del ranked

    kept, cluster_of = cluster_selection(
        correlation, select_ic, args.max_correlation
    )
    for index, entry in enumerate(entries):
        entry["selected"] = index in kept
        entry["cluster"] = cluster_of[index]
        siblings = [i for i in kept if i != index]
        entry["max_correlation"] = max(
            (abs(correlation[index][other]) for other in siblings
             if np.isfinite(correlation[index][other])),
            default=0.0,
        )
    sizes = collections.Counter(cluster_of.values())
    print(f"[combine] {len(aligned)} factors form {len(sizes)} correlation "
          f"clusters at |rho| >= {args.max_correlation}; "
          f"largest holds {max(sizes.values())}", flush=True)
    for index in kept:
        print(f"    cluster {cluster_of[index]:3d} "
              f"({sizes[cluster_of[index]]:3d} members) -> "
              f"{entries[index]['short']} selectRankIC={select_ic[index]:+.4f}",
              flush=True)

    components = [aligned[i] for i in kept]
    ic_weights = np.array([max(select_ic[i], 0.0) for i in kept], dtype=float)
    if ic_weights.sum() <= 0:
        ic_weights = np.ones(len(kept))
    composites = {
        "composite_equal": combine(components, np.ones(len(components))),
        "composite_ic_weighted": combine(components, ic_weights),
    }
    best_single = kept[0]
    tables = {"best_single": aligned[best_single], **composites}

    # A composite averages whatever components are present on a stock-day, so
    # a component whose export stops before a window simply drops out there and
    # the row silently describes a different portfolio. Count what actually
    # contributed per window and publish it beside the metrics.
    contributing = {
        key: sum(
            bool(torch.isfinite(_scope(components[c], masks[key])).any())
            for c in range(len(components))
        )
        for key, _label, _b, _f in windows
    }
    for key, count in contributing.items():
        if count < len(components):
            print(
                f"[combine] window {key}: only {count}/{len(components)} "
                "components have data; the composite there is not the same book",
                flush=True,
            )

    composite_table = []
    for name, value in tables.items():
        display = entries[best_single]["short"] if name == "best_single" else name
        for key, _label, _b, _f in windows:
            row = evaluate(value, fwd, masks[key], 1, args.cost_bps,
                           args.quantile, args.min_cross_section)
            composite_table.append({
                "name": display, "window": key,
                "components": 1 if name == "best_single" else contributing[key],
                **row,
            })

    # Smoothing sweep, from the unsmoothed panel so a window is measured in
    # absolute terms. Cost is charged on turnover, so the IC a longer average
    # gives up can still be a net gain.
    #
    # Placement is swept alongside the window because the two orderings are not
    # the same operation. Smoothing each component first averages a factor
    # against its own past, then lets the composite cancel what the components
    # disagree about; combining first cancels the disagreement immediately and
    # averages whatever survived. The second keeps more of the daily signal and
    # the first usually trades less, and which one nets more is an empirical
    # question about how correlated the components' turnover is -- not
    # something to settle by picking one and never measuring the other.
    unsmoothed = combine(
        [rank_z(raw[i]) * directions[i] for i in kept], np.ones(len(kept))
    )
    smoothing_days = [int(v) for v in args.smoothing.split(",") if v.strip()]
    smoothing_rows = []
    for days in smoothing_days:
        placements = {
            "combine→smooth": trailing_signal_mean(unsmoothed, days),
            "smooth→combine": combine(
                [
                    rank_z(trailing_signal_mean(raw[i], days)) * directions[i]
                    for i in kept
                ],
                np.ones(len(kept)),
            ),
        }
        for placement, value in placements.items():
            for key, _label, _b, _f in windows:
                row = evaluate(value, fwd, masks[key], 1, args.cost_bps,
                               args.quantile, args.min_cross_section)
                smoothing_rows.append({
                    "smoothing": days, "placement": placement,
                    "window": key, "best": False, **row,
                })
    for key, _label, _b, _f in windows:
        rows = [r for r in smoothing_rows if r["window"] == key
                and np.isfinite(r["net_annual"])]
        if rows:
            max(rows, key=lambda r: r["net_annual"])["best"] = True

    # A rank IC can stay positive while the extreme names move the wrong way.
    # Widening the leg dilutes the tails, so comparing leg sizes says which of
    # the two is happening -- and a factor whose ordering pays only in the
    # middle is still tradable, just not as a 20% quantile book.
    operational = composites["composite_equal"]
    quantile_rows = []
    for leg in [float(v) for v in args.quantile_sweep.split(",") if v.strip()]:
        for key, _label, _b, _f in windows:
            row = evaluate(operational, fwd, masks[key], 1, args.cost_bps,
                           leg, args.min_cross_section)
            quantile_rows.append({"quantile": leg, "window": key, **row})

    neutralization_rows = []
    if args.exposures:
        styles, industry, _levels = load_daily_exposures(
            args.exposures, dates, instruments, device=device
        )
        size = styles["ln_float_market_cap"]
        variants = {
            "raw": None,
            "industry": BatchedNeutralizer(
                operational.shape, (), industry, args.min_cross_section,
                device=device, rank_space=True,
            ),
            "industry+size": BatchedNeutralizer(
                operational.shape, (size,), industry, args.min_cross_section,
                device=device, rank_space=True,
            ),
        }
        for variant, neutralizer in variants.items():
            value = operational if neutralizer is None else neutralizer(operational)
            for key, _label, _b, _f in windows:
                row = evaluate(value, fwd, masks[key], 1, args.cost_bps,
                               args.quantile, args.min_cross_section)
                neutralization_rows.append({"variant": variant, "window": key, **row})
    else:
        print("[combine] no --exposures; neutralisation section skipped", flush=True)

    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cost_bps": args.cost_bps,
        "quantile": args.quantile,
        "base_smoothing": args.base_smoothing,
        "max_correlation": args.max_correlation,
        "train_end": args.train_end,
        "window_ranges": {k: [b, f] for k, _l, b, f in windows},
        "window_provenance": {
            k: "in-sample" if args.train_end and b <= args.train_end
            else "out-of-sample"
            for k, _l, b, _f in windows
        },
        "selection_rule": "average-linkage correlation clustering on the sample window; the strongest rank IC in each cluster represents it",
        "selected_count": len(kept),
        "factors": entries,
        "correlation": correlation,
        "composite_table": composite_table,
        "smoothing_table": smoothing_rows,
        "quantile_table": quantile_rows,
        "neutralization_table": neutralization_rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe = _safe(report)
    _build_html(safe, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\n[combine] {len(kept)}/{len(entries)} selected -> {destination}")
    print(f"{'series':26} {'window':7} {'n':>3} {'rankIC':>8} {'ICIR':>7} "
          f"{'turnover':>9} {'gross/yr':>9} {'net/yr':>9}")
    for row in composite_table:
        print(f"{row['name'][:26]:26} {row['window']:7} {row['components']:3d} "
              f"{row['rank_ic']:8.4f} {row['rank_icir']:7.3f} {row['turnover']:9.3f} "
              f"{row['gross_annual']:9.2%} {row['net_annual']:8.2%}")
    print("\nsmoothing sweep x placement (equal-weighted composite):")
    for row in smoothing_rows:
        star = " *" if row["best"] else "  "
        print(f"  {row['smoothing']:3d}d {row['placement']:15} {row['window']:7} "
              f"rankIC={row['rank_ic']:+.4f} turnover={row['turnover']:.3f} "
              f"gross/yr={row['gross_annual']:+.2%} "
              f"net/yr={row['net_annual']:+.2%}{star}")
    print("\nlong-short leg width (tail diagnosis):")
    for row in quantile_rows:
        print(f"  {row['quantile']:5.0%} {row['window']:7} "
              f"rankIC={row['rank_ic']:+.4f} turnover={row['turnover']:.3f} "
              f"gross/yr={row['gross_annual']:+.2%} net/yr={row['net_annual']:+.2%}")
    if neutralization_rows:
        print("\nneutralisation:")
        for row in neutralization_rows:
            print(f"  {row['variant']:14} {row['window']:7} rankIC={row['rank_ic']:+.4f} "
                  f"gross/yr={row['gross_annual']:+.2%} net/yr={row['net_annual']:+.2%}")


if __name__ == "__main__":
    main()
