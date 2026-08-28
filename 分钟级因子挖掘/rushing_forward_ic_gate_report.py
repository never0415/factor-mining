"""Three-stage RankIC gate for Rushing-Forward candidates.

The gate scores what the GP optimised.  ``robust_ic`` is a Spearman rank IC,
so the tradable direction is fixed from the sample-period *rank* IC and the
threshold is applied to rank IC.  The earlier revision locked direction and
applied the threshold to raw Pearson IC, which on these fat-tailed
minute-derived factors runs two to four times smaller than the rank IC that
produced it; a rank-IC-calibrated threshold therefore rejected every
candidate whose rank IC was in fact stable and correctly signed.

Pearson IC is still reported, and its sign agreement with the rank IC is a
separate published flag: a factor whose ordering pays but whose extreme
values do not is tradable only in quantile form, and that distinction should
be visible rather than folded into a single number.

Window provenance is explicit.  ``--train-end`` declares where the search
actually stopped, and any window overlapping it is marked in-sample.  A gate
whose validation window sits inside the training range proves nothing, so the
report says so instead of presenting the number as out-of-sample evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from min_gp.data import load_pit_daily_mask
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.factors.catalog import genome_from_export
from min_gp.factors.seed_tree import SeedTreeGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.operators import build_operator_registry
from min_gp.report_candidates import _pearson
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


# Windows are CLI-configurable; these defaults assume a 2022-12-31 training
# cut, which leaves 2023-2024 as a genuine validation range and 2025+ as the
# untouched hold-out.
RAW_MINUTE_FIELDS = {"open", "high", "low", "close", "volume"}

DEFAULT_WINDOWS = (
    ("select", "样本期", "2018-01-02", "2022-12-31"),
    ("valid", "验证期", "2023-01-01", "2024-12-31"),
    ("test", "测试期", "2025-01-02", "2026-07-31"),
)


def _load_metadata(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(x) for x in payload["instruments"]], [str(x) for x in payload["dates"]]


def _grid_from_parquet(path: Path) -> tuple[list[str], list[str]]:
    """Recover the (instrument, date) grid an exported factor was written on.

    ``factor_export`` writes instrument and trade_date beside every value, so
    the grid is self-describing and a separate metadata sidecar is optional.
    """
    table = pq.read_table(path, columns=["instrument", "trade_date"])
    instruments = table.column("instrument").combine_chunks().to_pylist()
    dates = table.column("trade_date").combine_chunks().to_pylist()
    ordered_dates = list(dict.fromkeys(dates))
    ordered_instruments = list(dict.fromkeys(instruments))
    if len(ordered_instruments) * len(ordered_dates) != table.num_rows:
        raise ValueError(f"{path} is not a dense instrument x date grid")
    return ordered_instruments, ordered_dates


def _load_context(directory: Path, device: str) -> dict[str, torch.Tensor]:
    """Every leaf the directory publishes, keyed by its archive name.

    Reading whatever is present rather than a fixed list lets the same gate
    score a free-form tree, whose leaves are not known until the sidecar is
    read, against the same hold-out panel as a fixed skeleton.
    """
    context = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                context[key] = torch.as_tensor(archive[key], device=device)
    if not context:
        raise SystemExit(f"no .npz leaves found in {directory}")
    return context


def _evaluate_genome(genome, context, registry, chunk_rows):
    """Skeleton genomes bind their own registry; seed trees take one."""
    if isinstance(genome, SeedTreeGenome):
        return genome.evaluate(context, registry, chunk_rows)
    return genome.evaluate(context, chunk_rows=chunk_rows)


def _load_candidates(directory: Path, registry):
    candidates = []
    for sidecar in sorted(directory.glob("candidate_*_rank0.parquet.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        candidates.append({
            "id": sidecar.name.removesuffix(".parquet.json"),
            "parquet": sidecar.with_suffix(""),
            "expression": record["expression"],
            "fitness": record["fitness"],
            "genome": genome_from_export(record["genome"], registry),
        })
    if not candidates:
        raise SystemExit(f"no rank-0 candidates found in {directory}")
    return candidates


def _load_exported_factor(path: Path, shape, device: str) -> torch.Tensor:
    table = pq.read_table(path, columns=["factor"])
    expected = int(np.prod(shape))
    if table.num_rows != expected:
        raise ValueError(f"{path} has {table.num_rows} rows, expected {expected}")
    values = table.column("factor").combine_chunks().to_numpy(zero_copy_only=False)
    return torch.as_tensor(values.reshape(shape), device=device)


def _series_stats(series: np.ndarray) -> dict:
    """Mean, IR and t-statistic of a non-overlapping weekly IC series."""
    finite = series[np.isfinite(series)]
    if finite.size < 2:
        return {
            "mean": float("nan"), "ir": float("nan"),
            "t_stat": float("nan"), "win_rate": float("nan"),
            "weeks": int(finite.size),
        }
    mean, sd = float(finite.mean()), float(finite.std(ddof=1))
    ir = mean / sd if sd > 0 else float("nan")
    return {
        "mean": mean,
        "ir": ir,
        # Weekly rebalance with a one-week horizon leaves the IC series
        # non-overlapping, so no autocorrelation correction is warranted.
        "t_stat": ir * float(np.sqrt(finite.size)) if np.isfinite(ir) else float("nan"),
        "win_rate": float((finite > 0).mean()),
        "weeks": int(finite.size),
    }


def _metric(factor, fwd, dates, begin: str, finish: str) -> dict:
    date_array = np.asarray(dates)
    mask = torch.as_tensor(
        (date_array >= begin) & (date_array <= finish), device=factor.device
    )
    scoped_factor = torch.where(
        mask.unsqueeze(0), factor, torch.full_like(factor, float("nan"))
    )
    scoped_fwd = torch.where(
        mask.unsqueeze(0), fwd, torch.full_like(fwd, float("nan"))
    )
    valid = torch.isfinite(scoped_factor) & torch.isfinite(scoped_fwd)
    ic = _pearson(scoped_factor, scoped_fwd, valid)
    rank_ic = _pearson(
        cross_section_rank(scoped_factor), cross_section_rank(scoped_fwd), valid
    )
    ic_np = ic.float().cpu().numpy()
    rank_np = rank_ic.float().cpu().numpy()
    ic_stats = _series_stats(ic_np)
    rank_stats = _series_stats(rank_np)
    return {
        "raw_ic": ic_stats["mean"],
        "raw_rank_ic": rank_stats["mean"],
        "ic_weeks": ic_stats["weeks"],
        "rank_ic_weeks": rank_stats["weeks"],
        # Direction-free dispersion statistics: sign flips with direction, so
        # only the magnitude of ir/t_stat is meaningful before alignment.
        "raw_rank_icir": rank_stats["ir"],
        "raw_rank_t_stat": rank_stats["t_stat"],
        "raw_rank_win_rate": rank_stats["win_rate"],
    }


def _window_is_in_sample(begin: str, finish: str, train_end: str | None) -> bool:
    """A window overlapping the training range cannot serve as validation."""
    return train_end is not None and begin <= train_end


class _Grid:
    """One aligned (instrument, date) panel plus its labels and pool mask.

    Windows are scored against whichever grid actually covers their dates, so
    moving the training cut earlier does not require the validation window to
    keep coming from the training panel. That coupling is what let a window
    inside the search range be published as validation evidence.
    """

    def __init__(self, name, instruments, dates, daily_parquet, pit, device):
        self.name = name
        self.instruments = instruments
        self.dates = dates
        self.shape = (len(instruments), len(dates))
        close = load_daily_close_tensor(
            daily_parquet, dates, instruments, device=device
        )
        self.pool = load_pit_daily_mask(
            pit, dates, instruments, device=device
        ) & torch.isfinite(close)
        forward = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
        self.fwd = torch.where(
            self.pool, forward, torch.full_like(forward, float("nan"))
        )
        self.first, self.last = min(dates), max(dates)

    def window_dates(self, begin: str, finish: str) -> set:
        """Trading days this grid holds inside a window.

        Containment cannot be tested on the range endpoints: a window boundary
        is a calendar date that is often a weekend, so a panel ending on the
        last trading day of 2022 would appear not to cover a window ending
        2022-12-31.
        """
        return {day for day in self.dates if begin <= day <= finish}

    def scoped(self, factor, signal_average_days):
        factor = trailing_signal_mean(factor, signal_average_days)
        return torch.where(
            self.pool, factor, torch.full_like(factor, float("nan"))
        )


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _number(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _build_html(report: dict, path: Path) -> None:
    rows = []
    for candidate in report["candidates"]:
        values = candidate["windows"]
        status = "通过" if candidate["passed"] else "未通过"
        row_class = "pass" if candidate["passed"] else "fail"
        pearson_flag = "一致" if candidate["pearson_sign_agrees"] else "背离"
        rows.append(
            f"<tr class='{row_class}'><td>{candidate['id']}</td>"
            f"<td>{candidate['direction']:+d}</td>"
            f"<td>{_number(candidate['train_robust_rank_ic'])}</td>"
            f"<td>{_number(values['select']['rank_ic'])}</td>"
            f"<td>{_number(values['valid']['rank_ic'])}</td>"
            f"<td>{_number(values.get('test', {}).get('rank_ic'))}</td>"
            f"<td>{_number(values['valid']['rank_icir'], 3)}</td>"
            f"<td>{_number(values['valid']['rank_t_stat'], 2)}</td>"
            f"<td>{_number(values['select']['ic'])}</td>"
            f"<td>{_number(values['valid']['ic'])}</td>"
            f"<td>{_number(values.get('test', {}).get('ic'))}</td>"
            f"<td>{pearson_flag}</td>"
            f"<td><b>{status}</b></td></tr>"
        )
    details = "".join(
        f"<details><summary>{candidate['id']}</summary><code>{candidate['expression']}</code></details>"
        for candidate in report["candidates"]
    )
    ranges = report["window_ranges"]
    provenance = report["window_provenance"]
    cards = "".join(
        f"<div class='metric'>{label}"
        f"<b>{ranges[key][0]}～{ranges[key][1]}</b>"
        f"<span class='tag {'insample' if provenance[key] == 'in-sample' else 'oos'}'>"
        f"{'样本内' if provenance[key] == 'in-sample' else '样本外'}</span></div>"
        for key, label in (("select", "样本期"), ("valid", "验证期"), ("test", "测试期"))
        if key in ranges
    )
    warning = ""
    if not report["gate_trustworthy"]:
        warning = (
            "<section class='card warn'><b>⚠ 验证期落在训练区间内</b>"
            f"（训练截止 {report['train_end']}）。此处的验证 IC 是样本内拟合值，"
            "不能作为样本外证据；请用早于训练截止的区间重新训练后再跑本门槛。</section>"
        )
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Rushing Forward RankIC门槛检查</title><style>
body{{margin:0;background:#f4f6f9;color:#172033;font:14px/1.55 system-ui,'Microsoft YaHei'}}
main{{max-width:1550px;margin:auto;padding:28px}}.card{{background:white;border:1px solid #dfe4ec;border-radius:12px;padding:18px;margin:14px 0}}
.card.warn{{background:#fff6ed;border-color:#f0c9a0;color:#8a4b12}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{border:1px solid #dfe4ec;border-radius:9px;padding:12px}}.metric b{{display:block;font-size:19px}}
.tag{{display:inline-block;margin-top:6px;padding:1px 8px;border-radius:20px;font-size:12px}}
.tag.oos{{background:#e6f4ee;color:#1c6b4a}}.tag.insample{{background:#fdecec;color:#a02c2c}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:9px 11px;border-bottom:1px solid #e4e8ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}
tr.pass{{background:#edf8f4}}tr.fail{{color:#71798a}}code{{word-break:break-all;white-space:normal}}.note{{color:#687187}}details{{padding:8px 0;border-bottom:1px solid #edf0f4}}
</style></head><body><main><h1>Rushing Forward：三阶段 RankIC 门槛检查</h1>
<p class='note'>方向由样本期 <b>RankIC</b> 符号锁定，门槛也施加在 RankIC 上——与 GP 优化的 <code>robust_ic</code> 同口径。普通 IC（Pearson）仅作报告，其符号是否与 RankIC 一致单独标注：背离说明收益只存在于排序中段，应以分组而非线性加权交易。测试期不参与通过判定。</p>
{warning}
<section class='card'><div class='grid'>{cards}<div class='metric'>硬门槛<b>样本与验证 RankIC ≥ {report['threshold']:.3f}</b><span class='tag oos'>同向</span></div></div></section>
<section class='card'><h2>筛选结果：{report['passed_count']} / {len(report['candidates'])} 通过</h2><div class='scroll'><table><thead><tr><th>因子</th><th>样本锁定方向</th><th>GP稳健RankIC</th><th>样本RankIC</th><th>验证RankIC</th><th>测试RankIC</th><th>验证ICIR</th><th>验证t值</th><th>样本IC</th><th>验证IC</th><th>测试IC</th><th>IC符号</th><th>结论</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='card'><h2>表达式</h2>{details}</section></main></body></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument(
        "--train-metadata",
        help="optional metadata.json; the exported parquet grid is used when omitted",
    )
    parser.add_argument(
        "--holdout-leaf-dir", "--test-leaf-dir", dest="holdout_leaf_dir",
        help="leaf directory covering dates after the training cut; windows it "
             "covers are re-evaluated from the genome instead of the export",
    )
    parser.add_argument("--daily-parquet", required=True)
    parser.add_argument("--pit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--threshold", type=float, default=0.025,
        help="minimum direction-aligned RankIC in both sample and validation",
    )
    parser.add_argument(
        "--no-pearson-sign", dest="require_pearson_sign", action="store_false",
        help="drop the second track and gate on rank IC alone",
    )
    parser.add_argument(
        "--pearson-floor", type=float, default=None,
        help="optional third gate: a numeric floor on direction-aligned "
             "Pearson IC, beyond the sign agreement already required",
    )
    parser.add_argument(
        "--train-end",
        help="last date the search saw; windows overlapping it are marked in-sample",
    )
    for key, label, begin, finish in DEFAULT_WINDOWS:
        parser.add_argument(f"--{key}-start", default=begin, help=f"{label}起始")
        parser.add_argument(f"--{key}-end", default=finish, help=f"{label}结束")
    parser.add_argument(
        "--holdout-minute-parquet",
        help="minute parquet used to supply raw OHLCV leaves a free-form "
             "tree needs on the hold-out grid",
    )
    parser.add_argument("--signal-average-days", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")

    windows = tuple(
        (key, label, getattr(args, f"{key}_start"), getattr(args, f"{key}_end"))
        for key, label, _begin, _finish in DEFAULT_WINDOWS
    )

    factor_dir = Path(args.factor_dir)
    registry = build_operator_registry()
    candidates = _load_candidates(factor_dir, registry)

    if args.train_metadata:
        train_instruments, train_dates = _load_metadata(Path(args.train_metadata))
    else:
        train_instruments, train_dates = _grid_from_parquet(candidates[0]["parquet"])
        print(
            f"[rushing-ic-gate] grid recovered from {candidates[0]['parquet'].name}: "
            f"I={len(train_instruments)} D={len(train_dates)}",
            flush=True,
        )
    train_grid = _Grid(
        "train", train_instruments, train_dates,
        args.daily_parquet, args.pit, device,
    )

    holdout_grid, holdout_context = None, None
    if args.holdout_leaf_dir:
        leaf_dir = Path(args.holdout_leaf_dir)
        holdout_instruments, holdout_dates = _load_metadata(leaf_dir / "metadata.json")
        holdout_context = _load_context(leaf_dir, device)
        # A free-form tree may also need raw minute prices, which live in
        # the minute parquet rather than the leaf archive.
        wanted = {
            name for candidate in candidates
            for name in candidate["genome"].required_fields
        }
        raw_fields = sorted(wanted & RAW_MINUTE_FIELDS - set(holdout_context))
        if raw_fields:
            if not args.holdout_minute_parquet:
                raise SystemExit(
                    f"candidates need raw minute fields {raw_fields} on the "
                    "hold-out grid; pass --holdout-minute-parquet"
                )
            minute, _meta = build_minute_slice(
                args.holdout_minute_parquet,
                min(holdout_dates), max(holdout_dates), fields=tuple(raw_fields),
                instruments=holdout_instruments, dates=holdout_dates,
                device=device,
            )
            holdout_context.update(minute)
        holdout_grid = _Grid(
            "holdout", holdout_instruments, holdout_dates,
            args.daily_parquet, args.pit, device,
        )
        print(
            f"[rushing-ic-gate] holdout grid {holdout_grid.first}..{holdout_grid.last} "
            f"I={len(holdout_instruments)} D={len(holdout_dates)}",
            flush=True,
        )
    else:
        print("[rushing-ic-gate] no --holdout-leaf-dir; only the exported grid is scored", flush=True)

    # Each window is scored on the panel that actually contains it. The export
    # wins where both cover a window, since re-evaluating the genome there
    # would only reproduce it.
    assignment = {}
    available = [("train", train_grid)]
    if holdout_grid is not None:
        available.append(("holdout", holdout_grid))
    for key, _label, begin, finish in windows:
        held = {name: grid.window_dates(begin, finish) for name, grid in available}
        union = set().union(*held.values())
        if not union:
            print(
                f"[rushing-ic-gate] window {key} ({begin}..{finish}) has no "
                "trading days in any supplied grid; skipped",
                flush=True,
            )
            continue
        # Prefer the export where both panels hold the whole window, since
        # re-evaluating the genome there would only reproduce it.
        complete = [name for name, days in held.items() if days == union]
        if not complete:
            raise SystemExit(
                f"window {key} ({begin}..{finish}) is split across grids "
                f"({ {name: len(days) for name, days in held.items()} }); "
                "supply a panel that spans it"
            )
        assignment[key] = complete[0]
    for required in ("select", "valid"):
        if required not in assignment:
            raise SystemExit(f"the {required} window must be covered to run the gate")

    output = []
    sources = set(assignment.values())
    for position, candidate in enumerate(candidates, 1):
        print(f"[rushing-ic-gate] {position}/{len(candidates)} {candidate['id']}", flush=True)
        train_factor = None
        if "train" in sources:
            train_factor = train_grid.scoped(
                _load_exported_factor(candidate["parquet"], train_grid.shape, device),
                args.signal_average_days,
            )
        holdout_factor = None
        if "holdout" in sources:
            holdout_factor = holdout_grid.scoped(
                _evaluate_genome(
                    candidate["genome"], holdout_context, registry,
                    args.chunk_rows,
                ).float(),
                args.signal_average_days,
            )
        raw_windows = {}
        for key, _label, begin, finish in windows:
            source = assignment.get(key)
            if source is None:
                continue
            grid = train_grid if source == "train" else holdout_grid
            factor = train_factor if source == "train" else holdout_factor
            raw_windows[key] = {
                **_metric(factor, grid.fwd, grid.dates, begin, finish),
                "source": source,
            }
        del train_factor, holdout_factor

        # Direction is fixed by the sample-period rank IC, the statistic the
        # search maximised. Locking it on Pearson IC could hand a factor the
        # opposite sign from the one its own fitness settled on.
        sample_rank_ic = raw_windows["select"]["raw_rank_ic"]
        valid_rank_ic = raw_windows["valid"]["raw_rank_ic"]
        direction = 1 if sample_rank_ic >= 0 else -1
        same_direction = bool(
            np.isfinite(sample_rank_ic) and np.isfinite(valid_rank_ic)
            and sample_rank_ic * valid_rank_ic > 0
        )
        result_windows = {}
        for key, raw in raw_windows.items():
            result_windows[key] = {
                **raw,
                "ic": direction * raw["raw_ic"],
                "rank_ic": direction * raw["raw_rank_ic"],
                "rank_icir": direction * raw["raw_rank_icir"],
                "rank_t_stat": direction * raw["raw_rank_t_stat"],
                "rank_win_rate": (
                    raw["raw_rank_win_rate"] if direction > 0
                    else 1.0 - raw["raw_rank_win_rate"]
                ),
            }
        # The two tracks answer different questions. Rank IC is ordering
        # quality and carries the threshold. Pearson IC only has to agree in
        # sign, and it is what catches a factor whose ordering pays while its
        # extreme values move the other way: on the 2025-2026 hold-out this
        # family kept a rank IC of +0.029 while its Pearson IC sat at 0.00001
        # and every long-short leg width lost money gross.
        pearson_sign_agrees = bool(
            all(
                np.isfinite(result_windows[key]["ic"])
                and result_windows[key]["ic"] > 0
                for key in ("select", "valid")
            )
        )
        passed = bool(
            same_direction
            and result_windows["select"]["rank_ic"] >= args.threshold
            and result_windows["valid"]["rank_ic"] >= args.threshold
            and (pearson_sign_agrees or not args.require_pearson_sign)
        )
        if args.pearson_floor is not None:
            passed = passed and all(
                np.isfinite(result_windows[key]["ic"])
                and result_windows[key]["ic"] >= args.pearson_floor
                for key in ("select", "valid")
            )
        output.append({
            "id": candidate["id"],
            "direction": direction,
            "sample_valid_same_direction": same_direction,
            "pearson_sign_agrees": pearson_sign_agrees,
            "passed": passed,
            "threshold": args.threshold,
            "train_robust_rank_ic": candidate["fitness"].get("robust_ic"),
            "stored_gp_direction": candidate["fitness"].get("direction"),
            "expression": candidate["expression"],
            "windows": result_windows,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    provenance = {
        key: "in-sample" if _window_is_in_sample(begin, finish, args.train_end)
        else "out-of-sample"
        for key, _label, begin, finish in windows
        if key in assignment
    }
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "threshold": args.threshold,
        "pearson_floor": args.pearson_floor,
        "direction_rule": "sample-period raw Spearman rank IC sign only",
        "require_pearson_sign": args.require_pearson_sign,
        "pass_rule": (
            "sample and validation raw rank IC same sign; both direction-aligned "
            "rank IC >= threshold"
            + ("; aligned Pearson IC positive in both"
               if args.require_pearson_sign else "")
            + ("" if args.pearson_floor is None
               else "; aligned Pearson IC >= pearson_floor")
        ),
        "test_used_for_selection": False,
        "train_end": args.train_end,
        "window_ranges": {
            key: [begin, finish] for key, _label, begin, finish in windows
            if key in assignment
        },
        "window_provenance": provenance,
        "window_source": assignment,
        # A validation window inside the training range is a fitted number, not
        # evidence. Say so in the artefact rather than letting a reader assume.
        "gate_trustworthy": provenance.get("valid") == "out-of-sample",
        "passed_count": sum(row["passed"] for row in output),
        "candidates": output,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_report = _safe(report)
    _build_html(safe_report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not report["gate_trustworthy"]:
        print(
            "[rushing-ic-gate] WARNING validation window is inside the training "
            f"range (train_end={args.train_end}); the gate is not out-of-sample",
            flush=True,
        )
    print(f"[rushing-ic-gate] {report['passed_count']}/{len(output)} passed")
    for row in output:
        w = row["windows"]
        test_rank = w["test"]["rank_ic"] if "test" in w else float("nan")
        test_ic = w["test"]["ic"] if "test" in w else float("nan")
        print(
            f"  {row['id']} dir={row['direction']:+d} "
            f"RankIC={w['select']['rank_ic']:+.4f}/{w['valid']['rank_ic']:+.4f}/{test_rank:+.4f} "
            f"IC={w['select']['ic']:+.4f}/{w['valid']['ic']:+.4f}/{test_ic:+.4f} "
            f"validICIR={w['valid']['rank_icir']:+.3f} t={w['valid']['rank_t_stat']:+.2f} "
            f"{'PASS' if row['passed'] else 'DROP'}"
        )


if __name__ == "__main__":
    main()
