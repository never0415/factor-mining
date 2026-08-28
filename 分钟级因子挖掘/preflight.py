"""Fail loudly when a search configuration cannot produce a scored genome.

Labels are only defined on rebalance dates, so a weekly run has roughly one
fifth of the label days a daily run has. Passing daily-calibrated window sizes
to a weekly run therefore starves every walk-forward fold: each one is skipped
for having too few IC days, the population scores -1e9 across the board, and
nothing in the output says why. This module turns that silent washout into a
message naming the parameter to change.
"""

from dataclasses import dataclass

from min_gp.evaluation.incremental import walk_forward_splits
from min_gp.label import week_end_mask


@dataclass(frozen=True)
class FoldFeasibility:
    total_days: int
    label_days: int
    splits: int
    scorable_folds: int
    fold_label_days: tuple[tuple[int, int], ...]
    min_train_label_days: int
    min_valid_label_days: int
    required_label_days: int
    required_folds: int

    @property
    def ok(self) -> bool:
        return self.scorable_folds >= self.required_folds


def label_day_flags(dates, rule, cfg):
    """Which dates carry a forward-return label under this rebalance rule."""
    if rule == "daily":
        horizon = cfg.holding_period
        flags = [index < len(dates) - horizon for index in range(len(dates))]
    elif rule == "week_end":
        mask = week_end_mask(dates, "cpu").tolist()
        # The final rebalance date has no subsequent exit, so it carries no
        # label even though it is a rebalance date.
        remaining = sum(mask)
        flags = []
        for flag in mask:
            if flag and remaining > 1:
                flags.append(True)
                remaining -= 1
            elif flag:
                flags.append(False)
            else:
                flags.append(False)
    else:
        raise ValueError(f"unknown rebalance rule: {rule}")
    # A strict N-day signal average is unavailable before N observations have
    # accumulated, even when a return label exists on those early dates.
    for index in range(min(cfg.signal_average_days - 1, len(flags))):
        flags[index] = False
    return flags


def assess_folds(dates, rule, cfg) -> FoldFeasibility:
    flags = label_day_flags(dates, rule, cfg)
    splits = walk_forward_splits(len(dates), cfg)
    per_fold = tuple(
        (sum(flags[train]), sum(flags[valid])) for train, valid in splits
    )
    scorable = sum(
        valid_days >= cfg.min_valid_ic_days
        and (
            cfg.direction_mode == "paper"
            or train_days >= cfg.min_valid_ic_days
        )
        for train_days, valid_days in per_fold
    )
    return FoldFeasibility(
        total_days=len(dates),
        label_days=sum(flags),
        splits=len(splits),
        scorable_folds=scorable,
        fold_label_days=per_fold,
        min_train_label_days=min(
            (train for train, _ in per_fold), default=0
        ),
        min_valid_label_days=min(
            (valid for _, valid in per_fold), default=0
        ),
        required_label_days=cfg.min_valid_ic_days,
        required_folds=cfg.min_folds,
    )


def describe(report: FoldFeasibility, rule: str) -> str:
    return (
        f"[preflight] rule={rule} days={report.total_days} "
        f"label_days={report.label_days} folds={report.splits}/"
        f"{report.required_folds} scorable={report.scorable_folds} "
        f"min_train/valid_label_days={report.min_train_label_days}/"
        f"{report.min_valid_label_days} required={report.required_label_days}"
    )


def check_or_exit(dates, rule, cfg, parser=None):
    """Print the fold budget, and refuse to start a run that cannot score."""
    report = assess_folds(dates, rule, cfg)
    print(describe(report, rule), flush=True)
    if report.ok:
        return report
    reasons = []
    if report.splits < report.required_folds:
        reasons.append(
            f"only {report.splits} walk-forward folds fit in {report.total_days} "
            f"trading days but --min-folds requires {report.required_folds}; "
            f"widen --start/--end, or lower --min-train-days/--valid-days"
        )
    short_train = [
        train for train, valid in report.fold_label_days
        if train < report.required_label_days
    ]
    short_valid = [
        valid for train, valid in report.fold_label_days
        if valid < report.required_label_days
    ]
    if cfg.direction_mode == "discovery" and short_train:
        reasons.append(
            f"discovery direction needs at least {report.required_label_days} "
            f"training labels per usable fold, but some folds have as few as "
            f"{min(short_train)}; raise --min-train-days or lower "
            f"--min-valid-ic-days"
        )
    if short_valid:
        reasons.append(
            f"the thinnest fold holds {report.min_valid_label_days} labelled "
            f"days but --min-valid-ic-days requires {report.required_label_days}; "
            f"under --rebalance {rule} a validation window of "
            f"{cfg.valid_days} trading days yields about "
            f"{report.min_valid_label_days} labels, so raise --valid-days or "
            f"lower --min-valid-ic-days"
        )
    reasons.append(
        f"only {report.scorable_folds} folds satisfy all label requirements "
        f"but --min-folds requires {report.required_folds}"
    )
    message = "configuration cannot score any genome: " + "; ".join(reasons)
    if parser is not None:
        parser.error(message)
    raise SystemExit(f"[preflight] {message}")
