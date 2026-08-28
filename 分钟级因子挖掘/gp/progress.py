"""Candidate-level progress for generations that take minutes, not seconds.

A generation prints one line when it finishes. At a minute per candidate that
is a silent half hour with no way to tell a slow run from a hung one, and no
way to see the cost distribution that made it slow. This reports as it goes and
names the slowest candidate, which is usually the one worth investigating.
"""

import time


def score_population(evaluate, population, label, verbose=True, every=10):
    """Evaluate a generation, reporting progress and the slowest candidate."""
    scored = []
    started = time.time()
    slowest_seconds = 0.0
    slowest_genome = None
    last_report = started
    total = len(population)
    for index, genome in enumerate(population, start=1):
        began = time.time()
        score = evaluate(genome)
        elapsed = time.time() - began
        if elapsed > slowest_seconds:
            slowest_seconds, slowest_genome = elapsed, genome
        scored.append((score, genome))
        if not verbose:
            continue
        due = index % every == 0 or index == total
        if due or time.time() - last_report >= 30:
            last_report = time.time()
            done = time.time() - started
            rate = done / index
            valid = sum(1 for value, _ in scored if value.valid)
            best = max(
                (value.robust_ic for value, _ in scored if value.valid),
                default=None,
            )
            best_text = "none yet" if best is None else f"{best:+.4f}"
            print(
                f"[{label}] {index}/{total} valid={valid} bestIC={best_text} "
                f"{rate:.1f}s/cand eta={rate * (total - index):.0f}s",
                flush=True,
            )
    if verbose and slowest_genome is not None and slowest_seconds > 5.0:
        print(
            f"[{label}] slowest candidate {slowest_seconds:.1f}s: "
            f"{str(slowest_genome)[:110]}",
            flush=True,
        )
    return scored
