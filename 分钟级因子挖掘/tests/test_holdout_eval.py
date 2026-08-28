"""Tests for hold-out candidate reconstruction and the out-of-sample guard."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from min_gp.factors import DrippingStoneTemplate, WaitRescueTemplate
from min_gp.factors.event_skeleton import moderate_risk_anchor, wait_rescue_anchor
from min_gp.holdout_eval import (
    assert_disjoint,
    build_genome,
    detect_family,
    load_candidates,
    newey_west_icir,
    required_minute_fields,
    resolve_rebalance,
    resolve_signal_average_days,
    resolve_period,
)


def _record(genome, rank=0, valid=True, family=None, train_end="2024-12-31"):
    record = {
        "pareto_rank": rank,
        "fitness": {"valid": valid, "direction": -1, "robust_ic": 0.05},
        "genome": genome,
        "expression": json.dumps(genome, sort_keys=True),
        "provenance": {"train_end": train_end, "period": 1},
    }
    if family:
        record["family"] = family
    return record


class FamilyDetectionTest(unittest.TestCase):
    def test_skeleton_fields_are_derived_from_genome_and_training_anchor(self):
        volume = _record(wait_rescue_anchor().to_dict(), family="event_skeleton")
        volume["provenance"].update({
            "minute_fields": ["volume"],
            "baseline_genome": wait_rescue_anchor().to_dict(),
        })
        self.assertEqual(required_minute_fields([volume]), ("volume",))
        close = _record(moderate_risk_anchor().to_dict(), family="event_skeleton")
        close["provenance"].update({
            "minute_fields": ["close", "volume"],
            "baseline_genome": moderate_risk_anchor().to_dict(),
        })
        self.assertEqual(
            required_minute_fields([volume, close]), ("close", "volume")
        )

    def test_event_skeleton_family_is_detected(self):
        genome = wait_rescue_anchor().to_dict()
        self.assertEqual(detect_family(_record(genome)), "event_skeleton")
        rebuilt, dropped = build_genome("event_skeleton", genome)
        self.assertEqual(rebuilt, wait_rescue_anchor())
        self.assertEqual(dropped, [])

    def test_explicit_family_tag_wins(self):
        record = _record({"top_k": 10}, family="wait_rescue")
        self.assertEqual(detect_family(record), "wait_rescue")

    def test_family_inferred_from_genome_fields(self):
        anchor = DrippingStoneTemplate.paper_anchor().to_dict()
        self.assertEqual(detect_family(_record(anchor)), "dripping_stone")
        self.assertEqual(
            detect_family(_record(WaitRescueTemplate().to_dict())), "wait_rescue"
        )

    def test_unrecognisable_genome_raises(self):
        with self.assertRaises(ValueError):
            detect_family(_record({"nonsense": 1}))


class GenomeRebuildTest(unittest.TestCase):
    def test_round_trip_preserves_the_genome(self):
        original = DrippingStoneTemplate(
            volume_transform="log1p", clip_k=2.0, session="pm",
            smooth_method="ema", smooth_window=10,
        )
        rebuilt, dropped = build_genome("dripping_stone", original.to_dict())
        self.assertEqual(rebuilt, original)
        self.assertEqual(dropped, [])

    def test_stale_fields_are_dropped_and_reported(self):
        payload = dict(WaitRescueTemplate().to_dict(), retired_knob=7)
        rebuilt, dropped = build_genome("wait_rescue", payload)
        self.assertEqual(rebuilt, WaitRescueTemplate())
        self.assertEqual(dropped, ["retired_knob"])

    def test_out_of_domain_value_is_rejected(self):
        payload = dict(WaitRescueTemplate().to_dict(), top_k=999)
        with self.assertRaises(ValueError):
            build_genome("wait_rescue", payload)


class CandidateLoadingTest(unittest.TestCase):
    def _write(self, records):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_invalid_and_deep_ranks_are_skipped(self):
        anchor = DrippingStoneTemplate.paper_anchor().to_dict()
        variant = DrippingStoneTemplate(session="am").to_dict()
        path = self._write([
            _record(anchor, rank=0),
            _record(variant, rank=3),
            _record(DrippingStoneTemplate(session="pm").to_dict(), valid=False),
        ])
        kept = load_candidates(path, max_rank=0, limit=None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(load_candidates(path, max_rank=3, limit=None)), 2)

    def test_duplicate_expressions_are_collapsed(self):
        anchor = DrippingStoneTemplate.paper_anchor().to_dict()
        path = self._write([_record(anchor), _record(anchor)])
        self.assertEqual(len(load_candidates(path, max_rank=0, limit=None)), 1)


class OutOfSampleGuardTest(unittest.TestCase):
    def test_overlapping_window_is_refused(self):
        records = [_record({"top_k": 10}, train_end="2025-06-30")]
        with self.assertRaises(SystemExit):
            assert_disjoint(records, "2025-01-02")

    def test_disjoint_window_is_accepted(self):
        records = [_record({"top_k": 10}, train_end="2024-12-31")]
        assert_disjoint(records, "2025-01-02")

    def test_mixed_periods_require_an_explicit_choice(self):
        a = _record({"top_k": 10})
        b = _record({"top_k": 15})
        b["provenance"]["period"] = 5
        with self.assertRaises(SystemExit):
            resolve_period([a, b], None)
        self.assertEqual(resolve_period([a, b], 5), 5)

    def test_holdout_reuses_the_training_rebalance_and_signal_window(self):
        row = _record({"top_k": 10})
        row["provenance"].update({
            "rebalance_rule": "week_end", "signal_average_days": 5,
        })
        self.assertEqual(resolve_rebalance([row], None), "week_end")
        self.assertEqual(resolve_signal_average_days([row], None), 5)
        with self.assertRaises(SystemExit):
            resolve_rebalance([row], "daily")
        with self.assertRaises(SystemExit):
            resolve_signal_average_days([row], 1)


class NeweyWestTest(unittest.TestCase):
    def test_daily_labels_use_the_plain_ratio(self):
        series = np.array([0.02, 0.01, 0.03, -0.01, 0.02])
        self.assertAlmostEqual(
            newey_west_icir(series, 1), float(series.mean() / series.std()), places=6
        )

    def test_overlapping_labels_shrink_the_icir(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0.02, 0.01, 400)
        # Overlapping windows induce positive autocorrelation.
        series = np.convolve(base, np.ones(5) / 5, mode="valid")
        plain = newey_west_icir(series, 1)
        corrected = newey_west_icir(series, 5)
        self.assertLess(corrected, plain)


if __name__ == "__main__":
    unittest.main()
