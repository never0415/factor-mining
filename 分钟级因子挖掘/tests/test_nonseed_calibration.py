"""Locks for both non-seed operator calibration archives."""

import json
from pathlib import Path
import unittest

import torch

from min_gp.benchmark_nonseed_operators import (
    benchmark_missing_specs, benchmark_target_specs, remaining_target_specs,
)
from min_gp.operators import build_operator_registry
from min_gp.factors.catalog import migration_progress


CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "calibrations" / "nonseed_operators.json"
)
REMAINING_CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "calibrations" / "remaining_operators.json"
)


class NonSeedCalibrationTest(unittest.TestCase):
    def test_gtx1070_archive_covers_all_anchor_targets(self):
        document = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        self.assertEqual(document["environment"]["device_name"], "NVIDIA GeForce GTX 1070")
        self.assertEqual(len(document["operators"]), 46)
        self.assertTrue(all(
            record["status"] == "ok"
            for record in document["operators"].values()
        ))
        registry = build_operator_registry()
        targets = benchmark_target_specs(registry)
        self.assertEqual(len(targets), 46)
        self.assertTrue(all(spec.calibration is not None for spec in targets))
        progress = migration_progress()
        self.assertEqual(progress["cost_calibration_missing"], 0)
        self.assertEqual(progress["registry_cost_calibration_missing"], 0)

    def test_gtx1070_archive_covers_alternatives_and_parameter_profiles(self):
        document = json.loads(REMAINING_CALIBRATION.read_text(encoding="utf-8"))
        self.assertEqual(document["environment"]["device_name"], "NVIDIA GeForce GTX 1070")
        self.assertEqual(len(document["operators"]), 48)
        self.assertTrue(all(
            record["status"] == "ok"
            for record in document["operators"].values()
        ))
        registry = build_operator_registry()
        targets = remaining_target_specs(registry)
        self.assertEqual(len(targets), 48)
        self.assertTrue(all(spec.calibration is not None for spec in targets))

    def test_rerun_still_targets_46_after_archive_is_loaded(self):
        document = benchmark_missing_specs(
            device=torch.device("cpu"),
            shape={"I": 4, "D": 12, "M": 240},
            warmup=0, repeats=1, progress=False,
        )
        self.assertEqual(len(document["operators"]), 46)
        self.assertTrue(all(
            record["status"] == "ok"
            for record in document["operators"].values()
        ))


if __name__ == "__main__":
    unittest.main()
