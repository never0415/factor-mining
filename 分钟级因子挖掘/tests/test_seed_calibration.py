"""Locks for persistent, device-measured seed-operator cost calibration."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from min_gp.benchmark_seed_operators import benchmark_seed_specs
from min_gp.dsl import load_calibration_file
from min_gp.factors.catalog import migration_progress
from min_gp.operators import build_operator_registry


CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "calibrations" / "seed_operators.json"
)


class SeedCalibrationTest(unittest.TestCase):
    def test_gtx1070_archive_covers_every_physical_seed_spec(self):
        self.assertTrue(CALIBRATION.is_file())
        document = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["environment"]["device_name"], "NVIDIA GeForce GTX 1070")
        self.assertEqual(len(document["operators"]), 135)
        self.assertTrue(all(
            record["status"] == "ok"
            for record in document["operators"].values()
        ))

        registry = build_operator_registry()
        specs = [spec for spec in registry.specs() if spec.name.startswith("seed_")]
        self.assertEqual(len(specs), 135)
        self.assertTrue(all(spec.calibration is not None for spec in specs))
        self.assertTrue(all(spec.complexity for spec in specs))

    def test_anchor_cost_coverage_includes_all_55_used_seed_specs(self):
        progress = migration_progress()
        self.assertEqual(progress["cost_calibrated_operators"], 121)
        self.assertEqual(progress["cost_calibration_required"], 121)
        self.assertEqual(progress["cost_zero_cost_exempt"], 2)
        self.assertEqual(progress["cost_calibration_missing"], 0)
        self.assertEqual(progress["cost_used_operators"], 123)

    def test_strict_loader_rejects_a_stale_operator_signature(self):
        document = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        first = next(iter(document["operators"].values()))
        first["spec_fingerprint"] = "stale"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale calibration"):
                load_calibration_file(build_operator_registry(), path, strict=True)

    def test_batch_runner_has_valid_inputs_for_all_specs(self):
        document = benchmark_seed_specs(
            device=torch.device("cpu"),
            shape={"I": 2, "D": 5, "M": 12},
            warmup=0, repeats=1, progress=False,
        )
        self.assertEqual(len(document["operators"]), 135)
        self.assertTrue(all(
            record["status"] == "ok"
            for record in document["operators"].values()
        ))


if __name__ == "__main__":
    unittest.main()
