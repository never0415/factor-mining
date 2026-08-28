"""Production cost admission must never silently pass unknown runtimes."""

import unittest

import torch

from min_gp.factors.dripping_stone import DrippingStoneTemplate
from min_gp.factors.event_factors import ModerateRiskTemplate, WaitRescueTemplate
from min_gp.gp.cost_control import estimate_genome_cost
from min_gp.gp.daily import DailyGPConfig
from min_gp.gp.dripping_stone import DrippingStoneGPConfig
from min_gp.gp.event import EventGPConfig
from min_gp.gp.exhaustive import ExhaustiveConfig
from min_gp.gp.handbook import HandbookGPConfig
from min_gp.gp.seed_tree import SeedTreeGPConfig
from min_gp.operators import build_operator_registry


class StrictCostAdmissionTest(unittest.TestCase):
    def test_every_production_gp_rejects_uncalibrated_cost_by_default(self):
        configs = (
            DailyGPConfig(), DrippingStoneGPConfig(), EventGPConfig(),
            ExhaustiveConfig(), HandbookGPConfig(), SeedTreeGPConfig(),
        )
        self.assertTrue(all(not value.allow_uncalibrated_cost for value in configs))

    def test_legacy_event_and_dripping_templates_have_complete_cost_dags(self):
        registry = build_operator_registry()
        context = {
            "close": torch.ones(4, 12, 240),
            "volume": torch.ones(4, 12, 240),
        }
        fwd_ret = torch.ones(4, 12)
        for genome in (
            ModerateRiskTemplate(), WaitRescueTemplate(), DrippingStoneTemplate(),
        ):
            with self.subTest(genome=type(genome).__name__):
                estimate = estimate_genome_cost(
                    genome, registry, context, fwd_ret, chunk_rows=4096
                )
                self.assertTrue(estimate.fully_calibrated)
                self.assertTrue(estimate.peak_fully_calibrated)
                self.assertGreater(estimate.calibrated_seconds, 0)


if __name__ == "__main__":
    unittest.main()
