import random
import unittest

import torch

from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.evaluation import WalkForwardConfig, evaluate_incremental_fitness
from min_gp.factors import DrippingStoneTemplate
from min_gp.gp.dripping_stone import (
    DrippingStoneGP,
    DrippingStoneGPConfig,
    crossover_template,
    mutate_template,
)
from min_gp.operators import build_operator_registry
from min_gp.operators.spectral import select_regular_session


class DrippingStoneSpectralTest(unittest.TestCase):
    def _volume(self, period, amplitude=20.0, rows=4):
        t = torch.arange(240, dtype=torch.float32)
        signal = 100.0 + amplitude * torch.sin(2 * torch.pi * t / period)
        return signal.repeat(rows, 1).reshape(2, rows // 2, 240)

    def test_three_minute_cycle_dominates_ten_minute_cycle(self):
        template = DrippingStoneTemplate.paper_anchor()
        in_band = template.evaluate(self._volume(3.0))
        out_band = template.evaluate(self._volume(10.0))
        self.assertGreater(float(in_band.nanmean()), 0.90)
        self.assertLess(float(out_band.nanmean()), 0.10)

    def test_power_ratio_is_scale_invariant(self):
        template = DrippingStoneTemplate.paper_anchor()
        a = template.evaluate(self._volume(3.0, amplitude=10.0))
        b = template.evaluate(self._volume(3.0, amplitude=100.0))
        self.assertTrue(torch.allclose(a, b, atol=2e-4, rtol=2e-4))

    def test_constant_volume_is_invalid(self):
        volume = torch.ones(2, 3, 240) * 100
        factor = DrippingStoneTemplate.paper_anchor().evaluate(volume)
        self.assertTrue(torch.isnan(factor).all())

    def test_legacy_241_grid_removes_only_the_empty_gap(self):
        regular = torch.arange(240).reshape(1, 1, 240)
        legacy = torch.cat((regular[..., :120], torch.tensor([[[999]]]), regular[..., 120:]), -1)
        restored = select_regular_session(legacy, "all")
        self.assertTrue(torch.equal(restored, regular))

    def test_semantic_types_reject_price_as_volume(self):
        registry = build_operator_registry()
        price = LeafNode("close", SemanticType.MINUTE_PRICE)
        node = OperatorNode("volume_transform", (price,), {"mode": "raw"})
        with self.assertRaises(TypeError):
            node.bind(registry)


class DrippingStoneEvolutionTest(unittest.TestCase):
    def test_mutation_and_crossover_keep_valid_templates(self):
        rng = random.Random(7)
        anchor = DrippingStoneTemplate.paper_anchor()
        variants = [mutate_template(anchor, rng) for _ in range(30)]
        for variant in variants:
            self.assertIsInstance(variant, DrippingStoneTemplate)
            child = crossover_template(anchor, variant, rng)
            self.assertIsInstance(child, DrippingStoneTemplate)

    def test_incremental_fitness_rewards_new_signal(self):
        torch.manual_seed(4)
        I, D = 40, 60
        signal = torch.randn(I, D)
        baseline = 0.2 * signal + torch.randn(I, D)
        candidate = signal + 0.1 * torch.randn(I, D)
        returns = signal + 0.2 * torch.randn(I, D)
        cfg = WalkForwardConfig(
            max_turnover=None,
            min_train_days=20, valid_days=10, n_splits=3,
            embargo_days=0, min_cross_section=20,
            min_valid_ic_days=5, cost_bps=0,
        )
        score = evaluate_incremental_fitness(
            candidate, baseline, returns, complexity=12, cfg=cfg
        )
        self.assertTrue(score.valid)
        self.assertGreater(score.robust_ic, 0.8)
        self.assertGreater(score.incremental_ic, 0.5)

    def test_incremental_fitness_rejects_constant_sections(self):
        I, D = 40, 40
        candidate = torch.ones(I, D)
        baseline = torch.randn(I, D)
        returns = torch.randn(I, D)
        cfg = WalkForwardConfig(
            max_turnover=None,
            min_train_days=10, valid_days=5, n_splits=3,
            embargo_days=0, min_cross_section=20,
            min_valid_ic_days=2,
        )
        score = evaluate_incremental_fitness(
            candidate, baseline, returns, complexity=1, cfg=cfg
        )
        self.assertFalse(score.valid)

    def test_small_gp_run_completes(self):
        torch.manual_seed(3)
        I, D, M = 30, 24, 240
        t = torch.arange(M, dtype=torch.float32).view(1, 1, M)
        strength = torch.rand(I, D, 1)
        volume = 100 + 20 * (
            strength * torch.sin(2 * torch.pi * t / 3)
            + (1 - strength) * torch.sin(2 * torch.pi * t / 10)
        )
        returns = strength.squeeze(2) + 0.05 * torch.randn(I, D)
        engine = DrippingStoneGP(
            volume, returns,
            gp_config=DrippingStoneGPConfig(
                population_size=6, generations=1, elite=2,
                chunk_rows=256, seed=2, verbose=False,
            ),
            fitness_config=WalkForwardConfig(
            max_turnover=None,
                min_train_days=8, valid_days=4, n_splits=3,
                embargo_days=0, min_cross_section=10,
                min_valid_ic_days=2, cost_bps=0,
            ),
        )
        ranked = engine.run()
        self.assertEqual(len(ranked), 6)
        self.assertTrue(any(score.valid for score, _, _, _ in ranked))


if __name__ == "__main__":
    unittest.main()
