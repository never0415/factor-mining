import random
import unittest

import torch

from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome, handbook_anchor
from min_gp.factors.rushing_skeleton import RushingForwardSkeletonGenome
from min_gp.gp.handbook import (
    crossover_handbook_genomes, mutate_handbook_genome,
    random_handbook_genome,
)


class RushingForwardSkeletonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.context = {
            "amount_share": torch.rand(5, 24, 16),
            "volume_share": torch.rand(5, 24, 16),
            "up_volume_down_price_mask": torch.rand(5, 24, 16) > .5,
        }

    def test_anchor_is_strict_skeleton_and_round_trips(self):
        anchor = handbook_anchor("rushing_forward")
        self.assertIsInstance(anchor, RushingForwardSkeletonGenome)
        self.assertEqual(anchor.required_fields, tuple(sorted(self.context)))
        self.assertIn("rushing_imbalance", anchor.operator_slots)
        restored = HandbookSkeletonGenome.from_dict(anchor.to_dict())
        self.assertEqual(anchor, restored)
        self.assertEqual(anchor.evaluate(self.context).shape, (5, 24))

    def test_random_mutation_and_crossover_preserve_core(self):
        fields = set(self.context)
        left = random_handbook_genome(random.Random(1), fields)
        right = mutate_handbook_genome(left, random.Random(2), fields)
        child = crossover_handbook_genomes(left, right, random.Random(3))
        for genome in (left, right, child):
            self.assertIsInstance(genome, RushingForwardSkeletonGenome)
            self.assertIn("rushing_imbalance", genome.operator_slots)
            self.assertEqual(genome.evaluate(self.context).shape, (5, 24))

    def test_missing_minutes_do_not_poison_the_daily_sum(self):
        anchor = handbook_anchor("rushing_forward", smooth_window=1)
        context = {name: value.clone() for name, value in self.context.items()}
        context["amount_share"][0, 3, :4] = float("nan")
        context["volume_share"][0, 3, :4] = float("nan")
        value = anchor.evaluate(context)
        self.assertTrue(torch.isfinite(value[0, 3]))
        context["amount_share"][0, 4] = float("nan")
        context["volume_share"][0, 4] = float("nan")
        value = anchor.evaluate(context)
        self.assertTrue(torch.isnan(value[0, 4]))


if __name__ == "__main__":
    unittest.main()
