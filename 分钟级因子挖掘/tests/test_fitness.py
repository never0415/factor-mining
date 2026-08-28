import unittest

import torch

from min_gp.fitness import batch_fitness, multi_fitness, ndcg_k


class DirectionAwareFitnessTest(unittest.TestCase):
    def test_cross_section_rank_uses_average_rank_for_ties(self):
        from min_gp.expr import _cs_rank
        values = torch.tensor([[1.0], [1.0], [3.0], [float("nan")]])
        ranked = _cs_rank(values)
        self.assertAlmostEqual(float(ranked[0, 0]), 0.25)
        self.assertAlmostEqual(float(ranked[1, 0]), 0.25)
        self.assertAlmostEqual(float(ranked[2, 0]), 1.0)
        self.assertTrue(torch.isnan(ranked[3, 0]))

    def setUp(self):
        generator = torch.Generator().manual_seed(20260819)
        self.factor = torch.randn(40, 60, generator=generator)
        noise = 0.05 * torch.randn(40, 60, generator=generator)
        self.fwd_ret = self.factor + noise

    def test_positive_and_negative_factors_are_symmetric(self):
        pos, pos_meta = multi_fitness(
            self.factor, self.fwd_ret, return_details=True)
        neg, neg_meta = multi_fitness(
            -self.factor, self.fwd_ret, return_details=True)

        self.assertEqual(pos_meta[1], 1)
        self.assertEqual(neg_meta[1], -1)
        for pos_obj, neg_obj in zip(pos, neg):
            self.assertAlmostEqual(pos_obj, neg_obj, places=6)

    def test_batch_matches_individual_evaluation_with_missing_values(self):
        factor_a = self.factor.clone()
        factor_b = (-self.factor).clone()
        factor_a[:3, :7] = float("nan")
        factor_b[5:9, 10:18] = float("nan")
        fwd_ret = self.fwd_ret.clone()
        fwd_ret[12:14, 20:30] = float("nan")

        expected = [multi_fitness(f, fwd_ret) for f in (factor_a, factor_b)]
        actual, details = batch_fitness(
            [factor_a, factor_b], fwd_ret, _chunk=2, return_details=True)

        self.assertEqual([item[1] for item in details], [1, -1])
        for expected_fit, actual_fit in zip(expected, actual):
            for expected_obj, actual_obj in zip(expected_fit, actual_fit):
                self.assertAlmostEqual(expected_obj, actual_obj, places=5)

    def test_ndcg_excludes_invalid_factor_positions_from_top_k(self):
        factor = -torch.arange(1, 31, dtype=torch.float32).unsqueeze(1).repeat(1, 60)
        fwd_ret = factor.clone()
        factor[:10] = float("nan")

        score = ndcg_k(factor, fwd_ret, k=5, direction=1)
        self.assertAlmostEqual(score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
