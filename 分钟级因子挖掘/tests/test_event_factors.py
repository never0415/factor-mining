import unittest

import torch

from min_gp.evaluation import WalkForwardConfig
from min_gp.factors import ModerateRiskTemplate, WaitRescueTemplate
from min_gp.gp.event import EventFactorGP, EventGPConfig
from min_gp.operators.event import (
    follow_ratio,
    intraday_sigma_event,
    topk_separated_events,
)


class EventOperatorTest(unittest.TestCase):
    def test_sigma_event_excluded_edges_do_not_enter_moments(self):
        signal = torch.tensor([[[100.0, 1.0, 2.0, 3.0, 100.0]]])
        event = intraday_sigma_event(
            signal, sigma=0.0, direction="above", exclude_edges=1, ddof=0
        )
        self.assertEqual(
            event.tolist(), [[[False, False, False, True, False]]]
        )

    def test_sigma_event_respects_edge_exclusion(self):
        signal = torch.zeros(1, 1, 20)
        signal[..., 0] = 100
        signal[..., 10] = 50
        event = intraday_sigma_event(
            signal, sigma=1.0, direction="above", exclude_edges=5
        )
        self.assertFalse(bool(event[..., 0].item()))
        self.assertTrue(bool(event[..., 10].item()))

    def test_topk_event_removes_later_nearby_event(self):
        volume = torch.full((1, 1, 40), float("nan"))
        volume[..., 20] = 100
        volume[..., 22] = 90
        volume[..., 30] = 80
        event = topk_separated_events(
            volume, k=5, exclude_before=15, min_gap=5
        )
        self.assertTrue(bool(event[..., 20].item()))
        self.assertFalse(bool(event[..., 22].item()))
        self.assertTrue(bool(event[..., 30].item()))
        self.assertEqual(int(event.sum().item()), 2)

    def test_follow_ratio_uses_next_minutes_only(self):
        volume = torch.ones(1, 1, 30)
        event = torch.zeros_like(volume, dtype=torch.bool)
        event[..., 10] = True
        result = follow_ratio(volume, event, window=5)
        self.assertAlmostEqual(float(result.item()), 5.0, places=6)


class EventTemplateTest(unittest.TestCase):
    def test_templates_produce_daily_factors(self):
        torch.manual_seed(5)
        I, D, M = 35, 30, 240
        volume = torch.rand(I, D, M) * 1000 + 1
        close = 10 * torch.cumprod(
            1 + torch.randn(I, D, M) * 0.001, dim=2
        )
        moderate = ModerateRiskTemplate(smooth_window=5).evaluate(close, volume)
        wait = WaitRescueTemplate(smooth_window=5).evaluate(volume)
        self.assertEqual(tuple(moderate.shape), (I, D))
        self.assertEqual(tuple(wait.shape), (I, D))
        self.assertGreater(int(torch.isfinite(moderate).sum()), 0)
        self.assertGreater(int(torch.isfinite(wait).sum()), 0)

    def test_small_wait_rescue_gp_completes(self):
        torch.manual_seed(8)
        I, D, M = 35, 28, 240
        volume = torch.rand(I, D, M) * 1000 + 1
        returns = torch.randn(I, D) * 0.02
        engine = EventFactorGP(
            {"volume": volume}, returns,
            gp_config=EventGPConfig(
                family="wait_rescue", population_size=6,
                generations=1, elite=2, chunk_rows=256,
                seed=4, verbose=False,
            ),
            fitness_config=WalkForwardConfig(
            max_turnover=None,
                min_train_days=8, valid_days=4, n_splits=3,
                embargo_days=0, min_cross_section=20,
                min_valid_ic_days=2, cost_bps=0,
                paper_direction=-1,
            ),
        )
        ranked = engine.run()
        self.assertEqual(len(ranked), 6)


if __name__ == "__main__":
    unittest.main()
