"""Tests for exhaustive event-family enumeration and its staged evaluation."""

import unittest
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from min_gp.evaluation import IncrementalFitness, WalkForwardConfig
from min_gp.experiment import configuration_fingerprint
from min_gp.factors import ModerateRiskTemplate, WaitRescueTemplate
from min_gp.gp.exhaustive import (
    ExhaustiveConfig,
    ExhaustiveEventSearch,
    _intraday_key,
    enumerate_event_templates,
)


def _synthetic(family, I=30, D=26, M=240, seed=1):
    torch.manual_seed(seed)
    volume = torch.rand(I, D, M) * 1000 + 1
    tensors = {"volume": volume}
    if family == "moderate_risk":
        tensors["close"] = 10 * torch.cumprod(
            1 + torch.randn(I, D, M) * 0.001, dim=2
        )
    returns = torch.randn(I, D) * 0.02
    pool_mask = torch.ones(I, D, dtype=torch.bool)
    pool_mask[0, :] = False          # exercise the masking path
    return tensors, returns, pool_mask


def _search(family, **overrides):
    tensors, returns, pool_mask = _synthetic(family)
    config = ExhaustiveConfig(
        family=family, chunk_rows=overrides.pop("chunk_rows", 128), verbose=False,
        **overrides,
    )
    fitness = WalkForwardConfig(
            max_turnover=None,
        min_train_days=8, valid_days=4, n_splits=3, embargo_days=0,
        min_cross_section=10, min_valid_ic_days=2, cost_bps=0.0,
        min_fold_consistency=0.0, min_folds=1, paper_direction=-1,
    )
    return ExhaustiveEventSearch(
        tensors, returns, pool_mask=pool_mask,
        config=config, fitness_config=fitness,
    )


class EnumerationTest(unittest.TestCase):
    def test_space_sizes_match_the_parameter_domains(self):
        self.assertEqual(len(enumerate_event_templates("moderate_risk")), 768)
        self.assertEqual(len(enumerate_event_templates("wait_rescue")), 324)

    def test_every_genome_is_distinct(self):
        for family in ("moderate_risk", "wait_rescue"):
            genomes = enumerate_event_templates(family)
            self.assertEqual(len(set(genomes)), len(genomes), family)

    def test_the_paper_anchor_is_enumerated(self):
        self.assertIn(
            ModerateRiskTemplate(), enumerate_event_templates("moderate_risk")
        )
        self.assertIn(
            WaitRescueTemplate(), enumerate_event_templates("wait_rescue")
        )

    def test_staging_factorisation_is_exact(self):
        """Intraday x daily must multiply back to the full space."""
        for family, expected_intraday in (
            ("moderate_risk", 96), ("wait_rescue", 81)
        ):
            genomes = enumerate_event_templates(family)
            keys = {_intraday_key(genome) for genome in genomes}
            self.assertEqual(len(keys), expected_intraday, family)
            self.assertEqual(len(genomes) % len(keys), 0, family)

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            enumerate_event_templates("no_such_family")


class StagedEquivalenceTest(unittest.TestCase):
    """The staged path must reproduce the template path exactly.

    If these drift apart the search optimises one function while the hold-out
    evaluator scores another, and nothing else in the pipeline would notice.
    """

    def _assert_matches_template(self, family, sample):
        search = _search(family)
        genomes = enumerate_event_templates(family)
        chosen = [genomes[i] for i in sample]
        stage1 = (
            search._moderate_risk_stage1({_intraday_key(g) for g in chosen})
            if family == "moderate_risk"
            else search._wait_rescue_stage1({_intraday_key(g) for g in chosen})
        )
        for genome in chosen:
            staged = search.build_factor(genome, stage1)
            direct = search._factor(genome)
            self.assertEqual(staged.shape, direct.shape, str(genome))
            self.assertTrue(
                torch.allclose(staged, direct, atol=1e-6, rtol=1e-6, equal_nan=True),
                f"staged != template for {genome}",
            )

    def test_moderate_risk_staging_matches_templates(self):
        genomes = enumerate_event_templates("moderate_risk")
        sample = list(range(0, len(genomes), len(genomes) // 12))
        self._assert_matches_template("moderate_risk", sample)

    def test_wait_rescue_staging_matches_templates(self):
        genomes = enumerate_event_templates("wait_rescue")
        sample = list(range(0, len(genomes), len(genomes) // 12))
        self._assert_matches_template("wait_rescue", sample)

    def test_shared_intraday_results_are_not_aliased(self):
        """Two genomes sharing an intraday key must still differ downstream."""
        search = _search("moderate_risk")
        a = ModerateRiskTemplate(smooth_window=5)
        b = ModerateRiskTemplate(smooth_window=40)
        self.assertEqual(_intraday_key(a), _intraday_key(b))
        stage1 = search._moderate_risk_stage1({_intraday_key(a)})
        fa, fb = search.build_factor(a, stage1), search.build_factor(b, stage1)
        finite = torch.isfinite(fa) & torch.isfinite(fb)
        self.assertTrue(finite.any())
        self.assertFalse(torch.allclose(fa[finite], fb[finite]))


class ExhaustiveRunTest(unittest.TestCase):
    def test_resume_rejects_a_scored_genome_with_the_wrong_identity(self):
        genomes = enumerate_event_templates("wait_rescue")
        probe = _search("wait_rescue")
        fingerprint = configuration_fingerprint({
            "family": "wait_rescue",
            "ordered_genomes": [genome.to_dict() for genome in genomes],
            "fitness": asdict(probe.fitness_config),
        })
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage1.pt"
            torch.save({
                "family": "wait_rescue", "space_fingerprint": fingerprint,
                "stage1": {},
            }, checkpoint)
            progress = checkpoint.with_suffix(".progress.json")
            progress.write_text(json.dumps({
                "family": "wait_rescue", "space_fingerprint": fingerprint,
                "scored": [{
                    "score": IncrementalFitness.invalid().__dict__,
                    "genome": genomes[1].to_dict(),
                }],
            }), encoding="utf-8")
            search = _search(
                "wait_rescue", checkpoint_path=str(checkpoint), resume=True
            )
            with self.assertRaisesRegex(ValueError, "genome prefix"):
                search.run()

    def test_wait_rescue_run_covers_the_whole_space(self):
        search = _search("wait_rescue")
        ranked = search.run()
        self.assertEqual(len(ranked), 324)
        genomes = {genome for _, genome, _, _ in ranked}
        self.assertEqual(genomes, set(enumerate_event_templates("wait_rescue")))
        self.assertEqual(min(rank for _, _, rank, _ in ranked), 0)

    def test_results_are_ordered_by_pareto_rank(self):
        search = _search("wait_rescue")
        ranks = [rank for _, _, rank, _ in search.run()]
        self.assertEqual(ranks, sorted(ranks))

    def test_store_device_does_not_change_the_result(self):
        a = _search("wait_rescue", store_device="cpu").run()
        b = _search("wait_rescue", store_device="cpu", chunk_rows=64).run()
        self.assertEqual(
            [str(g) for _, g, _, _ in a], [str(g) for _, g, _, _ in b]
        )


if __name__ == "__main__":
    unittest.main()
