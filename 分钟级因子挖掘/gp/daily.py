"""Second-layer GP that composes cached daily factor leaves."""

from dataclasses import asdict, dataclass
import random
from pathlib import Path
import json

import torch

from min_gp.gp.progress import score_population
from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.evaluation import WalkForwardConfig, evaluate_incremental_fitness
from min_gp.evaluation.incremental import IncrementalFitness
from min_gp.evaluation.neutralize import BatchedNeutralizer
from min_gp.gp.dripping_stone import _pareto_rank_and_crowding
from min_gp.gp.typed_tree import (
    TypedTreeGenome, crossover_trees, mutate_tree, random_tree,
)
from min_gp.operators import build_operator_registry
from min_gp.experiment import (
    append_failure, atomic_json, configuration_fingerprint,
)
from min_gp.gp.cost_control import (
    CostAdmission, DEFAULT_MAX_COST_UNITS, DEFAULT_MAX_ESTIMATED_SECONDS,
    DEFAULT_MAX_UNCALIBRATED_COST_UNITS,
)


DAILY_TREE_OPERATORS = {
    "daily_identity", "daily_neg", "daily_abs", "daily_add", "daily_sub",
    "daily_mul", "daily_div", "cross_section_distance",
    "cross_section_rank", "cross_section_identity", "smooth_daily",
    "rolling_daily_std", "mean_std_blend", "equal_blend",
}


@dataclass(frozen=True)
class DailyGPConfig:
    population_size: int = 80
    generations: int = 10
    elite: int = 8
    tournament: int = 5
    crossover_rate: float = 0.5
    mutation_rate: float = 0.8
    max_depth: int = 4
    seed: int = 0
    verbose: bool = True
    checkpoint_path: str | None = None
    error_log_path: str | None = None
    resume: bool = False
    chunk_rows: int = 4096
    max_cost_units: int | None = DEFAULT_MAX_COST_UNITS
    max_estimated_seconds: float | None = DEFAULT_MAX_ESTIMATED_SECONDS
    max_peak_bytes: int | None = None
    max_uncalibrated_cost_units: int | None = DEFAULT_MAX_UNCALIBRATED_COST_UNITS
    allow_uncalibrated_cost: bool = False


class DailyFactorGP:
    def __init__(
        self, leaves: dict[str, torch.Tensor], fwd_ret: torch.Tensor,
        pool_mask=None, gp_config=None, fitness_config=None,
        continuous_exposures=(), industry_exposure=None,
    ):
        if not leaves:
            raise ValueError("at least one daily factor leaf is required")
        shape = fwd_ret.shape
        if any(value.shape != shape for value in leaves.values()):
            raise ValueError("daily leaves and returns must share one grid")
        self.leaf_values = {name: value.float() for name, value in leaves.items()}
        self.fwd_ret = fwd_ret.float()
        self.pool_mask = pool_mask
        self.config = gp_config or DailyGPConfig()
        self.fitness_config = fitness_config or WalkForwardConfig()
        self.continuous_exposures = tuple(continuous_exposures)
        self.industry_exposure = industry_exposure
        self.neutralizer = None
        if self.continuous_exposures or self.industry_exposure is not None:
            self.neutralizer = BatchedNeutralizer(
                shape, self.continuous_exposures, self.industry_exposure,
                self.fitness_config.min_cross_section,
            )
        self.registry = build_operator_registry()
        self.leaves = tuple(
            LeafNode(name, SemanticType.DAILY_RAW_FACTOR)
            for name in sorted(leaves)
        )
        self.rng = random.Random(self.config.seed)
        self._fitness_cache = {}
        self.baseline_genome = self._leaf_genome(self.leaves[0])
        self.cost_admission = CostAdmission(
            self.config, self.registry, self.leaf_values, self.fwd_ret
        )
        if not self._admissible(self.baseline_genome):
            raise ValueError("daily baseline exceeds the configured cost budget")
        self.baseline = self._factor(self.baseline_genome)

    def _leaf_genome(self, leaf):
        return TypedTreeGenome(
            OperatorNode("daily_identity", (leaf,), {}).bind(self.registry)
        )

    def random_genome(self):
        root = random_tree(
            self.registry, SemanticType.DAILY_FACTOR, self.leaves,
            self.config.max_depth, self.rng, DAILY_TREE_OPERATORS,
        )
        return TypedTreeGenome(root)

    def _factor(self, genome):
        factor = genome.evaluate(self.leaf_values, self.registry).float()
        if self.neutralizer is not None:
            factor = self.neutralizer(factor)
        if self.pool_mask is not None:
            factor = torch.where(
                self.pool_mask, factor, torch.full_like(factor, float("nan"))
            )
        return factor

    def cost_estimate(self, genome):
        return self.cost_admission.estimate(genome)

    def _admissible(self, genome):
        return self.cost_admission.accepts(genome)

    def evaluate(self, genome):
        if genome in self._fitness_cache:
            return self._fitness_cache[genome]
        if not self._admissible(genome):
            score = IncrementalFitness.invalid()
            self._fitness_cache[genome] = score
            return score
        try:
            factor = self._factor(genome)
            score = evaluate_incremental_fitness(
                factor, self.baseline, self.fwd_ret,
                genome.complexity(self.registry), self.fitness_config,
            )
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            append_failure(self.config.error_log_path, "daily_factor_or_fitness", genome, exc)
            score = IncrementalFitness.invalid()
        self._fitness_cache[genome] = score
        return score

    def _initial_population(self):
        population = [self._leaf_genome(leaf) for leaf in self.leaves]
        attempts = 0
        while len(set(population)) < self.config.population_size:
            attempts += 1
            candidate = self.random_genome()
            if self._admissible(candidate):
                population.append(candidate)
            if attempts > self.config.population_size * 100:
                break
        return list(dict.fromkeys(population))[:self.config.population_size]

    def _resume_population(self):
        path = self.config.checkpoint_path
        if not self.config.resume or not path or not Path(path).exists():
            return 0, self._initial_population()
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = configuration_fingerprint(asdict(self.fitness_config))
        if payload.get("fitness_fingerprint") != expected:
            raise ValueError("daily checkpoint fitness configuration mismatch")
        population = [
            TypedTreeGenome.from_dict(item, self.registry)
            for item in payload["population"]
        ]
        if any(not self._admissible(genome) for genome in population):
            raise ValueError("checkpoint contains a genome exceeding the cost budget")
        return int(payload["generation"]), population

    def _checkpoint(self, generation, population, completed=False):
        if self.config.checkpoint_path:
            atomic_json(self.config.checkpoint_path, {
                "family": "daily_tree", "generation": generation,
                "completed": completed,
                "fitness_fingerprint": configuration_fingerprint(
                    asdict(self.fitness_config)
                ),
                "population": [genome.to_dict() for genome in population],
            })

    def _select(self, scored, ranks, crowd):
        choices = self.rng.sample(
            range(len(scored)), min(self.config.tournament, len(scored))
        )
        return scored[min(choices, key=lambda i: (ranks[i], -crowd[i]))][1]

    def run(self):
        start_generation, population = self._resume_population()
        for generation in range(start_generation, self.config.generations + 1):
            scored = score_population(
                self.evaluate, population,
                f"{'daily'} gen={generation}",
                verbose=self.config.verbose,
            )
            ranks, crowd = _pareto_rank_and_crowding(scored)
            order = sorted(range(len(scored)), key=lambda i: (ranks[i], -crowd[i]))
            if self.config.verbose:
                valid = [score for score, _ in scored if score.valid]
                best = max(valid, key=lambda score: score.robust_ic, default=scored[order[0]][0])
                print(
                    f"[daily-gp gen={generation}] pop={len(population)} "
                    f"front0={sum(rank == 0 for rank in ranks)} "
                    f"robustIC={best.robust_ic:+.4f}", flush=True,
                )
            if generation == self.config.generations:
                self._checkpoint(generation, population, completed=True)
                return [(scored[i][0], scored[i][1], ranks[i], crowd[i]) for i in order]
            next_population = [scored[i][1] for i in order[:self.config.elite]]
            attempts = 0
            while len(next_population) < self.config.population_size:
                attempts += 1
                parent = self._select(scored, ranks, crowd)
                child = parent
                if self.rng.random() < self.config.crossover_rate:
                    child = crossover_trees(
                        parent, self._select(scored, ranks, crowd),
                        self.registry, self.rng,
                    )
                if self.rng.random() < self.config.mutation_rate:
                    child = mutate_tree(
                        child, self.registry, self.leaves,
                        max(1, self.config.max_depth // 2), self.rng,
                        DAILY_TREE_OPERATORS,
                    )
                if not self._admissible(child):
                    continue
                if child not in next_population:
                    next_population.append(child)
                elif attempts > self.config.population_size * 50:
                    candidate = self.random_genome()
                    if candidate not in next_population and self._admissible(candidate):
                        next_population.append(candidate)
            population = next_population
            self._checkpoint(generation + 1, population)
        raise AssertionError("unreachable")
