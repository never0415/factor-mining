"""GP over the registered handbook core/cross-section/frequency slots."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import torch

from min_gp.evaluation import (
    BatchedIncrementalEvaluator, IncrementalFitness, WalkForwardConfig,
)
from min_gp.experiment import (
    append_failure, atomic_json, configuration_fingerprint,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.factors.handbook_skeleton import (
    HandbookSkeletonGenome, handbook_anchor, handbook_seed_population,
    handbook_slot_candidates,
)
from min_gp.gp.dripping_stone import _pareto_rank_and_crowding
from min_gp.gp.tide import (
    crossover_tide_genomes, mutate_tide_genome, random_tide_genome,
)
from min_gp.factors.tide_skeleton import CompleteTideSkeletonGenome
from min_gp.gp.climb import (
    crossover_climb_genomes, mutate_climb_genome, random_climb_genome,
)
from min_gp.factors.climb_skeleton import ClimbMountainSkeletonGenome
from min_gp.factors.rushing_skeleton import RushingForwardSkeletonGenome
from min_gp.gp.rushing import (
    crossover_rushing_genomes, mutate_rushing_genome,
    random_rushing_genome,
)
from min_gp.factors.handbook_composed import ComposedHandbookGenome
from min_gp.operators import build_operator_registry
from min_gp.gp.cost_control import (
    CostAdmission, DEFAULT_MAX_COST_UNITS, DEFAULT_MAX_ESTIMATED_SECONDS,
    DEFAULT_MAX_UNCALIBRATED_COST_UNITS,
)


@dataclass(frozen=True)
class HandbookGPConfig:
    population_size: int = 60
    generations: int = 8
    elite: int = 6
    tournament: int = 4
    crossover_rate: float = 0.4
    mutation_rate: float = 0.8
    seed: int = 0
    verbose: bool = True
    baseline_name: str | None = None
    error_log_path: str | None = None
    chunk_rows: int = 4096
    fitness_batch_size: int = 8
    progress_every: int = 1
    checkpoint_path: str | None = None
    resume: bool = False
    data_fingerprint: str | None = None
    max_cost_units: int | None = DEFAULT_MAX_COST_UNITS
    max_estimated_seconds: float | None = DEFAULT_MAX_ESTIMATED_SECONDS
    max_peak_bytes: int | None = None
    max_uncalibrated_cost_units: int | None = DEFAULT_MAX_UNCALIBRATED_COST_UNITS
    allow_uncalibrated_cost: bool = False

    def __post_init__(self):
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        if self.generations < 0:
            raise ValueError("generations must be non-negative")
        if self.fitness_batch_size < 1:
            raise ValueError("fitness_batch_size must be positive")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")


def _random_slot(kind, rng, registry, available_fields=None):
    candidates = handbook_slot_candidates(
        registry, kind, available_fields
    )
    if not candidates:
        raise ValueError(
            f"no handbook {kind} operator supports fields "
            f"{sorted(available_fields or ())}"
        )
    spec = rng.choice(candidates)
    return Slot.of(spec.name, **{
        name: rng.choice(tuple(domain))
        for name, domain in spec.parameter_domains.items()
    })


def random_handbook_genome(rng, available_fields=None, registry=None):
    registry = registry or build_operator_registry()
    fields = set(available_fields or ())
    anchors = [genome for genome in handbook_seed_population()
               if set(genome.required_fields) <= fields]
    if not anchors:
        raise ValueError(
            f"no handbook genome supports fields {sorted(fields)}"
        )
    anchor=rng.choice(anchors)
    if isinstance(anchor,CompleteTideSkeletonGenome): return random_tide_genome(rng,registry)
    if isinstance(anchor,ClimbMountainSkeletonGenome): return random_climb_genome(rng,registry)
    if isinstance(anchor,RushingForwardSkeletonGenome): return random_rushing_genome(rng,registry)
    return anchor.mutate(registry,fields,rng)


def mutate_handbook_genome(
    genome, rng, available_fields=None, registry=None,
):
    registry = registry or build_operator_registry()
    if isinstance(genome, CompleteTideSkeletonGenome):
        return mutate_tide_genome(genome, rng, registry)
    if isinstance(genome, ClimbMountainSkeletonGenome):
        return mutate_climb_genome(genome, rng, registry)
    if isinstance(genome, RushingForwardSkeletonGenome):
        return mutate_rushing_genome(genome, rng, registry)
    if isinstance(genome, ComposedHandbookGenome):
        return genome.mutate(registry,set(available_fields or ()),rng)
    kind = rng.choice(("core", "cross_section", "low_frequency"))
    return HandbookSkeletonGenome(
        core=(
            _random_slot("core", rng, registry, available_fields)
            if kind == "core" else genome.core
        ),
        cross_section=(
            _random_slot("cross_section", rng, registry)
            if kind == "cross_section" else genome.cross_section
        ),
        low_frequency=(
            _random_slot("low_frequency", rng, registry)
            if kind == "low_frequency" else genome.low_frequency
        ),
    )


def crossover_handbook_genomes(a, b, rng):
    if isinstance(a, CompleteTideSkeletonGenome) and isinstance(
        b, CompleteTideSkeletonGenome
    ):
        return crossover_tide_genomes(a, b, rng)
    if isinstance(a, ClimbMountainSkeletonGenome) and isinstance(
        b, ClimbMountainSkeletonGenome
    ):
        return crossover_climb_genomes(a, b, rng)
    if isinstance(a, RushingForwardSkeletonGenome) and isinstance(
        b, RushingForwardSkeletonGenome
    ):
        return crossover_rushing_genomes(a, b, rng)
    if isinstance(a,ComposedHandbookGenome) and isinstance(b,ComposedHandbookGenome):
        return a.crossover(b,build_operator_registry(),rng)
    if type(a) is not type(b):
        return rng.choice((a, b))
    return HandbookSkeletonGenome(
        core=rng.choice((a.core, b.core)),
        cross_section=rng.choice((a.cross_section, b.cross_section)),
        low_frequency=rng.choice((a.low_frequency, b.low_frequency)),
    )


class HandbookFactorGP:
    """Search all handbook anchors supported by the supplied leaf context."""

    @staticmethod
    def accepts_genome(genome) -> bool:
        return isinstance(genome, (
            HandbookSkeletonGenome, CompleteTideSkeletonGenome,
            ClimbMountainSkeletonGenome, RushingForwardSkeletonGenome,
            ComposedHandbookGenome,
        ))

    def __init__(
        self, context: dict[str, torch.Tensor], fwd_ret: torch.Tensor,
        pool_mask=None, gp_config=None, fitness_config=None,
    ):
        self.context = context
        self.available_fields = frozenset(context)
        self.fwd_ret = fwd_ret.float()
        self.pool_mask = pool_mask
        self.config = gp_config or HandbookGPConfig()
        self.fitness_config = fitness_config or WalkForwardConfig()
        self.registry = build_operator_registry()
        self.rng = random.Random(self.config.seed)
        self._fitness_cache = {}
        self.cost_admission = CostAdmission(
            self.config, self.registry, self.context, self.fwd_ret
        )
        seeds = [
            genome for genome in handbook_seed_population()
            if set(genome.required_fields) <= self.available_fields
            and self._admissible(genome)
        ]
        if not seeds:
            raise ValueError("no handbook factor is supported by the supplied fields")
        if self.config.baseline_name:
            baseline = handbook_anchor(self.config.baseline_name)
            if not set(baseline.required_fields) <= self.available_fields:
                raise ValueError("configured handbook baseline needs missing fields")
            if not self._admissible(baseline):
                raise ValueError("configured handbook baseline exceeds cost budget")
            self.baseline_genome = baseline
        else:
            self.baseline_genome = seeds[0]
        self.seed_genomes = tuple(seeds)
        self.baseline = self._factor(self.baseline_genome)
        self.fitness_evaluator = BatchedIncrementalEvaluator(
            self.baseline, self.fwd_ret, self.fitness_config
        )

    def _target_shape(self, genome):
        instruments, days = self.fwd_ret.shape
        minutes = max((
            value.shape[-1] for value in self.context.values()
            if isinstance(value, torch.Tensor) and value.ndim >= 3
            and value.shape[:2] == (instruments, days)
        ), default=1)
        shape = {"I": instruments, "D": days, "M": minutes}
        core = getattr(genome, "core", None)
        if core is not None:
            lags = dict(core.params).get("lags")
            if lags is not None:
                shape["P"] = lags + 2
        return shape

    def cost_estimate(self, genome):
        return self.cost_admission.estimate(genome)

    def _admissible(self, genome):
        return self.cost_admission.accepts(genome)

    def _factor(self, genome):
        factor = genome.evaluate(
            self.context, self.registry, chunk_rows=self.config.chunk_rows
        ).float()
        if factor.shape != self.fwd_ret.shape:
            raise ValueError(
                f"handbook factor shape {tuple(factor.shape)} does not match "
                f"return grid {tuple(self.fwd_ret.shape)}"
            )
        if self.pool_mask is not None:
            factor = torch.where(
                self.pool_mask, factor,
                torch.full_like(factor, float("nan")),
            )
        return factor

    def evaluate(self, genome):
        if genome in self._fitness_cache:
            return self._fitness_cache[genome]
        if not self._admissible(genome):
            score = IncrementalFitness.invalid()
            self._fitness_cache[genome] = score
            return score
        try:
            prepared = self.fitness_evaluator.prepare_candidate(
                self._factor(genome)
            )
            score = self.fitness_evaluator.evaluate_batch(
                prepared, (genome.complexity,), batch_size=1
            )[0]
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            append_failure(
                self.config.error_log_path, "handbook_factor_or_fitness",
                genome, exc,
            )
            score = IncrementalFitness.invalid()
        self._fitness_cache[genome] = score
        return score

    def _score_population(self, population, generation):
        """Build factors individually, then score weekly tensors in batches."""
        scored = [None] * len(population)
        pending_genomes, pending_values, pending_indices = [], [], []
        started = time.perf_counter()
        total = len(population)
        every = max(1, self.config.progress_every)
        for index, genome in enumerate(population, start=1):
            if genome in self._fitness_cache:
                scored[index - 1] = self._fitness_cache[genome]
                if self.config.verbose and (index % every == 0 or index == total):
                    print(
                        f"[handbook gen={generation} prepare] {index}/{total} "
                        "cached", flush=True,
                    )
                continue
            if not self._admissible(genome):
                score = IncrementalFitness.invalid()
                self._fitness_cache[genome] = score
                scored[index - 1] = score
                continue
            if self.config.verbose:
                print(
                    f"[handbook gen={generation} prepare] start {index}/{total} "
                    f"expr={str(genome)[:100]}", flush=True,
                )
            began = time.perf_counter()
            try:
                value = self.fitness_evaluator.prepare_candidate(
                    self._factor(genome)
                ).cpu()
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                append_failure(
                    self.config.error_log_path, "handbook_factor_prepare",
                    genome, exc,
                )
                score = IncrementalFitness.invalid()
                self._fitness_cache[genome] = score
                scored[index - 1] = score
            else:
                pending_genomes.append(genome)
                pending_values.append(value)
                pending_indices.append(index - 1)
            if self.config.verbose and (index % every == 0 or index == total):
                elapsed = time.perf_counter() - began
                total_elapsed = time.perf_counter() - started
                rate = total_elapsed / index
                print(
                    f"[handbook gen={generation} prepare] done {index}/{total} "
                    f"elapsed={elapsed:.2f}s eta={rate * (total-index):.0f}s",
                    flush=True,
                )

        if pending_values:
            fitness_started = time.perf_counter()

            def report(done, count):
                if not self.config.verbose:
                    return
                elapsed = time.perf_counter() - fitness_started
                rate = elapsed / max(done, 1)
                print(
                    f"[handbook gen={generation} fitness] {done}/{count} "
                    f"elapsed={elapsed:.1f}s eta={rate * (count-done):.0f}s",
                    flush=True,
                )

            try:
                batch_scores = self.fitness_evaluator.evaluate_batch(
                    pending_values,
                    tuple(genome.complexity for genome in pending_genomes),
                    batch_size=self.config.fitness_batch_size,
                    progress=report,
                )
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                # A batch failure must not silently invalidate unrelated
                # candidates. Fall back to isolated scoring so the failure log
                # identifies the exact expression.
                batch_scores = []
                for genome, value in zip(pending_genomes, pending_values):
                    try:
                        score = self.fitness_evaluator.evaluate_batch(
                            value, (genome.complexity,), batch_size=1
                        )[0]
                    except (RuntimeError, ValueError, TypeError, KeyError) as one_exc:
                        append_failure(
                            self.config.error_log_path,
                            "handbook_batched_fitness", genome, one_exc,
                        )
                        score = IncrementalFitness.invalid()
                    batch_scores.append(score)
                append_failure(
                    self.config.error_log_path,
                    "handbook_population_batch_fallback", pending_genomes[0], exc,
                )
            for genome, index, score in zip(
                pending_genomes, pending_indices, batch_scores
            ):
                self._fitness_cache[genome] = score
                scored[index] = score

        return [
            (score if score is not None else IncrementalFitness.invalid(), genome)
            for score, genome in zip(scored, population)
        ]

    def _initial_population(self):
        population = list(self.seed_genomes)
        attempts = 0
        while len(set(population)) < self.config.population_size:
            attempts += 1
            candidate = random_handbook_genome(
                self.rng, self.available_fields, self.registry
            )
            if self._admissible(candidate):
                population.append(candidate)
            if attempts > self.config.population_size * 100:
                raise RuntimeError("handbook search space cannot fill population")
        return list(dict.fromkeys(population))[:self.config.population_size]

    def _select(self, scored, ranks, crowd):
        choices = self.rng.sample(
            range(len(scored)), min(self.config.tournament, len(scored))
        )
        return scored[min(
            choices, key=lambda index: (ranks[index], -crowd[index])
        )][1]

    @staticmethod
    def _tupleize(value):
        if isinstance(value, list):
            return tuple(HandbookFactorGP._tupleize(item) for item in value)
        return value

    @staticmethod
    def _json_domain(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    def _checkpoint_fingerprint(self):
        operator_space = []
        for spec in self.registry.specs():
            calibration = spec.calibration
            operator_space.append({
                "name": spec.name,
                "inputs": [value.value for value in spec.input_types],
                "output": spec.output_type.value,
                "cost": spec.cost,
                "domains": {
                    name: [self._json_domain(value) for value in domain]
                    for name, domain in sorted(spec.parameter_domains.items())
                },
                "scope": int(spec.execution_scope),
                "complexity": dict(sorted(spec.complexity.items())),
                "calibration": None if calibration is None else {
                    "shape": dict(sorted(calibration.reference_shape.items())),
                    "seconds": calibration.seconds,
                    "peak_bytes": calibration.peak_bytes,
                    "device": calibration.device,
                    "parameters": {
                        name: self._json_domain(value)
                        for name, value in sorted(
                            calibration.parameter_values.items()
                        )
                    },
                },
            })
        config = self.config
        return configuration_fingerprint({
            "family": "handbook",
            "baseline_name": config.baseline_name,
            "available_fields": sorted(self.available_fields),
            "context": {
                name: {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                }
                for name, value in sorted(self.context.items())
                if isinstance(value, torch.Tensor)
            },
            "return_shape": list(self.fwd_ret.shape),
            "data_fingerprint": config.data_fingerprint,
            "population_size": config.population_size,
            "elite": config.elite,
            "tournament": config.tournament,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
            "chunk_rows": config.chunk_rows,
            "seed": config.seed,
            "fitness": asdict(self.fitness_config),
            "cost_budget": {
                "max_cost_units": config.max_cost_units,
                "max_estimated_seconds": config.max_estimated_seconds,
                "max_peak_bytes": config.max_peak_bytes,
                "max_uncalibrated_cost_units": config.max_uncalibrated_cost_units,
                "allow_uncalibrated_cost": config.allow_uncalibrated_cost,
            },
            "operator_space": operator_space,
        })

    def _restore_genome(self, payload):
        return HandbookSkeletonGenome.from_dict(payload)

    def _resume_population(self):
        path = self.config.checkpoint_path
        if not self.config.resume or not path or not Path(path).exists():
            return 0, self._initial_population()
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("family") != "handbook":
            raise ValueError("checkpoint factor family does not match handbook")
        if payload.get("configuration_fingerprint") != self._checkpoint_fingerprint():
            raise ValueError(
                "handbook checkpoint configuration, data, or operator-space "
                "mismatch; start a new run instead of resuming"
            )
        generation = int(payload["generation"])
        if generation < 0 or generation > self.config.generations:
            raise ValueError("checkpoint generation exceeds the requested run")
        population = [
            self._restore_genome(value) for value in payload["population"]
        ]
        if len(population) != self.config.population_size:
            raise ValueError("checkpoint population size does not match this run")
        if len(set(population)) != len(population):
            raise ValueError("checkpoint population contains duplicate genomes")
        if any(not self.accepts_genome(genome) for genome in population):
            raise ValueError("checkpoint contains a non-handbook genome")
        if any(
            set(genome.required_fields) - self.available_fields
            for genome in population
        ):
            raise ValueError("checkpoint genome requires unloaded fields")
        if any(not self._admissible(genome) for genome in population):
            raise ValueError("checkpoint genome exceeds the cost budget")
        for genome, score_payload in zip(
            population, payload.get("scores", [None] * len(population))
        ):
            if score_payload is not None:
                self._fitness_cache[genome] = IncrementalFitness(**score_payload)
        if "rng_state" in payload:
            self.rng.setstate(self._tupleize(payload["rng_state"]))
        return generation, population

    def _checkpoint(self, generation, population, completed=False):
        if not self.config.checkpoint_path:
            return
        atomic_json(self.config.checkpoint_path, {
            "schema_version": 1,
            "family": "handbook",
            "generation": generation,
            "completed": completed,
            "configuration_fingerprint": self._checkpoint_fingerprint(),
            "rng_state": self.rng.getstate(),
            "population": [genome.to_dict() for genome in population],
            "scores": [
                (
                    asdict(self._fitness_cache[genome])
                    if genome in self._fitness_cache else None
                )
                for genome in population
            ],
        })

    def run(self):
        start_generation, population = self._resume_population()
        for generation in range(start_generation, self.config.generations + 1):
            scored = self._score_population(population, generation)
            ranks, crowd = _pareto_rank_and_crowding(scored)
            order = sorted(
                range(len(scored)), key=lambda index: (ranks[index], -crowd[index])
            )
            if self.config.verbose:
                valid = [score for score, _ in scored if score.valid]
                best = max(
                    valid, key=lambda score: score.robust_ic,
                    default=scored[order[0]][0],
                )
                print(
                    f"[handbook-gp gen={generation}] pop={len(population)} "
                    f"front0={sum(rank == 0 for rank in ranks)} "
                    f"robustIC={best.robust_ic:+.4f}", flush=True,
                )
            if generation == self.config.generations:
                self._checkpoint(generation, population, completed=True)
                return [
                    (scored[index][0], scored[index][1], ranks[index],
                     crowd[index]) for index in order
                ]
            next_population = [
                scored[index][1] for index in order[:self.config.elite]
            ]
            attempts = 0
            while len(next_population) < self.config.population_size:
                attempts += 1
                parent = self._select(scored, ranks, crowd)
                child = parent
                if self.rng.random() < self.config.crossover_rate:
                    child = crossover_handbook_genomes(
                        parent, self._select(scored, ranks, crowd), self.rng
                    )
                if self.rng.random() < self.config.mutation_rate:
                    child = mutate_handbook_genome(
                        child, self.rng, self.available_fields, self.registry
                    )
                if not self._admissible(child):
                    continue
                if child not in next_population:
                    next_population.append(child)
                elif attempts > self.config.population_size * 50:
                    candidate = random_handbook_genome(
                        self.rng, self.available_fields, self.registry
                    )
                    if candidate not in next_population and self._admissible(candidate):
                        next_population.append(candidate)
            population = next_population
            self._checkpoint(generation + 1, population)
        raise AssertionError("unreachable")
