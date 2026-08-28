"""Structure-preserving GP islands for handbook event factors."""

from dataclasses import asdict, dataclass, replace
import random
from pathlib import Path
import json

import torch

from min_gp.gp.progress import score_population
from min_gp.evaluation import IncrementalFitness, WalkForwardConfig, evaluate_incremental_fitness
from min_gp.factors.event_factors import ModerateRiskTemplate, WaitRescueTemplate
from min_gp.factors.event_skeleton import (
    EventSkeletonGenome, moderate_risk_anchor, slot_candidates,
    wait_rescue_anchor,
)
from min_gp.gp.event_skeleton import (
    crossover_skeleton, mutate_skeleton, random_skeleton,
    skeleton_seed_population,
)
from min_gp.gp.dripping_stone import _pareto_rank_and_crowding
from min_gp.operators.event import (
    EVENT_SIGMAS,
    EDGE_EXCLUSIONS,
    DDOF_VALUES,
    EXCLUDE_OPEN_MINUTES,
    FORWARD_WINDOWS,
    MIN_EVENT_GAPS,
    TOP_K_EVENTS,
)
from min_gp.operators.temporal import SMOOTH_WINDOWS
from min_gp.experiment import (
    append_failure, atomic_json, configuration_fingerprint,
)
from min_gp.operators import build_operator_registry
from min_gp.gp.cost_control import (
    CostAdmission, DEFAULT_MAX_COST_UNITS, DEFAULT_MAX_ESTIMATED_SECONDS,
    DEFAULT_MAX_UNCALIBRATED_COST_UNITS,
)


EVENT_FAMILIES = ("moderate_risk", "wait_rescue", "event_skeleton")


@dataclass(frozen=True)
class EventGPConfig:
    family: str = "moderate_risk"
    population_size: int = 60
    generations: int = 8
    elite: int = 6
    tournament: int = 4
    crossover_rate: float = 0.35
    mutation_rate: float = 0.80
    chunk_rows: int = 4096
    seed: int = 0
    verbose: bool = True
    checkpoint_path: str | None = None
    error_log_path: str | None = None
    resume: bool = False
    max_cost_units: int | None = DEFAULT_MAX_COST_UNITS
    max_estimated_seconds: float | None = DEFAULT_MAX_ESTIMATED_SECONDS
    max_peak_bytes: int | None = None
    max_uncalibrated_cost_units: int | None = DEFAULT_MAX_UNCALIBRATED_COST_UNITS
    allow_uncalibrated_cost: bool = False

    def __post_init__(self):
        if self.family not in EVENT_FAMILIES:
            raise ValueError(f"unknown event family: {self.family}")


def anchor_template(family: str):
    if family == "moderate_risk":
        return ModerateRiskTemplate()
    if family == "wait_rescue":
        return WaitRescueTemplate()
    if family == "event_skeleton":
        return moderate_risk_anchor()
    raise ValueError(family)


def random_event_template(family: str, rng: random.Random, available_fields=None):
    if family == "event_skeleton":
        return random_skeleton(rng, available_fields=available_fields)
    if family == "moderate_risk":
        return ModerateRiskTemplate(
            sigma=rng.choice(EVENT_SIGMAS),
            response_window=rng.choice(FORWARD_WINDOWS),
            smooth_window=rng.choice(SMOOTH_WINDOWS[1:]),
            exclude_edges=rng.choice(EDGE_EXCLUSIONS),
            ddof=rng.choice(DDOF_VALUES),
            standardize_before_distance=rng.choice((False, True)),
        )
    return WaitRescueTemplate(
        top_k=rng.choice(TOP_K_EVENTS),
        exclude_before=rng.choice(EXCLUDE_OPEN_MINUTES),
        min_gap=rng.choice(MIN_EVENT_GAPS),
        follow_window=rng.choice(FORWARD_WINDOWS),
        smooth_window=rng.choice(SMOOTH_WINDOWS[1:]),
    )


def mutate_event_template(genome, rng: random.Random, available_fields=None):
    if isinstance(genome, EventSkeletonGenome):
        return mutate_skeleton(genome, rng, available_fields=available_fields)
    if isinstance(genome, ModerateRiskTemplate):
        field = rng.choice((
            "sigma", "response_window", "smooth_window", "exclude_edges", "ddof",
            "standardize"
        ))
        if field == "sigma":
            return replace(genome, sigma=rng.choice(EVENT_SIGMAS))
        if field == "response_window":
            return replace(genome, response_window=rng.choice(FORWARD_WINDOWS))
        if field == "smooth_window":
            return replace(genome, smooth_window=rng.choice(SMOOTH_WINDOWS[1:]))
        if field == "exclude_edges":
            return replace(genome, exclude_edges=rng.choice(EDGE_EXCLUSIONS))
        if field == "ddof":
            return replace(genome, ddof=rng.choice(DDOF_VALUES))
        return replace(
            genome, standardize_before_distance=not genome.standardize_before_distance
        )
    field = rng.choice(("top_k", "exclude_before", "min_gap", "follow_window", "smooth_window"))
    domain = {
        "top_k": TOP_K_EVENTS,
        "exclude_before": EXCLUDE_OPEN_MINUTES,
        "min_gap": MIN_EVENT_GAPS,
        "follow_window": FORWARD_WINDOWS,
        "smooth_window": SMOOTH_WINDOWS[1:],
    }[field]
    return replace(genome, **{field: rng.choice(domain)})


def crossover_event_template(a, b, rng: random.Random):
    if type(a) is not type(b):
        raise TypeError("event crossover requires the same factor family")
    if isinstance(a, EventSkeletonGenome):
        return crossover_skeleton(a, b, rng)
    if isinstance(a, ModerateRiskTemplate):
        return ModerateRiskTemplate(
            sigma=rng.choice((a.sigma, b.sigma)),
            response_window=rng.choice((a.response_window, b.response_window)),
            smooth_window=rng.choice((a.smooth_window, b.smooth_window)),
            exclude_edges=rng.choice((a.exclude_edges, b.exclude_edges)),
            ddof=rng.choice((a.ddof, b.ddof)),
            standardize_before_distance=rng.choice((
                a.standardize_before_distance, b.standardize_before_distance
            )),
        )
    return WaitRescueTemplate(
        top_k=rng.choice((a.top_k, b.top_k)),
        exclude_before=rng.choice((a.exclude_before, b.exclude_before)),
        min_gap=rng.choice((a.min_gap, b.min_gap)),
        follow_window=rng.choice((a.follow_window, b.follow_window)),
        smooth_window=rng.choice((a.smooth_window, b.smooth_window)),
    )


class EventFactorEvaluator:
    """Minute-tensor plumbing shared by every event-family search strategy.

    Holds the anchor baseline and the factor -> fitness path so that the
    genetic and exhaustive searches score genomes identically; only the way
    they choose which genomes to score differs.
    """

    def __init__(
        self,
        minute_tensors: dict[str, torch.Tensor],
        fwd_ret: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
        config=None,
        fitness_config: WalkForwardConfig | None = None,
        neutralizer=None,
    ):
        self.config = config or EventGPConfig()
        # Applied inside the fitness so the search optimises the residual it
        # will actually be judged on, rather than a style-loaded proxy.
        self.neutralizer = neutralizer
        self.fitness_config = fitness_config or WalkForwardConfig(paper_direction=-1)
        self.minute_tensors = minute_tensors
        self.available_fields = frozenset(minute_tensors)
        self.fwd_ret = fwd_ret.float()
        self.pool_mask = pool_mask
        self.registry = build_operator_registry()
        if self.config.family == "event_skeleton":
            self.baseline_genome = (
                moderate_risk_anchor()
                if "close" in self.available_fields else wait_rescue_anchor()
            )
        else:
            self.baseline_genome = anchor_template(self.config.family)
        self._fitness_cache = {}
        self._check_inputs()
        self.cost_admission = CostAdmission(
            self.config, self.registry, self.minute_tensors, self.fwd_ret
        )
        if not self._admissible(self.baseline_genome):
            raise ValueError("event baseline exceeds the configured cost budget")
        self.baseline = self._factor(self.baseline_genome)

    def _check_inputs(self):
        required = {"volume"}
        if self.config.family == "moderate_risk":
            required.add("close")
        missing = required - set(self.minute_tensors)
        if missing:
            raise ValueError(f"missing minute tensors: {sorted(missing)}")
        shape = self.minute_tensors["volume"].shape
        if shape[:2] != self.fwd_ret.shape:
            raise ValueError("minute and return grids do not align")

    def _mask(self, factor):
        if self.pool_mask is None:
            return factor
        return torch.where(
            self.pool_mask, factor, torch.full_like(factor, float("nan"))
        )

    def cost_estimate(self, genome):
        return self.cost_admission.estimate(genome)

    def _admissible(self, genome):
        return self.cost_admission.accepts(genome)

    def _factor(self, genome):
        if isinstance(genome, EventSkeletonGenome):
            factor = genome.evaluate(
                self.minute_tensors, self.config.chunk_rows
            )
        elif isinstance(genome, ModerateRiskTemplate):
            factor = genome.evaluate(
                self.minute_tensors["close"], self.minute_tensors["volume"],
                self.config.chunk_rows,
            )
        else:
            factor = genome.evaluate(
                self.minute_tensors["volume"], self.config.chunk_rows
            )
        return self._mask(factor)

    def score(self, factor, genome):
        """Fitness for an already-built factor, so callers can share stages."""
        try:
            return evaluate_incremental_fitness(
                factor, self.baseline, self.fwd_ret,
                complexity=genome.complexity, cfg=self.fitness_config,
               neutralizer=self.neutralizer,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            append_failure(getattr(self.config, "error_log_path", None), "fitness", genome, exc)
            return IncrementalFitness.invalid()

    def evaluate(self, genome):
        if genome in self._fitness_cache:
            return self._fitness_cache[genome]
        if not self._admissible(genome):
            score = IncrementalFitness.invalid()
            self._fitness_cache[genome] = score
            return score
        try:
            factor = self._factor(genome)
        except (RuntimeError, ValueError, TypeError) as exc:
            append_failure(getattr(self.config, "error_log_path", None), "factor", genome, exc)
            score = IncrementalFitness.invalid()
        else:
            score = self.score(factor, genome)
            del factor
        self._fitness_cache[genome] = score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score


class EventFactorGP(EventFactorEvaluator):
    @staticmethod
    def accepts_genome(genome) -> bool:
        return isinstance(
            genome, (EventSkeletonGenome, ModerateRiskTemplate, WaitRescueTemplate)
        )

    def __init__(
        self,
        minute_tensors: dict[str, torch.Tensor],
        fwd_ret: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
        gp_config: EventGPConfig | None = None,
        fitness_config: WalkForwardConfig | None = None,
        neutralizer=None,
    ):
        super().__init__(
            minute_tensors, fwd_ret, pool_mask,
            gp_config or EventGPConfig(), fitness_config, neutralizer,
        )
        self.rng = random.Random(self.config.seed)

    def _initial_population(self):
        population = (
            list(skeleton_seed_population(self.available_fields))
            if self.config.family == "event_skeleton"
            else [self.baseline_genome]
        )
        local_target = max(1, int(self.config.population_size * 0.75))
        population = [genome for genome in population if self._admissible(genome)]
        attempts = 0
        while len(set(population)) < local_target:
            attempts += 1
            candidate = self.baseline_genome
            for _ in range(self.rng.randint(1, 3)):
                candidate = mutate_event_template(
                    candidate, self.rng, self.available_fields
                )
            if self._admissible(candidate):
                population.append(candidate)
            if attempts > self.config.population_size * 100:
                raise RuntimeError("cost budget cannot fill event population")
        while len(set(population)) < self.config.population_size:
            attempts += 1
            candidate = random_event_template(
                self.config.family, self.rng, self.available_fields
            )
            if self._admissible(candidate):
                population.append(candidate)
            if attempts > self.config.population_size * 200:
                raise RuntimeError("cost budget cannot fill event population")
        return list(dict.fromkeys(population))[:self.config.population_size]

    def _restore_genome(self, payload):
        if self.config.family == "event_skeleton":
            return EventSkeletonGenome.from_dict(payload)
        cls = ModerateRiskTemplate if self.config.family == "moderate_risk" else WaitRescueTemplate
        return cls(**payload)

    def _resume_population(self):
        path = self.config.checkpoint_path
        if not self.config.resume or not path or not Path(path).exists():
            return 0, self._initial_population()
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("family") != self.config.family:
            raise ValueError("checkpoint factor family does not match this run")
        if payload.get("configuration_fingerprint") != self._checkpoint_fingerprint():
            raise ValueError(
                "checkpoint configuration or operator-domain mismatch; "
                "start a new run instead of resuming"
            )
        population = [self._restore_genome(value) for value in payload["population"]]
        if len(population) != self.config.population_size:
            raise ValueError("checkpoint population size does not match this run")
        if any(
            set(getattr(genome, "required_fields", ())) - self.available_fields
            for genome in population
        ):
            raise ValueError("checkpoint contains a genome requiring unloaded fields")
        if any(not self._admissible(genome) for genome in population):
            raise ValueError("checkpoint contains a genome exceeding the cost budget")
        return int(payload["generation"]), population

    def _checkpoint_fingerprint(self):
        config = self.config
        payload = {
            "family": config.family,
            "available_fields": sorted(self.available_fields),
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
            "legacy_domains": {
                "event_sigmas": EVENT_SIGMAS,
                "edge_exclusions": EDGE_EXCLUSIONS,
                "ddof": DDOF_VALUES,
                "exclude_open": EXCLUDE_OPEN_MINUTES,
                "forward_windows": FORWARD_WINDOWS,
                "min_event_gaps": MIN_EVENT_GAPS,
                "top_k_events": TOP_K_EVENTS,
                "smooth_windows": SMOOTH_WINDOWS,
            },
        }
        if config.family == "event_skeleton":
            registry = build_operator_registry()
            payload["operator_slots"] = {
                kind: [
                    {
                        "name": spec.name,
                        "inputs": [value.value for value in spec.input_types],
                        "output": spec.output_type.value,
                        "cost": spec.cost,
                        "domains": {
                            name: list(domain)
                            for name, domain in sorted(spec.parameter_domains.items())
                        },
                    }
                    for spec in slot_candidates(
                        registry, kind, self.available_fields
                    )
                ]
                for kind in (
                    "detector", "statistic", "aggregator", "cross_section",
                    "low_frequency", "combiner",
                )
            }
        return configuration_fingerprint(payload)

    def _checkpoint(self, generation, population, completed=False):
        if self.config.checkpoint_path:
            atomic_json(self.config.checkpoint_path, {
                "family": self.config.family, "generation": generation,
                "completed": completed,
                "configuration_fingerprint": self._checkpoint_fingerprint(),
                "population": [genome.to_dict() for genome in population],
            })

    def _select(self, scored, ranks, crowd):
        choices = self.rng.sample(
            range(len(scored)), min(self.config.tournament, len(scored))
        )
        index = min(choices, key=lambda i: (ranks[i], -crowd[i]))
        return scored[index][1]

    def run(self):
        start_generation, population = self._resume_population()
        for generation in range(start_generation, self.config.generations + 1):
            scored = score_population(
                self.evaluate, population,
                f"{self.config.family} gen={generation}",
                verbose=self.config.verbose,
            )
            ranks, crowd = _pareto_rank_and_crowding(scored)
            order = sorted(range(len(scored)), key=lambda i: (ranks[i], -crowd[i]))
            if self.config.verbose:
                valid = [score for score, _ in scored if score.valid]
                best = max(valid, key=lambda value: value.robust_ic, default=scored[order[0]][0])
                print(
                    f"[event-gp {self.config.family} gen={generation}] "
                    f"pop={len(population)} front0={sum(r == 0 for r in ranks)} "
                    f"robustIC={best.robust_ic:+.4f} "
                    f"incrementalIC={best.incremental_ic:+.4f} "
                    f"netLS={best.net_long_short:+.6f}", flush=True,
                )
            if generation == self.config.generations:
                self._checkpoint(generation, population, completed=True)
                return [(scored[i][0], scored[i][1], ranks[i], crowd[i]) for i in order]

            next_population = []
            for i in order:
                if len(next_population) >= self.config.elite:
                    break
                if scored[i][1] not in next_population:
                    next_population.append(scored[i][1])
            attempts = 0
            while len(next_population) < self.config.population_size:
                attempts += 1
                parent = self._select(scored, ranks, crowd)
                child = parent
                if self.rng.random() < self.config.crossover_rate:
                    child = crossover_event_template(
                        parent, self._select(scored, ranks, crowd), self.rng
                    )
                if self.rng.random() < self.config.mutation_rate:
                    child = mutate_event_template(
                        child, self.rng, self.available_fields
                    )
                if not self._admissible(child):
                    continue
                if child not in next_population:
                    next_population.append(child)
                elif attempts > self.config.population_size * 20:
                    candidate = random_event_template(
                        self.config.family, self.rng, self.available_fields
                    )
                    if candidate not in next_population and self._admissible(candidate):
                        next_population.append(candidate)
            population = next_population
            self._checkpoint(generation + 1, population)
        raise AssertionError("unreachable")
