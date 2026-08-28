"""Structure-preserving GP island for the 滴水穿石 factor family."""

from dataclasses import asdict, dataclass, replace
import random
from pathlib import Path
import json

import torch

from min_gp.gp.progress import score_population
from min_gp.evaluation import WalkForwardConfig, evaluate_incremental_fitness
from min_gp.factors.dripping_stone import (
    DrippingStoneTemplate,
    MIN_VALID_RATIOS,
    SESSIONS,
)
from min_gp.factors.dripping_skeleton import (
    DRIPPING_SLOT_OPERATORS, DrippingSkeletonGenome,
    dripping_slot_candidates, dripping_stone_anchor,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.operators import build_operator_registry
from min_gp.operators.spectral import (
    DETREND_METHODS,
    IQR_MULTIPLIERS,
    PERIOD_BANDS,
    SPECTRAL_WINDOWS,
    VOLUME_TRANSFORMS,
)
from min_gp.operators.temporal import SMOOTH_METHODS, SMOOTH_WINDOWS
from min_gp.experiment import (
    append_failure, atomic_json, configuration_fingerprint,
)
from min_gp.gp.cost_control import (
    CostAdmission, DEFAULT_MAX_COST_UNITS, DEFAULT_MAX_ESTIMATED_SECONDS,
    DEFAULT_MAX_UNCALIBRATED_COST_UNITS,
)


@dataclass(frozen=True)
class DrippingStoneGPConfig:
    population_size: int = 80
    generations: int = 8
    elite: int = 8
    tournament: int = 4
    crossover_rate: float = 0.35
    mutation_rate: float = 0.80
    chunk_rows: int = 4096
    seed: int = 0
    verbose: bool = True
    checkpoint_path: str | None = None
    error_log_path: str | None = None
    resume: bool = False
    genome_mode: str = "operator_slots"
    max_cost_units: int | None = DEFAULT_MAX_COST_UNITS
    max_estimated_seconds: float | None = DEFAULT_MAX_ESTIMATED_SECONDS
    max_peak_bytes: int | None = None
    max_uncalibrated_cost_units: int | None = DEFAULT_MAX_UNCALIBRATED_COST_UNITS
    allow_uncalibrated_cost: bool = False

    def __post_init__(self):
        if self.genome_mode not in ("operator_slots", "parameter_template"):
            raise ValueError("genome_mode must be operator_slots or parameter_template")


def _random_dripping_slot(kind, rng, registry):
    spec = rng.choice(dripping_slot_candidates(registry, kind))
    params = {
        name: rng.choice(tuple(domain))
        for name, domain in spec.parameter_domains.items()
    }
    if spec.name == "band_power_ratio":
        low, high = rng.choice(PERIOD_BANDS)
        params.update(period_low=low, period_high=high)
    if spec.name == "smooth_daily" and params.get("method") == "none":
        params["window"] = 1
    return Slot.of(spec.name, **params)


def random_dripping_skeleton(rng, registry=None):
    registry = registry or build_operator_registry()
    return DrippingSkeletonGenome(**{
        kind: _random_dripping_slot(kind, rng, registry)
        for kind in DRIPPING_SLOT_OPERATORS
    })


def mutate_dripping_skeleton(genome, rng, registry=None):
    registry = registry or build_operator_registry()
    kind = rng.choice(tuple(DRIPPING_SLOT_OPERATORS))
    values = {
        name: getattr(genome, name) for name in DRIPPING_SLOT_OPERATORS
    }
    values[kind] = _random_dripping_slot(kind, rng, registry)
    return DrippingSkeletonGenome(**values)


def crossover_dripping_skeleton(a, b, rng):
    return DrippingSkeletonGenome(**{
        kind: rng.choice((getattr(a, kind), getattr(b, kind)))
        for kind in DRIPPING_SLOT_OPERATORS
    })


def random_template(rng: random.Random) -> DrippingStoneTemplate:
    method = rng.choice(SMOOTH_METHODS)
    smooth_window = 1 if method == "none" else rng.choice(SMOOTH_WINDOWS[1:])
    low, high = rng.choice(PERIOD_BANDS)
    return DrippingStoneTemplate(
        volume_transform=rng.choice(VOLUME_TRANSFORMS),
        clip_k=rng.choice(IQR_MULTIPLIERS),
        min_valid_ratio=rng.choice(MIN_VALID_RATIOS),
        detrend=rng.choice(DETREND_METHODS),
        spectral_window=rng.choice(SPECTRAL_WINDOWS),
        period_low=low,
        period_high=high,
        session=rng.choice(SESSIONS),
        smooth_method=method,
        smooth_window=smooth_window,
    )


def mutate_template(
    genome: DrippingStoneTemplate, rng: random.Random
) -> DrippingStoneTemplate:
    field = rng.choice((
        "volume_transform", "clip_k", "min_valid_ratio", "detrend",
        "spectral_window", "period_band", "session", "smoothing",
    ))
    if field == "volume_transform":
        return replace(genome, volume_transform=rng.choice(VOLUME_TRANSFORMS))
    if field == "clip_k":
        return replace(genome, clip_k=rng.choice(IQR_MULTIPLIERS))
    if field == "min_valid_ratio":
        return replace(genome, min_valid_ratio=rng.choice(MIN_VALID_RATIOS))
    if field == "detrend":
        return replace(genome, detrend=rng.choice(DETREND_METHODS))
    if field == "spectral_window":
        return replace(genome, spectral_window=rng.choice(SPECTRAL_WINDOWS))
    if field == "period_band":
        low, high = rng.choice(PERIOD_BANDS)
        return replace(genome, period_low=low, period_high=high)
    if field == "session":
        return replace(genome, session=rng.choice(SESSIONS))
    method = rng.choice(SMOOTH_METHODS)
    window = 1 if method == "none" else rng.choice(SMOOTH_WINDOWS[1:])
    return replace(genome, smooth_method=method, smooth_window=window)


def crossover_template(
    a: DrippingStoneTemplate,
    b: DrippingStoneTemplate,
    rng: random.Random,
) -> DrippingStoneTemplate:
    """Uniform crossover across semantic slots; band and smoothing stay atomic."""
    band = rng.choice((
        (a.period_low, a.period_high), (b.period_low, b.period_high)
    ))
    smoothing = rng.choice((
        (a.smooth_method, a.smooth_window),
        (b.smooth_method, b.smooth_window),
    ))
    return DrippingStoneTemplate(
        volume_transform=rng.choice((a.volume_transform, b.volume_transform)),
        clip_k=rng.choice((a.clip_k, b.clip_k)),
        min_valid_ratio=rng.choice((a.min_valid_ratio, b.min_valid_ratio)),
        detrend=rng.choice((a.detrend, b.detrend)),
        spectral_window=rng.choice((a.spectral_window, b.spectral_window)),
        period_low=band[0], period_high=band[1],
        session=rng.choice((a.session, b.session)),
        smooth_method=smoothing[0], smooth_window=smoothing[1],
    )


def _dominates(a, b) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(
        x > y for x, y in zip(a, b)
    )


def _pareto_rank_and_crowding(scored):
    n = len(scored)
    dominates = [[] for _ in range(n)]
    count = [0] * n
    ranks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            oi, oj = scored[i][0].objectives, scored[j][0].objectives
            if _dominates(oi, oj):
                dominates[i].append(j); count[j] += 1
            elif _dominates(oj, oi):
                dominates[j].append(i); count[i] += 1
    fronts = []
    front = [i for i, value in enumerate(count) if value == 0]
    rank = 0
    while front:
        fronts.append(front)
        next_front = []
        for i in front:
            ranks[i] = rank
            for j in dominates[i]:
                count[j] -= 1
                if count[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1

    crowd = [0.0] * n
    for front in fronts:
        if len(front) <= 2:
            for i in front:
                crowd[i] = float("inf")
            continue
        n_obj = len(scored[0][0].objectives)
        for k in range(n_obj):
            order = sorted(front, key=lambda i: scored[i][0].objectives[k])
            low = scored[order[0]][0].objectives[k]
            high = scored[order[-1]][0].objectives[k]
            if high == low:
                continue
            crowd[order[0]] = crowd[order[-1]] = float("inf")
            for p in range(1, len(order) - 1):
                crowd[order[p]] += (
                    scored[order[p + 1]][0].objectives[k]
                    - scored[order[p - 1]][0].objectives[k]
                ) / (high - low)
    return ranks, crowd


class DrippingStoneGP:
    @staticmethod
    def accepts_genome(genome) -> bool:
        return isinstance(genome, (DrippingSkeletonGenome, DrippingStoneTemplate))

    def __init__(
        self,
        volume: torch.Tensor,
        fwd_ret: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
        gp_config: DrippingStoneGPConfig | None = None,
        fitness_config: WalkForwardConfig | None = None,
    ):
        if volume.ndim != 3 or fwd_ret.ndim != 2:
            raise ValueError("expected volume=(I,D,M), fwd_ret=(I,D)")
        if volume.shape[:2] != fwd_ret.shape:
            raise ValueError("volume and fwd_ret grids do not align")
        self.volume = volume
        self.fwd_ret = fwd_ret.float()
        self.pool_mask = pool_mask
        self.config = gp_config or DrippingStoneGPConfig()
        self.fitness_config = fitness_config or WalkForwardConfig()
        self.rng = random.Random(self.config.seed)
        self.registry = build_operator_registry()
        self.baseline_genome = (
            dripping_stone_anchor()
            if self.config.genome_mode == "operator_slots"
            else DrippingStoneTemplate.paper_anchor()
        )
        self.cost_admission = CostAdmission(
            self.config, self.registry, {"volume": self.volume}, self.fwd_ret
        )
        if not self._admissible(self.baseline_genome):
            raise ValueError("dripping baseline exceeds the configured cost budget")
        self.baseline = self._factor(self.baseline_genome)
        self._fitness_cache = {}

    def _factor(self, genome) -> torch.Tensor:
        if isinstance(genome, DrippingSkeletonGenome):
            factor = genome.evaluate(
                {"volume": self.volume}, self.config.chunk_rows, self.registry
            )
        else:
            factor = genome.evaluate(self.volume, self.config.chunk_rows)
        if self.pool_mask is not None:
            factor = torch.where(
                self.pool_mask, factor, torch.full_like(factor, float("nan"))
            )
        return factor

    def cost_estimate(self, genome):
        return self.cost_admission.estimate(genome)

    def _admissible(self, genome):
        return self.cost_admission.accepts(genome)

    def evaluate(self, genome: DrippingStoneTemplate):
        if genome in self._fitness_cache:
            return self._fitness_cache[genome]
        if not self._admissible(genome):
            from min_gp.evaluation import IncrementalFitness
            score = IncrementalFitness.invalid()
            self._fitness_cache[genome] = score
            return score
        try:
            factor = self._factor(genome)
            score = evaluate_incremental_fitness(
                factor, self.baseline, self.fwd_ret,
                complexity=genome.complexity,
                cfg=self.fitness_config,
            )
            del factor
        except (RuntimeError, ValueError) as exc:
            from min_gp.evaluation import IncrementalFitness
            append_failure(self.config.error_log_path, "factor_or_fitness", genome, exc)
            score = IncrementalFitness.invalid()
        self._fitness_cache[genome] = score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score

    def _initial_population(self):
        pop = [self.baseline_genome]
        attempts = 0
        # Anchor-local variants dominate initialization; fully random trees are
        # kept as a smaller exploratory component.
        local_target = max(1, int(self.config.population_size * 0.75))
        while len(set(pop)) < local_target:
            attempts += 1
            candidate = self.baseline_genome
            for _ in range(self.rng.randint(1, 3)):
                candidate = (
                    mutate_dripping_skeleton(candidate, self.rng, self.registry)
                    if isinstance(candidate, DrippingSkeletonGenome)
                    else mutate_template(candidate, self.rng)
                )
            if self._admissible(candidate):
                pop.append(candidate)
            if attempts > self.config.population_size * 100:
                raise RuntimeError("cost budget cannot fill dripping population")
        while len(set(pop)) < self.config.population_size:
            attempts += 1
            candidate = (
                random_dripping_skeleton(self.rng, self.registry)
                if self.config.genome_mode == "operator_slots"
                else random_template(self.rng)
            )
            if self._admissible(candidate):
                pop.append(candidate)
            if attempts > self.config.population_size * 200:
                raise RuntimeError("cost budget cannot fill dripping population")
        return list(dict.fromkeys(pop))[:self.config.population_size]

    def _resume_population(self):
        path = self.config.checkpoint_path
        if not self.config.resume or not path or not Path(path).exists():
            return 0, self._initial_population()
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = configuration_fingerprint(asdict(self.fitness_config))
        if payload.get("fitness_fingerprint") != expected:
            raise ValueError("dripping checkpoint fitness configuration mismatch")
        if payload.get("genome_mode", "parameter_template") != self.config.genome_mode:
            raise ValueError("dripping checkpoint genome mode mismatch")
        population = [
            DrippingSkeletonGenome.from_dict(item)
            if self.config.genome_mode == "operator_slots"
            else DrippingStoneTemplate(**item)
            for item in payload["population"]
        ]
        if any(not self._admissible(genome) for genome in population):
            raise ValueError("checkpoint contains a genome exceeding the cost budget")
        return int(payload["generation"]), population

    def _checkpoint(self, generation, population, completed=False):
        if self.config.checkpoint_path:
            atomic_json(self.config.checkpoint_path, {
                "family": "dripping_stone", "generation": generation,
                "completed": completed,
                "genome_mode": self.config.genome_mode,
                "fitness_fingerprint": configuration_fingerprint(
                    asdict(self.fitness_config)
                ),
                "population": [genome.to_dict() for genome in population],
            })

    def _select(self, scored, ranks, crowd):
        choices = self.rng.sample(
            range(len(scored)), min(self.config.tournament, len(scored))
        )
        best = min(choices, key=lambda i: (ranks[i], -crowd[i]))
        return scored[best][1]

    def run(self):
        start_generation, population = self._resume_population()
        for generation in range(start_generation, self.config.generations + 1):
            scored = score_population(
                self.evaluate, population,
                f"{'dripping'} gen={generation}",
                verbose=self.config.verbose,
            )
            ranks, crowd = _pareto_rank_and_crowding(scored)
            order = sorted(range(len(scored)), key=lambda i: (ranks[i], -crowd[i]))
            if self.config.verbose:
                best = max(
                    (s for s, _ in scored if s.valid),
                    key=lambda s: s.robust_ic,
                    default=scored[order[0]][0],
                )
                print(
                    f"[dripping-gp gen={generation}] pop={len(population)} "
                    f"front0={sum(r == 0 for r in ranks)} "
                    f"robustIC={best.robust_ic:+.4f} "
                    f"incrementalIC={best.incremental_ic:+.4f} "
                    f"netLS={best.net_long_short:+.6f}",
                    flush=True,
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
                    mate = self._select(scored, ranks, crowd)
                    child = (
                        crossover_dripping_skeleton(parent, mate, self.rng)
                        if isinstance(parent, DrippingSkeletonGenome)
                        else crossover_template(parent, mate, self.rng)
                    )
                if self.rng.random() < self.config.mutation_rate:
                    child = (
                        mutate_dripping_skeleton(child, self.rng, self.registry)
                        if isinstance(child, DrippingSkeletonGenome)
                        else mutate_template(child, self.rng)
                    )
                if not self._admissible(child):
                    continue
                if child not in next_population:
                    next_population.append(child)
                elif attempts > self.config.population_size * 20:
                    candidate = (
                        random_dripping_skeleton(self.rng, self.registry)
                        if self.config.genome_mode == "operator_slots"
                        else random_template(self.rng)
                    )
                    if candidate not in next_population and self._admissible(candidate):
                        next_population.append(candidate)
            population = next_population
            self._checkpoint(generation + 1, population)
        raise AssertionError("unreachable")
