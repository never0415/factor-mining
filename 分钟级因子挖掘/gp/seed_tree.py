"""Unified production typed-tree GP with migrated seeds and reusable organs."""

from dataclasses import dataclass
import random
import torch

from min_gp.gp.progress import score_population
from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.evaluation import IncrementalFitness, WalkForwardConfig, evaluate_incremental_fitness
from min_gp.evaluation.incremental import mean_rank_correlation
from min_gp.numeric.ranking import cross_section_rank
from min_gp.factors.seed_tree import SeedTreeGenome, typed_seed_population
from min_gp.gp.dripping_stone import _pareto_rank_and_crowding
from min_gp.gp.organs import graft_organ
from min_gp.gp.typed_tree import TypedTreeGenome, random_tree
from min_gp.operators import build_operator_registry
from min_gp.gp.cost_control import (
    CostAdmission, DEFAULT_MAX_COST_UNITS, DEFAULT_MAX_ESTIMATED_SECONDS,
    DEFAULT_MAX_UNCALIBRATED_COST_UNITS,
)


@dataclass(frozen=True)


class SeedTreeGPConfig:
    population_size: int = 60
    generations: int = 8
    elite: int = 6
    tournament: int = 4
    crossover_rate: float = .4
    mutation_rate: float = .8
    max_depth: int = 5
    chunk_rows: int = 4096
    seed: int = 0
    verbose: bool = True
    include_catalog_anchors: bool = True
    # None searches all 59 migrated anchors.  A CLI may select a smaller
    # island to keep the resident minute-leaf set within the GPU budget.
    anchor_names: tuple[str, ...] | None = None
    max_cost_units: int | None = DEFAULT_MAX_COST_UNITS
    max_estimated_seconds: float | None = DEFAULT_MAX_ESTIMATED_SECONDS
    max_peak_bytes: int | None = None
    max_uncalibrated_cost_units: int | None = DEFAULT_MAX_UNCALIBRATED_COST_UNITS
    allow_uncalibrated_cost: bool = False
    # Unified grammar: every fine-grained registered operator is available;
    # whole-factor compatibility wrappers remain forbidden.
    unified_operator_space: bool = True
    # A fraction of generation zero is grown afresh instead of being occupied
    # by the 59 historical seed expressions.
    random_initialization_fraction: float = .30
    # Maximum absolute rank correlation between two elites carried forward, and
    # between two candidates in the returned front.
    #
    # NSGA-II keeps its diversity in objective space: crowding distance is
    # measured across (robust_ic, incremental_ic, net_long_short, -complexity,
    # -pool_correlation), and duplicate rejection compares genome structure via
    # ``node_key``. Neither looks at the factor values. Two expressions that
    # differ by one constant therefore score as two distinct points and both
    # survive, which is how a single economic signal comes to occupy several
    # slots on the front - an observed run returned three variants correlated
    # 0.99 with each other. ``factor_pool`` does not cover this: it gates
    # against externally supplied factors, never against the live population.
    #
    # None preserves the historical behaviour exactly.
    intra_run_max_correlation: float | None = None
    # Offer every minute panel and session mask in the context to random trees,
    # not only those a seed expression happens to mention.
    #
    # The leaf pool is otherwise harvested by walking the seed expressions, so
    # its size is a property of the seed library rather than of the data. A
    # measured run had 37 tensors resident and 15 reachable: `open`, `is_jump`,
    # `has_gap`, eighteen multi-scale session masks and `w_time` were computed,
    # held in memory, and impossible for `random_tree` to select. Widening the
    # pool costs nothing to compute - the tensors are already there - and
    # fixes no parameter, unlike precomputing parametric masks into leaves.
    leaves_from_context: bool = False
    # For each fresh tree or mutation, independently try one typed organ graft.
    organ_graft_probability: float = .30
    # External factor roots donate internal organs, not complete initial
    # candidates. This prevents a failed factor being smuggled back wholesale.
    organs_replace_catalog_anchors: bool = True


def _context_leaf_type(value):
    """Semantic type for a context tensor no seed expression mentions.

    ``build_slice`` returns ``(I,D,M)`` minute panels and ``(M,)`` session
    masks. Every seed that does mention one of those declares exactly
    ``LEGACY_MINUTE`` or ``SESSION_MASK``, so this reads the mapping off the
    existing declarations rather than inventing one.

    Anything else is skipped deliberately. The ``(I,D)`` daily grids installed
    by ``--daily-leaf`` and ``--leaf-npz`` share one shape across several
    distinct semantic types, so a shape test cannot tell them apart; they
    already reach the population as anchors, with their types declared.
    """
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim == 3:
        return SemanticType.LEGACY_MINUTE
    if value.ndim == 1:
        return SemanticType.SESSION_MASK
    return None


class SeedTreeFactorGP:
    @staticmethod
    def accepts_genome(genome) -> bool:
        return isinstance(genome, SeedTreeGenome)

    def __init__(
        self, context, fwd_ret, pool_mask=None, gp_config=None,
        fitness_config=None, extra_genomes=(), factor_pool=(), organ_library=None,
    ):
        self.context, self.fwd_ret, self.pool_mask = context, fwd_ret.float(), pool_mask
        # Factors already accepted for trading. They widen the incremental-IC
        # residual and gate rediscoveries; ``pool_mask`` above is the unrelated
        # PIT universe mask.
        self.factor_pool = tuple(
            factor.float().to(self.fwd_ret.device) for factor in factor_pool
        )
        for factor in self.factor_pool:
            if factor.shape != self.fwd_ret.shape:
                raise ValueError("pooled factor and return grids differ")
        self.config = gp_config or SeedTreeGPConfig()
        if not 0 <= self.config.organ_graft_probability <= 1:
            raise ValueError("organ_graft_probability must be in [0, 1]")
        if not 0 <= self.config.random_initialization_fraction < 1:
            raise ValueError("random_initialization_fraction must be in [0, 1)")
        self.organ_library = organ_library
        self.fitness_config = fitness_config or WalkForwardConfig()
        self.registry, self.rng, self.cache = build_operator_registry(), random.Random(self.config.seed), {}
        seed_anchors = [
            genome for genome in typed_seed_population(self.registry)
            if set(genome.required_fields) <= set(context)
            and (
                self.config.anchor_names is None
                or genome.name in self.config.anchor_names
            )
        ]
        if self.config.anchor_names is not None:
            known = {genome.name for genome in typed_seed_population(self.registry)}
            unknown = set(self.config.anchor_names) - known
            if unknown:
                raise ValueError(f"unknown seed-tree anchors: {sorted(unknown)}")
        candidates = list(extra_genomes)
        injected_count = len(candidates)
        if self.config.include_catalog_anchors and not (
            self.organ_library and self.config.organs_replace_catalog_anchors
        ):
            from min_gp.factors.catalog import build_factor_catalog
            candidates.extend(
                entry.genome for entry in build_factor_catalog()
                if entry.source != "seed_catalog"
                and set(entry.leaves) <= set(context)
            )
        shared, injected = [], []
        for index, genome in enumerate(candidates):
            if isinstance(genome, SeedTreeGenome):
                converted = genome
            else:
                expression = getattr(genome, "expression", None)
                if expression is None:
                    continue
                root, _ = expression(self.registry)
                if not self.registry.accepts(
                    root.output_type, SemanticType.LEGACY_DAILY
                ):
                    continue
                name = getattr(genome, "anchor_name", None) or type(genome).__name__
                converted = SeedTreeGenome(f"shared:{name}:{index}", root)
            if set(converted.required_fields) <= set(context):
                (injected if index < injected_count else shared).append(converted)
        # Explicitly requested anchors lead. ``_initial`` fills the population
        # from this order and truncates at ``population_size``, so appending
        # them after the 59 migrated seeds meant any population at or below the
        # seed count silently searched without the anchor that was asked for.
        seeds = injected + seed_anchors + shared
        if not seeds:
            raise ValueError("no migrated seed tree is supported by the supplied fields")
        self.cost_admission = CostAdmission(
            self.config, self.registry, self.context, self.fwd_ret
        )
        self.seeds = tuple(genome for genome in seeds if self._admissible(genome))
        if not self.seeds:
            raise ValueError("all supported seed trees exceed the cost budget")
        self.injected = tuple(
            genome for genome in injected if genome in self.seeds
        )
        dropped = [g.name for g in injected if g not in self.seeds]
        if dropped:
            print(
                f"[seed-tree-gp] injected anchors over the cost budget: {dropped}",
                flush=True,
            )
        leaves = set()
        for genome in self.seeds:
            pending = [genome.root]
            while pending:
                node = pending.pop()
                if isinstance(node, LeafNode):
                    leaves.add(node)
                elif isinstance(node, OperatorNode):
                    pending.extend(node.children)
        if self.organ_library:
            for block in self.organ_library.blocks:
                pending = [block.root]
                while pending:
                    node = pending.pop()
                    if isinstance(node, LeafNode):
                        leaves.add(node)
                    elif isinstance(node, OperatorNode):
                        pending.extend(node.children)
        harvested = {leaf.name for leaf in leaves}
        if self.config.leaves_from_context:
            for name in sorted(context):
                if name in harvested:
                    continue
                semantic = _context_leaf_type(context[name])
                if semantic is not None:
                    leaves.add(LeafNode(name, semantic))
        self.leaves = tuple(sorted(leaves, key=lambda leaf: (leaf.name, leaf.output_type.value)))
        if self.config.verbose:
            opened = sorted({leaf.name for leaf in self.leaves} - harvested)
            detail = f" (+{len(opened)} from context: {', '.join(opened)})" if opened else ""
            print(
                f"[seed-tree-gp] leaf pool={len(self.leaves)} "
                f"harvested-from-seeds={len(harvested)}{detail}",
                flush=True,
            )
        self.allowed_operators = {
            spec.name for spec in self.registry.specs()
            if not spec.factor_wrapper
            and (
                self.config.unified_operator_space
                or spec.name.startswith("seed_")
            )
        }
        baseline_genome = next(
            (genome for genome in self.seeds if not genome.name.startswith("shared:")),
            self.seeds[0],
        )
        self.baseline = self._factor(baseline_genome)

    def _factor(self, genome):
        factor = genome.evaluate(self.context, self.registry, self.config.chunk_rows).float()
        if factor.shape != self.fwd_ret.shape:
            raise ValueError("seed factor and return grids differ")
        if self.pool_mask is not None:
            factor = torch.where(self.pool_mask, factor, torch.full_like(factor, float("nan")))
        return factor

    def cost_estimate(self, genome):
        return self.cost_admission.estimate(genome)

    def _admissible(self, genome):
        return self.cost_admission.accepts(genome)

    def evaluate(self, genome):
        if genome not in self.cache:
            if not self._admissible(genome):
                self.cache[genome] = IncrementalFitness.invalid()
                return self.cache[genome]
            try:
                score = evaluate_incremental_fitness(
                    self._factor(genome), self.baseline, self.fwd_ret,
                    genome.complexity(self.registry), self.fitness_config,
                    pool=self.factor_pool,
                )
            except (RuntimeError, ValueError, TypeError, KeyError):
                score = IncrementalFitness.invalid()
            self.cache[genome] = score
        return self.cache[genome]

    def _mutate(self, genome):
        try:
            return genome.mutate(
                self.registry, self.leaves, self.rng, self.config.max_depth,
                allowed_operators=self.allowed_operators,
                organ_library=self.organ_library,
                organ_probability=self.config.organ_graft_probability,
            )
        except ValueError:
            # A specialised root may have no seed_* producer of exactly that
            # semantic type. It remains crossable through its generic daily
            # parent, but this particular mutation point is infeasible.
            return genome

    def _random_genome(self):
        """Grow one genuinely new unified tree, with at most one organ graft."""
        root = random_tree(
            self.registry, SemanticType.LEGACY_DAILY, self.leaves,
            self.config.max_depth, self.rng, self.allowed_operators,
        )
        tree = TypedTreeGenome(root)
        if (
            self.organ_library
            and self.rng.random() < self.config.organ_graft_probability
        ):
            tree = graft_organ(
                tree, self.registry, self.organ_library, self.rng,
                max_levels=self.config.max_depth + 1,
            )
        return SeedTreeGenome("unified_generated", tree.root)

    def _maybe_graft(self, genome):
        if (
            not self.organ_library
            or self.rng.random() >= self.config.organ_graft_probability
        ):
            return genome
        tree = graft_organ(
            TypedTreeGenome(genome.root), self.registry, self.organ_library,
            self.rng, max_levels=self.config.max_depth + 1,
        )
        return SeedTreeGenome(genome.name, tree.root)

    def _initial(self):
        local = [genome for genome in self.seeds if not genome.name.startswith("shared:")]
        shared = [genome for genome in self.seeds if genome.name.startswith("shared:")]
        population = list(self.injected[:self.config.population_size])
        if local and local[0] not in population:
            population.append(local[0])
        if shared and self.config.population_size > 1 and shared[0] not in population:
            population.append(shared[0])
        random_target = round(
            self.config.population_size
            * self.config.random_initialization_fraction
        )
        seed_limit = max(len(population), self.config.population_size - random_target)
        for genome in self.seeds:
            if genome in population or len(population) >= seed_limit:
                continue
            candidate = self._maybe_graft(genome)
            if candidate not in population and self._admissible(candidate):
                population.append(candidate)
        attempts = 0
        while len(population) < self.config.population_size:
            attempts += 1
            try:
                candidate = self._random_genome()
            except ValueError:
                candidate = self._mutate(self.rng.choice(self.seeds))
            if candidate not in population and self._admissible(candidate):
                population.append(candidate)
            if attempts > self.config.population_size * 200:
                raise RuntimeError("cost/type constraints cannot fill seed population")
        return population

    def _select(self, scored, ranks, crowd):
        choices = self.rng.sample(range(len(scored)), min(self.config.tournament, len(scored)))
        return scored[min(choices, key=lambda i: (ranks[i], -crowd[i]))][1]

    def _demote_duplicate_front(self, scored, ranks, order):
        """Push near-duplicates off rank zero, every generation.

        Deduplicating elites is not enough. Elites are the handful of genomes
        carried forward unchanged; the rest of each generation is bred from
        parents drawn by ``_select``, which reads rank. A cluster occupying most
        of the front is therefore drawn most of the time and breeds its own
        likeness, so the population keeps resampling one attractor. Measured on
        a run with only the elite cut active, the top fifteen of the front
        collapsed into three clusters with one pair at 0.998.

        Walking the front best-first, a genome whose factor correlates at or
        above the threshold with one already kept moves to the next rank. It
        stays in the population with its metrics intact - demotion changes
        which genomes look attractive to selection, it does not discard them.
        Cost is one factor rebuild per front member per generation, which is
        why the test runs on the front rather than on every pair in the
        population.
        """
        threshold = self.config.intra_run_max_correlation
        if threshold is None:
            return ranks
        demoted = list(ranks)
        kept = []
        for i in order:
            if ranks[i] != 0:
                continue
            try:
                factor = cross_section_rank(self._factor(scored[i][1]))
            except (RuntimeError, ValueError, TypeError, KeyError):
                continue
            if any(
                abs(mean_rank_correlation(factor, other, pre_ranked=True)) >= threshold
                for other in kept
            ):
                demoted[i] += 1
                continue
            kept.append(factor)
        return demoted

    def _distinct(self, scored, order, limit):
        """Best genomes in ``order`` that are not near-duplicates of each other.

        Walks the already-sorted order and keeps a genome only when its factor
        is below the correlation threshold against every genome kept so far,
        so the survivor of a near-duplicate group is its best-ranked member.
        Falls back to filling any shortfall from the same order, because
        returning fewer elites than requested would shrink the population.
        """
        threshold = self.config.intra_run_max_correlation
        if threshold is None:
            return [scored[i][1] for i in order[:limit]]
        kept, ranked = [], []
        for i in order:
            if len(kept) >= limit:
                break
            genome = scored[i][1]
            try:
                factor = cross_section_rank(self._factor(genome))
            except (RuntimeError, ValueError, TypeError, KeyError):
                continue
            if any(
                abs(mean_rank_correlation(factor, other, pre_ranked=True)) >= threshold
                for other in ranked
            ):
                continue
            kept.append(genome)
            ranked.append(factor)
        if len(kept) < limit:
            for i in order:
                if len(kept) >= limit:
                    break
                genome = scored[i][1]
                if genome not in kept:
                    kept.append(genome)
        return kept

    def run(self):
        population = self._initial()
        for generation in range(self.config.generations + 1):
            scored = score_population(
                self.evaluate, population,
                f"{'seed-tree'} gen={generation}",
                verbose=self.config.verbose,
            )
            ranks, crowd = _pareto_rank_and_crowding(scored)
            order = sorted(range(len(scored)), key=lambda i: (ranks[i], -crowd[i]))
            raw_front = sum(r == 0 for r in ranks)
            # Demote every generation, not only the last. Selection reads rank,
            # so a cluster holding most of the front is drawn as a parent most
            # of the time and breeds its own likeness; deduplicating only the
            # elites leaves that intact, because elites are a small fraction of
            # each generation and the rest is bred. Cleaning the front once at
            # the end would tidy the report while the search had already spent
            # every generation resampling one attractor.
            ranks = self._demote_duplicate_front(scored, ranks, order)
            order = sorted(range(len(scored)), key=lambda i: (ranks[i], -crowd[i]))
            if self.config.verbose:
                kept = sum(r == 0 for r in ranks)
                print(
                    f"[seed-tree-gp gen={generation}] pop={len(population)} "
                    f"front0={raw_front} distinct={kept}",
                    flush=True,
                )
            if generation == self.config.generations:
                return [(scored[i][0], scored[i][1], ranks[i], crowd[i]) for i in order]
            next_population = self._distinct(scored, order, self.config.elite)
            attempts = 0
            while len(next_population) < self.config.population_size:
                attempts += 1
                parent = self._select(scored, ranks, crowd)
                child = parent
                if self.rng.random() < self.config.crossover_rate:
                    child = parent.crossover(self._select(scored, ranks, crowd), self.registry, self.rng)
                if self.rng.random() < self.config.mutation_rate:
                    child = self._mutate(child)
                if not self._admissible(child):
                    continue
                if child not in next_population:
                    next_population.append(child)
                elif attempts > self.config.population_size * 50:
                    candidate = self._mutate(self.rng.choice(self.seeds))
                    if candidate not in next_population and self._admissible(candidate):
                        next_population.append(candidate)
                    attempts = 0
            population = next_population
        raise AssertionError("unreachable")
