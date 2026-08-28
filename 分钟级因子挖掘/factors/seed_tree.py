"""Migration of the 59 seed expressions into the registered typed DSL."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import random

from min_gp.dsl import (
    ConstantNode, LeafNode, OperatorNode, SemanticType,
    evaluate_daily_expression,
)
from min_gp.gp.typed_tree import (
    TypedTreeGenome, crossover_trees, mutate_tree, node_from_dict,
    node_key, node_to_dict,
)


ALIASES = {"o":"open", "h":"high", "l":"low", "c":"close", "v":"volume"}
SESSION_MASKS = {"mask_am", "mask_pm"}

# Constants in these positions configure an operator and recover their GP
# domain.  All other constants are literal ConstantNodes.
PARAMETERS = {
    "intra_mean": {1: ("window", (1, 5, 10, 20, 40))},
    "intra_std": {1: ("window", (1, 5, 10, 20, 40))},
    "intra_shift": {1: ("shift", (-1, 1))},
    "ts_mean": {1: ("window", (1, 5, 10, 20, 40))},
    "ts_sum": {1: ("window", (1, 5, 10, 20, 40))},
    "ts_std": {1: ("window", (1, 5, 10, 20, 40))},
    "ts_min": {1: ("window", (1, 5, 10, 20, 40))},
    "ts_max": {1: ("window", (1, 5, 10, 20, 40))},
    "ts_delay": {1: ("window", (1, 5, 10, 20, 40))},
    "ts_corr": {2: ("window", (1, 5, 10, 20, 40))},
    "ts_quantile": {
        1: ("quantile", (0.25, 0.5, 0.75, 0.8)),
        2: ("window", (1, 5, 10, 20, 40)),
    },
    "day_quantile": {1: ("quantile", (0.25, 0.5, 0.75, 0.8))},
    "mask_agg": {2: ("statistic", tuple(range(9)))},
    "mask_ratio": {3: ("window", (1, 5, 10, 20, 40))},
    "roll_cut": {
        2: ("window", (1, 5, 10, 20, 40)),
        3: ("quantile", (0.25, 0.5, 0.75, 0.8)),
    },
}


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    raise TypeError("operator configuration must be a numeric literal")


def _variant(base, children):
    kinds = {
        SemanticType.LEGACY_MINUTE: "minute",
        SemanticType.LEGACY_DAILY: "daily",
        SemanticType.SAME_MINUTE_HISTORY: "same_minute",
    }
    non_scalar = [child.output_type for child in children if child.output_type != SemanticType.SCALAR]
    if not non_scalar or len(set(non_scalar)) != 1:
        raise TypeError(f"{base}: incompatible child types {[c.output_type for c in children]}")
    suffix = kinds[non_scalar[0]]
    if len(children) == 2 and children[0].output_type == SemanticType.SCALAR:
        suffix += "_scalar_left"
    elif len(children) == 2 and children[1].output_type == SemanticType.SCALAR:
        suffix += "_scalar_right"
    return f"seed_{base}_{suffix}"


def _operator_name(old_name, children):
    if old_name in {"add", "sub", "mul", "div", "abs", "sqrt", "neg", "f",
                    "ge", "le", "gt", "lt"}:
        return _variant(old_name, children)
    if old_name == "or_":
        return _variant("or", children)
    if old_name in {"day_sum", "day_mean", "day_std", "day_min", "day_max"}:
        suffix = "same_minute" if children[0].output_type == SemanticType.SAME_MINUTE_HISTORY else "minute"
        return f"seed_{old_name}_{suffix}"
    if old_name in {"ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max", "ts_delay", "ts_corr"}:
        suffix = "same_minute" if children[0].output_type == SemanticType.SAME_MINUTE_HISTORY else "daily"
        return f"seed_{old_name}_{suffix}"
    fixed = {
        "intra_shift":"seed_intra_shift", "intra_mean":"seed_intra_mean",
        "intra_std":"seed_intra_std", "day_last":"seed_day_last",
        "day_first":"seed_day_first", "day_ratio":"seed_day_ratio",
        "day_median":"seed_day_median", "day_quantile":"seed_day_quantile",
        "day_corr":"seed_day_corr", "time_barycenter":"seed_time_barycenter",
        "day_istd":"seed_day_istd", "day_iskew":"seed_day_iskew",
        "day_ikurt":"seed_day_ikurt", "ts_quantile":"seed_ts_quantile_daily",
        "to_B":"seed_to_same_minute", "bcast":"seed_broadcast_daily",
        "mask_mul":"seed_mask_mul", "mask_ratio":"seed_mask_ratio",
        "dist_to_event":"seed_dist_to_event", "cs_resid":"seed_cs_resid",
        "cs_rank":"seed_cs_rank", "roll_cut":"seed_roll_cut",
    }
    if old_name == "mask_agg":
        return "seed_mask_agg_all" if len(children) == 1 else "seed_mask_agg"
    try:
        return fixed[old_name]
    except KeyError as exc:
        raise KeyError(f"seed operator has not been migrated: {old_name}") from exc


def _convert(node, registry):
    if isinstance(node, ast.Name):
        name = ALIASES.get(node.id, node.id)
        kind = SemanticType.SESSION_MASK if name in SESSION_MASKS else SemanticType.LEGACY_MINUTE
        return LeafNode(name, kind)
    if isinstance(node, ast.Constant):
        return ConstantNode(_literal(node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return ConstantNode(_literal(node))
    if isinstance(node, ast.BinOp):
        mapping = {ast.Add:"add", ast.Sub:"sub", ast.Mult:"mul", ast.Div:"div"}
        old_name = mapping[type(node.op)]
        raw_args = (node.left, node.right)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        old_name = node.func.id
        raw_args = tuple(node.args)
    else:
        raise TypeError(f"unsupported seed syntax: {ast.dump(node)}")

    parameters, children = {}, []
    configured = PARAMETERS.get(old_name, {})
    for index, raw in enumerate(raw_args):
        if old_name == "mask_agg" and index == 1:
            try:
                is_all_minutes = _literal(raw) == 1
            except TypeError:
                is_all_minutes = False
            if is_all_minutes:
                parameters["all_minutes"] = True
                continue
        if index in configured:
            parameter, domain = configured[index]
            value = _literal(raw)
            if value not in domain:
                raise ValueError(f"{old_name}.{parameter}={value!r} outside {domain!r}")
            parameters[parameter] = int(value) if float(value).is_integer() else float(value)
        else:
            children.append(_convert(raw, registry))
    name = _operator_name(old_name, children)
    return OperatorNode(name, tuple(children), parameters).bind(registry)


def parse_seed_expression(expression, registry):
    return _convert(ast.parse(expression, mode="eval").body, registry)


def _nodes(node):
    yield node
    if isinstance(node, OperatorNode):
        for child in node.children:
            yield from _nodes(child)


@dataclass(frozen=True)
class SeedTreeGenome:
    name: str
    root: LeafNode | ConstantNode | OperatorNode

    def __hash__(self):
        return hash((self.name, node_key(self.root)))

    def evaluate(self, context, registry, chunk_rows=4096):
        return evaluate_daily_expression(self.root, context, registry, chunk_rows)

    def expression(self, registry):
        return self.root, registry

    def complexity(self, registry):
        return (
            self.root.complexity_with(registry)
            if isinstance(self.root, OperatorNode) else self.root.complexity
        )

    @property
    def required_fields(self):
        return tuple(sorted({node.name for node in _nodes(self.root) if isinstance(node, LeafNode)}))

    @property
    def operator_slots(self):
        return tuple(node.name for node in _nodes(self.root) if isinstance(node, OperatorNode))

    def mutate(
        self, registry, leaves, rng=None, max_depth=5,
        allowed_operators=None, organ_library=None, organ_probability=0.0,
    ):
        rng = rng or random.Random()
        if allowed_operators is None:
            allowed_operators = {
                name for name in registry.names() if name.startswith("seed_")
            }
        tree = mutate_tree(
            TypedTreeGenome(self.root), registry, leaves, max_depth, rng,
            allowed_operators=allowed_operators,
            organ_library=organ_library,
            organ_probability=organ_probability,
        )
        return SeedTreeGenome(self.name, tree.root)

    def crossover(self, other, registry, rng=None):
        rng = rng or random.Random()
        tree = crossover_trees(
            TypedTreeGenome(self.root), TypedTreeGenome(other.root), registry, rng
        )
        return SeedTreeGenome(self.name, tree.root)

    def to_dict(self):
        return {"kind":"typed_seed_tree", "name":self.name, "root":node_to_dict(self.root)}

    @classmethod
    def from_dict(cls, payload, registry):
        return cls(payload["name"], node_from_dict(payload["root"], registry))

    def __str__(self):
        return str(self.root)


def typed_seed_population(registry):
    from min_gp.seeds import SEEDS
    return tuple(
        SeedTreeGenome(name, parse_seed_expression(expression, registry))
        for name, expression in SEEDS.items()
    )
