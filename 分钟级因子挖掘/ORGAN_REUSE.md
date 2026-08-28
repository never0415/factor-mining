# External-factor organ reuse

The organ mechanism uses the typed definitions corresponding to
`factor_reproduction_handbook.pdf` as its only donor set. It never reads
`all76`, Pareto logs, exported candidates, or materialised daily factors.

## Genetic representation

An organ is an internal operator subtree with 2–3 levels, counting a terminal
as level 1. For example, `sqrt(volume)` has two levels and
`div(sub(high, low), close)` has three. The complete failed-factor root is
excluded.

Organs remain ordinary `OperatorNode` trees in every genome. Their children
are visible to traversal, mutation and crossover. A source name such as
`rushing_forward` is provenance only; it is never installed as a genetic leaf.

Exact duplicate structures are stored once and retain all donor names. At run
time the library is filtered by available fields and by the semantic type of
the receiving parent slot.

## Unified GP behaviour

The production seed-tree GP now:

- permits all registered fine-grained operators and excludes whole-factor
  wrappers;
- reserves 30% of generation zero for genuinely new trees;
- attempts one compatible organ graft with probability 30% for a fresh tree,
  an admitted seed tree, or a mutation;
- uses external factor trees as organ donors instead of complete initial
  candidates;
- writes `organ_manifest.json` beside each run with source, structure, fields,
  levels, type and expression.

Defaults can be changed with:

```text
--organ-source NAME                 repeat to restrict donor factors
--organ-min-levels 2
--organ-max-levels 3
--organ-graft-probability 0.30
--random-initialization-fraction 0.30
```

`--seed-only-operators` restores the old 135-operator generation space.
`--no-external-organs` restores the legacy mode. Opaque `--extra-anchor` and
`--daily-leaf` terminals are rejected while unified organ mode is enabled.
