"""Fine-grained typed expression trees for eight handbook anchors."""

from dataclasses import dataclass

from min_gp.dsl import LeafNode, OperatorNode, SemanticType, evaluate_daily_expression
from min_gp.operators import build_operator_registry


def _leaf(name, kind):
    return LeafNode(name, kind)


def _op(registry, name, *children, **params):
    return OperatorNode(name, tuple(children), params).bind(registry)


def _smooth(registry, raw, window, method="mean"):
    return _op(registry, "smooth_daily", raw, method=method, window=window)


def _mean_std(registry, raw, window):
    return _op(registry, "mean_std_blend", raw, window=window)


def _rank_factor(registry, value):
    return _op(registry, "rank_daily_factor", value)


def _hidden(registry, lags, smooth_window, align_component_directions):
    close=_leaf("close",SemanticType.MINUTE_CLOSE); volume=_leaf("volume",SemanticType.MINUTE_VOLUME)
    ret=_op(registry,"close_minute_return",close,horizon=1)
    delta=_op(registry,"minute_delta_volume",volume,horizon=1)
    bundle=_op(registry,"minute_ols_bundle",ret,delta,lags=lags)
    intercept=_op(registry,"ols_intercept",bundle)
    morning=_op(registry,"ols_lag_dispersion",bundle)
    noon=_op(registry,"ols_conditional_abs",intercept,bundle)
    night=_op(registry,"rolling_market_abs_corr",intercept,window=smooth_window)
    night=_op(registry,"signed_raw",night,sign=-1 if align_component_directions else 1)
    components=(
        _rank_factor(registry,_smooth(registry,morning,smooth_window)),
        _rank_factor(registry,_smooth(registry,noon,smooth_window)),
        _rank_factor(registry,_op(registry,"daily_identity",night)),
    )
    return _op(registry,"blend_three_factors",*components)


def _long_short(registry, return_window, smooth_window):
    high=_leaf("high",SemanticType.MINUTE_HIGH); low=_leaf("low",SemanticType.MINUTE_LOW)
    close=_leaf("close",SemanticType.MINUTE_CLOSE); volume=_leaf("volume",SemanticType.MINUTE_VOLUME)
    q=_op(registry,"window_return",close,window=return_window)
    bvr=_op(registry,"sort_cumulative_difference_volume",volume,q)
    vr=_mean_std(registry,_op(registry,"cross_section_distance",bvr,standardize=True),smooth_window)
    position=_op(registry,"prior_range_position",low,high,close,window=return_window)
    bvp=_op(registry,"sort_cumulative_difference_volume",volume,position)
    vp=_mean_std(registry,_op(registry,"cross_section_distance",bvp,standardize=False),smooth_window)
    volume_battle=_op(registry,"equal_blend",vr,vp)
    amplitude=_op(registry,"close_amplitude",high,low,close)
    bamp=_op(registry,"sort_cumulative_difference_signal",amplitude,q)
    amp_battle=_mean_std(registry,_op(registry,"cross_section_distance",bamp,standardize=False),smooth_window)
    return _op(registry,"equal_blend",volume_battle,amp_battle)


def _equal_treatment(registry,response_window,exclude_edges,smooth_window):
    open_=_leaf("open",SemanticType.MINUTE_OPEN); close=_leaf("close",SemanticType.MINUTE_CLOSE)
    volume=_leaf("volume",SemanticType.MINUTE_VOLUME)
    transformed=_op(registry,"boxcox_grid_mle",volume)
    delta=_op(registry,"minute_delta_volume",transformed,horizon=1)
    common=dict(sigma=1.0,exclude_edges=exclude_edges,ddof=0)
    spike=_op(registry,"intraday_sigma_event",delta,direction="above",**common)
    drop=_op(registry,"intraday_sigma_event",delta,direction="below",**common)
    ret=_op(registry,"close_minute_return",close,horizon=1)
    vol=_op(registry,"forward_window_std",ret,window=response_window,ddof=0)
    fair_vol=_op(registry,"raw_abs_difference",
        _op(registry,"masked_daily_mean_signal",vol,spike),
        _op(registry,"masked_daily_mean_signal",vol,drop))
    fair_ret=_op(registry,"raw_abs_difference",
        _op(registry,"masked_daily_mean_return",ret,spike),
        _op(registry,"masked_daily_mean_return",ret,drop))
    day_ret=_op(registry,"daily_open_close_return",open_,close)
    a=_smooth(registry,_op(registry,"raw_multiply",day_ret,fair_vol),smooth_window)
    b=_smooth(registry,_op(registry,"raw_multiply",day_ret,fair_ret),smooth_window)
    return _op(registry,"equal_blend",a,b)


def _dark_flow(registry,bins,lookback,multiple,smooth_window):
    open_=_leaf("open",SemanticType.MINUTE_OPEN); high=_leaf("high",SemanticType.MINUTE_HIGH)
    low=_leaf("low",SemanticType.MINUTE_LOW); volume=_leaf("volume",SemanticType.MINUTE_VOLUME)
    entropy=_op(registry,"relative_volume_entropy",volume,bins=bins)
    entropy=_mean_std(registry,_op(registry,"cross_section_distance",entropy,standardize=False),smooth_window)
    spike=_op(registry,"relative_volume_event",volume,lookback=lookback,multiple=multiple,exclude_edges=0)
    amplitude=_op(registry,"open_amplitude",open_,high,low)
    spike_mean=_op(registry,"masked_daily_mean_signal",amplitude,spike)
    normal_mean=_op(registry,"masked_daily_mean_signal",amplitude,_op(registry,"inverse_event",spike))
    elasticity=_op(registry,"liquidity_elasticity",spike_mean,normal_mean)
    elasticity=_mean_std(registry,_op(registry,"cross_section_distance",elasticity,standardize=False),smooth_window)
    return _op(registry,"equal_blend",entropy,elasticity)


def _raw_panic(registry,smooth_window):
    raw=_op(registry,"raw_panic_weight",_leaf("daily_close",SemanticType.DAILY_PRICE),
            _leaf("market_close",SemanticType.MARKET_DAILY_PRICE))
    return _op(registry,"equal_blend",_smooth(registry,raw,smooth_window),
               _op(registry,"rolling_daily_std",raw,window=smooth_window))


def _rushing(registry,smooth_window):
    raw=_op(registry,"rushing_imbalance",
        _leaf("amount_share",SemanticType.MINUTE_AMOUNT_SHARE),
        _leaf("volume_share",SemanticType.MINUTE_VOLUME_SHARE),
        _leaf("up_volume_down_price_mask",SemanticType.MINUTE_MASK))
    return _smooth(registry,raw,smooth_window)


def _water(registry):
    spread=_op(registry,"amount_spread",_leaf("high_amount",SemanticType.MINUTE_HIGH_AMOUNT),
               _leaf("low_amount",SemanticType.MINUTE_LOW_AMOUNT))
    raw=_op(registry,"cap_scale",spread,_leaf("float_market_cap",SemanticType.DAILY_FLOAT_MARKET_CAP))
    return _op(registry,"daily_identity",raw)


def _cooperation(registry,peer_count,smooth_window):
    volume=_leaf("volume_share",SemanticType.MINUTE_VOLUME_SHARE)
    peers=_op(registry,"state_peer_volume",volume,_leaf("price_state",SemanticType.MINUTE_PRICE_STATE))
    corr=_op(registry,"minute_path_correlation",volume,peers)
    volume_component=_mean_std(registry,corr,smooth_window)
    spread=_op(registry,"peer_return_spread",_leaf("daily_return",SemanticType.DAILY_RETURN),
               _leaf("pair_similarity",SemanticType.PAIR_SIMILARITY),peer_count=peer_count)
    spread_component=_mean_std(registry,spread,smooth_window)
    return _op(registry,"equal_blend",volume_component,spread_component)


BUILDERS={
    "hidden_flower":_hidden,"long_short_battle":_long_short,
    "equal_treatment":_equal_treatment,"dark_flow":_dark_flow,
    "raw_panic":_raw_panic,"rushing_forward":_rushing,
    "water_boat":_water,"cooperation_effect":_cooperation,
}


def _walk(node):
    yield node
    if isinstance(node,OperatorNode):
        for child in node.children: yield from _walk(child)


@dataclass(frozen=True)
class ComposedHandbookGenome:
    anchor_name: str
    root: OperatorNode

    def __hash__(self):
        from min_gp.gp.typed_tree import node_key
        return hash((self.anchor_name,node_key(self.root)))

    def __eq__(self,other):
        from min_gp.gp.typed_tree import node_key
        return isinstance(other,ComposedHandbookGenome) and self.anchor_name==other.anchor_name and node_key(self.root)==node_key(other.root)

    def expression(self,registry=None):
        return self.root, registry or build_operator_registry()

    def evaluate(self,context,registry=None,chunk_rows=4096):
        registry=registry or build_operator_registry()
        missing=set(self.required_fields)-set(context)
        if missing: raise ValueError(f"{self.anchor_name} requires fields {sorted(missing)}")
        return evaluate_daily_expression(self.root,context,registry,chunk_rows)

    @property
    def required_fields(self):
        return tuple(sorted({n.name for n in _walk(self.root) if isinstance(n,LeafNode)}))

    @property
    def operator_slots(self):
        return tuple(n.name for n in _walk(self.root) if isinstance(n,OperatorNode))

    @property
    def execution_scope(self):
        return self.root.execution_scope_with(build_operator_registry())

    @property
    def history_days(self):
        return self.root.history_days_with(build_operator_registry())

    @property
    def intraday_lookahead_minutes(self):
        return self.root.intraday_lookahead_with(build_operator_registry())

    @property
    def complexity_profile(self):
        return self.root.complexity_profile_with(build_operator_registry())

    @property
    def complexity(self):
        return self.root.complexity_with(build_operator_registry())

    def mutate(self,registry,available_fields,rng,max_depth=5):
        from min_gp.gp.typed_tree import TypedTreeGenome, mutate_tree
        leaves=tuple(
            LeafNode(node.name,node.output_type) for node in _walk(self.root)
            if isinstance(node,LeafNode) and node.name in available_fields
        )
        allowed={name for name in registry.names() if not name.startswith(("handbook_","seed_"))}
        tree=mutate_tree(TypedTreeGenome(self.root),registry,leaves,max_depth,rng,allowed)
        return ComposedHandbookGenome(self.anchor_name,tree.root)

    def crossover(self,other,registry,rng):
        from min_gp.gp.typed_tree import TypedTreeGenome, crossover_trees
        tree=crossover_trees(TypedTreeGenome(self.root),TypedTreeGenome(other.root),registry,rng)
        return ComposedHandbookGenome(f"{self.anchor_name}__{other.anchor_name}",tree.root)

    def to_dict(self):
        from min_gp.gp.typed_tree import node_to_dict
        return {"kind":"composed_handbook","anchor_name":self.anchor_name,"root":node_to_dict(self.root)}

    @classmethod
    def from_dict(cls,payload,registry=None):
        from min_gp.gp.typed_tree import node_from_dict
        registry=registry or build_operator_registry()
        return cls(payload["anchor_name"],node_from_dict(payload["root"],registry))

    def __str__(self): return str(self.root)


def composed_handbook_anchor(name,**params):
    registry=build_operator_registry()
    try: root=BUILDERS[name](registry,**params)
    except KeyError as exc: raise ValueError(f"unknown composed handbook factor: {name}") from exc
    return ComposedHandbookGenome(name,root)
