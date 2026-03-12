from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ._delta_metrics_core import DeltaGraph, EdgeRecord


class RoutingPolicy(Protocol):
    name: str

    def edge_score(self, graph: "DeltaGraph", edge: "EdgeRecord") -> float:
        ...


class InletPolicy(Protocol):
    name: str

    def source_weights(self, graph: "DeltaGraph") -> Mapping[int, float]:
        ...


@dataclass(frozen=True)
class WidthRoutingPolicy:
    attr: str = "wid_adj"
    name: str = "width"

    def edge_score(self, graph: "DeltaGraph", edge: "EdgeRecord") -> float:
        try:
            score = float(edge.attrs[self.attr])
        except KeyError as exc:
            raise KeyError(
                f"Edge {edge.edge_id} missing routing attribute '{self.attr}'."
            ) from exc
        if score < 0:
            raise ValueError(
                f"Edge {edge.edge_id} has negative routing score {score}."
            )
        return score


@dataclass(frozen=True)
class UniformRoutingPolicy:
    name: str = "uniform"

    def edge_score(self, graph: "DeltaGraph", edge: "EdgeRecord") -> float:
        return 1.0


@dataclass(frozen=True)
class EqualInletPolicy:
    name: str = "equal"

    def source_weights(self, graph: "DeltaGraph") -> Mapping[int, float]:
        if len(graph.inlets) == 0:
            raise ValueError("Graph has no inlet nodes.")
        w = 1.0 / len(graph.inlets)
        return {nid: w for nid in graph.inlets}


@dataclass(frozen=True)
class WidthInletPolicy:
    attr: str = "wid_adj"
    name: str = "width"

    def source_weights(self, graph: "DeltaGraph") -> Mapping[int, float]:
        if len(graph.inlets) == 0:
            raise ValueError("Graph has no inlet nodes.")

        raw = {}
        for inlet in graph.inlets:
            score = 0.0
            for edge in graph.out_edges(inlet):
                val = float(edge.attrs.get(self.attr, 0.0))
                if val < 0:
                    raise ValueError(
                        f"Inlet edge {edge.edge_id} has negative '{self.attr}'={val}."
                    )
                score += val
            raw[inlet] = score

        total = sum(raw.values())
        if total <= 0:
            raise ValueError(
                f"Width-based inlet partition failed; summed inlet '{self.attr}' is zero."
            )
        return {nid: val / total for nid, val in raw.items()}


@dataclass(frozen=True)
class UserInletPolicy:
    weights: Mapping[int, float]
    normalize: bool = True
    name: str = "user"

    def source_weights(self, graph: "DeltaGraph") -> Mapping[int, float]:
        missing = set(graph.inlets) - set(self.weights.keys())
        extra = set(self.weights.keys()) - set(graph.inlets)
        if missing:
            raise ValueError(f"Missing user-specified inlet weights for: {sorted(missing)}")
        if extra:
            raise ValueError(f"User-specified inlet weights include non-inlets: {sorted(extra)}")

        raw = {nid: float(self.weights[nid]) for nid in graph.inlets}
        if any(v < 0 for v in raw.values()):
            raise ValueError("Inlet weights must be nonnegative.")

        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Inlet weights must sum to a positive value.")

        if self.normalize:
            return {nid: val / total for nid, val in raw.items()}
        return raw


def make_routing_policy(policy: str | RoutingPolicy) -> RoutingPolicy:
    if isinstance(policy, str):
        key = policy.lower()
        if key == "width":
            return WidthRoutingPolicy()
        if key == "uniform":
            return UniformRoutingPolicy()
        raise ValueError(f"Unknown routing policy '{policy}'.")
    return policy


def make_inlet_policy(
    policy: str | InletPolicy | None,
    *,
    inlet_weights: Mapping[int, float] | None = None,
) -> InletPolicy:
    if policy is None:
        if inlet_weights is not None:
            return UserInletPolicy(inlet_weights)
        return WidthInletPolicy()

    if isinstance(policy, str):
        key = policy.lower()
        if key == "width":
            return WidthInletPolicy()
        if key == "equal":
            return EqualInletPolicy()
        if key == "user":
            if inlet_weights is None:
                raise ValueError("inlet='user' requires inlet_weights.")
            return UserInletPolicy(inlet_weights)
        raise ValueError(f"Unknown inlet policy '{policy}'.")

    return policy