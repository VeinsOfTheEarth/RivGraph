from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ._delta_metrics_policies import (
    InletPolicy,
    RoutingPolicy,
    make_inlet_policy,
    make_routing_policy,
)


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: int
    u: int
    v: int
    attrs: Mapping[str, Any]


@dataclass
class DeltaGraph:
    nodes: list[int]
    edges: list[EdgeRecord]
    inlets: list[int]
    outlets: list[int]
    node_attrs: dict[int, dict[str, Any]] = field(default_factory=dict)

    def node_index(self) -> dict[int, int]:
        return {nid: i for i, nid in enumerate(self.nodes)}

    def out_edges(self, node: int) -> list[EdgeRecord]:
        return [e for e in self.edges if e.u == node]

    def in_edges(self, node: int) -> list[EdgeRecord]:
        return [e for e in self.edges if e.v == node]

    def successors(self, node: int) -> list[int]:
        return list(dict.fromkeys(e.v for e in self.out_edges(node)))

    def predecessors(self, node: int) -> list[int]:
        return list(dict.fromkeys(e.u for e in self.in_edges(node)))

    def topological_order(self) -> list[int]:
        indeg = {nid: 0 for nid in self.nodes}
        succ = {nid: [] for nid in self.nodes}
        for e in self.edges:
            indeg[e.v] += 1
            succ[e.u].append(e.v)

        order = []
        queue = [nid for nid in self.nodes if indeg[nid] == 0]

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for v in succ[nid]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)

        if len(order) != len(self.nodes):
            raise ValueError("DeltaGraph is not acyclic.")
        return order

    def reverse_topological_order(self) -> list[int]:
        return list(reversed(self.topological_order()))


@dataclass
class TransitionModel:
    edge_prob: dict[int, float]
    source: dict[int, float]


@dataclass
class SteadyStateResult:
    node_flux: dict[int, float]
    edge_flux: dict[int, float]
    outlet_flux: dict[int, float]
    subnetwork_membership: np.ndarray
    node_order: list[int]
    outlet_order: list[int]
    transition: TransitionModel
    mass_balance_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_delta_graph(links: dict, nodes: dict) -> DeltaGraph:
    edges = []
    for edge_id, conn_idx in zip(links["id"], links["conn"]):
        lidx = links["id"].index(edge_id)
        attrs = {k: links[k][lidx] for k in links.keys() if isinstance(links[k], list) and len(links[k]) == len(links["id"])}
        edges.append(EdgeRecord(edge_id=edge_id, u=conn_idx[0], v=conn_idx[1], attrs=attrs))

    node_ids = list(nodes["id"])
    inlet_ids = list(nodes.get("inlets", []))

    # outlets inferred structurally: no outgoing edges
    outdeg = {nid: 0 for nid in node_ids}
    for e in edges:
        outdeg[e.u] += 1
    outlet_ids = [nid for nid, deg in outdeg.items() if deg == 0]

    return DeltaGraph(
        nodes=node_ids,
        edges=edges,
        inlets=inlet_ids,
        outlets=outlet_ids,
    )


def _validate_graph_basic(graph: DeltaGraph) -> None:
    if len(graph.inlets) == 0:
        raise ValueError("Graph has no inlet nodes.")
    if len(graph.outlets) == 0:
        raise ValueError("Graph has no outlet nodes.")
    graph.topological_order()  # raises if cyclic


def compute_edge_probabilities(
    graph: DeltaGraph,
    routing: RoutingPolicy,
) -> dict[int, float]:
    probs: dict[int, float] = {}
    for nid in graph.nodes:
        out_edges = graph.out_edges(nid)
        if len(out_edges) == 0:
            continue

        scores = np.array([routing.edge_score(graph, e) for e in out_edges], dtype=float)
        if np.any(scores < 0):
            raise ValueError(f"Negative routing scores found for outgoing edges of node {nid}.")
        total = float(np.sum(scores))
        if total <= 0:
            raise ValueError(
                f"Outgoing routing scores for node {nid} sum to {total}; cannot normalize."
            )

        for e, p in zip(out_edges, scores / total):
            probs[e.edge_id] = float(p)

    return probs


def compute_source_vector(
    graph: DeltaGraph,
    inlet: InletPolicy,
) -> dict[int, float]:
    source = dict(inlet.source_weights(graph))
    total = float(sum(source.values()))
    if total <= 0:
        raise ValueError("Inlet source weights must sum to a positive value.")
    return {nid: float(val) / total for nid, val in source.items()}


def propagate_node_fluxes(
    graph: DeltaGraph,
    transition: TransitionModel,
) -> dict[int, float]:
    flux = {nid: 0.0 for nid in graph.nodes}
    for nid, val in transition.source.items():
        flux[nid] += val

    for nid in graph.topological_order():
        f = flux[nid]
        if f == 0:
            continue
        for e in graph.out_edges(nid):
            p = transition.edge_prob[e.edge_id]
            flux[e.v] += f * p

    return flux


def compute_edge_fluxes(
    graph: DeltaGraph,
    node_flux: dict[int, float],
    edge_prob: dict[int, float],
) -> dict[int, float]:
    return {
        e.edge_id: float(node_flux[e.u] * edge_prob[e.edge_id])
        for e in graph.edges
    }


def compute_outlet_fluxes(
    graph: DeltaGraph,
    node_flux: dict[int, float],
) -> dict[int, float]:
    return {nid: float(node_flux[nid]) for nid in graph.outlets}


def propagate_subnetwork_membership(
    graph: DeltaGraph,
    transition: TransitionModel,
) -> tuple[np.ndarray, list[int]]:
    node_order = graph.nodes
    outlet_order = list(graph.outlets)
    node_to_i = {nid: i for i, nid in enumerate(node_order)}
    out_to_j = {nid: j for j, nid in enumerate(outlet_order)}

    subn = np.zeros((len(node_order), len(outlet_order)), dtype=float)

    for oid in outlet_order:
        subn[node_to_i[oid], out_to_j[oid]] = 1.0

    for nid in graph.reverse_topological_order():
        if nid in out_to_j:
            continue
        out_edges = graph.out_edges(nid)
        if len(out_edges) == 0:
            continue
        i = node_to_i[nid]
        for e in out_edges:
            p = transition.edge_prob[e.edge_id]
            subn[i, :] += p * subn[node_to_i[e.v], :]

    return subn, outlet_order


def solve_steady_state(
    graph: DeltaGraph,
    *,
    routing: str | RoutingPolicy = "width",
    inlet: str | InletPolicy | None = None,
    inlet_weights: Mapping[int, float] | None = None,
    validate: bool = True,
    atol: float = 1e-12,
) -> SteadyStateResult:
    if validate:
        _validate_graph_basic(graph)

    routing_policy = make_routing_policy(routing)
    inlet_policy = make_inlet_policy(inlet, inlet_weights=inlet_weights)

    edge_prob = compute_edge_probabilities(graph, routing_policy)
    source = compute_source_vector(graph, inlet_policy)
    transition = TransitionModel(edge_prob=edge_prob, source=source)

    node_flux = propagate_node_fluxes(graph, transition)
    edge_flux = compute_edge_fluxes(graph, node_flux, edge_prob)
    outlet_flux = compute_outlet_fluxes(graph, node_flux)
    subn, outlet_order = propagate_subnetwork_membership(graph, transition)

    total_source = sum(source.values())
    total_outlet = sum(outlet_flux.values())
    mbe = abs(total_source - total_outlet)

    if mbe > atol:
        raise ValueError(
            f"Mass balance check failed: source={total_source}, outlets={total_outlet}, "
            f"abs diff={mbe}"
        )

    return SteadyStateResult(
        node_flux=node_flux,
        edge_flux=edge_flux,
        outlet_flux=outlet_flux,
        subnetwork_membership=subn,
        node_order=list(graph.nodes),
        outlet_order=outlet_order,
        transition=transition,
        mass_balance_error=mbe,
        diagnostics={
            "routing_policy": getattr(routing_policy, "name", type(routing_policy).__name__),
            "inlet_policy": getattr(inlet_policy, "name", type(inlet_policy).__name__),
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_inlets": len(graph.inlets),
            "n_outlets": len(graph.outlets),
        },
    )


def attach_edge_values(
    links: dict,
    values_by_edge_id: Mapping[int, float],
    *,
    attr_name: str,
) -> dict:
    out = dict(links)
    out[attr_name] = [float(values_by_edge_id[eid]) for eid in links["id"]]
    return out