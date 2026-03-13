from __future__ import annotations

from collections import deque
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
    _out_edges_by_node: dict[int, list[EdgeRecord]] = field(init=False, repr=False)
    _in_edges_by_node: dict[int, list[EdgeRecord]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._out_edges_by_node = {nid: [] for nid in self.nodes}
        self._in_edges_by_node = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            self._out_edges_by_node.setdefault(edge.u, []).append(edge)
            self._in_edges_by_node.setdefault(edge.v, []).append(edge)

    def node_index(self) -> dict[int, int]:
        return {nid: i for i, nid in enumerate(self.nodes)}

    def out_edges(self, node: int) -> list[EdgeRecord]:
        return list(self._out_edges_by_node.get(node, []))

    def in_edges(self, node: int) -> list[EdgeRecord]:
        return list(self._in_edges_by_node.get(node, []))

    def successors(self, node: int) -> list[int]:
        return list(dict.fromkeys(e.v for e in self._out_edges_by_node.get(node, [])))

    def predecessors(self, node: int) -> list[int]:
        return list(dict.fromkeys(e.u for e in self._in_edges_by_node.get(node, [])))

    def topological_order(self) -> list[int]:
        indeg = {nid: len(self._in_edges_by_node.get(nid, [])) for nid in self.nodes}
        succ = {
            nid: [edge.v for edge in self._out_edges_by_node.get(nid, [])]
            for nid in self.nodes
        }

        order: list[int] = []
        queue = deque(nid for nid in self.nodes if indeg[nid] == 0)

        while queue:
            nid = queue.popleft()
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


def _is_edge_attr_column(values: Any, n_edges: int) -> bool:
    if isinstance(values, (str, bytes, dict)):
        return False
    if not hasattr(values, "__len__"):
        return False
    try:
        return len(values) == n_edges
    except TypeError:
        return False


def build_delta_graph(links: dict, nodes: dict) -> DeltaGraph:
    edge_ids = list(links["id"])
    edge_conns = list(links["conn"])
    n_edges = len(edge_ids)

    attr_columns = {
        key: value
        for key, value in links.items()
        if _is_edge_attr_column(value, n_edges)
    }

    edges = []
    for edge_idx, (edge_id, conn_idx) in enumerate(zip(edge_ids, edge_conns)):
        attrs = {key: values[edge_idx] for key, values in attr_columns.items()}
        edges.append(
            EdgeRecord(
                edge_id=edge_id,
                u=conn_idx[0],
                v=conn_idx[1],
                attrs=attrs,
            )
        )

    node_ids = list(nodes["id"])
    inlet_ids = list(nodes.get("inlets", []))

    if "outlets" in nodes:
        outlet_ids = list(nodes["outlets"])
    else:
        outdeg = {nid: 0 for nid in node_ids}
        for edge in edges:
            outdeg[edge.u] += 1
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

        scores = np.array([routing.edge_score(graph, edge) for edge in out_edges], dtype=float)
        if np.any(~np.isfinite(scores)):
            raise ValueError(f"Non-finite routing scores found for outgoing edges of node {nid}.")
        if np.any(scores < 0):
            raise ValueError(f"Negative routing scores found for outgoing edges of node {nid}.")

        total = float(np.sum(scores))
        if total <= 0:
            raise ValueError(
                f"Outgoing routing scores for node {nid} sum to {total}; cannot normalize."
            )

        for edge, p in zip(out_edges, scores / total):
            probs[edge.edge_id] = float(p)

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
        for edge in graph.out_edges(nid):
            p = transition.edge_prob[edge.edge_id]
            flux[edge.v] += f * p

    return flux


def compute_edge_fluxes(
    graph: DeltaGraph,
    node_flux: dict[int, float],
    edge_prob: dict[int, float],
) -> dict[int, float]:
    return {
        edge.edge_id: float(node_flux[edge.u] * edge_prob[edge.edge_id])
        for edge in graph.edges
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
    node_order = list(graph.nodes)
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
        for edge in out_edges:
            p = transition.edge_prob[edge.edge_id]
            subn[i, :] += p * subn[node_to_i[edge.v], :]

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

    total_source = float(sum(source.values()))
    total_outlet = float(sum(outlet_flux.values()))
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



@dataclass
class AdjacencySteadyStateResult:
    node_flux: np.ndarray
    subnetwork_membership: np.ndarray
    apex: int
    outlets: np.ndarray
    transition_matrix: np.ndarray


def _column_normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Normalize adjacency so each non-outlet column sums to one."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('Adjacency matrix must be square.')
    colsum = np.sum(A, axis=0, dtype=float)
    P = np.zeros_like(A, dtype=float)
    nz = colsum > 0
    if np.any(nz):
        P[:, nz] = A[:, nz] / colsum[nz]
    return P


def solve_adjacency_steady_state(A: np.ndarray, *, atol: float = 1e-12) -> AdjacencySteadyStateResult:
    """Solve steady-state node fluxes and subnetwork membership on a DAG adjacency.

    Parameters
    ----------
    A : np.ndarray
        Square adjacency-like matrix using the historical delta-metrics convention
        where ``A[v, u]`` is the edge score from upstream node ``u`` to downstream
        node ``v``. Columns are normalized internally.
    atol : float, optional
        Tolerance used only for validation and zero-detection.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('Adjacency matrix must be square.')
    if np.any(~np.isfinite(A)):
        raise ValueError('Adjacency matrix contains non-finite values.')
    if np.any(A < -atol):
        raise ValueError('Adjacency matrix contains negative values.')

    # zero-out tiny numerical noise; metrics code often passes boolean or normalized matrices
    A = A.copy()
    A[np.abs(A) <= atol] = 0.0

    # Historical convention: rows with no incoming define the unique apex, columns with
    # no outgoing define outlets.
    row_sum = np.sum(A, axis=1)
    col_sum = np.sum(A, axis=0)
    apexes = np.where(np.abs(row_sum) <= atol)[0]
    if apexes.size != 1:
        raise RuntimeError('The graph contains more than one apex.')
    apex = int(apexes[0])
    outlets = np.where(np.abs(col_sum) <= atol)[0]
    if outlets.size == 0:
        raise ValueError('Adjacency matrix has no outlet nodes.')

    # Build DAG structure from the nonzero pattern of A[v, u].
    out_neighbors = {u: list(np.where(np.abs(A[:, u]) > atol)[0]) for u in range(A.shape[0])}
    indeg = {u: int(np.count_nonzero(np.abs(A[u, :]) > atol)) for u in range(A.shape[0])}

    order: list[int] = []
    queue = deque([u for u, deg in indeg.items() if deg == 0])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in out_neighbors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != A.shape[0]:
        raise ValueError('Adjacency matrix is not acyclic.')

    P = _column_normalize_adjacency(A)

    F = np.zeros(A.shape[0], dtype=float)
    F[apex] = 1.0
    for u in order:
        if F[u] == 0:
            continue
        for v in out_neighbors[u]:
            F[v] += F[u] * P[v, u]

    subn = np.zeros((A.shape[0], outlets.size), dtype=float)
    for j, outlet in enumerate(outlets):
        subn[outlet, j] = 1.0
    outlet_set = set(int(o) for o in outlets.tolist())
    for u in reversed(order):
        if u in outlet_set:
            continue
        for v in out_neighbors[u]:
            subn[u, :] += P[v, u] * subn[v, :]

    # QA only; outlet flux should sum to one.
    mbe = abs(float(np.sum(F[outlets])) - 1.0)
    if mbe > max(atol, 1e-10):
        raise ValueError(f'Adjacency steady-state mass balance failed; abs diff={mbe}.')

    return AdjacencySteadyStateResult(
        node_flux=F,
        subnetwork_membership=subn,
        apex=apex,
        outlets=outlets.astype(int),
        transition_matrix=P,
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
