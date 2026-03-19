from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class NetworkValidationReport:
    n_nodes: int = 0
    n_links: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def _as_list(values):
    if values is None:
        return []
    return list(values)


def _ensure_required_keys(data: dict, required: tuple[str, ...], label: str, errors: list[str]) -> bool:
    missing = [key for key in required if key not in data]
    if missing:
        errors.append(f"{label} is missing required key(s): {missing}.")
        return False
    return True


def _is_column_aligned(values: Any, expected_len: int) -> bool:
    if isinstance(values, (str, bytes, dict)):
        return False
    if not hasattr(values, '__len__'):
        return False
    try:
        return len(values) == expected_len
    except TypeError:
        return False


def _build_cycle_check_graph(link_ids, link_conns, node_ids):
    G = nx.MultiDiGraph()
    G.add_nodes_from(node_ids)
    for lid, conn in zip(link_ids, link_conns):
        G.add_edge(conn[0], conn[1], key=lid, link_id=lid)
    return G



def dag_diagnostics_from_network(links: dict, nodes: dict) -> dict[str, Any]:
    """Return DAG diagnostics for a structurally valid RivGraph network."""
    link_ids = _as_list(links['id'])
    link_conns = _as_list(links['conn'])
    node_ids = _as_list(nodes['id'])

    G = _build_cycle_check_graph(link_ids, link_conns, node_ids)
    diagnostics = {
        'is_dag': nx.is_directed_acyclic_graph(G),
        'n_nodes': len(node_ids),
        'n_links': len(link_ids),
    }

    if diagnostics['is_dag'] is True:
        diagnostics['cyclic_regions'] = []
        return diagnostics

    cyclic_regions = []
    for comp in nx.strongly_connected_components(G):
        comp = set(comp)
        has_self_loop = any(G.has_edge(nid, nid) for nid in comp)
        if len(comp) == 1 and has_self_loop is False:
            continue

        comp_links = sorted({
            data['link_id']
            for u, v, _k, data in G.edges(keys=True, data=True)
            if u in comp and v in comp
        })

        cyclic_regions.append({
            'nodes': sorted(comp),
            'links': comp_links,
        })

    cyclic_regions.sort(key=lambda item: (-len(item['nodes']), -len(item['links']), item['nodes']))
    diagnostics['cyclic_regions'] = cyclic_regions
    return diagnostics



def validate_rivgraph_network(
    links: dict,
    nodes: dict,
    *,
    require_inlets: bool = False,
    require_outlets: bool = False,
    require_dag: bool = False,
    required_link_weight: str | None = None,
) -> NetworkValidationReport:
    """Validate internal RivGraph link/node dictionaries.

    This validator is intentionally internal and focused on preflight checks for
    downstream algorithms. It verifies structural consistency between the link
    and node dictionaries and, optionally, algorithm prerequisites such as DAG
    topology and routing-weight availability.
    """
    report = NetworkValidationReport()

    links_ok = _ensure_required_keys(links, ('id', 'conn'), 'links', report.errors)
    nodes_ok = _ensure_required_keys(nodes, ('id', 'conn'), 'nodes', report.errors)
    if not links_ok or not nodes_ok:
        return report

    link_ids = _as_list(links['id'])
    link_conns = _as_list(links['conn'])
    node_ids = _as_list(nodes['id'])
    node_conns = _as_list(nodes['conn'])

    report.n_nodes = len(node_ids)
    report.n_links = len(link_ids)

    if len(link_conns) != len(link_ids):
        report.errors.append(
            f"links['conn'] has length {len(link_conns)} but links['id'] has length {len(link_ids)}."
        )
    if len(node_conns) != len(node_ids):
        report.errors.append(
            f"nodes['conn'] has length {len(node_conns)} but nodes['id'] has length {len(node_ids)}."
        )

    if len(set(link_ids)) != len(link_ids):
        report.errors.append('links[\'id\'] contains duplicate link ids.')
    if len(set(node_ids)) != len(node_ids):
        report.errors.append('nodes[\'id\'] contains duplicate node ids.')

    if report.errors:
        return report

    node_id_set = set(node_ids)
    link_id_set = set(link_ids)
    link_endpoints: dict[int, tuple[int, int]] = {}

    for lid, conn in zip(link_ids, link_conns):
        if not hasattr(conn, '__len__') or len(conn) != 2:
            report.errors.append(f"Link {lid} must connect exactly two node ids; got {conn!r}.")
            continue
        u, v = conn[0], conn[1]
        link_endpoints[lid] = (u, v)
        if u not in node_id_set or v not in node_id_set:
            report.errors.append(
                f"Link {lid} references node ids {conn!r} that are not all present in nodes['id']."
            )

    if report.errors:
        return report

    node_conn_map = {nid: list(conn) for nid, conn in zip(node_ids, node_conns)}

    for nid, conn_links in node_conn_map.items():
        for lid in conn_links:
            if lid not in link_id_set:
                report.errors.append(
                    f"Node {nid} references link id {lid}, but that link is not present in links['id']."
                )
                continue
            u, v = link_endpoints[lid]
            if nid not in (u, v):
                report.errors.append(
                    f"Node {nid} lists link {lid} in nodes['conn'], but link {lid} connects nodes {(u, v)}."
                )

    for lid, (u, v) in link_endpoints.items():
        missing = [nid for nid in (u, v) if lid not in node_conn_map[nid]]
        if missing:
            report.errors.append(
                f"Link {lid} connects nodes {(u, v)}, but nodes {missing} do not reference that link in nodes['conn']."
            )

    inlets = _as_list(nodes.get('inlets'))
    outlets = _as_list(nodes.get('outlets'))

    unknown_inlets = [nid for nid in inlets if nid not in node_id_set]
    if unknown_inlets:
        report.errors.append(f"nodes['inlets'] contains unknown node ids: {unknown_inlets}.")

    if require_inlets and len(inlets) == 0:
        report.errors.append("Network has no inlet nodes.")

    if outlets:
        unknown_outlets = [nid for nid in outlets if nid not in node_id_set]
        if unknown_outlets:
            report.errors.append(f"nodes['outlets'] contains unknown node ids: {unknown_outlets}.")
    else:
        outdeg = {nid: 0 for nid in node_ids}
        for u, _v in link_endpoints.values():
            outdeg[u] += 1
        inferred_outlets = [nid for nid, deg in outdeg.items() if deg == 0]
        report.diagnostics['outlets_inferred'] = inferred_outlets
        outlets = inferred_outlets

    if require_outlets and len(outlets) == 0:
        report.errors.append("Network has no outlet nodes.")

    if required_link_weight is not None:
        if required_link_weight not in links:
            report.errors.append(
                f"links is missing required routing-weight field '{required_link_weight}'."
            )
        else:
            values = links[required_link_weight]
            if not _is_column_aligned(values, len(link_ids)):
                report.errors.append(
                    f"links['{required_link_weight}'] does not align with links['id']."
                )
            else:
                arr = np.asarray(values, dtype=float)
                if np.any(~np.isfinite(arr)):
                    report.errors.append(
                        f"links['{required_link_weight}'] contains non-finite values."
                    )
                if np.any(arr <= 0):
                    report.errors.append(
                        f"links['{required_link_weight}'] must be strictly positive for routing."
                    )

    if report.errors:
        return report

    if require_dag:
        dag = dag_diagnostics_from_network(links, nodes)
        report.diagnostics.update(dag)
        if dag['is_dag'] is False:
            regions = dag['cyclic_regions']
            example = regions[0] if len(regions) > 0 else {'nodes': [], 'links': []}
            report.errors.append(
                'Directed graph is not acyclic. '
                f"Detected {len(regions)} cyclic region(s); example nodes={example['nodes']} links={example['links']}."
            )

    return report



def raise_if_invalid_network(links: dict, nodes: dict, *, context: str, **kwargs) -> NetworkValidationReport:
    report = validate_rivgraph_network(links, nodes, **kwargs)
    if report.is_valid:
        return report

    joined = '; '.join(report.errors)
    raise ValueError(f"Invalid network for {context}: {joined}")
