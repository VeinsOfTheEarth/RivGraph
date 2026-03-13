"""Public-facing tests for delta_metrics wrappers and legacy metric plumbing."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
import networkx as nx


class _DummyModule(types.ModuleType):
    """Minimal stub for GDAL/OGR/OSR modules during import-only tests."""

    def __getattr__(self, name):  # pragma: no cover - trivial shim
        return 0



def _import_delta_metrics_with_gdal_stubs():
    """Import delta_metrics without requiring GDAL in lightweight test envs."""
    for name in ("gdal", "ogr", "osr"):
        if name not in sys.modules:
            sys.modules[name] = _DummyModule(name)

    if "osgeo" not in sys.modules:
        osgeo = types.ModuleType("osgeo")
        osgeo.gdal = sys.modules["gdal"]
        osgeo.ogr = sys.modules["ogr"]
        osgeo.osr = sys.modules["osr"]
        sys.modules["osgeo"] = osgeo

    return importlib.import_module("rivgraph.deltas.delta_metrics")



def test_compute_steady_state_link_fluxes_public_wrapper():
    """Public wrapper should expose new solver behavior through the old API."""
    delta_metrics = _import_delta_metrics_with_gdal_stubs()

    links = {
        "id": [1, 2, 3],
        "conn": [[1, 2], [1, 2], [2, 3]],
        "wid_adj": [2.0, 1.0, 5.0],
    }
    nodes = {
        "id": [1, 2, 3],
        "inlets": [1],
    }

    out = delta_metrics.compute_steady_state_link_fluxes(
        None,
        links,
        nodes,
        weight_name="flux_ss",
    )

    assert out["flux_ss"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 1.0])



def test_delta_subn_f_uses_stable_dag_propagation():
    """Legacy delta_subN_F API should still return expected F and SubN."""
    delta_metrics = _import_delta_metrics_with_gdal_stubs()

    # Historical convention: A[v, u] is the score from upstream node u to downstream node v.
    A = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )

    F, subn = delta_metrics.delta_subN_F(A)

    assert F == pytest.approx([1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0])
    assert subn.shape == (4, 1)
    assert np.allclose(subn, np.ones((4, 1)))


def test_intermediate_vars_preserves_legacy_keys_and_values():
    """intermediate_vars should keep the legacy dict contract while using the new core."""
    delta_metrics = _import_delta_metrics_with_gdal_stubs()

    links = {
        "id": [1, 2, 3, 4],
        "conn": [[1, 2], [1, 3], [2, 4], [3, 4]],
        "wid_adj": [2.0, 1.0, 1.0, 1.0],
    }
    nodes = {
        "id": [1, 2, 3, 4],
        "inlets": [1],
    }

    G = delta_metrics.graphiphy(links, nodes, weight="wid_adj")
    deltavars = delta_metrics.intermediate_vars(G)

    expected_keys = {
        "A_w", "F_w", "SubN_w",
        "A_w_trans", "F_w_trans", "SubN_w_trans",
        "A_uw", "F_uw", "SubN_uw",
        "A_uw_trans", "F_uw_trans", "SubN_uw_trans",
        "apex", "outlets",
    }
    assert set(deltavars.keys()) == expected_keys

    assert deltavars["apex"].tolist() == [0]
    assert deltavars["outlets"].tolist() == [3]

    assert deltavars["F_w"] == pytest.approx([1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0])
    assert deltavars["F_uw_trans"] == pytest.approx([1.0, 0.5, 0.5, 1.0])
    assert np.allclose(deltavars["SubN_w"], np.ones((4, 1)))
    assert np.allclose(deltavars["SubN_uw_trans"], np.ones((4, 1)))

    # Weighted adjacency should preserve width-based splits; unweighted transitional
    # should split equally across the apex's two downstream links.
    assert deltavars["A_w"][1, 0] == pytest.approx(2.0 / 3.0)
    assert deltavars["A_w"][2, 0] == pytest.approx(1.0 / 3.0)
    assert deltavars["A_uw_trans"][1, 0] == pytest.approx(0.5)
    assert deltavars["A_uw_trans"][2, 0] == pytest.approx(0.5)



def _stub_metric_outputs(monkeypatch, delta_metrics):
    """Replace heavy metric functions with deterministic lightweight stubs."""
    monkeypatch.setattr(delta_metrics, 'delta_nER', lambda *_args, **_kwargs: (1.0, 2.0, np.array([3.0])))
    monkeypatch.setattr(delta_metrics, 'top_entropy_based_topo', lambda *_args, **_kwargs: (4.0, 5.0))
    monkeypatch.setattr(delta_metrics, 'top_link_sharing_index', lambda *_args, **_kwargs: 6.0)
    monkeypatch.setattr(delta_metrics, 'top_number_alternative_paths', lambda *_args, **_kwargs: 7.0)
    monkeypatch.setattr(delta_metrics, 'top_resistance_distance', lambda *_args, **_kwargs: 8.0)
    monkeypatch.setattr(delta_metrics, 'top_s2s_topo_pairwise_dep', lambda *_args, **_kwargs: 9.0)
    monkeypatch.setattr(delta_metrics, 'dyn_flux_sharing_index', lambda *_args, **_kwargs: 10.0)
    monkeypatch.setattr(delta_metrics, 'dyn_leakage_index', lambda *_args, **_kwargs: 11.0)
    monkeypatch.setattr(delta_metrics, 'dyn_pairwise_dep', lambda *_args, **_kwargs: 12.0)
    monkeypatch.setattr(delta_metrics, 'dyn_entropy_based_dyn', lambda *_args, **_kwargs: (13.0, 14.0))


def _simple_multi_inlet_network():
    links = {
        'id': [10, 11, 12],
        'conn': [[1, 2], [3, 2], [2, 4]],
        'idx': [[10, 20], [30, 20], [20, 40]],
        'wid_adj': [2.0, 1.0, 3.0],
        'wid': [2.0, 1.0, 3.0],
        'wid_med': [2.0, 1.0, 3.0],
        'sinuosity': [1.0, 1.0, 1.0],
        'len': [5.0, 5.0, 5.0],
        'len_adj': [5.0, 5.0, 5.0],
    }
    nodes = {
        'id': [1, 2, 3, 4],
        'idx': [10, 20, 30, 40],
        'conn': [[10], [10, 11, 12], [11], [12]],
        'inlets': [1, 3],
    }
    return links, nodes


def test_compute_delta_metrics_warns_and_uses_legacy_single_inlet(monkeypatch):
    """Multi-inlet default should keep backward compatibility via warned pruning."""
    delta_metrics = _import_delta_metrics_with_gdal_stubs()
    _stub_metric_outputs(monkeypatch, delta_metrics)

    links, nodes = _simple_multi_inlet_network()
    captured = {}

    def fake_ensure_single_inlet(links_arg, nodes_arg):
        captured['ensure_called'] = True
        return ({'id': [1], 'conn': [[1, 2]], 'wid_adj': [1.0]}, {'id': [1, 2], 'inlets': [1]})

    def fake_graphiphy(links_arg, nodes_arg, weight=None, inletweights=None):
        captured['weight'] = weight
        captured['inletweights'] = inletweights
        G = nx.DiGraph()
        G.add_edge(1, 2, weight=1.0)
        return G

    monkeypatch.setattr(delta_metrics, 'ensure_single_inlet', fake_ensure_single_inlet)
    monkeypatch.setattr(delta_metrics, 'graphiphy', fake_graphiphy)
    monkeypatch.setattr(delta_metrics, 'intermediate_vars', lambda G: {})

    with pytest.warns(UserWarning, match='legacy single-inlet pruning'):
        metrics = delta_metrics.compute_delta_metrics(links, nodes)

    assert captured['ensure_called'] is True
    assert captured['weight'] == 'wid_adj'
    assert captured['inletweights'] is None
    assert metrics['nonlin_entropy_rate'] == 1.0
    assert metrics['dyn_conditional_entropy'] == 14.0


def test_compute_delta_metrics_equal_inlet_policy_preserves_all_inlets(monkeypatch):
    """Explicit inlet policy should avoid pruning and pass normalized shares to the graph build."""
    delta_metrics = _import_delta_metrics_with_gdal_stubs()
    _stub_metric_outputs(monkeypatch, delta_metrics)

    links, nodes = _simple_multi_inlet_network()
    captured = {}

    def fail_ensure_single_inlet(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError('ensure_single_inlet should not be called when inlet policy is explicit.')

    def fake_graphiphy(links_arg, nodes_arg, weight=None, inletweights=None):
        captured['weight'] = weight
        captured['inletweights'] = inletweights
        captured['has_super_apex'] = 'super_apex' in nodes_arg
        G = nx.DiGraph()
        G.add_edge(1, 2, weight=1.0)
        return G

    monkeypatch.setattr(delta_metrics, 'ensure_single_inlet', fail_ensure_single_inlet)
    monkeypatch.setattr(delta_metrics, 'graphiphy', fake_graphiphy)
    monkeypatch.setattr(delta_metrics, 'intermediate_vars', lambda G: {})

    metrics = delta_metrics.compute_delta_metrics(links, nodes, inlet='equal', routing='uniform')

    assert captured['has_super_apex'] is True
    assert captured['inletweights'] == pytest.approx([0.5, 0.5])
    assert captured['weight'] is None
    assert metrics['top_mutual_info'] == 4.0
    assert metrics['flux_sharing_idx'] == 10.0


def test_compute_delta_metrics_user_inlet_policy_orders_weights_by_nodes_inlets(monkeypatch):
    """User inlet weights should follow the public nodes['inlets'] ordering in graphiphy."""
    delta_metrics = _import_delta_metrics_with_gdal_stubs()
    _stub_metric_outputs(monkeypatch, delta_metrics)

    links, nodes = _simple_multi_inlet_network()
    captured = {}

    def fake_graphiphy(links_arg, nodes_arg, weight=None, inletweights=None):
        captured['inletweights'] = inletweights
        G = nx.DiGraph()
        G.add_edge(1, 2, weight=1.0)
        return G

    monkeypatch.setattr(delta_metrics, 'graphiphy', fake_graphiphy)
    monkeypatch.setattr(delta_metrics, 'intermediate_vars', lambda G: {})

    delta_metrics.compute_delta_metrics(
        links,
        nodes,
        inlet='user',
        inlet_weights={3: 1.0, 1: 3.0},
    )

    assert captured['inletweights'] == pytest.approx([0.75, 0.25])
