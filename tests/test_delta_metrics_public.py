"""Public-facing tests for delta_metrics wrappers and legacy metric plumbing."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


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
