"""Unit tests for the new delta-metrics steady-state core."""
from pathlib import Path
import pickle

import numpy as np
import pytest

from rivgraph.deltas._delta_metrics_core import build_delta_graph, solve_steady_state
from rivgraph.ordered_set import OrderedSet



def test_build_delta_graph_accepts_nonlist_columns():
    """Builder should accept OrderedSet ids and numpy-array attributes."""
    links = {
        "id": OrderedSet([10, 11]),
        "conn": [[1, 2], [2, 3]],
        "wid_adj": np.array([4.0, 2.0]),
    }
    nodes = {
        "id": OrderedSet([1, 2, 3]),
        "inlets": [1],
    }

    graph = build_delta_graph(links, nodes)

    assert graph.nodes == [1, 2, 3]
    assert graph.inlets == [1]
    assert graph.outlets == [3]
    assert graph.edges[0].attrs["wid_adj"] == pytest.approx(4.0)
    assert graph.edges[1].attrs["wid_adj"] == pytest.approx(2.0)



def test_solve_steady_state_single_inlet_width_partition():
    """Width-based routing should recover the expected steady-state fluxes."""
    links = {
        "id": [1, 2, 3, 4],
        "conn": [[1, 2], [1, 3], [2, 4], [3, 4]],
        "wid_adj": [2.0, 1.0, 1.0, 1.0],
    }
    nodes = {
        "id": [1, 2, 3, 4],
        "inlets": [1],
    }

    result = solve_steady_state(build_delta_graph(links, nodes))

    assert result.node_flux[1] == pytest.approx(1.0)
    assert result.node_flux[2] == pytest.approx(2.0 / 3.0)
    assert result.node_flux[3] == pytest.approx(1.0 / 3.0)
    assert result.node_flux[4] == pytest.approx(1.0)

    assert result.edge_flux[1] == pytest.approx(2.0 / 3.0)
    assert result.edge_flux[2] == pytest.approx(1.0 / 3.0)
    assert result.edge_flux[3] == pytest.approx(2.0 / 3.0)
    assert result.edge_flux[4] == pytest.approx(1.0 / 3.0)

    assert result.outlet_flux[4] == pytest.approx(1.0)
    assert result.mass_balance_error == pytest.approx(0.0)
    assert np.allclose(result.subnetwork_membership, np.ones((4, 1)))



def test_solve_steady_state_handles_parallel_links():
    """Parallel links should remain distinct and receive separate fluxes."""
    links = {
        "id": [1, 2, 3],
        "conn": [[1, 2], [1, 2], [2, 3]],
        "wid_adj": [2.0, 1.0, 3.0],
    }
    nodes = {
        "id": [1, 2, 3],
        "inlets": [1],
    }

    result = solve_steady_state(build_delta_graph(links, nodes))

    assert result.edge_flux[1] == pytest.approx(2.0 / 3.0)
    assert result.edge_flux[2] == pytest.approx(1.0 / 3.0)
    assert result.edge_flux[3] == pytest.approx(1.0)
    assert result.node_flux[2] == pytest.approx(1.0)
    assert result.outlet_flux[3] == pytest.approx(1.0)



def test_solve_steady_state_multiple_inlets_user_weights():
    """User-specified inlet weights should define the source vector."""
    links = {
        "id": [1, 2],
        "conn": [[1, 3], [2, 3]],
        "wid_adj": [5.0, 5.0],
    }
    nodes = {
        "id": [1, 2, 3],
        "inlets": [1, 2],
    }

    result = solve_steady_state(
        build_delta_graph(links, nodes),
        inlet="user",
        inlet_weights={1: 0.75, 2: 0.25},
    )

    assert result.transition.source[1] == pytest.approx(0.75)
    assert result.transition.source[2] == pytest.approx(0.25)
    assert result.edge_flux[1] == pytest.approx(0.75)
    assert result.edge_flux[2] == pytest.approx(0.25)
    assert result.node_flux[3] == pytest.approx(1.0)
    assert result.outlet_flux[3] == pytest.approx(1.0)



def test_solve_steady_state_real_fixture_smoke_test():
    """The new solver should work on an actual RivGraph fixture."""
    fixture = Path("tests/integration/data/Colville/Colville_network.pkl")
    with fixture.open("rb") as f:
        links, nodes = pickle.load(f)

    result = solve_steady_state(build_delta_graph(links, nodes))

    assert len(result.edge_flux) == len(links["id"])
    assert sum(result.transition.source.values()) == pytest.approx(1.0)
    assert sum(result.outlet_flux.values()) == pytest.approx(1.0)
    assert result.mass_balance_error == pytest.approx(0.0)



def test_user_inlet_policy_requires_all_inlets():
    """Partial user inlet weights should fail loudly."""
    links = {
        "id": [1, 2],
        "conn": [[1, 3], [2, 3]],
        "wid_adj": [1.0, 1.0],
    }
    nodes = {
        "id": [1, 2, 3],
        "inlets": [1, 2],
    }

    with pytest.raises(ValueError, match="Missing user-specified inlet weights"):
        solve_steady_state(
            build_delta_graph(links, nodes),
            inlet="user",
            inlet_weights={1: 1.0},
        )
