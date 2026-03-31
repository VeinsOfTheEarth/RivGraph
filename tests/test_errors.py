"""Focused exception/error tests."""
from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import REGRESSION_DATA_ROOT, require_raster_runtime, require_rivgraph_classes


def _delta_metrics_module():
    require_raster_runtime()
    from rivgraph.deltas import delta_metrics

    return delta_metrics


def test_graphiphy():
    delta_metrics = _delta_metrics_module()
    links = {"id": [1]}
    nodes = {"id": [2]}
    with pytest.raises(RuntimeError):
        delta_metrics.graphiphy(links, nodes, weight="bad")


def test_inlet_outlet():
    delta_metrics = _delta_metrics_module()
    A = np.ones((5, 5))
    with pytest.raises(RuntimeError):
        delta_metrics.find_inlet_outlet_nodes(A)


def test_river_noexit():
    _, river, _ = require_rivgraph_classes()
    mask = REGRESSION_DATA_ROOT / "river_brahma_clipped" / "inputs" / "mask.tif"
    with pytest.raises(ValueError, match="Must provide exit_sides"):
        river("synth_river", str(mask), "tests/results/synthetic_cycles/")