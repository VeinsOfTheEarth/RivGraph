"""Testing RivGraph Exceptions/Errors."""
import pytest
import numpy as np
from rivgraph.classes import river
from rivgraph.deltas import delta_metrics


def test_graphiphy():
    """Raise RuntimeError due to missing weight."""
    links = {'id': [1]}
    nodes = {'id': [2]}
    with pytest.raises(RuntimeError):
        delta_metrics.graphiphy(links, nodes, weight='bad')


def test_inlet_outlet():
    """Raise RuntimeError due to multiple apexes."""
    A = np.ones((5, 5))
    with pytest.raises(RuntimeError):
        delta_metrics.find_inlet_outlet_nodes(A)


def test_river_noexit():
    """Raise Warning when river created without exit sides."""
    with pytest.raises(Warning):
        river('synth_river',
              'tests/data/SyntheticCycle/skeleton.tif',
              'tests/results/synthetic_cycles/')
