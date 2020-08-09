"""Testing RivGraph Exceptions/Errors."""
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import delta
from rivgraph.classes import river
from rivgraph import ln_utils
from rivgraph.deltas import delta_metrics


def test_geovectorType(known_net):
    """Raise TypeError in to_geovectors() function."""
    with pytest.raises(TypeError):
        known_net.to_geovectors(ftype='invalid')


def test_no_shoreline(test_net):
    """Raise attribute error."""
    with pytest.raises(AttributeError):
        test_net.prune_network(path_inletnodes='tests/data/Colville/Colville_inlet_nodes.shp')


def test_no_inlet(test_net):
    """Raise attribute error."""
    with pytest.raises(AttributeError):
        test_net.prune_network(path_shoreline='tests/data/Colville/Colville_shoreline.shp')


def test_bad_shore_prune(test_net):
    """Raise AttributeError because shoreline shapefile can't be found."""
    with pytest.raises(AttributeError):
        test_net.prune_network(path_shoreline='bad', path_inletnodes='tests/data/Colville/Colville_inlet_nodes.shp')


def test_bad_inlet_prune(test_net):
    """Raise AttributeError because inlet shapefile can't be found."""
    with pytest.raises(AttributeError):
        test_net.prune_network(path_shoreline='tests/data/Colville/Colville_shoreline.shp', path_inletnodes='bad')


def test_flow_no_network(test_net):
    """Raise attribute error due to lack of links (network not computed)."""
    with pytest.raises(AttributeError):
        test_net.assign_flow_directions()


def test_compute_metrics(test_net):
    """Raise attribute error due to lack of links (network not computed)."""
    with pytest.raises(AttributeError):
        test_net.compute_topologic_metrics()


def test_river_prune(test_river):
    """Raise attribute error related to river pruning."""
    with pytest.raises(AttributeError):
        test_river.prune_network()


def test_river_flow_bad(test_river):
    """Raise attribute error associated with assign_flow_directions."""
    with pytest.raises(AttributeError):
        test_river.assign_flow_directions()


def test_junction_angles(test_river):
    """Raise KeyError due to lack of flow directions."""
    test_river.compute_network()
    with pytest.raises(KeyError):
        ln_utils.junction_angles(test_river.links, test_river.nodes,
                                 test_river.Iskel.shape, 5)


def test_graphiphy():
    """Raise RuntimeError due to missing weight."""
    links = dict()
    nodes = dict()
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
