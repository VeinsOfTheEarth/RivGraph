"""Testing RivGraph Exceptions/Errors."""
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import delta
from rivgraph.classes import river
from rivgraph import ln_utils


def test_geovectorType(known_net):
    """Raise TypeError in to_geovectors() function."""
    with pytest.raises(Exception):
        known_net.to_geovecors(ftype='invalid')


def test_no_shoreline(test_net):
    """Raise attribute error."""
    with pytest.raises(Exception):
        test_net.prune_network(path_inletnodes='tests/data/Colville/Colville_inlet_nodes.shp')


def test_no_inlet(test_net):
    """Raise attribute error."""
    with pytest.raises(Exception):
        test_net.prune_network(path_shoreline='tests/data/Colville/Colville_shoreline.shp')


def test_flow_no_network(test_net):
    """Raise attribute error due to lack of links (network not computed)."""
    with pytest.raises(Exception):
        test_net.assign_flow_direcions()


def test_compute_metrics(test_net):
    """Raise attribute error due to lack of links (network not computed)."""
    with pytest.raises(Exception):
        test_net.compute_topologic_metrics()


def test_river_prune(test_river):
    """Raise attribute error related to river pruning."""
    with pytest.raises(Exception):
        test_river.prune_network()


def test_river_flow_bad(test_river):
    """Raise attribute error associated with assign_flow_directions."""
    with pytest.raises(Exception):
        test_river.assign_flow_directions()


def test_junction_angles(test_river):
    """Raise KeyError due to lack of flow directions."""
    test_river.compute_network()
    with pytest.raises(Exception):
        ln_utils.junction_angles(test_river.links, test_river.nodes,
                                 test_river.Iskel.shape, 5)
