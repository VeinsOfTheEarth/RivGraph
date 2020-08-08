"""Unit tests for a simple synthetic case."""
import pytest
import sys
import os
import io
import numpy as np
import networkx as nx
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import directionality as di
from rivgraph import mask_utils as mu


def test_check_props(synthetic_cycles):
    """check that synthetic system was properly established."""
    assert type(synthetic_cycles) is not None
    assert synthetic_cycles.gt == (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)
    assert synthetic_cycles.imshape == (15, 10)
    assert synthetic_cycles.pixarea == 1.0
    assert synthetic_cycles.pixlen == 1.0
    assert synthetic_cycles.unit == 'degree'


def test_skiptolinks(synthetic_cycles):
    """Test compute_link_width_and_length without computing network first."""
    synthetic_cycles.compute_link_width_and_length()
    # make assertions
    assert synthetic_cycles.nodes['id'].items == [0, 1, 2, 3]
    assert synthetic_cycles.links['id'].items == [0, 1, 2, 4]


def test_get_islands(synthetic_cycles):
    """Test get_islands()."""
    islands = synthetic_cycles.get_islands()
    assert np.shape(islands) == (2,)
    assert islands[1].shape == (15, 10)
    # make it into a dictionary for easier testing
    idict = islands[0].to_dict()
    assert idict['area'] == {1: 8.0}
    assert ('major_axis_length' in idict.keys()) is True
    assert ('minor_axis_length' in idict.keys()) is True
    assert ('geometry' in idict.keys()) is True
    assert ('maxwid' in idict.keys()) is True
    assert ('sur_area' in idict.keys()) is True
    assert ('sur_avg_wid' in idict.keys()) is True
    assert ('sur_max_wid' in idict.keys()) is True
    assert ('sur_min_wid' in idict.keys()) is True
    assert ('sur_link_ids' in idict.keys()) is True
    assert ('remove' in idict.keys()) is True


def test_get_islands_verbose(synthetic_cycles):
    """Test get_islands() with verbosity."""
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    synthetic_cycles.verbose = True
    islands = synthetic_cycles.get_islands()
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Getting island properties...done.'


@pytest.mark.xfail
def test_assigning_inletshore(synthetic_cycles):
    """Test setting inlet/shoreline."""
    synthetic_cycles.prune_network(path_shoreline='tests/data/SyntheticCycle/shoreline.shp',
                                   path_inletnodes='tests/data/SyntheticCycle/inlet_node.shp')
    # breaks in one of the geopandas functions
    # suggests that synthetic case handling is not working correctly


# Delete data created by tests in this file ...

def test_delete_files():
    """Delete created files at the end."""
    for i in os.listdir('tests/results/synthetic_cycles/'):
        os.remove('tests/results/synthetic_cycles/'+i)
    # check directory is empty
    assert os.listdir('tests/results/synthetic_cycles/') == []
