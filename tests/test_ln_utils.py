"""Unit tests for ln_utils.py."""
# import pytest
import numpy as np
from scipy.ndimage import distance_transform_edt as dte
try:
    from osgeo import gdal
except ImportError:
    import gdal

from rivgraph import ln_utils
from rivgraph.ordered_set import OrderedSet


def test_add_node():
    """Check catch for node already in set."""
    # initialize node
    nodes = dict()
    nodes['idx'] = OrderedSet([])
    nodes['id'] = OrderedSet([])
    nodes['conn'] = []
    # set node values
    nodes['idx'].append(1)
    nodes['id'].append(10)
    nodes['conn'].append([5])
    # set the other input parameters up
    idx = 1
    linkconn = [5]
    # run the function
    new_nodes = ln_utils.add_node(nodes, idx, linkconn)
    # make assertion
    assert new_nodes.keys() == nodes.keys()
    assert new_nodes['idx'] == nodes['idx']
    assert new_nodes['id'] == nodes['id']
    assert new_nodes['conn'] == nodes['conn']


def test_delete_node():
    """Check print warning about node still having connections."""
    # initialize node
    nodes = dict()
    nodes['idx'] = OrderedSet([])
    nodes['id'] = OrderedSet([])
    nodes['conn'] = []
    # set node values
    nodes['idx'].append(1)
    nodes['id'].append(10)
    nodes['conn'].append([5])
    # run delete_node function
    new_nodes = ln_utils.delete_node(nodes, 10, warn=False)
    # do assertion
    assert new_nodes['conn'] == []


def test_link_widths_lengths():
    """Test the link widths and lengths calculation."""
    links = dict()
    Idt = np.zeros((10, 10))
    Idt[1:-1, 5] = 1.0
    inds = np.where(Idt == 1)
    links['idx'] = [list(np.ravel_multi_index(inds, Idt.shape))]
    Idt[4, 4:8] = 1.0
    links = ln_utils.link_widths_and_lengths(links, dte(Idt))
    # assert things about computed link properties
    assert links['sinuosity'][0] == 1.0
    assert len(links['idx'][0]) == 8
    assert links['wid_med'][0] == 2.0
    assert links['len_adj'][0] == 5.0
    assert links['len'][0] == 7.0
    assert links['wid_adj'][0] > 2.0
    assert links['wid_adj'][0] > links['wid'][0]  # b/c single end px clipped
