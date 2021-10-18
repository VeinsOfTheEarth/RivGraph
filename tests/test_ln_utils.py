"""Unit tests for ln_utils.py."""
# import pytest
import sys
import os
import io
try:
    from osgeo import gdal
except ImportError:
    import gdal
# import numpy as np
# import matplotlib.pyplot as plt

from inspect import getsourcefile
basepath = os.path.dirname(os.path.dirname(os.path.abspath(getsourcefile(lambda:0))))
sys.path.insert(0, basepath)

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
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
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
    new_nodes = ln_utils.delete_node(nodes, 10, warn=True)
    # grab output
    sys.stdout = sys.__stdout__
    # do assertion
    assert capturedOutput.getvalue()[:-1] == 'You are deleting node 10 which still has connections to links.'
    assert new_nodes['conn'] == []
