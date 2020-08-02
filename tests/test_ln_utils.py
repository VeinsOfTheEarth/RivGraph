"""Unit tests for ln_utils.py."""
import pytest
import sys
import os
import io
import gdal
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
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


def test_conn_links(test_net):
    """Test out the conn_links() function."""
    test_net.skeletonize()
    test_net.compute_network()
    # test some and make assertions
    link_pix_01 = ln_utils.conn_links(test_net.nodes, test_net.links, 139249)
    link_pix_02 = ln_utils.conn_links(test_net.nodes, test_net.links, 614511)
    link_pix_03 = ln_utils.conn_links(test_net.nodes, test_net.links, 762395)
    assert link_pix_01 == [139249, 139259, 129862, 139249, 151567, 139249]
    assert link_pix_02 == [744008, 740931, 1320184, 1411025, 2003991, 2000915]
    assert link_pix_03 == [2304222, 2301073]


def test_append_link_len(test_net):
    """Test append_link_lengths() function."""
    gobj = gdal.Open('tests/data/Colville/Colville_islands_filled.tif')
    test_net.skeletonize()
    test_net.compute_network()
    # assert keys in links before getting lengths
    assert ('len' in test_net.links) is False
    # then run the function
    links = ln_utils.append_link_lengths(test_net.links, gobj)
    # then assert that lengths is in links
    assert ('len' in links) is True


def test_add_art_nodes(test_net):
    """Testing add_artificial_nodes() function."""
    gobj = gdal.Open('tests/data/Colville/Colville_islands_filled.tif')
    test_net.skeletonize()
    test_net.compute_network()
    # assert no artificial links or nodes before function
    assert ('arts' in test_net.links) is False
    assert ('arts' in test_net.nodes) is False
    # then run the function
    links, nodes = ln_utils.add_artificial_nodes(test_net.links,
                                                 test_net.nodes,
                                                 gobj)
    # assert that arts is now in nodes and links
    assert ('arts' in links) is True
    assert ('arts' in nodes) is True
