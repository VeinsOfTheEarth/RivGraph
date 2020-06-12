"""Unit tests for directionality.py."""
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import directionality as di

# Functions on the 'known_net' of the extracted Colville network. This
# network does not have any cycles, so functionality related to cycles is just
# tested here to show that it does not break a correct/working set of links and
# nodes. More complicated cycle tests still need to be constructed to catch all
# of the types of cycles and the methods for handling them.


def test_dir_known_link_angles(known_net):
    """Test dir_known_link_angles()."""
    known_net.skeletonize()
    dims = np.shape(known_net.Iskel)
    links = known_net.links
    nodes = known_net.nodes
    links, nodes = di.dir_known_link_angles(links, nodes, dims)
    # make assertions
    assert links['guess'] == known_net.links['guess']
    assert links['guess_alg'] == known_net.links['guess_alg']


def test_cycle_get_original_orient(known_net):
    """Test cycle_get_original_orientation()."""
    orig = di.cycle_get_original_orientation(known_net.links,
                                             known_net.links['id'])
    # make assertions
    assert orig['id'] == known_net.links['id']
    assert orig['conn'] == known_net.links['conn']
    assert orig['idx'] == known_net.links['idx']
    assert np.all(orig['wid_pix'][0] == known_net.links['wid_pix'][0])
    assert np.all(orig['certain_alg'][0] == known_net.links['certain_alg'][0])
    assert np.all(orig['certain_order'][0]
                  == known_net.links['certain_order'][0])


def test_return_orient(known_net):
    """Test cycle_return_to_original_orientation()."""
    orig = di.cycle_get_original_orientation(known_net.links,
                                             known_net.links['id'])
    links = known_net.links
    re_links = di.cycle_return_to_original_orientation(links, orig)
    # make assertions
    assert re_links['id'] == links['id']
    assert re_links['conn'] == links['conn']
    assert re_links['idx'] == links['idx']
    assert np.all(re_links['wid_pix'][0] == links['wid_pix'][0])
    assert np.all(re_links['certain_alg'][0] == links['certain_alg'][0])
    assert np.all(re_links['certain_order'][0] == links['certain_order'][0])


def test_no_backtrack(known_net):
    """Test set_no_backtrack()."""
    links = known_net.links
    nodes = known_net.nodes
    nb_links, nb_nodes = di.set_no_backtrack(links, nodes)
    # make assertions
    assert nb_links['id'] == links['id']
    assert nb_links['conn'] == links['conn']
    assert nb_links['idx'] == links['idx']
    assert np.all(nb_links['wid_pix'][0] == links['wid_pix'][0])
    assert np.all(nb_links['certain_alg'][0] == links['certain_alg'][0])
    assert np.all(nb_links['certain_order'][0] == links['certain_order'][0])
    assert nb_nodes['id'] == nodes['id']
    assert nb_nodes['conn'] == nodes['conn']
    assert nb_nodes['inlets'] == nodes['inlets']
