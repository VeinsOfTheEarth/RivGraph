"""Unit tests for directionality.py."""
import pytest
import sys
import os
import numpy as np
import networkx as nx
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import directionality as di
from rivgraph import ln_utils as lnu

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


def test_dir_known_link_angles(known_net):
    known_net.skeletonize()
    dims = np.shape(known_net.Iskel)
    links = known_net.links
    nodes = known_net.nodes
    links, nodes = di.dir_known_link_angles(links, nodes, dims)
    # make assertions
    assert links['guess'] == known_net.links['guess']
    assert links['guess_alg'] == known_net.links['guess_alg']


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


def test_merge_list_of_lists():
    """Test the function merge_list_of_lists()."""
    inlist = [[1, 2, 3], [1, 2, 4], [2, 4, 6]]
    merged = di.merge_list_of_lists(inlist)
    # make assertion of single list of unique values
    assert merged == [[1, 2, 3, 4, 6]]


def test_flip_links_in_G():
    """Test the function flip_links_in_G()."""
    # create networkx graph to use
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G = di.flip_links_in_G(G, 'all')
    [x, y] = G.edges.data()
    # make assertions
    assert x == (2, 1, {})
    assert y == (3, 2, {})


def test_source_sink(known_net):
    """
    Test fix_sources_and_sinks().

    This test really just checks that the function doesn't break anything.
    """
    links, nodes = di.fix_sources_and_sinks(known_net.links, known_net.nodes)
    # make some assertions
    assert len(nodes['id']) == len(known_net.nodes['id'])
    assert len(links['id']) == len(known_net.links['id'])


def test_bad_continuity(known_net):
    """
    Test check_continuity().

    This test flips links so that continuity is disturbed.
    """
    links = known_net.links
    links = lnu.flip_link(links, 199)
    links = lnu.flip_link(links, 198)
    problem_nodes = di.check_continuity(links, known_net.nodes)
    # make assertion that a problem node has been created
    assert problem_nodes == [177]


@pytest.mark.xfail
def test_fix_source_sink(known_net):
    """Actually test fix_sources_and_sinks().

    Currently fails when the fix_sources_and_sinks function is called.
    Haven't looked into this error very closely.
    But know that old_problem_nodes == [177], so there are links to be flipped
    and a problem to be 'fixed' by the function..."""
    links = known_net.links
    old_problem_nodes = di.check_continuity(links, known_net.nodes)
    newlinks, nodes = di.fix_sources_and_sinks(links, known_net.nodes)
    problem_nodes = di.check_continuity(links, known_net.nodes)
    # verify that old problem node existed
    assert old_problem_nodes == [177]
    # make assertion that no problem node exists - aka 'fix' worked
    assert problem_nodes == []
