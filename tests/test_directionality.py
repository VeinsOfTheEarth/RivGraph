"""Unit tests for directionality.py."""
import pytest
import sys
import os
import io
import numpy as np
import networkx as nx
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import directionality as di
from rivgraph import ln_utils as lnu
from rivgraph.deltas import delta_directionality as dd

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


# def test_dir_known_link_angles(known_net):
#     known_net.skeletonize()
#     dims = np.shape(known_net.Iskel)
#     links = known_net.links
#     nodes = known_net.nodes
#     links, nodes = di.dir_known_link_angles(links, nodes, dims)
#     # make assertions
#     assert links['guess'] == known_net.links['guess']
#     assert links['guess_alg'] == known_net.links['guess_alg']


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


def test_get_link_vec(known_net):
    """Test function get_link_vector()."""
    links = known_net.links
    nodes = known_net.nodes
    imshape = known_net.imshape
    linkid = 968
    pixlen = known_net.pixlen
    link_vec = di.get_link_vector(links, nodes, linkid, imshape, pixlen)
    # make assertion
    assert link_vec == pytest.approx(np.array([0.19371217, 0.98105841]))


def test_get_link_vec_trim(known_net):
    """Test function get_link_vector()."""
    links = known_net.links
    nodes = known_net.nodes
    imshape = known_net.imshape
    linkid = 968
    pixlen = known_net.pixlen
    link_vec = di.get_link_vector(links, nodes, linkid,
                                  imshape, pixlen, trim=True)
    # make assertion
    assert link_vec == pytest.approx(np.array([0.37003286, 0.92901867]))


def test_source_sink(known_net):
    """
    Test fix_sources_and_sinks().

    This test really just checks that the function doesn't break anything.
    """
    links, nodes = di.fix_sources_and_sinks(known_net.links, known_net.nodes)
    # make some assertions
    assert len(nodes['id']) == len(known_net.nodes['id'])
    assert len(links['id']) == len(known_net.links['id'])


def test_find_cycle(known_net):
    """Test find_a_cycle() by creating one."""
    links = known_net.links
    links = lnu.flip_link(links, 964)
    c_nodes, c_links = di.find_a_cycle(links, known_net.nodes)
    # make assertions
    assert c_nodes == [761, 784]
    assert c_links == [964, 964]


@pytest.mark.xfail
def test_fix_cycles(known_net):
    """Test fix_cycles()."""
    links = known_net.links
    nodes = known_net.nodes
    # check that cycle exists
    c_nodes, c_links = di.find_a_cycle(links, nodes)
    assert c_nodes == [761, 784]
    assert c_links == [964, 964]
    # check that there are no continuity issues
    problem_nodes = di.check_continuity(links, nodes)
    assert problem_nodes == []
    # now try to fix the cycles
    links, nodes, n_cycles = di.fix_cycles(links, nodes)
    # test fails at di.fix_cycles(links, nodes)
    # mystery given that c_nodes & c_links exist, aka there is a cycle
    # error: networkx.exception.NetworkXError: The edge 784-761 not in graph.


@pytest.mark.xfail
def test_fix_delta_cycles(known_net):
    """Test fix_delta_cycles()."""
    links = known_net.links
    nodes = known_net.nodes
    imshape = known_net.Imask.shape
    # check if a cycle exists
    c_nodes, c_links = di.find_a_cycle(links, nodes)
    assert c_nodes == [761, 784]
    assert c_links == [964, 964]
    # verify that no continuity issues exist
    problem_nodes = di.check_continuity(links, nodes)
    assert problem_nodes == []
    # try to fix the cycle
    links, nodes, allfixed = dd.fix_delta_cycles(links, nodes, imshape)
    # get a KeyError in OrderedSet - KeyError: 1081
    # happens when the 'set_parallel_links' function is run


@pytest.mark.xfail
def test_fix_cycles_river(known_river):
    """Test fix_cycles() with river example."""
    links = known_river.links
    nodes = known_river.nodes
    # flip link to create a cycle
    links = lnu.flip_link(links, 2494)
    # check that cycle exists
    c_nodes, c_links = di.find_a_cycle(links, nodes)
    assert c_nodes == [1794, 1805]
    assert c_links == [2494, 2494]
    # check that there are no continuity issues
    problem_nodes = di.check_continuity(links, nodes)
    assert problem_nodes == []
    # now try to fix the cycles
    links, nodes, n_cycles = di.fix_cycles(links, nodes)
    # test fails at di.fix_cycles(links, nodes)
    # mystery given that c_nodes & c_links exist, aka there is a cycle
    # error: networkx.exception.NetworkXError: The edge 1065-1051 not in graph.


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
    """Actually test fix_sources_and_sinks()."""
    links = known_net.links
    # verify that old problem node existed
    old_problem_nodes = di.check_continuity(links, known_net.nodes)
    assert old_problem_nodes == [177]
    # try to fix the continuity issue
    newlinks, nodes = di.fix_sources_and_sinks(links, known_net.nodes)
    # test fails at di.fix_sources_and_sinks(links, known_net.nodes)
    # we know an issue exists because we verified the problem node
    # error message: KeyError: 1081
    problem_nodes = di.check_continuity(links, known_net.nodes)
    # make assertion that no problem node exists - aka 'fix' worked
    assert problem_nodes == []


@pytest.mark.xfail
def test_set_link_directions(known_net):
    """Test set_link_directions()."""
    links = known_net.links
    nodes = known_net.nodes
    imshape = known_net.Imask.shape
    # verify that old problem node exists or create it
    old_problem_nodes = di.check_continuity(links, nodes)
    if 177 in old_problem_nodes:
        assert old_problem_nodes == [177]
    else:
        links = lnu.flip_link(links, 198)
        old_problem_nodes = di.check_continuity(links, nodes)
        assert old_problem_nodes == [177]
    # set up capture string
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    # apply the function
    newlinks, newnodes = dd.set_link_directions(links, nodes, imshape)
    # grab output
    sys.stdout = sys.__stdout__
    # assert output
    assert capturedOutput.getvalue()[:-1] == 'Nodes 177 violate continuity. Check connected links and fix manually.'
    # currently getting an error when setting the initial directionality to links/nodes...


def test_fix_source_sink_river(known_river):
    """Test fix_sources_and_sinks() with river example."""
    links = known_river.links
    # flip links to create sources/sinks and check that it happened
    links = lnu.flip_link(links, 2635)
    links = lnu.flip_link(links, 2634)
    bad_nodes = di.check_continuity(links, known_river.nodes)
    assert bad_nodes == [1897, 1899]
    # now try to fix them
    newlinks, nodes = di.fix_sources_and_sinks(links, known_river.nodes)
    # re-check for sources and sinks and verify that there are less bad nodes
    fixed_nodes = di.check_continuity(newlinks, known_river.nodes)
    assert len(fixed_nodes) < len(bad_nodes)
