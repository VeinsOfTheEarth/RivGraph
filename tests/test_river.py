"""Tests for broader `rivgraph.classes.river` functions."""
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import river


def test_compute_network(test_river, known_river):
    """Test compute_network()."""
    test_river.compute_network()

    # check that nodes and links are created
    assert len(test_river.nodes['id']) >= len(known_river.nodes['id'])
    assert len(test_river.links['id']) >= len(known_river.links['id'])


def test_prune_network(test_river, known_river):
    """Test prune_network()."""
    test_river.compute_mesh()
    test_river.prune_network()

    # check that nodes and links match known case
    assert len(test_river.nodes['id']) == len(known_river.nodes['id'])
    assert len(test_river.links['id']) == len(known_river.links['id'])

    # check that meshlines and meshpolys were created
    assert hasattr(test_river, 'meshlines') == True
    assert hasattr(test_river, 'meshpolys') == True
    assert hasattr(test_river, 'centerline_smooth') == True


def test_assign_flow_directions(test_river, known_river):
    """Test assigning flow directions."""
    test_river.assign_flow_directions()

    # make some simple assertions
    # node assertions
    assert len(test_river.nodes['conn']) == len(known_river.nodes['conn'])
    assert len(test_river.nodes['inlets']) == len(known_river.nodes['inlets'])
    assert len(test_river.nodes['outlets']) == len(known_river.nodes['outlets'])
    # link assertions
    assert len(test_river.links['conn']) == len(known_river.links['conn'])
    assert len(test_river.links['parallels']) == len(known_river.links['parallels'])

    # check that 90% of directions match known case
    # identify list of indices to check
    ind_list = range(0, len(known_river.nodes['idx']))

    # create list of connected idx values
    test_dirs = []
    known_dirs = []
    for j in ind_list:
        test_ind = test_river.nodes['idx'].index(known_river.nodes['idx'][j])
        # interrogate the 'conn' values to find corresponding 'idx' values
        t_inds = test_river.nodes['conn'][test_ind]
        t_idx = []
        for i in t_inds:
            t_idx.append(test_river.links['id'].index(i))

        k_inds = known_river.nodes['conn'][j]
        k_idx = []
        for i in k_inds:
            k_idx.append(known_river.links['id'].index(i))
        # add to the overall dirs lists
        test_dirs.append(test_river.links['idx'][t_idx[0]])
        known_dirs.append(known_river.links['idx'][k_idx[0]])

    # check how many sets of idx values match between the test and known case
    match_counter = 0
    for i in range(0, len(test_dirs)):
        if test_dirs[i] == known_dirs[i]:
            match_counter += 1

    # "soft" unit test -- check that over 90% of the idx values match
    assert match_counter / len(ind_list) > 0.9
