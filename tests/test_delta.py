"""Tests for broader `rivgraph.classes.delta` functions."""
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import delta
from rivgraph import directionality


def test_skeletonize(test_net):
    """Test skeletonization."""
    # do skeletonization
    test_net.skeletonize()
    # check that all Iskel 1 values are 1s in the Imask array aka no created 1s
    mask = np.where(test_net.Iskel == 1)
    assert np.all(test_net.Imask[mask] == True)
    # ensure the size of the input mask is equal to size of skeletonized output
    assert np.shape(test_net.Imask) == np.shape(test_net.Iskel)
    # ensure max value in skeleton are 1
    assert np.max(test_net.Iskel) == 1
    # ensure min value in skeleton is 0
    assert np.min(test_net.Iskel) == 0
    # test specific pixels near junctions
    # pixel that is part of mask and skeleton
    assert test_net.Iskel[920,384] == True
    assert test_net.Imask[920,384] == True
    # pixel that is part of mask but not skeleton
    assert test_net.Iskel[920,383] == False
    assert test_net.Imask[920,383] == True
    # pixel in mask and skeleton
    assert test_net.Iskel[446,962] == True
    assert test_net.Imask[446,962] == True
    # pixel in mask not skeleton
    assert test_net.Iskel[448,960] == False
    assert test_net.Imask[448,960] == True
    # pixel in mask and skeleton
    assert test_net.Iskel[1297,457] == True
    assert test_net.Imask[1297,457] == True
    # pixel in mask not skeleton
    assert test_net.Iskel[1298,457] == False
    assert test_net.Imask[1298,457] == True


def test_compute_network(test_net, known_net):
    """Test compute network."""
    # compute network
    test_net.compute_network()

    # check that nodes and links are created
    assert len(test_net.nodes['id']) >= len(known_net.nodes['id'])
    assert len(test_net.links['id']) >= len(known_net.links['id'])


@pytest.mark.xfail
def test_prune_network(test_net, known_net):
    """Test network pruning."""
    # prune the network
    test_net.prune_network(path_shoreline='tests/data/Colville/Colville_shoreline.shp',
                           path_inletnodes='tests/data/Colville/Colville_inlet_nodes.shp')
    # now the number of nodes and links should be exactly the same
    # Currently x-failing because changes in directionality computation altered the number of nodes/links identified
    assert len(test_net.nodes['id']) == len(known_net.nodes['id'])
    assert len(test_net.links['id']) == len(known_net.links['id'])


def test_flowdir(test_net, known_net):
    """Check that 90% of directions are assigned to match known case."""
    # set directions
    test_net.assign_flow_directions()

    # identify list of indices to check
    ind_list = range(0, len(known_net.nodes['idx']))

    # create list of connected idx values
    test_dirs = []
    known_dirs = []
    for j in ind_list:
        test_ind = test_net.nodes['idx'].index(known_net.nodes['idx'][j])
        # interrogate the 'conn' values to find corresponding 'idx' values
        t_inds = test_net.nodes['conn'][test_ind]
        t_idx = []
        for i in t_inds:
            t_idx.append(test_net.links['id'].index(i))

        k_inds = known_net.nodes['conn'][j]
        k_idx = []
        for i in k_inds:
            k_idx.append(known_net.links['id'].index(i))
        # add to the overall dirs lists
        test_dirs.append(test_net.links['idx'][t_idx[0]])
        known_dirs.append(known_net.links['idx'][k_idx[0]])

    # check how many sets of idx values match between the test and known case
    match_counter = 0
    for i in range(0, len(test_dirs)):
        if test_dirs[i] == known_dirs[i]:
            match_counter += 1

    # "soft" unit test -- check that over 90% of the idx values match
    assert match_counter / len(ind_list) > 0.9


def test_junction_angles(test_net, known_net):
    """Check that 90% of junction angles agree."""
    # compute the junction angles
    test_net.compute_junction_angles(weight=None)
    known_net.compute_junction_angles(weight=None)

    # identify list of indices to check
    ind_list = range(0, len(known_net.nodes['idx']))

    # create lists to store junction angles
    test_angles = []
    known_angles = []
    test_types = []
    known_types = []

    # grab all angles and junction types
    for j in ind_list:
        t_ind = test_net.nodes['idx'].index(known_net.nodes['idx'][j])
        # store angles
        test_angles.append(np.floor(test_net.nodes['int_ang'][t_ind]))
        known_angles.append(np.floor(known_net.nodes['int_ang'][j]))
        # store junction types
        test_types.append(test_net.nodes['jtype'][t_ind])
        known_types.append(known_net.nodes['jtype'][j])

    # count number of matches of angles and junction types
    match_angle = 0
    match_jct = 0
    for i in range(0, len(ind_list)):
        if test_angles[i] == known_angles[i]:
            match_angle += 1
        if test_types[i] == known_types[i]:
            match_jct += 1

    # "soft" unit test -- check that over 90% of the values match
    assert match_angle / len(ind_list) > 0.9
    assert match_jct / len(ind_list) > 0.9



# currently the compute_topologic_metrics() method is
# creating a memory overflow warning (sometimes)

def test_metrics(test_net,known_net):
    # compute metrics
    test_net.compute_topologic_metrics()
    known_net.compute_topologic_metrics()

    assert len(test_net.topo_metrics.keys()) == len(known_net.topo_metrics.keys())


def test_adj(test_net,known_net):
    # define adjacency matrices
    test_adj = test_net.adjacency_matrix()
    known_adj = known_net.adjacency_matrix()

    assert np.shape(test_adj) == np.shape(known_adj)
    assert np.sum(test_adj) == np.sum(known_adj)


def test_adj_norm(test_net):
    # test normalization of the adjacency matrix
    t_norm = test_net.adjacency_matrix(normalized=True)

    assert np.max(np.sum(t_norm, axis=1)) == 1
