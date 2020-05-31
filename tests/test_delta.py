import pytest
import sys, os
import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph.classes import delta

def test_skeletonize(test_net):
    # do skeletonization
    test_net.skeletonize()
    # ensure the size of the input mask is equal to size of skeletonized output
    assert np.shape(test_net.Imask) == np.shape(test_net.Iskel)
    # ensure max value in skeleton are 1
    assert np.max(test_net.Iskel) == 1
    # ensure min value in skeleton is 0
    assert np.min(test_net.Iskel) == 0


def test_compute_network(test_net,known_net):
    # compute network
    test_net.compute_network()

    # check that nodes and links are created
    assert len(test_net.nodes['id']) >= len(known_net.nodes['id'])
    assert len(test_net.links['id']) >= len(known_net.links['id'])


def test_prune_network(test_net,known_net):
    # prune the network
    test_net.prune_network(path_shoreline='tests/data/Colville/Colville_shoreline.shp',
                          path_inletnodes='tests/data/Colville/Colville_inlet_nodes.shp')

    # now the number of nodes and links should be exactly the same
    assert len(test_net.nodes['id']) == len(known_net.nodes['id'])
    assert len(test_net.links['id']) == len(known_net.links['id'])


def test_flowdir(test_net,known_net):
    # set directions
    test_net.assign_flow_directions()
    # identify corresponding idx value in test network to known 'idx' [0]
    test_ind = test_net.nodes['idx'].index(known_net.nodes['idx'][0])
    # interrogate the 'conn' values to find corresponding 'idx' values
    t_inds = test_net.nodes['conn'][test_ind]
    t_idx = []
    for i in t_inds:
        t_idx.append(test_net.links['id'].index(i))

    k_inds = known_net.nodes['conn'][0]
    k_idx = []
    for i in k_inds:
        k_idx.append(known_net.links['id'].index(i))

    # expect same node in test network and known network to have same conns
    # have to use the idx values from one of the conn links to verify this
    assert test_net.links['idx'][t_idx[0]] == known_net.links['idx'][k_idx[0]]


def test_junction_angles(test_net,known_net):
    # compute the junction angles
    test_net.compute_junction_angles(weight=None)
    known_net.compute_junction_angles(weight=None)
    # comparison of calculated junction angle (rounded to integer)
    t_ind = test_net.nodes['idx'].index(known_net.nodes['idx'][0])
    assert np.floor(test_net.nodes['int_ang'][t_ind]) == np.floor(known_net.nodes['int_ang'][0])
    # comparison of junction type
    assert test_net.nodes['jtype'][t_ind] == known_net.nodes['jtype'][0]


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
