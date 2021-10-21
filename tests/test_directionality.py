"""Unit tests for directionality.py."""
import pytest
import sys
import os
import io
import numpy as np
import networkx as nx
from rivgraph import directionality as di
from rivgraph import ln_utils as lnu
from rivgraph.deltas import delta_directionality as dd


def test_directionlity_trackers():
    """Test of directionality tracker function."""
    links = {'id': [0, 1]}
    nodes = {'id': [0, 1]}
    ntype = 'delta'
    di.add_directionality_trackers(links, nodes, ntype)
    # assertions
    assert 'certain' in links.keys()
    assert 'certain_order' in links.keys()
    assert 'certain_alg' in links.keys()
    assert 'guess' in links.keys()
    assert 'guess_alg' in links.keys()
    assert 'maxang' not in links.keys()
    # pretend its a river
    di.add_directionality_trackers(links, nodes, 'river')
    assert 'maxang' in links.keys()

class Test_algmap:
    """Tests for algmap function."""

    def test_sourcesinkfix(self):
        assert di.algmap('sourcesinkfix') == -2

    def test_manual_set(self):
        assert di.algmap('manual_set') == -1

    def test_inletoutlet(self):
        assert di.algmap('inletoutlet') == 0

    def test_continuity(self):
        assert di.algmap('continuity') == 1

    def test_parallels(self):
        assert di.algmap('parallels') == 2

    def test_artificials(self):
        assert di.algmap('artificials') == 2.1

    def test_main_chans(self):
        assert di.algmap('main_chans') == 4

    def test_bridges(self):
        assert di.algmap('bridges') == 5

    def test_known_fdr(self):
        assert di.algmap('known_fdr') == 6

    def test_known_fdr_rs(self):
        assert di.algmap('known_fdr_rs') == 6.1

    def test_syn_dem(self):
        assert di.algmap('syn_dem') == 10

    def test_syn_dem_med(self):
        assert di.algmap('syn_dem_med') == 10.1

    def test_sym_dem_leftover(self):
        assert di.algmap('sym_dem_leftover') == 10.2

    def test_sp_links(self):
        assert di.algmap('sp_links') == 11

    def test_sp_nodes(self):
        assert di.algmap('sp_nodes') == 12

    def test_longest_steepest(self):
        assert di.algmap('longest_steepest') == 13

    def test_three_agree(self):
        assert di.algmap('three_agree') == 15

    def test_syn_dem_and_sp(self):
        assert di.algmap('syn_dem_and_sp') == 16

    def test_cl_dist_guess(self):
        assert di.algmap('cl_dist_guess') == 20

    def test_cl_ang_guess(self):
        assert di.algmap('cl_ang_guess') == 21

    def test_cl_dist_set(self):
        assert di.algmap('cl_dist_set') == 22

    def test_cl_ang_set(self):
        assert di.algmap('cl_ang_set') == 23

    def test_cl_ang_rs(self):
        assert di.algmap('cl_ang_rs') == 23.1

    def test_cl_dist_and_ang(self):
        assert di.algmap('cl_dist_and_ang') == 24

    def test_short_no_bktrck(self):
        assert di.algmap('short_no_bktrck') == 25

    def test_wid_pctdiff(self):
        assert di.algmap('wid_pctdiff') == 26


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
