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
