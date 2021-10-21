"""Unit tests for `rivgraph.classes` classes and methods."""
import pytest
import unittest.mock as mock
import sys
import os
import io
import numpy as np
from rivgraph.classes import rivnetwork
from rivgraph.classes import delta
from rivgraph.classes import river
import rivgraph.mask_to_graph as m2g
import rivgraph.ln_utils as lnu
import rivgraph.deltas.delta_metrics as dm

class Test_rivnetwork:
    """Set up some variable names."""
    name = 'demo'
    path_to_mask = os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif')

    def test_init(self, tmp_path):
        """Test initialization of a rivnetwork class."""
        results_folder = os.path.join(tmp_path, 'results')
        _rivnetwork = rivnetwork(self.name, self.path_to_mask, results_folder)

        # assert attributes of class
        assert _rivnetwork.name == self.name
        assert _rivnetwork.verbose is False
        assert type(_rivnetwork.paths) == dict
        assert type(_rivnetwork.imshape) == tuple
        assert type(_rivnetwork.unit) == str
        assert type(_rivnetwork.pixarea) == float
        assert type(_rivnetwork.pixlen) == float
        assert type(_rivnetwork.Imask) == np.ndarray

    def test_compute_network(self, tmp_path):
        """Test compute network calls skeletonize and skel to graph."""
        # define network object
        results_folder = os.path.join(tmp_path, 'results')
        _rivnetwork = rivnetwork(self.name, self.path_to_mask, results_folder)
        # mock methods to test
        _rivnetwork.Iskel = mock.MagicMock()
        _rivnetwork.skeletonize = mock.MagicMock()
        # patch skel to graph method
        def _patched_skel_to_graph(Iskel):
            return {'id': [0]}, {'id': [1]}
        patcher = mock.patch(
            'rivgraph.mask_to_graph.skel_to_graph',
            new=_patched_skel_to_graph)
        patcher.start()
        _rivnetwork.compute_network()
        # count function calls
        assert _rivnetwork.skeletonize.call_count == 0  # mocked a skeleton
        assert _rivnetwork.links['id'] == [0]
        assert _rivnetwork.nodes['id'] == [1]
        patcher.stop()

    def test_compute_distance_transform(self, tmp_path):
        """Test general distance transform function."""
        # define network object
        results_folder = os.path.join(tmp_path, 'results')
        _rivnetwork = rivnetwork(self.name, self.path_to_mask, results_folder)
        assert hasattr(_rivnetwork, 'Idist') is False
        # mock Imask attribute
        _rivnetwork.Imask = mock.MagicMock()
        # call method
        _rivnetwork.compute_distance_transform()
        # make assertions
        assert hasattr(_rivnetwork, 'Idist') is True

    def test_compute_link_width_and_length(self, tmp_path):
        """Test computation width/length assuming network exists."""
        # define network object
        results_folder = os.path.join(tmp_path, 'results')
        _rivnetwork = rivnetwork(self.name, self.path_to_mask, results_folder)
        # mock methods to test
        _rivnetwork.links = mock.MagicMock()
        _rivnetwork.Idist = mock.MagicMock()
        _rivnetwork.pixlen = mock.MagicMock()
        # patch link width and length function
        def _patched_link_widths_and_lengths(links, Idist, pixlen):
            return {'id': [0]}
        patcher = mock.patch(
            'rivgraph.ln_utils.link_widths_and_lengths',
            new=_patched_link_widths_and_lengths)
        patcher.start()
        # call the method to test
        _rivnetwork.compute_link_width_and_length()
        # assertions
        assert _rivnetwork.links == {'id': [0]}
        patcher.stop()

    def test_compute_junction_angles(self, tmp_path):
        """Test junction angle function."""
        # define network object
        results_folder = os.path.join(tmp_path, 'results')
        _rivnetwork = rivnetwork(self.name, self.path_to_mask, results_folder)
        # set up links/keys
        _rivnetwork.links = {'certain': 0}
        # mock function
        _rivnetwork.nodes = mock.MagicMock()
        _rivnetwork.imshape = mock.MagicMock()
        _rivnetwork.pixlen = mock.MagicMock()
        # patch junction angles function
        def _patched_junction_angles(links, nodes, imshape, pixlen, weight):
            return {'id': [1]}
        patcher = mock.patch(
            'rivgraph.ln_utils.junction_angles',
            new=_patched_junction_angles)
        patcher.start()
        # call method
        _rivnetwork.compute_junction_angles()
        # assert internal calls
        assert _rivnetwork.nodes == {'id': [1]}
        patcher.stop()

    def test_adj_matrix(self, tmp_path):
        """Test adjacency matrix function."""
        # define network object
        results_folder = os.path.join(tmp_path, 'results')
        _rivnetwork = rivnetwork(
            self.name, self.path_to_mask, results_folder)
        _rivnetwork.links = []
        _rivnetwork.nodes = []
        # patch functions
        def _patched_graphiphy(links, nodes, weight):
            return 1
        patcher1 = mock.patch(
            'rivgraph.deltas.delta_metrics.graphiphy',
            new=_patched_graphiphy)
        patcher1.start()
        def _patched_normalize_adj_matrix(A):
            return A + 1
        patcher2 = mock.patch(
            'rivgraph.deltas.delta_metrics.normalize_adj_matrix',
            new=_patched_normalize_adj_matrix)
        patcher2.start()
        # call
        A = _rivnetwork.adjacency_matrix(normalized=True)
        # assertions
        assert A == 2
        patcher1.stop()
        patcher2.stop()
