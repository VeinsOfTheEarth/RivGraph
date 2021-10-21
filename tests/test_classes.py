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
        m2g.skel_to_graph = mock.MagicMock(return_value=(1, 2))
        # call the method
        _rivnetwork.compute_network()
        # count function calls
        assert _rivnetwork.skeletonize.call_count == 0  # mocked a skeleton
        assert m2g.skel_to_graph.call_count == 1

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
        lnu.link_widths_and_lengths = mock.MagicMock()
        # call the method to test
        _rivnetwork.compute_link_width_and_length()
        # count function calls
        assert lnu.link_widths_and_lengths.call_count == 1

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
        lnu.junction_angles = mock.MagicMock()
        # call method
        _rivnetwork.compute_junction_angles()
        # assert internal calls
        assert lnu.junction_angles.call_count == 1

    def test_adj_matrix(self, tmp_path):
        """Test adjacency matrix function."""
        # define network object
        results_folder = os.path.join(tmp_path, 'results')
        _rivnetwork = rivnetwork(
            self.name, self.path_to_mask, results_folder)
        _rivnetwork.links = []
        _rivnetwork.nodes = []
        # mocking functions
        dm.graphiphy = mock.MagicMock()
        dm.normalize_adj_matrix = mock.MagicMock()
        # call
        _rivnetwork.adjacency_matrix(normalized=True)
        # assertions
        assert dm.graphiphy.call_count == 1
        assert dm.normalize_adj_matrix.call_count == 1
