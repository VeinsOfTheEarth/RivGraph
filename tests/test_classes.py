"""Focused tests for ``rivgraph.classes`` base behaviors."""
from __future__ import annotations

import io
import os
import sys
import unittest.mock as mock

import numpy as np

from tests._helpers import REGRESSION_DATA_ROOT, require_rivgraph_classes


class TestRivnetwork:
    name = "demo"
    path_to_mask = REGRESSION_DATA_ROOT / "delta_mossy" / "inputs" / "mask.tif"

    def test_init(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder)

        assert network.name == self.name
        assert network.verbose is False
        assert isinstance(network.paths, dict)
        assert isinstance(network.imshape, tuple)
        assert isinstance(network.unit, str)
        assert isinstance(network.pixarea, float)
        assert isinstance(network.pixlen, float)
        assert isinstance(network.Imask, np.ndarray)
        assert hasattr(network, 'gdobj') is False

    def test_compute_network(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder)
        network.Iskel = mock.MagicMock()
        network.skeletonize = mock.MagicMock()

        def _patched_skel_to_graph(_iskel):
            return {"id": [0]}, {"id": [1]}

        with mock.patch("rivgraph.mask_to_graph.skel_to_graph", new=_patched_skel_to_graph):
            network.compute_network()

        assert network.skeletonize.call_count == 0
        assert network.links["id"] == [0]
        assert network.nodes["id"] == [1]

    def test_compute_distance_transform(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder)
        assert hasattr(network, "Idist") is False

        network.Imask = mock.MagicMock()
        network.compute_distance_transform()
        assert hasattr(network, "Idist") is True

    def test_compute_link_width_and_length(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder)
        network.links = mock.MagicMock()
        network.Idist = mock.MagicMock()
        network.pixlen = mock.MagicMock()

        def _patched_link_widths_and_lengths(links, Idist, pixlen):
            return {"id": [0]}

        with mock.patch(
            "rivgraph.ln_utils.link_widths_and_lengths",
            new=_patched_link_widths_and_lengths,
        ):
            network.compute_link_width_and_length()

        assert network.links == {"id": [0]}

    def test_compute_junction_angles(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder)
        network.links = {"certain": 0}
        network.nodes = mock.MagicMock()
        network.imshape = mock.MagicMock()
        network.pixlen = mock.MagicMock()

        def _patched_junction_angles(links, nodes, imshape, pixlen, weight):
            return {"id": [1]}

        with mock.patch(
            "rivgraph.ln_utils.junction_angles",
            new=_patched_junction_angles,
        ):
            network.compute_junction_angles()

        assert network.nodes == {"id": [1]}

    def test_adj_matrix(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder)
        network.links = []
        network.nodes = []

        def _patched_graphiphy(links, nodes, weight):
            return 1

        def _patched_normalize_adj_matrix(A):
            return A + 1

        with mock.patch("rivgraph.deltas.delta_metrics.graphiphy", new=_patched_graphiphy), mock.patch(
            "rivgraph.deltas.delta_metrics.normalize_adj_matrix",
            new=_patched_normalize_adj_matrix,
        ):
            A = network.adjacency_matrix(normalized=True)

        assert A == 2

    def test_logger_off(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        captured_output = io.StringIO()
        sys.stdout = captured_output
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder)
        assert os.path.isfile(network.paths["log"]) is True
        sys.stdout = sys.__stdout__
        assert captured_output.getvalue() == ""

    def test_logger_on(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        captured_output = io.StringIO()
        sys.stdout = captured_output
        results_folder = os.path.join(tmp_path, "results")
        network = rivnetwork(self.name, str(self.path_to_mask), results_folder, verbose=True)
        assert os.path.isfile(network.paths["log"]) is True
        assert captured_output.getvalue() == "---------- New Run ----------\n"
