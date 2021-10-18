# -*- coding: utf-8 -*-
"""Unit tests for im_utils.py."""
import pytest
import numpy as np
from rivgraph import im_utils


class TestSkelBranchpts:
    """Tests for skel_branchpoints method."""
    def test_colville_branchpoints(self, known_net):
        """Test branchpoints on colville skeleton."""
        known_net.skeletonize()
        Iskel = known_net.Iskel
        Ipbs = im_utils.skel_branchpoints(Iskel)
        # make assertions
        assert np.max(Ipbs) == 1
        assert np.min(Ipbs) == 0
        assert np.sum(Ipbs) == 570

    def test_brahma_branchpoints(self, known_river):
        """Tests branchpoints on brahma skeleton."""
        known_river.skeletonize()
        Iskel = known_river.Iskel
        Ipbs = im_utils.skel_branchpoints(Iskel)
        # make assertions
        assert np.max(Ipbs) == 1
        assert np.min(Ipbs) == 0
        assert np.sum(Ipbs) == 1371


class TestSkelCurvature:
    """Tests for skel_pixel_curvature method."""
    def test_colville_curvature(self, known_net):
        """Test curvature on colville skeleton."""
        known_net.skeletonize()
        Iskel = known_net.Iskel
        Icur = im_utils.skel_pixel_curvature(Iskel)
        # make assertions
        assert np.nanmax(Icur) == 180.0
        assert np.nanmin(Icur) == 0.0
        assert pytest.approx(np.nansum(Icur)) == 899437.7559864743

    def test_brahma_curvature(self, known_river):
        """Tests curvature on brahma skeleton."""
        known_river.skeletonize()
        Iskel = known_river.Iskel
        Icur = im_utils.skel_pixel_curvature(Iskel)
        # make assertions
        assert np.nanmin(Icur) == 0.0
        assert np.nanmax(Icur) == 180.0
        assert pytest.approx(np.nansum(Icur)) == 1238328.4876505535
