"""Tests for the mask_utils.py functions."""
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import mask_utils


class TestIslandProps:
    """Tests associated with get_island_properties()."""

    def test_no_props(self, test_net):
        """Test, no island properties."""
        props = []
        gdf, Ilabel = mask_utils.get_island_properties(test_net.Imask,
                                                       test_net.pixlen,
                                                       test_net.pixarea,
                                                       test_net.crs,
                                                       test_net.gt,
                                                       props)
        # assert image mask size same as labeled image size
        assert Ilabel.shape == test_net.Imask.shape
        # make assertion about gdf shape
        assert np.shape(gdf) == (99, 2)

    @pytest.mark.xfail
    def test_maxwidth(self, test_net):
        """Test given maxwidth."""
        # not implemented yet
        pass

    @pytest.mark.xfail
    def test_perimeter(self, test_net):
        """Test given perimeter."""
        # not implemented yet
        pass

    @pytest.mark.xfail
    def test_islandprops(self, test_net):
        """Test given island properties."""
        # not implemented yet
        pass
