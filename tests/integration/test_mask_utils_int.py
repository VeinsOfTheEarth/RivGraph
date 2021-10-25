"""Tests for the mask_utils.py functions."""
import numpy as np
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

    def test_maxwidth(self, test_net):
        """Test given maxwidth."""
        props = ['maxwidth']
        gdf, Ilabel = mask_utils.get_island_properties(test_net.Imask,
                                                       test_net.pixlen,
                                                       test_net.pixarea,
                                                       test_net.crs,
                                                       test_net.gt,
                                                       props)
        # assert image mask size same as labeled image size
        assert Ilabel.shape == test_net.Imask.shape
        # make assertion about gdf shape
        assert np.shape(gdf) == (99, 3)


    def test_islandprops(self, test_net):
        """Test given island properties."""
        props = ['area', 'major_axis_length', 'minor_axis_length',
                 'perim_len', 'convex_area']
        gdf, Ilabel = mask_utils.get_island_properties(test_net.Imask,
                                                       test_net.pixlen,
                                                       test_net.pixarea,
                                                       test_net.crs,
                                                       test_net.gt,
                                                       props)
        # assert image mask size same as labeled image size
        assert Ilabel.shape == test_net.Imask.shape
        # make assertion about gdf shape
        assert np.shape(gdf) == (99, 7)
