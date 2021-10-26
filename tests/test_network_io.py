"""Unit tests to check Input/Output functionality."""
import os
import numpy as np
from rivgraph import io_utils as iu

# Explicit testing of functions from io_utils.py
class TestColortable:
    """Testing options related to colortable() function."""

    def test_binary(self):
        """Test binary colormap."""
        color_table = iu.colortable('binary')
        assert np.all(color_table.GetColorEntry(0) == (0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1) == (255, 255, 255, 100))

    def test_mask(self):
        """Test mask."""
        color_table = iu.colortable('mask')
        assert np.all(color_table.GetColorEntry(0) == (0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1) == (0, 128, 0, 100))

    def test_tile(self):
        """Test tile."""
        color_table = iu.colortable('tile')
        assert np.all(color_table.GetColorEntry(0) == (0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1) == (0, 0, 255, 100))

    def test_GSW(self):
        """Test GSW."""
        color_table = iu.colortable('GSW')
        assert np.all(color_table.GetColorEntry(0) == (0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1) == (0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(2) == (176, 224, 230, 100))

# Function iu.coords_from_shapefile() no longer exists
# def test_coords_from_shp():
#     coordspath = 'tests/data/Colville/Colville_inlet_nodes.shp'
#     coords = iu.coords_from_shapefile(coordspath)
#     assert np.all(coords==[(352136.9198745673,
#                             7780855.6952854665),
#                            (345692.87413494784,
#                             7781342.648680793)])


def test_create_manual_dir_csv(tmp_path):
    """Test creation of manual direction csv."""
    csvpath = os.path.join(tmp_path, 'csvtest.csv')
    iu.create_manual_dir_csv(csvpath)
    assert os.path.isfile(csvpath) is True


def test_prep_paths(tmp_path):
    """Test prepare_paths()."""
    resultsfolder = tmp_path
    name = 'new'
    basetiff = os.path.normpath(
        'tests/data/Colville/Colville_islands_filled.tif')
    paths = iu.prepare_paths(resultsfolder, name, basetiff)
    # assertions
    assert type(paths) == dict
    assert paths['basepath'] == os.path.normpath(resultsfolder)
    assert paths['maskpath'] == basetiff
    assert paths['Iskel'] == os.path.join(tmp_path, 'new_skel.tif')
    assert paths['Idist'] == os.path.join(tmp_path, 'new_dist.tif')
    assert paths['network_pickle'] == os.path.join(tmp_path, 'new_network.pkl')
    assert paths['fixlinks_csv'] == os.path.join(tmp_path, 'new_fixlinks.csv')
    assert paths['linkdirs'] == os.path.join(
        tmp_path, 'new_link_directions.tif')
    assert paths['metrics'] == os.path.join(tmp_path, 'new_metrics.pkl')
    assert paths['shoreline'] == os.path.join(tmp_path, 'new_shoreline.shp')
    assert paths['inlet_nodes'] == os.path.join(
        tmp_path, 'new_inlet_nodes.shp')
