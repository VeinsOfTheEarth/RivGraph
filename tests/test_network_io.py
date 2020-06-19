"""Unit tests to check Input/Output functionality."""
import pytest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import io_utils as iu

# Testing of functions that do I/O within the rivgraph.classes.delta class


def test_save(known_net):
    """Test saving functionality."""
    known_net.save_network()
    # assert that the file now exists
    assert os.path.isfile('tests/results/known/known_network.pkl') == True


def test_load(known_net):
    """Test loading functionality."""
    known_net.load_network()
    # assert that path used is correct
    assert known_net.paths['network_pickle'] == 'tests/results/known/known_network.pkl'


def test_outvec_json(known_net):
    """Test default functionality should write network to json."""
    known_net.to_geovectors()
    # check that files exist
    assert os.path.isfile(known_net.paths['links']) == True
    assert os.path.isfile(known_net.paths['nodes']) == True


def test_outvec_shp(known_net):
    """Test default functionality should write network to shp."""
    known_net.to_geovectors(export='network', ftype='shp')
    # check that files exist
    assert os.path.isfile(known_net.paths['links']) == True
    assert os.path.isfile(known_net.paths['nodes']) == True


def test_to_geotiff(known_net):
    """Have to re-create skeleton."""
    known_net.skeletonize()
    # have to generate distances
    known_net.compute_distance_transform()
    # test writing of geotiff to disk
    known_net.to_geotiff('directions')
    known_net.to_geotiff('distance')
    known_net.to_geotiff('skeleton')
    # check that expected files exist
    assert os.path.isfile(known_net.paths['linkdirs']) == True
    assert os.path.isfile(known_net.paths['Idist']) == True
    assert os.path.isfile(known_net.paths['Iskel']) == True


def test_plotnetwork(known_net):
    """Make plots with various kwargs specified."""
    # default
    f1 = known_net.plot()
    plt.savefig('tests/results/known/testall.png')
    plt.close()
    # network
    f2 = known_net.plot('network')
    plt.savefig('tests/results/known/testnetwork.png')
    plt.close()
    # directions
    f3 = known_net.plot('directions')
    plt.savefig('tests/results/known/testdirections.png')
    plt.close()
    # assert that figures were made
    assert os.path.isfile('tests/results/known/testall.png') == True
    assert os.path.isfile('tests/results/known/testnetwork.png') == True
    assert os.path.isfile('tests/results/known/testdirections.png') == True


# Explicit testing of functions from io_utils.py
class TestColortable:
    """Testing options related to colortable() function."""

    def test_binary(self):
        """Test binary colormap."""
        color_table = iu.colortable('binary')
        assert np.all(color_table.GetColorEntry(0)==(0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1)==(255, 255, 255, 100))

    def test_mask(self):
        """Test mask."""
        color_table = iu.colortable('mask')
        assert np.all(color_table.GetColorEntry(0)==(0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1)==(0, 128, 0, 100))

    def test_tile(self):
        """Test tile."""
        color_table = iu.colortable('tile')
        assert np.all(color_table.GetColorEntry(0)==(0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1)==(0, 0, 255, 100))

    def test_JRCmo(self):
        """Test JRCmo."""
        color_table = iu.colortable('JRCmo')
        assert np.all(color_table.GetColorEntry(0)==(0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(1)==(0, 0, 0, 0))
        assert np.all(color_table.GetColorEntry(2)==(176, 224, 230, 100))

# Function iu.coords_from_shapefile() no longer exists
# def test_coords_from_shp():
#     coordspath = 'tests/data/Colville/Colville_inlet_nodes.shp'
#     coords = iu.coords_from_shapefile(coordspath)
#     assert np.all(coords==[(352136.9198745673,
#                             7780855.6952854665),
#                            (345692.87413494784,
#                             7781342.648680793)])


def test_create_manual_dir_csv():
    """Test creation of manual direction csv."""
    csvpath = 'tests/results/known/csvtest.csv'
    iu.create_manual_dir_csv(csvpath)
    assert os.path.isfile('tests/results/known/csvtest.csv') == True


@pytest.mark.xfail
def tests_coords_to_shp():
    """Test function coords_to_shp()."""
    coords = [(352136.9198745673, 7780855.6952854665),
              (345692.87413494784, 7781342.648680793)]
    epsg = 32606
    outpath = 'test/results/known/test_coords_to_shp.shp'
    iu.coords_to_shapefile(coords, epsg, outpath)
    # function was breaking and returning NoneType...
    # fix to remove xfail

# Delete data created by tests in this file ...


def test_delete_files():
    """Delete created files at the end."""
    for i in os.listdir('tests/results/known/'):
        os.remove('tests/results/known/'+i)
    # check directory is empty
    assert os.listdir('tests/results/known/') == []
