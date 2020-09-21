"""Unit tests to check Input/Output functionality."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from inspect import getsourcefile
basepath = os.path.dirname(os.path.dirname(os.path.abspath(getsourcefile(lambda:0))))
sys.path.insert(0, basepath)

from rivgraph import io_utils as iu

# Testing of functions that do I/O within the rivgraph.classes.delta class


def test_save(known_net):
    """Test saving functionality."""
    known_net.save_network()
    # assert that the file now exists
    assert os.path.isfile(os.path.join(basepath, os.path.normpath('tests/results/known/known_network.pkl'))) == True


def test_load(known_net):
    """Test loading functionality."""
    known_net.load_network()
    
    # assert that network was loaded
    assert hasattr(known_net, 'links')
    assert hasattr(known_net, 'nodes')


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


def test_outvec_all(known_river):
    """Test writing all variables out as json."""
    known_river.compute_mesh()
    known_river.to_geovectors(export='all')
    # check that files exist
    assert os.path.isfile(known_river.paths['links']) == True
    assert os.path.isfile(known_river.paths['nodes']) == True
    assert os.path.isfile(known_river.paths['meshlines']) == True
    assert os.path.isfile(known_river.paths['centerline']) == True
    assert os.path.isfile(known_river.paths['centerline_smooth']) == True


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

    def test_GSW(self):
        """Test GSW."""
        color_table = iu.colortable('GSW')
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


def test_coords_to_from_geovector(known_net):
    """Test coords_from_geovector and coords_to_geovector."""
    # coords from geovector
    known_net.to_geovectors()
    geopath = known_net.paths['links']
    coords = iu.coords_from_geovector(geopath)
    # assert some things about the coords
    assert np.shape(coords) == (246, 2)
    assert coords[0] == (353827.29959433706, 7821290.700172625)
    assert coords[50] == (361360.8647424011, 7812192.341367241)
    assert coords[-1] == (375090.0, 7815300.0)

    # coords to geovector
    epsg = 32606
    outpath = os.path.normpath('tests/results/known/geo_test.shp')
    iu.coords_to_geovector(coords, epsg, outpath)
    # assert file is created
    assert os.path.isfile(outpath) == True


def test_prep_paths():
    """Test prepare_paths()."""
    resultsfolder = os.path.normpath('tests/results/new')
    name = 'new'
    basetiff = os.path.normpath('tests/data/Colville/Colville_islands_filled.tif')
    paths = iu.prepare_paths(resultsfolder, name, basetiff)
    # assertions
    assert type(paths) == dict
    assert paths['basepath'] == resultsfolder
    assert paths['maskpath'] == basetiff
    assert paths['Iskel'] == os.path.normpath('tests/results/new/new_skel.tif')
    assert paths['Idist'] == os.path.normpath('tests/results/new/new_dist.tif')
    assert paths['network_pickle'] == os.path.normpath('tests/results/new/new_network.pkl')
    assert paths['fixlinks_csv'] == os.path.normpath('tests/results/new/new_fixlinks.csv')
    assert paths['linkdirs'] == os.path.normpath('tests/results/new/new_link_directions.tif')
    assert paths['metrics'] == os.path.normpath('tests/results/new/new_metrics.pkl')
    assert paths['shoreline'] == os.path.normpath('tests/results/new/new_shoreline.shp')
    assert paths['inlet_nodes'] == os.path.normpath('tests/results/new/new_inlet_nodes.shp')


# Delete data created by tests in this file ...

def test_delete_files():
    """Delete created files at the end."""
    for i in os.listdir('tests/results/known/'):
        os.remove('tests/results/known/'+i)
    for i in os.listdir('tests/results/brahma/'):
        os.remove('tests/results/brahma/'+i)
    os.rmdir('tests/results/new')
    # check directory is empty
    assert os.listdir('tests/results/known/') == []
    assert os.listdir('tests/results/brahma/') == []
    assert os.path.isdir('tests/results/new') is False
