"""Unit tests to check Input/Output functionality."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from rivgraph import io_utils as iu

# Testing of functions that do I/O within the rivgraph.classes.delta class


def test_save(known_net, tmp_path):
    """Test saving functionality."""
    # assert file does not exist
    assert os.path.isfile(known_net.paths['network_pickle']) == False
    # save it
    known_net.save_network()
    # assert that the file now exists
    assert os.path.isfile(known_net.paths['network_pickle']) == True


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


def test_plotnetwork(known_net, tmp_path):
    """Make plots with various kwargs specified."""
    # default
    f1 = known_net.plot()
    plt.savefig(os.path.join(tmp_path, 'testall.png'))
    plt.close()
    # network
    f2 = known_net.plot('network')
    plt.savefig(os.path.join(tmp_path, 'testnetwork.png'))
    plt.close()
    # directions
    f3 = known_net.plot('directions')
    plt.savefig(os.path.join(tmp_path, 'testdirections.png'))
    plt.close()
    # assert that figures were made
    assert os.path.isfile(os.path.join(tmp_path, 'testall.png')) == True
    assert os.path.isfile(os.path.join(tmp_path, 'testnetwork.png')) == True
    assert os.path.isfile(os.path.join(tmp_path, 'testdirections.png')) == True


def test_coords_to_from_geovector(known_net, tmp_path):
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
    outpath = os.path.join(tmp_path, 'geo_test.shp')
    iu.coords_to_geovector(coords, epsg, outpath)
    # assert file is created
    assert os.path.isfile(outpath) == True
