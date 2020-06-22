"""Tests for geo_utils.py."""
import pytest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from rivgraph import geo_utils
import osr
import ogr
import gdal
from pyproj import CRS

# function geo_utils.get_EPSG() no longer exists
# def test_getEPSG_fromshp():
#         shp_file = 'tests/data/Colville/Colville_inlet_nodes.shp'
#         epsg = geo_utils.get_EPSG(shp_file, returndict=True)
#         print(epsg)
#         # make assertion
#         assert type(epsg) == dict
#         assert len(epsg.keys()) == 1
#         assert epsg['init'] == 'epsg:32606'


def test_get_unit():
    """Test get_unit() function."""
    # Projected CRS
    epsg = 32606
    crs = CRS.from_epsg(epsg)
    unit = geo_utils.get_unit(crs)
    assert unit == 'meter'
    
    # Unprojected CRS
    epsg = 4326
    crs = CRS.from_epsg(epsg)
    unit = geo_utils.get_unit(crs)
    assert unit == 'degree'


def test_geotiff_vals_from_coords():
    """Test geotiff_vals_from_coords()."""
    # read in origin point
    coords = np.array([[336885.0000000000000000, 7826415.0000000000000000]])
    gobj = gdal.Open('tests/data/Colville/Colville_islands_filled.tif')
    vals = geo_utils.geotiff_vals_from_coords(coords,gobj)
    # assert that value at origin is 0, this is what it is in the mask
    assert vals[0] == 0


def test_coords_to_xy():
    """Test coords_to_xy()."""
    # use corners or extents of the geotiff to test
    xs = [336885.0000000000000000, 383085.0000000000000000]
    ys = [7780215.0000000000000000, 7826415.0000000000000000]
    gobj = gdal.Open('tests/data/Colville/Colville_islands_filled.tif')
    gt = gobj.GetGeoTransform()
    vals = geo_utils.coords_to_xy(xs, ys, gt)
    # make assertions
    assert np.all(vals[0] == [0, 1540])
    assert np.all(vals[1] == [1540, 0])


def test_transform_coords():
    """Test transform_coords()."""
    xs = [336885.0000000000000000, 383085.0000000000000000]
    ys = [7780215.0000000000000000, 7826415.0000000000000000]
    input_EPSG = 32606
    output_EPSG = 4326
    xy = geo_utils.transform_coords(xs, ys, input_EPSG, output_EPSG)
    # make assertions
    assert np.all(xy[0] == pytest.approx([-151.2921657688682,
                                          -150.1418733861601]))
    assert np.all(xy[1] == pytest.approx([70.07706865567432,
                                          70.51576941571783]))


# function geo_utils.transform_coordinates() no longer exists
# def test_transform_coodinates():
#     xs = [336885.0000000000000000, 383085.0000000000000000]
#     ys = [7780215.0000000000000000, 7826415.0000000000000000]
#     input_EPSG = 32606
#     output_EPSG = 4326
#     xy = geo_utils.transform_coordinates(xs, ys, input_EPSG, output_EPSG)
#     # make assertions
#     assert np.all(xy[0]==pytest.approx([-151.2921657688682,
#                                         -150.1418733861601]))
#     assert np.all(xy[1]==pytest.approx([70.07706865567432,
#                                         70.51576941571783]))

# function geo_utils.transform_coordinates() no longer exists
# def test_notransform_coodinates():
#     xs = [336885.0000000000000000, 383085.0000000000000000]
#     ys = [7780215.0000000000000000, 7826415.0000000000000000]
#     input_EPSG = 32606
#     output_EPSG = 32606
#     xy = geo_utils.transform_coordinates(xs, ys, input_EPSG, output_EPSG)
#     # make assertions
#     assert np.all(xy[0]==pytest.approx(xs))
#     assert np.all(xy[1]==pytest.approx(ys))

def test_crop_geotiff():
    """Test crop_geotif()."""
    g_path = 'tests/data/Colville/Colville_islands_filled.tif'
    outpath = 'tests/results/known/cropped.tif'
    # run cropping function
    o_path = geo_utils.crop_geotif(g_path, outpath=outpath)
    # assert file name and existance
    assert o_path == outpath
    assert os.path.isfile(o_path) == True


def test_croppad_geotiff():
    """Test crop_geotif() with padding."""
    g_path = 'tests/data/Colville/Colville_islands_filled.tif'
    outpath = 'tests/results/known/croppedpad.tif'
    # run cropping function
    o_path = geo_utils.crop_geotif(g_path, npad=6, outpath=outpath)
    # assert file name and existance
    assert o_path == outpath
    assert os.path.isfile(o_path) == True


def test_crop_geotiff_output():
    """Test output of crop_geotif()."""
    crop_file = gdal.Open('tests/results/known/cropped.tif')
    o_file = crop_file.ReadAsArray()
    # check that first/last rows and columns not all 0s
    assert np.sum(o_file[0, :]) > 0
    assert np.sum(o_file[-1, :]) > 0
    assert np.sum(o_file[:, 0]) > 0
    assert np.sum(o_file[:, -1]) > 0


def test_croppad_geotiff_output():
    """Test padded crop_geotif() output."""
    crop_file = gdal.Open('tests/results/known/croppedpad.tif')
    o_file = crop_file.ReadAsArray()
    # check that first/last rows and columns are all 0s
    assert np.sum(o_file[0, :]) == 0
    assert np.sum(o_file[-1, :]) == 0
    assert np.sum(o_file[:, 0]) == 0
    assert np.sum(o_file[:, -1]) == 0
    # check that first/last rows and columns after pads not all 0s
    assert np.sum(o_file[6, :]) > 0
    assert np.sum(o_file[-7, :]) > 0
    assert np.sum(o_file[:, 6]) > 0
    assert np.sum(o_file[:, -7]) > 0


def test_delete_files():
    """Delete files created by tests."""
    # delete created files at the end
    for i in os.listdir('tests/results/known/'):
        os.remove('tests/results/known/'+i)
    # check directory is empty
    assert os.listdir('tests/results/known/') == []
