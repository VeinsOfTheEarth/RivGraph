"""Tests for geo_utils.py."""
import pytest
import sys
import os
import numpy as np
# import matplotlib.pyplot as plt
from rivgraph import geo_utils
# import osr
# import ogr
try:
    from osgeo import gdal
except ImportError:
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
    gobj = gdal.Open(
        os.path.normpath(
            'tests/integration/data/Colville/Colville_islands_filled.tif'))
    vals = geo_utils.geotiff_vals_from_coords(coords, gobj)
    # assert that value at origin is 0, this is what it is in the mask
    assert vals[0] == 0


def test_coords_to_xy():
    """Test coords_to_xy()."""
    # use corners or extents of the geotiff to test
    xs = [336885.0000000000000000, 383085.0000000000000000]
    ys = [7780215.0000000000000000, 7826415.0000000000000000]
    gobj = gdal.Open(os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif'))
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

def test_crop_geotiff(tmp_path):
    """Test crop_geotif()."""
    g_path = os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif')
    outpath = os.path.join(tmp_path, 'cropped.tif')
    # run cropping function
    o_path = geo_utils.crop_geotif(g_path, outpath=outpath)
    # assert file name and existance
    assert o_path == outpath
    assert os.path.isfile(o_path) == True

    # check output of cropped file
    crop_file = gdal.Open(o_path)
    o_file = crop_file.ReadAsArray()
    # check that first/last rows and columns not all 0s
    assert np.sum(o_file[0, :]) > 0
    assert np.sum(o_file[-1, :]) > 0
    assert np.sum(o_file[:, 0]) > 0
    assert np.sum(o_file[:, -1]) > 0


# def test_crop_geotiff_out():
#     """Test crop_geotif() with no outpath."""
#     g_path = os.path.normpath(
#         'tests/integration/data/Colville/Colville_islands_filled.tif')
#     # run cropping function
#     o_path = geo_utils.crop_geotif(g_path)
#     # assert file name and existance
#     assert o_path == g_path.split('.')[-2] + '_cropped.tif'
#     assert os.path.isfile(o_path) == True


def test_croppad_geotiff(tmp_path):
    """Test crop_geotif() with padding."""
    g_path = os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif')
    outpath = os.path.join(tmp_path, 'cropped.tif')
    # run cropping function
    o_path = geo_utils.crop_geotif(g_path, npad=6, outpath=outpath)
    # assert file name and existance
    assert o_path == outpath
    assert os.path.isfile(o_path) == True

    # check padded crop_geotif() output
    crop_file = gdal.Open(o_path)
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


def test_downsample_bad_factor(tmp_path):
    """Test downsampling geotiff with invalid ds_factor."""
    g_path = os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif')
    outpath = os.path.join(tmp_path, 'downsampled.tif')
    # run downsampling function and throw error
    with pytest.raises(ValueError):
        geo_utils.downsample_binary_geotiff(g_path, 2.0, outpath)


def test_downsample_default_thresh(tmp_path):
    """Test downsampling geotiff with default thresh."""
    g_path = os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif')
    outpath = os.path.join(tmp_path, 'downsampled.tif')
    # run downsampling function
    ofile = geo_utils.downsample_binary_geotiff(g_path, 0.5, outpath)
    # assert output
    assert ofile == outpath


def test_downsample_input_thresh(tmp_path):
    """Test downsampling geotiff with defined thresh."""
    g_path = os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif')
    outpath = os.path.join(tmp_path, 'downsampled.tif')
    # run downsampling function
    outfile = geo_utils.downsample_binary_geotiff(g_path, 0.5,
                                                  outpath, thresh=0.1)
    # assert output
    assert outfile == outpath

    # check downsampled output
    # original
    og_file = gdal.Open(g_path)
    og_img = og_file.ReadAsArray()
    # downsampled
    ds_file = gdal.Open(outpath)
    o_file = ds_file.ReadAsArray()
    # check that downsampled size is smaller
    assert np.shape(og_img)[0] > np.shape(o_file)[0]
    assert np.shape(og_img)[1] > np.shape(o_file)[1]
    # check pixel resolution - should be double the original
    assert (ds_file.GetGeoTransform()[1]) == (og_file.GetGeoTransform()[1]*2)
    assert (ds_file.GetGeoTransform()[5]) == (og_file.GetGeoTransform()[5]*2)


def test_downsample_w_pad(tmp_path):
    """Test downsampling geotiff with some padding due to division."""
    g_path = os.path.normpath(
        'tests/integration/data/Colville/Colville_islands_filled.tif')
    outpath = os.path.join(tmp_path, 'downsampled.tif')
    # run downsampling function
    ofile = geo_utils.downsample_binary_geotiff(g_path, 0.76, outpath)
    # assert output
    assert ofile == outpath
