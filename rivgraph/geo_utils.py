"""
Georeferencing Utilities (geo_utils.py)
=======================================

Utilities for reading, writing, managing, processing, manipulating, etc.
geographic data including tiffs, vrts, shapefiles, etc.

6/2/2020 - Consider merging this into io_utils and im_utils. Not much actual
functionality here, and some of these functions are simply unused.

"""
import numpy as np
from pyproj import Transformer
import warnings
import rivgraph.io_utils as io
import rivgraph.rasters as rasters

def get_unit(crs):
    """
    Returns the units for a projection defined by an EPSG code.
    See https://en.wikibooks.org/wiki/PROJ.4#Units for a list of unit string
    maps.

    Parameters
    ----------
    crs : pyproj CRS object
        Defines the coordinate reference system.

    Returns
    ----------
    unit : str
        The unit of the provided epsg code.

    """
    # The to_proj4() function generates a warning.
    warnings.simplefilter(action='ignore', category=UserWarning)
    p4 = crs.to_proj4()
    warnings.simplefilter(action='default', category=UserWarning)

    projkey = p4[p4.index('+proj=') + len('+proj='):].split(' ')[0]

    if projkey == 'longlat':
        unit = 'degree'
    else:
        unitstr = p4[p4.index('+units=') + len('+units='):].split(' ')[0]

        p4units = {'m' : 'meter',
                   'cm' : 'centimeter',
                   'dm' : 'decimenter',
                   'ft' : 'foot',
                   'in' : 'inch',
                   'km' : 'kilometer',
                   'mi' : 'international statute mile',
                   'mm' : 'millimeter',
                   'yd' : 'international yard'}

        if unitstr in p4units.keys():
            unit = p4units[unitstr]
        else:
            unit = unitstr
            raise Warning('Unit type {} not understood.'.format(unitstr))

    return unit


def geotiff_vals_from_coords(coords, gd_obj):
    """
    Returns pixel values at specific coordinates from a geotiff object.

    Arguments
    ---------
    coords : np.array()
        An Nx2 numpy array, where each row is a (lon, lat) pair.
    gd_obj : osgeo.gdal.Dataset
        Geotiff object created with gdal.Open().

    Returns
    ----------
    vals : list
        The value of the pixels of the geotiff for each coordinate.
    """


    # Lat/lon to row/col
    rowcol = coords_to_xy(coords[:, 0], coords[:, 1], gd_obj.GetGeoTransform())

    # Pull value from raster at row/col
    vals = []
    for rc in rowcol:
        vals.append(gd_obj.ReadAsArray(int(rc[0]), int(rc[1]), int(1), int(1))[0, 0])

    return vals


def coords_to_xy(xs, ys, gt):
    """Transforms projected coordinates to pixel columns/rows."""
    return rasters.coords_to_xy(xs, ys, gt)


def idx_to_coords(idx, gd_obj):
    """Transforms flat raster indices to projected coordinates."""
    return rasters.idx_to_coords(idx, (gd_obj.RasterYSize, gd_obj.RasterXSize), gd_obj.GetGeoTransform())


def xy_to_coords(xs, ys, gt):
    """Transforms pixel columns/rows to projected coordinates."""
    return rasters.xy_to_coords(xs, ys, gt)


def transform_coords(xs, ys, inputEPSG, outputEPSG):
    """
    Transforms a set of coordinates from one epsg to another.
    This implementation differs from above by using pyproj.

    Arguments
    ---------
    xs : np.array
        Specifies the x-coordinates to transform.
    ys : np.array
        Specifies the y-coordinates to transform.
    inputEPSG : int
        epsg code corresponding to xs, ys
    outputEPSG : int
        epsg code corresponding to desired CRS.

    Returns
    ----------
    xy : np.array()
        Two element array of transformed (x, y) coordinates. xy[0] are
        transformed x coordinates, xy[1] are transformed y coordinates.
    """

    proj = Transformer.from_crs(inputEPSG, outputEPSG, always_xy=True)
    xt, yt = proj.transform(xs, ys)
    xy = np.array((xt, yt))

    return xy


def crop_geotif(tif, cropto='first_nonzero', npad=0, outpath=None):
    """Backward-compatible wrapper around the raster backend crop helper."""
    return rasters.crop_geotiff(tif, cropto=cropto, npad=npad, outpath=outpath)


def downsample_binary_geotiff(input_file, ds_factor, output_name, thresh=None):
    """Backward-compatible wrapper around the raster backend downsampler."""
    return rasters.downsample_binary_geotiff(input_file, ds_factor, output_name, thresh=thresh)
