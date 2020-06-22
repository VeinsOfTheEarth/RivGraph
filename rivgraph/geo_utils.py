# -*- coding: utf-8 -*-
"""
geo_utils
=========

Created on Tue Sep 11 11:24:42 2018

@author: Jon

Utilities for reading, writing, managing, processing, manipulating, etc.
geographic data including tiffs, vrts, shapefiles, etc.

6/2/2020 - Consider merging this into io_utils and im_utils. Not much actual
functionality here, and some of these functions are simply unused.

"""
import gdal
import numpy as np
from pyproj import Transformer
import warnings
import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)))
import io_utils as io


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
        An Nx2 numpy array, where each row is a (lat, lon) pair.
    gd_obj : osgeo.gdal.Dataset
        Geotiff object created with gdal.Open().

    Returns
    ----------
    vals : list
        The value of the pixels of the geotiff for each coordinate.
    """


    # Lat/lon to row/col
    rowcol = coords_to_xy(coords[:,0], coords[:,1], gd_obj.GetGeoTransform())

    # Pull value from vrt at row/col
    vals = []
    for rc in rowcol:
           vals.append(gd_obj.ReadAsArray(int(rc[0]), int(rc[1]), int(1), int(1))[0,0])

    return vals


def coords_to_xy(xs, ys, gt):
    """
    Transforms a set of xs, ys in projected coordinates to rows, columns within
    a geotiff.

    Arguments
    ---------
    xs : list or np.array()
        Specifies the E-W coordinates (longitude).
    ys : list or np.array()
        Specifies the N-S coordinates (latitude).
    gt : tuple
        6-element tuple gdal GeoTransform. (uL_x, x_res, rotation, ul_y, rotation, y_res).
        Automatically created by gdal's GetGeoTransform() method.

    Returns
    ----------
    rowcols : np.array()
        Nx2 array of (row, col) indices corresponding to the inpute coordinates. N = len(xs).
    """

    xs = np.array(xs)
    ys = np.array(ys)

    xs = ((xs - gt[0]) / gt[1]).astype(int)
    ys = ((ys - gt[3]) / gt[5]).astype(int)

    return np.column_stack((xs, ys))


def idx_to_coords(idx, gd_obj):
    """
    Transforms a set of indices from a geotiff image to their corresponding
    coordinates.

    Arguments
    ---------
    idx : np.array()
        Specifies the indices to transform. See np.ravel_index for more info.
    gd_obj : osego.gdal.Dataset
        gdal object of the geotiff from which indices were computed.

    Returns
    ----------
    cx, cy : tuple
        x and y coordinates of the provided indices.
    """

    yx = np.unravel_index(idx, (gd_obj.RasterYSize, gd_obj.RasterXSize))
    cx, cy = xy_to_coords(yx[1], yx[0], gd_obj.GetGeoTransform())

    return cx, cy


def xy_to_coords(xs, ys, gt):
    """
    Transforms a set of x and y coordinates to their corresponding coordinates
    within a geotiff image.

    Arguments
    ---------
    (xs, ys) : (np.array(), np.array())
        Specifies the coordinates to transform.
    gt : tuple
        6-element tuple gdal GeoTransform. (uL_x, x_res, rotation, ul_y, rotation, y_res).
        Automatically created by gdal's GetGeoTransform() method.

    Returns
    ----------
    cx, cy : tuple
        Column and row indices of the provided coordinates.
    """

    cx = gt[0] + (xs + 0.5) * gt[1]
    cy = gt[3] + (ys + 0.5) * gt[5]

    return cx, cy


def transform_coords(xs, ys, inputEPSG, outputEPSG):
    """
    Transforms a set of coordinates from one epsg to another.
    This implementation differs from above by using pyproj.

    Arguments
    ---------
    (xs, ys) : (np.array(), np.array())
        Specifies the coordinates to transform.
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
    """
    Crops a geotiff to the minimum bounding box as defined by the first
    nonzero pixels along each direction. The cropped image is written to
    disk.

    Arguments
    ---------
    tif : str
        Path to geotiff to crop.
    cropto : str
        [first_nonzero] is currently the only choice.
    npad : int
        Number of pixels to add to each direction of the cropped image.
    outpath : str
        Defines the path where the cropped image will be written to disk. If
        [None], the file will be written to the same directory as the input
        geotiff.

    Returns
    ----------
    output_file : str
        Path to the saved, cropped geotiff.
    """

    # Prepare output file path
    if outpath is None:
        output_file = tif.split('.')[-2] + '_cropped.tif'
    else:
        output_file = outpath

    tif_obj = gdal.Open(tif)
    tiffull = tif_obj.ReadAsArray()

    if cropto == 'first_nonzero':
        idcs = np.where(tiffull>0)
        t = np.min(idcs[0])
        b = np.max(idcs[0]) + 1
        l = np.min(idcs[1])
        r = np.max(idcs[1]) + 1

    # Crop the tiff
    tifcropped = tiffull[t:b,l:r]

    # Pad the tiff (if necessary)
    if npad != 0:
        tifcropped = np.pad(tifcropped, npad, mode='constant', constant_values=False)

    # Create a new geotransform by adjusting the origin (upper-left-most point)
    gt = tif_obj.GetGeoTransform()
    ulx = gt[0] + (l - npad) * gt[1]
    uly = gt[3] + (t - npad) * gt[5]
    crop_gt = (ulx, gt[1], gt[2], uly, gt[4], gt[5])

    # Prepare datatype and options for saving...
    datatype = tif_obj.GetRasterBand(1).DataType

    options = ['BLOCKXSIZE=128',
               'BLOCKYSIZE=128',
               'TILED=YES']

    # Only compress if we're working with a non-float
    if datatype in [1, 2, 3, 4, 5]: # Int types: see the list at the end of this file
        options.append('COMPRESS=LZW')

    io.write_geotiff(tifcropped, crop_gt, tif_obj.GetProjection(), output_file, dtype=datatype, options=options)

    return output_file