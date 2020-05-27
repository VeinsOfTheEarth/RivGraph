# -*- coding: utf-8 -*-
"""
geo_utils
=========

Created on Tue Sep 11 11:24:42 2018

@author: Jon

Utilities for reading, writing, managing, processing, manipulating, etc.
geographic data including tiffs, vrts, shapefiles, etc.

"""
import osr, ogr, gdal
import numpy as np
import geopandas as gpd
from pyproj import Proj, transform
import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)))
import io_utils as io



def get_EPSG(file_or_obj, returndict=False):
    """
    Returns the EPSG code from a path to a geotiff (or vrt) or shapefile/GeoJSON/etc.
    EPSG can be returned as a dict (e.g. geopandas/fiona format) if desired.

    Parameters
    ----------
    file_or_obj : str or gdal object
        Either the path to a georeferenced datafile (geotiff, shapefile, etc.),
        or an object created by gdal.Open().
    returndict : bool
        True or [False] to return the epsg code in a diciionary form for easy
        passing to fiona/geopandas.

    Returns
    ----------
    epsg : int or dict
        If returndict is False, returns the epsg code. If returndict is True,
        returns epsg code in dictionary format for easy passing to fiona/
        geopandas.
    """

    if type(file_or_obj) is str:
        rast_obj = gdal.Open(file_or_obj)
    else:
        rast_obj = file_or_obj

    if rast_obj is not None: # we have a raster
        wkt = rast_obj.GetProjection()
        epsg = wkt2epsg(wkt)
    else: # we have a shapefile
        vec = gpd.read_file(file_or_obj)
        epsg = int(vec.crs['init'].strip('epsg:'))

    if returndict is True:
        epsg_dict = dict()
        epsg_dict['init'] = 'epsg:' + str(epsg)
        return epsg_dict
    else:
        return epsg


def get_unit(epsg):
    """
    Returns the units for a projection defined by an EPSG code.

    Parameters
    ----------
    epsg : int, float, or str
        The epsg code.

    Returns
    ----------
    unit : str
        The unit of the provided epsg code.

    """


    # Units of projection
    srs = osr.SpatialReference()
    srs.SetFromUserInput("EPSG:" + str(epsg))
    unit = srs.GetAttrValue("UNIT",0)

    return unit.lower()



def wkt2epsg(wkt):
    """
    Determines the epsg code for a provided WKT definition. Code was mostly
    taken from https://gis.stackexchange.com/questions/20298/is-it-possible-to-get-the-epsg-value-from-an-osr-spatialreference-class-using-th

    Arguments
    ---------
    wkt : str
        WKT definition

    Returns
    ----------
    epsg : int
        The epsg code.
    """

    p_in = osr.SpatialReference()
    s = p_in.ImportFromWkt(wkt)
    if wkt[8:23] == 'World_Mollweide':
        return(54009)
    if s == 5:  # invalid WKT
        return None
    if p_in.IsLocal() == 1:  # this is a local definition
        return p_in.ExportToWkt()
    if p_in.IsGeographic() == 1:  # this is a geographic srs
        cstype = 'GEOGCS'
    else:  # this is a projected srs
        cstype = 'PROJCS'
    an = p_in.GetAuthorityName(cstype)
    ac = p_in.GetAuthorityCode(cstype)
    if an is not None and ac is not None:  # return the EPSG code
#        return str(p_in.GetAuthorityName(cstype)), str(p_in.GetAuthorityCode(cstype))
        return int(p_in.GetAuthorityCode(cstype))


#def geotiff_vals_from_idcs(idcs, I_orig, I_pull):
#    ## TODO: change the name of this function as geotiffs are no longer handled here
#    """
#    Pulls pixel values from an image
#    Uses the get_array function to pull individual indices from
#    a vrt or tiff.
#    I_orig is the image corresponding to the input indices
#    I_pull is the image that you want to pull values from
#    """
#    if I_orig == I_pull:
#        vals = []
#        for i in idcs:
#            val = iu.get_array(i, I_pull, (1,1))[0][0][0]
#            vals.append(val)
#
#    return vals


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


def transform_coordinates(xs, ys, inputEPSG, outputEPSG):
    """
    Transforms a set of coordinates from one epsg to another.

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
    xyout : (np.array(), np.array())
        N-element arrays of transformed (x, y) coordinates.
    """

    if inputEPSG == outputEPSG:
        return xs, ys

    # Create an ogr object of multipoints
    points = ogr.Geometry(ogr.wkbMultiPoint)

    for x,y in zip(xs,ys):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(x), float(y))
        points.AddGeometry(point)

    # Create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # transform point
    points.Transform(coordTransform)

    xyout = np.array([0,0,0])
    for i in range(len(xs)):
        xyout = np.vstack((xyout, points.GetGeometryRef(i).GetPoints()))
    xyout = xyout[1:,0:2]

    return xyout[:,0], xyout[:,1]


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
        N-element array of transformed (x, y) coordinates.
    """

    in_proj = Proj(init='epsg:'+str(inputEPSG))
    out_proj = Proj(init='epsg:'+str(outputEPSG))
    xy = transform(in_proj, out_proj, xs, ys)

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
