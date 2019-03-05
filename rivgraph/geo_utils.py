# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:24:42 2018

@author: Jon

Utilities for reading, writing, managing, processing, manipulating, etc. 
geographic data including tiffs, vrts, shapefiles, etc.

"""
import osr, ogr, gdal
import numpy as np
import rivgraph.im_utils as iu
import geopandas as gpd
from pyproj import Proj, transform



def get_EPSG(file_or_obj, returndict=False):
    """
    Returns the EPSG code from a path to a geotiff (or vrt) or shapefile/GeoJSON/etc.
    EPSG can be returned as a dict (e.g. geopandas/fiona format) if desired.
    
    file_or_obj can be either a path to a file or an opened gdal object
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

    # Units of projection
    srs = osr.SpatialReference()
    srs.SetFromUserInput("EPSG:" + str(epsg))
    unit = srs.GetAttrValue("UNIT",0)
    
    return unit.lower()


    
def wkt2epsg(wkt):
    
    """
    From https://gis.stackexchange.com/questions/20298/is-it-possible-to-get-the-epsg-value-from-an-osr-spatialreference-class-using-th
    Transform a WKT string to an EPSG code

    Arguments
    ---------
    
    wkt: WKT definition
    
    Returns: EPSG code

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
    
    
def geotiff_vals_from_idcs(idcs, I_orig, I_pull):
    """
    Uses the get_array function to pull individual indices from
    a vrt or tiff. 
    I_orig is the image corresponding to the input indices
    I_pull is the image that you want to pull values from
    """
    if I_orig == I_pull:
        vals = []
        for i in idcs:
            val = iu.get_array(i, I_pull, (1,1))[0][0][0]
            vals.append(val)
            
#    else:
#        ll = idx_to_coords(idcs, idx_gdobj)
#        vals = geotiff_vals_from_coords(ll, val_gdobj)
        
    return vals


def geotiff_vals_from_coords(coords, gt_obj):
    
    # Lat/lon to row/col
    rowcol = coords_to_xy(coords[:,0], coords[:,1], gt_obj.GetGeoTransform())
    
    # Pull value from vrt at row/col
    vals = []
    for rc in rowcol:
           vals.append(gt_obj.ReadAsArray(int(rc[0]), int(rc[1]), int(1), int(1))[0,0])
           
    return vals


def coords_to_xy(xs, ys, gt):
            
    xs = np.array(xs) 
    ys = np.array(ys) 

    xs = ((xs - gt[0]) / gt[1]).astype(int)
    ys = ((ys - gt[3]) / gt[5]).astype(int)
    
    return np.column_stack((xs, ys))


def idx_to_coords(idx, gd_obj, printout=False):
    
    yx = np.unravel_index(idx, (gd_obj.RasterYSize, gd_obj.RasterXSize))
    cx, cy = xy_to_coords(yx[1], yx[0], gd_obj.GetGeoTransform())
    
    return cx, cy


def xy_to_coords(xs, ys, gt):
    """
    xs and ys should be numpy arrays of same length.
    """
                
    cx = gt[0] + (xs + 0.5) * gt[1]
    cy = gt[3] + (ys + 0.5) * gt[5]
    
    return cx, cy


def transform_coordinates(xs, ys, inputEPSG, outputEPSG):
    
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


def transform_coords(in_epsg, out_epsg, xs, ys):
    """
    Another way of transforming coordinates using pyproj.
    """
        
    in_proj = Proj(init='epsg:'+str(in_epsg))
    out_proj = Proj(init='epsg:'+str(out_epsg))
    xy = transform(in_proj, out_proj, xs, ys)
    
    return xy



""" Utilities below here are not explicitly called by RivGraph functions """

def crop_geotif(tif, cropto='first_nonzero', npad=0, outpath=None):
 
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
    if npad is not 0:
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
        
    write_geotiff(tifcropped, crop_gt, tif_obj.GetProjection(), output_file, dtype=datatype, options=options)
    
    return output_file
