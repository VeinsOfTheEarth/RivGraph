# -*- coding: utf-8 -*-
"""
io_utils
========

Created on Sun Sep 16 15:15:18 2018

@author: Jon
"""
import os
import pickle
import ogr, osr
import gdal
import numpy as np
import pandas as pd
import geopandas as gpd
import rivgraph.geo_utils as gu
from shapely.geometry import Point, LineString


def prepare_paths(resultsfolder, name, basetiff):
    """
    Given a results folder, a delta or river name, and a filetype, generates
    paths for saving results or intermediate files.
    """
    basepath =os.path.normpath(resultsfolder)

    # Create results folder if it doesn't exist
    if os.path.isdir(basepath) is False:
        os.makedirs(basepath)

    # Create dictionary of directories
    paths = dict()

    paths['basepath'] = basepath
    paths['maskpath'] = basetiff                                                     # geotiff binary mask; must be input by user
    paths['Iskel'] = os.path.join(basepath, name + "_skel.tif")                      # geotiff of skeletonized mask
    paths['Idist'] = os.path.join(basepath, name + "_dist.tif")                      # geotiff of distance transform of mask
    paths['network_pickle'] = os.path.join(basepath, name + "_network.pkl")          # links and nodes dictionaries, pickled
    paths['fixlinks_csv'] = os.path.join(basepath, name + "_fixlinks.csv")           # csv file to manually fix link directionality, must be created by user
    paths['linkdirs'] = os.path.join(basepath, name + "_link_directions.tif")        # tif file that shows link directionality
    paths['metrics'] = os.path.join(basepath, name + "_metrics.pkl")                 # metrics dictionary

    # The files at the following paths are not created by RivGraph, but by the user.
    paths['shoreline'] = os.path.join(basepath, name + "_shoreline.shp")     # shoreline shapefile, must be created by user
    paths['inlet_nodes'] = os.path.join(basepath, name + "_inlet_nodes.shp") # inlet nodes shapefile, must be created by user

    return paths


def pickle_links_and_nodes(links, nodes, outpath):

    with open(outpath, 'wb') as f:
        pickle.dump([links, nodes], f)


def unpickle_links_and_nodes(lnpath):

    import sys
    from rivgraph import ordered_set    
    sys.modules['ordered_set'] = ordered_set

    with open(lnpath, 'rb') as f:
        links, nodes = pickle.load(f)

    return links, nodes


def get_driver(filename):

    # Write geodataframe to file
    ext = filename.split('.')[-1]
    if ext == 'json':
        driver = 'GeoJSON'
    elif ext == 'shp':
        driver = 'ESRI Shapefile'

    return driver


def nodes_to_geofile(nodes, dims, gt, crs, outpath):

    nodexy = np.unravel_index(nodes['idx'], dims)
    x, y = gu.xy_to_coords(nodexy[1], nodexy[0], gt)
    all_nodes = [Point(x,y) for x, y in zip(x,y)]

    # Create GeoDataFrame for storing geometries and attributes
    gdf = gpd.GeoDataFrame(geometry=all_nodes)
    gdf.crs = crs

    # Store attributes as strings (numpy types give fiona trouble)
    dontstore = ['idx']
    storekeys = [k for k in nodes.keys() if len(nodes[k]) == len(nodes['id']) and k not in dontstore]
    store_as_num = ['id', 'idx', 'logflux', 'flux']
    for k in storekeys:
        if k in store_as_num:
            gdf[k] = [c for c in nodes[k]]
        else:
            gdf[k] = [str(c).replace('[','').replace(']','') for c in nodes[k]]

    # Write geodataframe to file
    gdf.to_file(outpath, driver=get_driver(outpath))


def links_to_geofile(links, dims, gt, crs, outpath):

    # Create line objects to write to shapefile
    all_links = []
    for link in links['idx']:
        xy = np.unravel_index(link, dims)
        x, y = gu.xy_to_coords(xy[1], xy[0], gt)
        all_links.append(LineString(zip(x,y)))

    # Create GeoDataFrame for storing geometries and attributes
    gdf = gpd.GeoDataFrame(geometry=all_links)
    gdf.crs = crs

    # Store attributes as strings (numpy types give fiona trouble)
    dontstore = ['idx', 'n_networks']
    storekeys = [k for k in links.keys() if k not in dontstore]
    storekeys = [k for k in storekeys if len(links[k]) == len(links['id'])]
    store_as_num = ['id', 'flux', 'logflux']
    for k in storekeys:
        if k in store_as_num:
            gdf[k] = [c for c in links[k]]
        elif k == 'wid_pix':
            gdf[k] = [str(c.tolist()).replace('[','').replace(']','') for c in links[k]]
        else:
            gdf[k] = [str(c).replace('[','').replace(']','') for c in links[k]]

    # Write geodataframe to file
    gdf.to_file(outpath, driver=get_driver(outpath))


def centerline_to_geovector(cl, crs, outpath):
    """
    Centerline is already-projected Nx2 numpy array.
    """
    # Put points into shapely LineString
    if type(cl) is not LineString:
        cl = LineString(zip(cl[0], cl[1]))

    # Geopandas dataframe
    cl_df = gpd.GeoDataFrame(geometry=[cl])
    cl_df.crs = crs

    # Save
    cl_df.to_file(outpath, driver=get_driver(outpath))


def write_geotiff(raster, gt, wkt, outputpath, dtype=gdal.GDT_UInt16, options=['COMPRESS=LZW'], nbands=1, nodata=None, color_table=None):

    width = np.shape(raster)[1]
    height = np.shape(raster)[0]

    # Add empty dimension for single-band images
    if len(raster.shape) == 2:
        raster = np.expand_dims(raster, -1)

    # Prepare destination file
    driver = gdal.GetDriverByName("GTiff")
    if options != None:
        dest = driver.Create(outputpath, width, height, nbands, dtype, options)
    else:
        dest = driver.Create(outputpath, width, height, nbands, dtype)

    # Write output raster
    for b in range(nbands):
        dest.GetRasterBand(b+1).WriteArray(raster[:,:,b])

        if nodata is not None:
            dest.GetRasterBand(b+1).SetNoDataValue(nodata)

        if color_table != None:
            dest.GetRasterBand(1).SetRasterColorTable(color_table)

    # Set transform and projection
    dest.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())

    # Close output raster dataset
    dest = None



def colortable(ctype):

    color_table = gdal.ColorTable()

    if ctype == 'binary':
        # Some examples / last value is alpha (transparency). See http://www.gdal.org/structGDALColorEntry.html
        # and https://gis.stackexchange.com/questions/158195/python-gdal-create-geotiff-from-array-with-colormapping
        color_table.SetColorEntry( 0, (0, 0, 0, 0) )
        color_table.SetColorEntry( 1, (255, 255, 255, 100))
    elif ctype == 'skel':
        color_table.SetColorEntry( 0, (0, 0, 0, 0) )
        color_table.SetColorEntry( 1, (255, 0, 255, 100))
    elif ctype == 'mask':
        color_table.SetColorEntry( 0, (0, 0, 0, 0) )
        color_table.SetColorEntry( 1, (0, 128, 0, 100))
    elif ctype == 'tile':
        color_table.SetColorEntry( 0, (0, 0, 0, 0) )
        color_table.SetColorEntry( 1, (0, 0, 255, 100))
    elif ctype == 'JRCmo':
        color_table.SetColorEntry( 0, (0, 0, 0, 0) )
        color_table.SetColorEntry( 1, (0, 0, 0, 0) )
        color_table.SetColorEntry( 2, (176, 224, 230, 100))

    return color_table


def coords_from_geovector(coordspath):
    """
    Not called in classes.py
    
    Retrieves centerline coordinates from shapefile.
    """
    xy_gdf = gpd.read_file(coordspath)
    coords = []
    for i in xy_gdf.index:
        coords_obj = xy_gdf['geometry'][i].centroid.xy
        coords.append((coords_obj[0][0], coords_obj[1][0]))

    return coords


def coords_to_geovector(coords, epsg, outpath):
    """
    Not called in classes.py
    
    Given a list or tuple of (x,y) coordinates and the EPSG code, writes the
    coordinates to a shapefile.
    
    This should be replaced by a geodataframe creation, but no use cases
    yet...
    """

    all_coords = []
    for c in coords:
        pt = ogr.Geometry(type=ogr.wkbPoint)
        pt.AddPoint_2D(c[1], c[0])
        all_coords.append(pt)

    # Write the shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(outpath)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    layer = datasource.CreateLayer("Coords", srs, ogr.wkbPoint)
    defn = layer.GetLayerDefn()

    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    layer.CreateField(idField)

    for i, p in enumerate(all_coords):

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', int(i))

        # Make a geometry
        geom = ogr.CreateGeometryFromWkb(p.ExportToWkb())
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

    # Save and close everything
    datasource = layer = feat = geom = None


def meshlines_to_geovectors(lines, crs, outpath):

    gdf = gpd.GeoDataFrame(geometry=lines)
    gdf.crs = crs
    gdf.to_file(outpath, driver=get_driver(outpath))


def meshpolys_to_geovectors(meshpolys, crs, outpath):
    """
    Exports the meshpolys returned by centerline_mesh as a shapefile.
    """

    gdf = gpd.GeoDataFrame(geometry=meshpolys)
    gdf.crs = crs
    gdf.to_file(outpath, driver=get_driver(outpath))


def write_linkdirs_geotiff(links, gd_obj, writepath):
    """
    Creates a geotiff where links are colored according to their directionality.
    Pixels in each link are interpolated between 0 and 1 such that the upstream
    pixel is 0 and the downstream-most pixel is 1. In a GIS, color can then
    be set to see directionality.
    """

    # Initialize plotting raster
    I = gd_obj.ReadAsArray()
    I = np.zeros((gd_obj.RasterYSize, gd_obj.RasterXSize), dtype=np.float32)

    # Loop through links and store each pixel's interpolated value
    for lidcs in links['idx']:
        n = len(lidcs)
        vals = np.linspace(0,1, n)
        rcidcs = np.unravel_index(lidcs, I.shape)
        I[rcidcs] = vals

    # Save the geotiff
    write_geotiff(I, gd_obj.GetGeoTransform(), gd_obj.GetProjection(), writepath, dtype=gdal.GDT_Float32, nodata=0)

    return


def create_manual_dir_csv(csvpath):
    """
    Creates a csv file for fixing links manually.
    """
    df = pd.DataFrame(columns=['link_id','usnode'])
    df.to_csv(csvpath, index=False)


