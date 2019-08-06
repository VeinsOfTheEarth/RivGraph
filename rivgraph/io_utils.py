# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 15:15:18 2018

@author: Jon
"""
import os
import pickle
import ogr, osr
import gdal
import numpy as np
import geopandas as gpd
import pandas as pd
import rivgraph.geo_utils as gu
from shapely.geometry import Point, LineString, Polygon
import fiona


def prepare_paths(resultsfolder, name, basetiff, ftype='GeoJSON'):
    """
    Given a results folder, a delta or river name, and a filetype, generates 
    paths for saving results or intermediate files.
    """   
    basepath = os.path.join(os.path.normpath(resultsfolder), name)
    
    if ftype == 'GeoJSON':
        ext = 'json'
    elif ftype == 'shapefile':
        ext = 'shp'
    
    # Create results folder if it doesn't exist
    if os.path.isdir(basepath) is False:
        os.mkdir(basepath)
    
    # Create dictionary of directories
    paths = dict()

    paths['maskpath'] = basetiff                                                     # geotiff binary mask; must be input by user
    paths['Iskel'] = os.path.join(basepath, name + "_skel.tif")                      # geotiff of skeletonized mask 
    paths['Idist'] = os.path.join(basepath, name + "_dist.tif")                      # geotiff of distance transform of mask
    paths['network_pickle'] = os.path.join(basepath, name + "_network.pkl")          # links and nodes dictionaries, pickled
    paths['links'] = os.path.join(basepath, name + "_links." + ext)                  # links geovectors
    paths['nodes'] = os.path.join(basepath, name + "_nodes."  + ext)                 # nodes geovectors
    paths['fixlinks_csv'] = os.path.join(basepath, name + "_fixlinks.csv")           # csv file to manually fix link directionality, must be created by user
    paths['linkdirs'] = os.path.join(basepath, name + "_link_directions.tif")        # tif file that shows link directionality
    paths['metrics'] = os.path.join(basepath, name + "_metrics.pkl")                 # metrics dictionary
    paths['meshlines'] = os.path.join(basepath, name + "_meshlines."  + ext)         # meshlines geovectors
    paths['meshpolys'] = os.path.join(basepath, name + "_meshpolys." + ext)          # meshpolys geovectors
    paths['centerline'] = os.path.join(basepath, name + "_centerline." + ext)        # centerline geovector
    paths['centerline_smooth'] = os.path.join(basepath, name + "_centerline_smooth."  + ext)     # smooth centerline geovector
    
    # The following paths a    
    paths['shoreline'] = os.path.join(basepath, name + "_shoreline.shp")     # shoreline shapefile, must be created by user
    paths['inlet_nodes'] = os.path.join(basepath, name + "_inlet_nodes.shp") # inlet nodes shapefile, must be created by user

    return paths


def pickle_links_and_nodes(links, nodes, outpath):
    
    with open(outpath, 'wb') as f:
        pickle.dump([links, nodes], f)
        

def unpickle_links_and_nodes(lnpath):
    
    with open(lnpath, 'rb') as f:
        links, nodes = pickle.load(f)
        
    return links, nodes


def get_driver(filename):
    
    # Write geodataframe to file
    ext = filename.split('.')[-1]
    if ext == 'json':
        driver = 'GeoJSON'
    elif ext == 'shp':
        driver = 'shapefile'
        
    return driver


def nodes_to_geofile(nodes, dims, gt, epsg, outpath):
                        
    nodexy = np.unravel_index(nodes['idx'], dims)
    x, y = gu.xy_to_coords(nodexy[1], nodexy[0], gt)
    all_nodes = [Point(x,y) for x, y in zip(x,y)]
        
    # Create GeoDataFrame for storing geometries and attributes
    gdf = gpd.GeoDataFrame(geometry=all_nodes)
    gdf.crs = fiona.crs.from_epsg(epsg)
    
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


def links_to_geofile(links, dims, gt, epsg, outpath):
       
    # Create line objects to write to shapefile
    all_links = []
    for link in links['idx']:
        xy = np.unravel_index(link, dims)
        x, y = gu.xy_to_coords(xy[1], xy[0], gt)
        all_links.append(LineString(zip(x,y)))
        
    # Create GeoDataFrame for storing geometries and attributes
    gdf = gpd.GeoDataFrame(geometry=all_links)
    gdf.crs = fiona.crs.from_epsg(epsg)
    
    # Store attributes as strings (numpy types give fiona trouble)
    dontstore = ['idx', 'n_networks']
    storekeys = [k for k in links.keys() if k not in dontstore]
    storekeys = [k for k in storekeys if len(links[k]) == len(links['id'])]
    store_as_num = ['id', 'flux', 'logflux']
    for k in storekeys:
        if k in store_as_num:            
            gdf[k] = [c for c in links[k]]
        else:
            gdf[k] = [str(c).replace('[','').replace(']','') for c in links[k]]
                        
    # Write geodataframe to file        
    gdf.to_file(outpath, driver=get_driver(outpath))
    
        
def centerline_to_geovector(cl, epsg, outpath):
    
    """
    Centerline is already-projected Nx2 numpy array.
    """    
    # Put points into shapely LineString   
    cl_ls = LineString(zip(cl[0], cl[1]))
    
    # Geopandas dataframe
    cl_df = gpd.GeoDataFrame(geometry=[cl_ls])
    cl_df.crs = fiona.crs.from_epsg(epsg)
    
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
    if options != 0:
        dest = driver.Create(outputpath, width, height, nbands, dtype, options)
    else:
        dest = driver.Create(outputpath, width, height, nbands, dtype)
          
    # Write output raster   
    for b in range(nbands):
        dest.GetRasterBand(b+1).WriteArray(raster[:,:,b])
 
        if nodata is not None:
            dest.GetRasterBand(b+1).SetNoDataValue(nodata)

        if color_table != None:
            dest.GetRasterBand(1).SetColorTable(color_table)
        
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


def coords_from_shapefile(coordspath):
    """
    Retrieves centerline coordinates from shapefile.
    """
    xy_gdf = gpd.read_file(coordspath)
    coords = []
    for i in xy_gdf.index:
        coords_obj = xy_gdf['geometry'][i].centroid.xy
        coords.append((coords_obj[0][0], coords_obj[1][0]))

    return coords


def coords_to_shapefile(coords, epsg, outpath):
    """
    Given a list or tuple of (x,y) coordinates and the EPSG code, writes the
    coordinates to a shapefile.
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
    
    
def meshlines_to_shapefile(lines, EPSG, outpath, nameid=None):
    """
    Given meshlines (output by RivMesh), ouput a shapefile.
    """
    
    # Create line objects to write to shapefile
    all_lines = []
    for l in lines:
        line = ogr.Geometry(type=ogr.wkbLineString)
        line.AddPoint_2D(l[0][0], l[0][1])
        line.AddPoint_2D(l[1][0], l[1][1])
        all_lines.append(line)

    # Write the shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(outpath)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    
    layer = datasource.CreateLayer("meshlines", srs, ogr.wkbLineString)
    defn = layer.GetLayerDefn()

    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    layer.CreateField(idField)
    if nameid is not None:
        nameField = ogr.FieldDefn('River_Name', ogr.OFTString)
        layer.CreateField(nameField)
    

    for i, p in enumerate(all_lines):
            
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', int(i))
        
        # Set upstream and downstream node attributes
        if nameid is not None:
            fieldstr = str(nameid)
            feat.SetField('River_Name', fieldstr)
    
        # Make a geometry
        geom = ogr.CreateGeometryFromWkb(p.ExportToWkb())
        feat.SetGeometry(geom)
    
        layer.CreateFeature(feat)
        
        feat = geom = None  # destroy these
    
    # Save and close everything
    datasource = layer = feat = geom = None


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
    

def meshpolys_to_geovectors(meshpolys, epsg, outpath):
    """
    Exports the meshopolys returned by centerline_mesh as a shapefile.
    """
        
    pgons = []
    for mp in meshpolys:
        pgons.append(Polygon(mp))
    
    gdf = gpd.GeoDataFrame(geometry=pgons)
    gdf.crs = fiona.crs.from_epsg(epsg)
    gdf.to_file(outpath, driver=get_driver(outpath))



""" Replaced or outdated functions """

#def nodes_to_shapefile(nodes, dims, gt, epsg, outpath):
#    
#    # Read metadata from links dictionary
#    dims = nodes['imshape']
#    gt = nodes['gt']
#    epsg = nodes['epsg']
#    outpath = nodes['savepath']
#
#    # Create point objects to write to shapefile
#    if 'id' not in nodes.keys():
#        ids = list(range(0,len(nodes['idx'])))
#    else:
#        ids = nodes['id']
#        
#    all_nodes = []
#    for node in nodes['idx']:
#        pt = ogr.Geometry(type=ogr.wkbPoint)
#        xy = np.unravel_index(node, dims)
#        ll = gu.xy_to_coords(xy[1]+.5, xy[0]+.5, gt, inputEPSG=epsg, outputEPSG=epsg)[0]
#        pt.AddPoint_2D(ll[1], ll[0])
#        all_nodes.append(pt)
#
#    # Write the shapefile
#    driver = ogr.GetDriverByName('ESRI Shapefile')
#    datasource = driver.CreateDataSource(outpath)
#    
#    srs = osr.SpatialReference()
#    srs.ImportFromEPSG(epsg)
#    
#    layer = datasource.CreateLayer("Nodes", srs, ogr.wkbPoint)
#    defn = layer.GetLayerDefn()
#    
#    idField = ogr.FieldDefn('id', ogr.OFTInteger)
#    connField = ogr.FieldDefn('conn', ogr.OFTString)
#  
#    layer.CreateField(idField)
#    layer.CreateField(connField)
#
#    for p, i in zip(all_nodes, ids):
#    
#        # Create a new feature (attribute and geometry)
#        feat = ogr.Feature(defn)
#        feat.SetField('id', int(i))
#        
#        fieldstr = str(nodes['conn'][nodes['id'].index(i)])
#        fieldstr = fieldstr[1:-1]
#        feat.SetField('conn', fieldstr)
#    
#        # Make a geometry
#        geom = ogr.CreateGeometryFromWkb(p.ExportToWkb())
#        feat.SetGeometry(geom)
#    
#        layer.CreateFeature(feat)
#        feat = geom = None  # destroy these
#    
#    # Save and close everything
#    datasource = layer = feat = geom = None


#def links_to_shapefile(links):
#    
#    # Read metadata from links dictionary
#    dims = links['imshape']
#    gt = links['gt']
#    epsg = links['epsg']
#    outpath = links['savepath']
#    
#
#    # Write the shapefile
#    driver = ogr.GetDriverByName('ESRI Shapefile')
#    datasource = driver.CreateDataSource(outpath)
#    
#    srs = osr.SpatialReference()
#    srs.ImportFromEPSG(epsg)
#    
#    layer = datasource.CreateLayer("Links", srs, ogr.wkbLineString)
#    defn = layer.GetLayerDefn()
#
#    # Create fields
#    idField = ogr.FieldDefn('id', ogr.OFTInteger)
#    usField = ogr.FieldDefn('us node', ogr.OFTInteger)
#    dsField = ogr.FieldDefn('ds node', ogr.OFTInteger)
#    layer.CreateField(idField)
#    layer.CreateField(usField)
#    layer.CreateField(dsField)
#    
#    # Include other attributes if available
#    if 'len' in links.keys():
#        lenField = ogr.FieldDefn('length', ogr.OFTReal)
#        layer.CreateField(lenField)
#    if 'wid' in links.keys():
#        lenField = ogr.FieldDefn('width', ogr.OFTReal)
#        layer.CreateField(lenField)
#    if 'len_adj' in links.keys():
#        widField = ogr.FieldDefn('len_adj', ogr.OFTReal)
#        layer.CreateField(widField)
#    if 'wid_adj' in links.keys():
#        widField = ogr.FieldDefn('width_adj', ogr.OFTReal)
#        layer.CreateField(widField)
#
#    usnodes = [c[0] for c in links['conn']]
#    dsnodes = [c[1] for c in links['conn']]
#    for i, p in enumerate(all_links):
#            
#        # Create a new feature (attribute and geometry)
#        feat = ogr.Feature(defn)
#        feat.SetField('id', int(links['id'][i]))
#        
#        # Set upstream and downstream node attributes
#        fieldstr = str(usnodes[i])
#        feat.SetField('us node', fieldstr)
#        fieldstr = str(dsnodes[i])
#        feat.SetField('ds node', fieldstr)
#    
#        # Set other attributes if available
#        if 'len' in links.keys():
#            fieldstr = str(links['len'][i])
#            feat.SetField('length', fieldstr)
#        
#        if 'len_adj' in links.keys():
#            fieldstr = str(links['len_adj'][i])
#            feat.SetField('len_adj', fieldstr)
#        
#        if 'wid' in links.keys():
#            fieldstr = str(links['wid'][i])
#            feat.SetField('width', fieldstr)
#
#        if 'wid_adj' in links.keys():
#            fieldstr = str(links['wid_adj'][i])
#            feat.SetField('wid_adj', fieldstr)
#    
#        # Make a geometry
#        geom = ogr.CreateGeometryFromWkb(p.ExportToWkb())
#        feat.SetGeometry(geom)
#    
#        layer.CreateFeature(feat)
#        
#        feat = geom = None  # destroy these
#    
#    # Save and close everything
#    datasource = layer = feat = geom = None

