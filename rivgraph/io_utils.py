# -*- coding: utf-8 -*-
"""
Input/Output Utilities (io_utils.py)
====================================

Functions for input/output.

Created on Sun Sep 16 15:15:18 2018

@author: Jon
"""
import os
import pickle
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

import rivgraph.geo_utils as gu
from rivgraph.rivers import centerline_utils as cu



def prepare_paths(path_results, name, path_mask):
    """
    Creates a dictionary of paths for most of the RivGraph-exportable files.

    Parameters
    ----------
    path_results : str
        The directory of the path where results are exported. Will be created
        if it does not exist.
    name : str
        Name of the analysis that is prepended to exported results.
    path_mask : str
        Path to the mask geotiff, including extension.

    Returns
    -------
    paths : dict
        Contains all the export paths. Not all will be necessarily used, but
        all possible exports' paths are contained.

    """
    basepath = os.path.normpath(path_results)

    # Create results folder if it doesn't exist
    if os.path.isdir(basepath) is False:
        os.makedirs(basepath)

    # Create dictionary of directories
    paths = dict()

    paths['basepath'] = basepath
    # geotiff binary mask; must be input by user
    paths['maskpath'] = path_mask
    # geotiff of skeletonized mask
    paths['Iskel'] = os.path.join(basepath, name + "_skel.tif")
    # geotiff of distance transform of mask
    paths['Idist'] = os.path.join(basepath, name + "_dist.tif")
    # links and nodes dictionaries, pickled
    paths['network_pickle'] = os.path.join(basepath, name + "_network.pkl")
    # csv file to manually fix link directionality, must be created by user
    paths['fixlinks_csv'] = os.path.join(basepath, name + "_fixlinks.csv")
    # tif file that shows link directionality
    paths['linkdirs'] = os.path.join(basepath, name + "_link_directions.tif")
    # metrics dictionary
    paths['metrics'] = os.path.join(basepath, name + "_metrics.pkl")
    # log file path
    paths['log'] = os.path.join(basepath, name + "_log.log")

    # The files at the following paths are not created by RivGraph,
    # but by the user.
    # shoreline shapefile, must be created by user
    paths['shoreline'] = os.path.join(basepath, name + "_shoreline.shp")
    # inlet nodes shapefile, must be created by user
    paths['inlet_nodes'] = os.path.join(basepath, name + "_inlet_nodes.shp")

    return paths


def pickle_links_and_nodes(links, nodes, path_out):
    """
    Saves the links and nodes dictionaries to a pickle file for easy loading.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    path_out : str
        Where to save the pickle.

    Returns
    -------
    None.

    """
    with open(path_out, 'wb') as f:
        pickle.dump([links, nodes], f)


def unpickle_links_and_nodes(path_pickle):
    """
    Unpickles a previously pickled network. Note that ordered_set is reloaded
    within this function so that pickle can interpret OrderedSet object fields.

    Parameters
    ----------
    path_pickle : str
        Path to the pickled network.

    Returns
    -------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    """
    import sys
    from rivgraph import ordered_set
    sys.modules['ordered_set'] = ordered_set

    with open(path_pickle, 'rb') as f:
        links, nodes = pickle.load(f)

    return links, nodes


def get_driver(path_file):
    """
    Finds the proper geopandas driver for saving a geodataframe. Keys off the
    extension in the filename, and supports either shapefiles or geojsons.

    Parameters
    ----------
    path_file : str
        Where the file will be saved.

    Returns
    -------
    driver : str
        Driver string specifying file format when using geopandas' to_file().

    """
    # Write geodataframe to file
    ext = path_file.split('.')[-1]
    if ext == 'json':
        driver = 'GeoJSON'
    elif ext == 'shp':
        driver = 'ESRI Shapefile'

    return driver


def nodes_to_geofile(nodes, dims, gt, crs, path_export):
    """
    Saves the nodes of the network to a georeferencedshapefile or geojson.
    Computed node properties are appended as attributes when available.
    The filetype is specified by the export path.

    Parameters
    ----------
    nodes : dict
        Network nodes and associated properties.
    dims : tuple
        (nrows, ncols) of the original mask from which nodes were derived.
    gt : tuple
        GDAL geotransform of the original mask from which nodes were derived.
    crs : pyrpoj.CRS
        CRS object specifying the coordinate reference system of the original
        mask from which nodes were derived.
    path_export : str
        Path, including extension, where to save the nodes export.

    Returns
    -------
    None.

    """
    nodexy = np.unravel_index(nodes['idx'], dims)
    x, y = gu.xy_to_coords(nodexy[1], nodexy[0], gt)
    all_nodes = [Point(x, y) for x, y in zip(x, y)]

    # Create GeoDataFrame for storing geometries and attributes
    gdf = gpd.GeoDataFrame(geometry=all_nodes)
    gdf.crs = crs

    # Store attributes as strings (numpy types give fiona trouble)
    dontstore = ['idx']
    storekeys = [k for k in nodes.keys() if len(nodes[k]) == len(nodes['id']) and k not in dontstore]
    store_as_num = ['id', 'idx', 'logflux', 'flux', 'outletflux']
    for k in storekeys:
        if k in store_as_num:
            gdf[k] = [c for c in nodes[k]]
        else:
            gdf[k] = [str(c).replace('[', '').replace(']', '') for c in nodes[k]]

    # Write geodataframe to file
    gdf.to_file(path_export, driver=get_driver(path_export))


def links_to_geofile(links, dims, gt, crs, path_export):
    """
    Saves the links of the network to a georeferencedshapefile or geojson.
    Computed link properties are saved as attributes when available. Note that
    the 'wid_pix' property, which stores the width at each pixel along the
    link, may be truncated depending on its length and the filetype.
    The filetype is specified by the export path.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    dims : tuple
        (nrows, ncols) of the original mask from which links were derived.
    gt : tuple
        GDAL geotransform of the original mask from which links were derived.
    crs : pyrpoj.CRS
        CRS object specifying the coordinate reference system of the original
        mask from which links were derived.
    path_export : str
        Path, including extension, specifying where to save the links export.

    Returns
    -------
    None.

    """
    # Create line objects to write to shapefile
    all_links = []
    for link in links['idx']:
        xy = np.unravel_index(link, dims)
        x, y = gu.xy_to_coords(xy[1], xy[0], gt)
        all_links.append(LineString(zip(x, y)))

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
            gdf[k] = [str(c.tolist()).replace('[', '').replace(']', '') for c in links[k]]
        else:
            gdf[k] = [str(c).replace('[', '').replace(']', '') for c in links[k]]

    # Write geodataframe to file
    gdf.to_file(path_export, driver=get_driver(path_export))


def centerline_to_geovector(cl, crs, path_export):
    """
    Exports centerline coordinates as a georeferenced linestring. Can be used
    with any set of coordinates.

    Parameters
    ----------
    cl : np.array
        ((xs), (ys)) array of coordinates to write. The coordinates should already be
        in terms of the provided crs.
    crs : pyproj.CRS
        CRS object specifying the coordinate reference system of the provided
        coordinates
    path_export : str
        Path, including extension, specifying where to save the coordinates export.

    Returns
    -------
    None.

    """
    # Put points into shapely LineString
    if type(cl) is not LineString:
        cl = LineString(zip(cl[0], cl[1]))

    # Geopandas dataframe
    cl_df = gpd.GeoDataFrame(geometry=[cl])
    cl_df.set_crs(crs, inplace=True)

    # Save
    cl_df.to_file(path_export, driver=get_driver(path_export))


def write_geotiff(raster, gt, wkt, path_export, dtype=gdal.GDT_UInt16,
                  options=['COMPRESS=LZW'], nbands=1, nodata=None,
                  color_table=None):
    """
    Writes a georeferenced raster to disk.

    Parameters
    ----------
    raster : np.array
        Image to be written. Shape is (nrows, ncols, nbands), although if only
        one band is present the shape can be just (nrows, ncols).
    gt : tuple
        GDAL geotransform for the raster. Often this can simply be copied from
        another geotiff via gdal.Open(path_to_geotiff).GetGeoTransform(). Can
        also be constructed following the gdal convention of
        (leftmost coordinate, pixel width, xskew, uppermost coordinate, pixel height, yskew).
        For non-rotated images, the skews will be zero.
    wkt : str
        Well-known text describing the coordinate reference system of the raster.
        Can be copied from another geotiff with gdal.Open(path_to_geotiff).GetProjection().
    path_export : str
        Path with extension of the geotiff to export.
    dtype : gdal.GDT_XXX, optional
        Gdal data type. Options for XXX include Byte, UInt16, UInt32, Int32,
        Float32, Float64 and complex types CInt16, Cint32, CFloat32 and CFloat64.
        If storing decimal data, use a Float type, binary data use Byte type.
        The default is gdal.GDT_UInt16 (non-float).
    options : list of strings, optional
        Options that can be fed to gdal dataset creator. See YYY for what
        can be specified by options.
        The default is ['COMPRESS=LZW'].
    nbands : int, optional
        Number of bands of the raster. The default is 1.
    nodata : numeric, optional
        Pixels with this value will be written as nodata. If None, no nodata
        value will be considered. The default is None.
    color_table : gdal.ColorTable, optional
        Color table to append to the geotiff. Can use colortable() function
        to create, or create a custom type with gdal.ColorTable().
        Note that color_tables can only be specified for Byte and UInt16 datatypes.
        The default is None.

    Returns
    -------
    None.

    """
    height = np.shape(raster)[0]
    width = np.shape(raster)[1]

    # Add empty dimension for single-band images
    if len(raster.shape) == 2:
        raster = np.expand_dims(raster, -1)

    # Prepare destination file
    driver = gdal.GetDriverByName("GTiff")
    if options != None:
        dest = driver.Create(path_export, width, height, nbands, dtype,
                             options)
    else:
        dest = driver.Create(path_export, width, height, nbands, dtype)

    # Write output raster
    for b in range(nbands):
        dest.GetRasterBand(b+1).WriteArray(raster[:, :, b])

        if nodata is not None:
            dest.GetRasterBand(b+1).SetNoDataValue(nodata)

        if color_table != None:
            dest.GetRasterBand(1).SetRasterColorTable(color_table)

    # Set transform and projection
    dest.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())

    # Close and save output raster dataset
    dest = None


def colortable(ctype):
    """
    Generates a gdal-ingestible color table for a set of pre-defined options.
    Can add your own colortable options. See https://gdal.org/doxygen/structGDALColorEntry.html
    and https://gis.stackexchange.com/questions/158195/python-gdal-create-geotiff-from-array-with-colormapping
    for guidance.

    Parameters
    ----------
    ctype : str
        Specifies the type of colortable to return. Choose from
        {'binary', 'skel', 'mask', 'tile', or 'GSW'}.

    Returns
    -------
    color_table : gdal.ColorTable()
        Color table that can be supplied to gdal when creating a raster.

    """

    color_table = gdal.ColorTable()

    if ctype == 'binary':
        # Some examples / last value is alpha (transparency).
        color_table.SetColorEntry(0, (0, 0, 0, 0))
        color_table.SetColorEntry(1, (255, 255, 255, 100))
    elif ctype == 'skel':
        color_table.SetColorEntry(0, (0, 0, 0, 0))
        color_table.SetColorEntry(1, (255, 0, 255, 100))
    elif ctype == 'mask':
        color_table.SetColorEntry(0, (0, 0, 0, 0))
        color_table.SetColorEntry(1, (0, 128, 0, 100))
    elif ctype == 'tile':
        color_table.SetColorEntry(0, (0, 0, 0, 0))
        color_table.SetColorEntry(1, (0, 0, 255, 100))
    elif ctype == 'GSW':
        color_table.SetColorEntry(0, (0, 0, 0, 0))
        color_table.SetColorEntry(1, (0, 0, 0, 0))
        color_table.SetColorEntry(2, (176, 224, 230, 100))

    return color_table


def shapely_list_to_geovectors(shplist, crs, path_export):
    """
    Exports a list of shapely geometries to a GIS-ingestible format.

    Parameters
    ----------
    shplist : list
        A list of shapely.geometry objects defining components of the mesh.
    crs : pyproj.CRS
        CRS object specifying the coordinate reference system of the geometries
        to export.
    path_export : str
        Path, including extension, where the geovector data should be written.
        Extensions can be either '.shp' or '.geojson'.

    Returns
    -------
    None.

    """
    gdf = gpd.GeoDataFrame(geometry=shplist)
    gdf.crs = crs
    gdf.to_file(path_export, driver=get_driver(path_export))


def write_linkdirs_geotiff(links, gdobj, path_export):
    """
    Creates a geotiff where links are colored according to their directionality.
    Pixels in each link are interpolated between 0 and 1 such that the upstream
    pixel is 0 and the downstream-most pixel is 1. In a GIS, color can then
    be set to visualize flow directionality.


    Parameters
    ----------
    links : dict
        Network links and associated properties.
    gdobj :  osgeo.gdal.Dataset
        GDAL object correspondng to the original mask from which links were
        derived.
    path_export : str
        Path, including .tif extension, where the directions geotiff is
        written.

    Returns
    -------
    None.

    """
    # Initialize plotting raster
    I = gdobj.ReadAsArray()
    I = np.ones((gdobj.RasterYSize, gdobj.RasterXSize), dtype=np.float32)*-1

    # Loop through links and store each pixel's interpolated value
    for lidcs in links['idx']:
        n = len(lidcs)
        vals = np.linspace(0, 1, n)
        rcidcs = np.unravel_index(lidcs, I.shape)
        I[rcidcs] = vals

    # Save the geotiff
    write_geotiff(I, gdobj.GetGeoTransform(), gdobj.GetProjection(), path_export, dtype=gdal.GDT_Float32, nodata=-1)

    return


def create_manual_dir_csv(path_csv):
    """
    Creates a .csv file for fixing links manually.

    Parameters
    ----------
    path_csv : str
        Path, including .csv extension, where the .csv is written.

    Returns
    -------
    None.

    """
    df = pd.DataFrame(columns=['link_id', 'usnode'])
    df.to_csv(path_csv, index=False)


def coords_from_geovector(path_geovector):
    """
    Retreives coordinates from a shapefile containing a LineString or Points.

    Parameters
    ----------
    path_geovector : str
        Path, including .shp extension, of the file containing coordinates.

    Returns
    -------
    coords : list of tuples
        Coordinates (x, y) of the vertices in the provided geovector.

    """
    xy_gdf = gpd.read_file(path_geovector)
    coords = []
    for i in xy_gdf.index:
        coords_obj = xy_gdf['geometry'][i].centroid.xy
        coords.append((coords_obj[0][0], coords_obj[1][0]))

    return coords


def coords_to_geovector(coords, epsg, path_export):
    """
    Exports coordinates to a Point shapefile.

    Parameters
    ----------
    coords : list-like of list-likes
        List or tuple of (x, y) coordinates to export.
    epsg : int
        EPSG code of the coordinate reference system of the coordinates.
    path_export : str
        Path with .shp extension where the shapefile should be saved.

    Returns
    -------
    None.

    """
    # TODO: This should be replaced by a geodataframe creation, but no use cases
    # yet...


    all_coords = []
    for c in coords:
        pt = ogr.Geometry(type=ogr.wkbPoint)
        pt.AddPoint_2D(c[1], c[0])
        all_coords.append(pt)

    # Write the shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(path_export)

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

    return

def export_for_sword(links, gdobj, crs, paths, unit, metadata={}):
    """
    Exports a reaches and nodes shapefiles that are formatted
    consistently with the SWORD database to the best of RivGraph's ability.

    Currently the notion of directionality (upstream/downstream) is not
    included. An additional flag called dfir_set indicates if the exported
    data have considered flow directaionality. This will always be False
    until directionality is implemented into this function.

    Georeferenced fields are exported in WGS84 (EPSG:4326).
    """
    if unit != 'meter':
        raise TypeError('Reproject your mask to a meters-based CRS for SWORD exports. Or raise an issue for RivGraph to handle more unit types.')

    node_spacing = 200 # meters, a SWORD default

    # Initialize dictionary to store all the segment nodes' properties, including their geometries
    segprops = ['geometry', 'x', 'y', 'node_id_rg', 'node_len', 'reach_id_R', 'width', 'width_var', 'max_width', 'sinuosity']
    segs = {prop: [] for prop in segprops}
    reachprops = ['geometry', 'x', 'y', 'reach_id_R', 'reach_len', 'n_nodes', 'width', 'width_var', 'max_width', 'rch_id_up', 'rch_id_dn', 'n_rch_up', 'n_rch_down', 'fdir_set', 'conn_reach']
    reaches = {prop:[] for prop in reachprops}

    # Define attributes that RG will not compute (some of these are computable by RG) to ensure matching with existing SWORD structure.
    sword_empty_segprops = ['node_id', 'reach_id', 'wse', 'wse_var', 'facc', 'n_chan_max', 'n_chan_mod', 'obstr_type', 'grod_id', 'hfalls_id',
                    'dist_out', 'lakeflag', 'manual_add', 'meand_len', 'type', 'river_name', 'edit_flag', 'trib_flag',
                    'path_freq', 'path_order', 'path_segs', 'main_side', 'strm_order', 'end_reach', 'network']
    sword_empty_reachprops = ['wse', 'wse_var', 'facc', 'n_chan_max', 'n_chan_mod', 'obstr_type', 'grod_id', 'hfalls_id',
                            'dist_out', 'lakeflag', 'swot_orbit', 'swot_obs',
                            'type', 'river_name', 'edit_flag', 'trib_flag', 'path_freq', 'path_order', 'path_segs',
                            'main_side', 'strm_order', 'end_reach', 'network']

    # Make nodes for each link that are at least node_spacing apart
    # SWORD calls these nodes, but RG uses nodes for something different so here we call them segs/segments
    for i in range(len(links['idx'])):
        this_idx = links['idx'][i]
        this_x, this_y = gu.idx_to_coords(this_idx, gdobj)
        this_s, _ = cu.s_ds(this_x, this_y)

        # Segment the link, storing the indices along it
        segments = []
        start_idx = 0
        for j in range(1, len(this_s)):
            if this_s[j] - this_s[start_idx] >= node_spacing:
                segments.append((start_idx, j))
                start_idx = j
        if len(segments) == 0: # If the segment is too short, use the whole thing
            segments = [[0, len(this_idx)-1]]

        # Find a central vertex to use as the representative SWORD node (this defines the coordinate of the SWORD node)
        for seg in segments:
            seg_idx = int(sum(seg)/2)
            segs['geometry'].append(Point(this_x[seg_idx], this_y[seg_idx]))
            lon, lat = gu.transform_coords(this_x[seg_idx], this_y[seg_idx], crs.to_epsg(), 4326)
            segs['x'].append(lon)
            segs['y'].append(lat)
            segs['node_id_rg'].append(this_idx[seg_idx])
            segs['node_len'].append(this_s[seg[1]] - this_s[seg[0]])
            segs['reach_id_R'].append(links['id'][i])
            seg_widths = links['wid_pix'][i][seg[0]:seg[1]]
            segs['width'].append(np.mean(seg_widths))
            segs['width_var'].append(np.var(seg_widths))
            segs['max_width'].append(np.max(seg_widths))
            segs['sinuosity'].append(max(0, segs['node_len'][-1] / np.hypot(this_x[seg[1]] - this_x[seg[0]], this_y[seg[1]] - this_y[seg[0]])))

        # Handle the SWORD reaches
        reaches['geometry'].append(LineString(zip(this_x, this_y)))
        reaches['reach_id_R'].append(links['id'][i])
        reaches['reach_len'].append(links['len'][i])
        reaches['n_nodes'].append(len(segments))
        reaches['width'].append(links['wid_adj'][i])
        reaches['width_var'].append(np.var(links['wid_pix'][i]))
        reaches['max_width'].append(max(links['wid_pix'][i]))
        
        # Need a representative x, y (lon, lat) for each reach; use midpoint along reach
        # Must be in WGS84 (EPSG:4326)
        line = reaches['geometry'][-1]
        midpoint = line.interpolate(0.5 * line.length)
        lon, lat = gu.transform_coords(midpoint.x, midpoint.y, crs.to_epsg(), 4326)
        reaches['x'].append(lon)
        reaches['y'].append(lat)

        # Handle directionality if set
        if 'certain' not in links.keys():
            reaches['n_rch_up'].append(links['conn'][i][0])
            reaches['n_rch_down'].append('NA')
            reaches['rch_id_up'].append(' ')
            reaches['reach_id_down'].append(' ')
            reaches['fdir_set'].append(False)
        else:
            # Get upstream/downstream reaches
            conn_nodes = links['conn'][i]
            us_node, ds_node = conn_nodes[0], conn_nodes[1]
            conn_links = links['link_conn'][i]
            us_links, ds_links = [], []
            for cl in conn_links:
                this_link_idx = links['id'].index(cl)
                this_conn_nodes = links['conn'][this_link_idx]
                if this_conn_nodes[-1] == conn_nodes[0]: # Upstream links
                    us_links.append(links['id'][this_link_idx])
                elif this_conn_nodes[0] == conn_nodes[-1]: # Downstream links
                    ds_links.append(links['id'][this_link_idx])
            reaches['n_rch_up'].append(len(us_links))
            reaches['n_rch_down'].append(len(ds_links))
            reaches['rch_id_up'].append(' '.join([str(s) for s in us_links]))
            reaches['rch_id_dn'].append(' '.join([str(s) for s in ds_links]))
            reaches['fdir_set'].append(True)

        reaches['conn_reach'].append(', '.join(str(x) for x in links['link_conn'][i]))

    # Convert to GeoDataFrames and write to disk
    sword_nodes = gpd.GeoDataFrame(segs, crs=crs)
    sword_reaches = gpd.GeoDataFrame(reaches, crs=crs)

    # SWORD expects EPSG:4326
    sword_nodes = sword_nodes.to_crs(epsg=4326)
    sword_reaches = sword_reaches.to_crs(epsg=4326)

    # Append metadata
    if metadata:
        for k in metadata.keys():
            sword_reaches[k] = metadata[k]
            sword_nodes[k] = metadata[k]

    # Add all the empty (non-RG-computed but exist in SWORD properties
    for segempty in sword_empty_segprops:
        sword_nodes[segempty] = None
    for reachempty in sword_empty_reachprops:
        sword_reaches[reachempty] = None

    sword_nodes.to_file(paths['nodes_sword'])
    sword_reaches.to_file(paths['reaches_sword'])
    return
