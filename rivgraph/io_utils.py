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
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

import rivgraph.geo_utils as gu
import rivgraph.rasters as rasters
from rivgraph.export_schema import (
    EXPORT_SCHEMA_VERSION,
    RG_LINK_RESERVED_INPUT_KEYS,
    RG_LINK_SCHEMA_COLUMNS,
    RG_NODE_RESERVED_INPUT_KEYS,
    RG_NODE_SCHEMA_COLUMNS,
    get_driver_for_path,
    ordered_export_columns,
)
from rivgraph.rivers import centerline_utils as cu
import rivgraph.ln_utils as lnu




def _warn_if_network_ids_not_finalized(links=None, nodes=None):
    """Warn once when exporting provisional network IDs."""
    if lnu.network_ids_are_finalized(links=links, nodes=nodes):
        return

    warnings.warn(
        'Exporting a network with provisional IDs. Call finalize_ids() after topology-changing operations and before exporting if you need deterministic, final IDs.',
        UserWarning,
        stacklevel=3,
    )


def _shapefile_export_warnings(gdf):
    """Return a concise warning message about likely lossy shapefile exports."""
    long_name_cols = [str(c) for c in gdf.columns if c != gdf.geometry.name and len(str(c)) > 10]

    long_value_cols = []
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        series = gdf[col]
        try:
            needs_truncation = series.astype(str).map(len).gt(254).any()
        except Exception:
            needs_truncation = False
        if needs_truncation:
            long_value_cols.append(str(col))

    parts = []
    if long_name_cols:
        parts.append(
            'field names longer than 10 characters will be shortened by the shapefile driver: ' +
            ', '.join(long_name_cols)
        )
    if long_value_cols:
        parts.append(
            'stringified attribute values longer than 254 characters may be truncated: ' +
            ', '.join(long_value_cols)
        )

    if not parts:
        return None

    return (
        'Exporting to ESRI Shapefile may be lossy due to format constraints; ' +
        '; '.join(parts) +
        '. Prefer GPKG when you need full attribute fidelity.'
    )


def _write_gdf(gdf, path_export, reproject=False):
    """Write a GeoDataFrame while surfacing concise, format-aware warnings."""
    gdf = _prepare_gdf_for_export(gdf, path_export, reproject=reproject)
    driver = get_driver(path_export)

    if driver != 'ESRI Shapefile':
        gdf.to_file(path_export, driver=driver)
        return

    message = _shapefile_export_warnings(gdf)
    if message is not None:
        warnings.warn(message, UserWarning, stacklevel=2)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Column names longer than 10 characters will be truncated when saved to ESRI Shapefile\.',
            category=UserWarning,
        )
        warnings.filterwarnings(
            'ignore',
            message='Normalized/laundered field name: .*',
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            'ignore',
            message=r"Value '.*' of field .* has been truncated to 254 characters\..*",
            category=RuntimeWarning,
        )
        gdf.to_file(path_export, driver=driver)


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
    """Return the geopandas/OGR driver implied by *path_file*."""
    return get_driver_for_path(path_file)


def _rg_io_type_from_flags(is_inlet, is_outlet):
    """Return a compact inlet/outlet classification label for exports."""
    if is_inlet and is_outlet:
        return 'both'
    if is_inlet:
        return 'inlet'
    if is_outlet:
        return 'outlet'
    return 'neither'


def _crs_is_epsg_4326(crs):
    """Return True when *crs* resolves to EPSG:4326 / OGC:CRS84-style lon/lat coordinates."""
    if crs is None:
        return False

    try:
        epsg = crs.to_epsg()
    except AttributeError:
        epsg = None
    if epsg == 4326:
        return True

    try:
        axis_info = getattr(crs, 'axis_info', None) or []
        axis_dirs = [getattr(axis, 'direction', '').lower() for axis in axis_info]
        if axis_dirs[:2] == ['east', 'north'] and getattr(crs, 'is_geographic', False):
            return True
    except Exception:
        pass

    return False


def _prepare_gdf_for_export(gdf, path_export, reproject=False):
    """Validate and optionally reproject a GeoDataFrame before writing."""
    driver = get_driver(path_export)

    if driver != 'GeoJSON':
        return gdf

    if gdf.crs is None:
        raise ValueError(
            'GeoJSON export requires a defined CRS. Supply data with an attached CRS '
            'or choose a format that preserves native projection metadata, such as GPKG.'
        )

    if _crs_is_epsg_4326(gdf.crs):
        return gdf

    if reproject is not True:
        raise ValueError(
            'GeoJSON export requires EPSG:4326 coordinates. The native CRS is not EPSG:4326; '
            'pass reproject=True to export this dataset as GeoJSON.'
        )

    return gdf.to_crs(epsg=4326)


def _scalarize_export_value(value):
    """Convert numpy scalars to native Python scalars for Fiona/pyogrio compatibility."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stringify_export_sequence(value):
    """Flatten a list-like export value to a stable comma-separated string."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, pd.Series):
        value = value.tolist()

    if isinstance(value, set):
        value = sorted(value)

    if isinstance(value, (list, tuple)):
        return ', '.join(str(_scalarize_export_value(v)) for v in value)

    return str(_scalarize_export_value(value))


def _coerce_export_value(value):
    """Prepare an attribute value for vector export without destroying scalar types."""
    value = _scalarize_export_value(value)

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return _stringify_export_sequence(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return str(value)


def _aligned_export_keys(data, *, exclude=None, id_key='id'):
    """Return data keys whose values align one-to-one with the exported features."""
    exclude = set() if exclude is None else set(exclude)
    if id_key not in data:
        return []

    n_items = len(data[id_key])
    keys = []
    for key in data.keys():
        if key in exclude:
            continue
        try:
            if len(data[key]) == n_items:
                keys.append(key)
        except TypeError:
            continue
    return keys


def _finalize_export_gdf(records, *, crs, canonical_columns, extra_columns):
    """Build a GeoDataFrame with stable RG export column ordering."""
    columns = ordered_export_columns(canonical_columns, extra_columns)
    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=crs)
    return gdf.loc[:, columns]


def nodes_to_geofile(nodes, dims, gt, crs, path_export, reproject=False):
    """
    Save network nodes to a georeferenced vector file using the canonical RG schema.

    Parameters
    ----------
    nodes : dict
        Network nodes and associated properties.
    dims : tuple
        (nrows, ncols) of the original mask from which nodes were derived.
    gt : tuple
        Geotransform tuple of the original mask from which nodes were derived.
    crs : pyproj.CRS
        CRS object specifying the coordinate reference system of the original
        mask from which nodes were derived.
    path_export : str
        Path, including extension, where to save the nodes export.
    reproject : bool, optional
        When exporting GeoJSON, reproject to EPSG:4326 if True. If False, GeoJSON
        export will fail unless the native CRS is already EPSG:4326.

    Returns
    -------
    None.

    """
    _warn_if_network_ids_not_finalized(nodes=nodes)
    nodexy = np.unravel_index(nodes['idx'], dims)
    x, y = gu.xy_to_coords(nodexy[1], nodexy[0], gt)
    inlet_nodes = set(nodes.get('inlets', []))
    outlet_nodes = set(nodes.get('outlets', []))

    records = []
    extra_keys = _aligned_export_keys(nodes, exclude=RG_NODE_RESERVED_INPUT_KEYS)
    for i, node_id in enumerate(nodes['id']):
        conn_links = nodes['conn'][i] if 'conn' in nodes else []
        is_inlet = node_id in inlet_nodes
        is_outlet = node_id in outlet_nodes

        row = {
            'id_node': _coerce_export_value(node_id),
            'idx_node': _coerce_export_value(nodes['idx'][i]),
            'id_links': _stringify_export_sequence(conn_links),
            'n_links': len(conn_links),
            'is_inlet': is_inlet,
            'is_outlet': is_outlet,
            'type_io': _rg_io_type_from_flags(is_inlet, is_outlet),
            'geometry': Point(x[i], y[i]),
        }
        for key in extra_keys:
            row[key] = _coerce_export_value(nodes[key][i])
        records.append(row)

    records = [dict(rec, schema_rg=EXPORT_SCHEMA_VERSION) for rec in records]
    gdf = _finalize_export_gdf(
        records,
        crs=crs,
        canonical_columns=(*RG_NODE_SCHEMA_COLUMNS, 'schema_rg'),
        extra_columns=extra_keys,
    )
    _write_gdf(gdf, path_export, reproject=reproject)


def links_to_geofile(links, dims, gt, crs, path_export, nodes=None, reproject=False):
    """
    Save network links to a georeferenced vector file using the canonical RG schema.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    dims : tuple
        (nrows, ncols) of the original mask from which links were derived.
    gt : tuple
        Geotransform tuple of the original mask from which links were derived.
    crs : pyproj.CRS
        CRS object specifying the coordinate reference system of the original
        mask from which links were derived.
    path_export : str
        Path, including extension, specifying where to save the links export.
    nodes : dict, optional
        Network nodes and associated properties. When provided, inlet/outlet
        flags are derived for each link in a direction-agnostic way.
    reproject : bool, optional
        When exporting GeoJSON, reproject to EPSG:4326 if True. If False, GeoJSON
        export will fail unless the native CRS is already EPSG:4326.

    Returns
    -------
    None.

    """
    _warn_if_network_ids_not_finalized(links=links, nodes=nodes)
    all_links = []
    for link in links['idx']:
        xy = np.unravel_index(link, dims)
        x, y = gu.xy_to_coords(xy[1], xy[0], gt)
        all_links.append(LineString(zip(x, y)))

    inlet_nodes = set()
    outlet_nodes = set()
    if nodes is not None:
        inlet_nodes = set(nodes.get('inlets', []))
        outlet_nodes = set(nodes.get('outlets', []))

    flow_dirs_available = 'certain' in links and len(links['certain']) == len(links['id'])
    extra_keys = _aligned_export_keys(links, exclude=RG_LINK_RESERVED_INPUT_KEYS)
    records = []
    for i, link_id in enumerate(links['id']):
        conn_nodes = links['conn'][i] if 'conn' in links else []
        is_inlet = any(node_id in inlet_nodes for node_id in conn_nodes)
        is_outlet = any(node_id in outlet_nodes for node_id in conn_nodes)

        if flow_dirs_available and bool(links['certain'][i]) and len(conn_nodes) >= 2:
            id_us_node, id_ds_node = conn_nodes[0], conn_nodes[1]
        else:
            id_us_node, id_ds_node = None, None

        row = {
            'id_link': _coerce_export_value(link_id),
            'idx_link': _coerce_export_value(links['idx'][i]),
            'id_nodes': _stringify_export_sequence(conn_nodes),
            'n_nodes': len(conn_nodes),
            'id_us_node': _coerce_export_value(id_us_node),
            'id_ds_node': _coerce_export_value(id_ds_node),
            'is_inlet': is_inlet,
            'is_outlet': is_outlet,
            'type_io': _rg_io_type_from_flags(is_inlet, is_outlet),
            'geometry': all_links[i],
        }
        for key in extra_keys:
            row[key] = _coerce_export_value(links[key][i])
        records.append(row)

    records = [dict(rec, schema_rg=EXPORT_SCHEMA_VERSION) for rec in records]
    gdf = _finalize_export_gdf(
        records,
        crs=crs,
        canonical_columns=(*RG_LINK_SCHEMA_COLUMNS, 'schema_rg'),
        extra_columns=extra_keys,
    )
    _write_gdf(gdf, path_export, reproject=reproject)


def centerline_to_geovector(cl, crs, path_export, reproject=False):
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
    _write_gdf(cl_df, path_export, reproject=reproject)


def write_geotiff(raster, gt, wkt, path_export, dtype='uint16',
                  options=['COMPRESS=LZW'], nbands=1, nodata=None,
                  color_table=None):
    """Writes a georeferenced raster to disk using Rasterio."""
    rasters.write_geotiff(
        raster,
        gt,
        wkt,
        path_export,
        dtype=dtype,
        options=options,
        nbands=nbands,
        nodata=nodata,
        color_table=color_table,
    )


def colortable(ctype):
    """Generates a color table for a set of pre-defined options."""
    color_table = rasters.ColorTable()

    if ctype == 'binary':
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


def shapely_list_to_geovectors(shplist, crs, path_export, reproject=False):
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
    _write_gdf(gdf, path_export, reproject=reproject)


def write_linkdirs_geotiff(links, imshape, gt, wkt, path_export):
    """
    Creates a geotiff where links are colored according to their directionality.
    Pixels in each link are interpolated between 0 and 1 such that the upstream
    pixel is 0 and the downstream-most pixel is 1. In a GIS, color can then
    be set to visualize flow directionality.


    Parameters
    ----------
    links : dict
        Network links and associated properties.
    imshape : tuple
        Shape of the source mask raster.
    gt : tuple
        Geotransform tuple of the source mask raster.
    wkt : str
        WKT representation of the source mask CRS.
    path_export : str
        Path, including .tif extension, where the directions geotiff is
        written.

    Returns
    -------
    None.

    """
    # Initialize plotting raster
    I = np.ones(imshape, dtype=np.float32) * -1

    # Loop through links and store each pixel's interpolated value
    for lidcs in links['idx']:
        n = len(lidcs)
        vals = np.linspace(0, 1, n)
        rcidcs = np.unravel_index(lidcs, I.shape)
        I[rcidcs] = vals

    # Save the geotiff
    write_geotiff(I, gt, wkt, path_export, dtype='float32', nodata=-1)

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


def coords_to_geovector(coords, epsg, path_export, reproject=False):
    """Exports coordinates to a Point geovector file."""
    geometry = [Point(c[0], c[1]) for c in coords]
    gdf = gpd.GeoDataFrame({'id': list(range(len(geometry)))}, geometry=geometry, crs=f'EPSG:{epsg}')
    _write_gdf(gdf, path_export, reproject=reproject)
    return


def _resolve_sword_flux_attr(links, flux_attr=None):
    """Return the link attribute to use for SWORD-exported fluxes."""
    if flux_attr is not None:
        if flux_attr not in links:
            raise KeyError(f"Requested flux attribute '{flux_attr}' was not found in links.")
        if len(links[flux_attr]) != len(links['id']):
            raise ValueError(f"Requested flux attribute '{flux_attr}' does not align with links['id'].")
        return flux_attr

    for candidate in ('flux_ss', 'flux'):
        if candidate in links and len(links[candidate]) == len(links['id']):
            return candidate

    return None


def build_sword_geodataframes(links, nodes, imshape, gt, crs, unit, metadata=None, flux_attr=None):
    """
    Build SWORD-style reaches and nodes GeoDataFrames from a RivGraph network.

    Directionality and fluxes are exported when available using RG-specific
    fields so they do not conflict with canonical SWORD attributes.
    """
    _warn_if_network_ids_not_finalized(links=links, nodes=nodes)
    if unit != 'meter':
        raise TypeError('Reproject your mask to a meters-based CRS for SWORD exports. Or raise an issue for RivGraph to handle more unit types.')

    metadata = {} if metadata is None else dict(metadata)
    node_spacing = 200  # meters, a SWORD default

    # Initialize dictionary to store all the segment nodes' properties, including their geometries
    segprops = ['geometry', 'x', 'y', 'node_id_rg', 'node_len', 'reach_id_R', 'width', 'width_var', 'max_width', 'sinuosity',
                'fdir_set', 'rg_flux']
    segs = {prop: [] for prop in segprops}
    reachprops = ['geometry', 'x', 'y', 'reach_id_R', 'reach_len', 'n_nodes', 'width', 'width_var', 'max_width',
                  'rch_id_up', 'rch_id_dn', 'n_rch_up', 'n_rch_down', 'fdir_set', 'conn_reach',
                  'rg_us_nd', 'rg_ds_nd', 'rg_inlet', 'rg_outlet', 'rg_flux', 'rg_outflx']
    reaches = {prop: [] for prop in reachprops}

    # Define attributes that RG will not compute (some of these are computable by RG) to ensure matching with existing SWORD structure.
    sword_empty_segprops = ['node_id', 'reach_id', 'wse', 'wse_var', 'facc', 'n_chan_max', 'n_chan_mod', 'obstr_type', 'grod_id', 'hfalls_id',
                    'dist_out', 'lakeflag', 'manual_add', 'meand_len', 'type', 'river_name', 'edit_flag', 'trib_flag',
                    'path_freq', 'path_order', 'path_segs', 'main_side', 'strm_order', 'end_reach', 'network']
    sword_empty_reachprops = ['wse', 'wse_var', 'facc', 'n_chan_max', 'n_chan_mod', 'obstr_type', 'grod_id', 'hfalls_id',
                            'dist_out', 'lakeflag', 'swot_orbit', 'swot_obs',
                            'type', 'river_name', 'edit_flag', 'trib_flag', 'path_freq', 'path_order', 'path_segs',
                            'main_side', 'strm_order', 'end_reach', 'network']

    node_id_to_index = {nid: i for i, nid in enumerate(nodes['id'])}
    link_id_to_index = {lid: i for i, lid in enumerate(links['id'])}
    flow_dirs_available = 'certain' in links and len(links['certain']) == len(links['id'])
    flux_attr = _resolve_sword_flux_attr(links, flux_attr=flux_attr)

    # Make nodes for each link that are at least node_spacing apart
    # SWORD calls these nodes, but RG uses nodes for something different so here we call them segs/segments
    for i in range(len(links['idx'])):
        this_idx = links['idx'][i]
        this_x, this_y = gu.idx_to_coords(this_idx, imshape, gt)
        this_s, _ = cu.s_ds(this_x, this_y)

        link_id = links['id'][i]
        link_flux = None if flux_attr is None else float(links[flux_attr][i])
        link_is_directed = bool(links['certain'][i]) if flow_dirs_available else False

        if link_is_directed:
            us_node, ds_node = links['conn'][i]
            outlet_reach = ds_node in nodes.get('outlets', [])
            inlet_reach = us_node in nodes.get('inlets', [])
        else:
            us_node, ds_node = None, None
            outlet_reach = False
            inlet_reach = False

        # Segment the link, storing the indices along it
        segments = []
        start_idx = 0
        for j in range(1, len(this_s)):
            if this_s[j] - this_s[start_idx] >= node_spacing:
                segments.append((start_idx, j))
                start_idx = j
        if len(segments) == 0:  # If the segment is too short, use the whole thing
            segments = [(0, len(this_idx)-1)]

        # Find a central vertex to use as the representative SWORD node (this defines the coordinate of the SWORD node)
        for seg in segments:
            seg_idx = int(sum(seg)/2)
            segs['geometry'].append(Point(this_x[seg_idx], this_y[seg_idx]))
            lon, lat = gu.transform_coords(this_x[seg_idx], this_y[seg_idx], crs.to_epsg(), 4326)
            segs['x'].append(lon)
            segs['y'].append(lat)
            segs['node_id_rg'].append(this_idx[seg_idx])
            segs['node_len'].append(this_s[seg[1]] - this_s[seg[0]])
            segs['reach_id_R'].append(link_id)
            seg_widths = links['wid_pix'][i][seg[0]:seg[1]]
            segs['width'].append(np.mean(seg_widths))
            segs['width_var'].append(np.var(seg_widths))
            segs['max_width'].append(np.max(seg_widths))
            segs['sinuosity'].append(max(0, segs['node_len'][-1] / np.hypot(this_x[seg[1]] - this_x[seg[0]], this_y[seg[1]] - this_y[seg[0]])))
            segs['fdir_set'].append(link_is_directed)
            segs['rg_flux'].append(link_flux)

        # Handle the SWORD reaches
        reaches['geometry'].append(LineString(zip(this_x, this_y)))
        reaches['reach_id_R'].append(link_id)
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

        # Always compute connected reaches.
        conn_reach = []
        for cn in links['conn'][i]:
            for connected_link in nodes['conn'][node_id_to_index[cn]]:
                if connected_link != link_id and connected_link not in conn_reach:
                    conn_reach.append(connected_link)

        if link_is_directed:
            us_links, ds_links = [], []
            for cl in nodes['conn'][node_id_to_index[us_node]]:
                if cl == link_id:
                    continue
                this_link_idx = link_id_to_index[cl]
                if bool(links['certain'][this_link_idx]) and links['conn'][this_link_idx][1] == us_node:
                    us_links.append(cl)
            for cl in nodes['conn'][node_id_to_index[ds_node]]:
                if cl == link_id:
                    continue
                this_link_idx = link_id_to_index[cl]
                if bool(links['certain'][this_link_idx]) and links['conn'][this_link_idx][0] == ds_node:
                    ds_links.append(cl)

            reaches['n_rch_up'].append(len(us_links))
            reaches['n_rch_down'].append(len(ds_links))
            reaches['rch_id_up'].append(' '.join([str(s) for s in us_links]))
            reaches['rch_id_dn'].append(' '.join([str(s) for s in ds_links]))
            reaches['fdir_set'].append(True)
            reaches['rg_us_nd'].append(us_node)
            reaches['rg_ds_nd'].append(ds_node)
            reaches['rg_inlet'].append(inlet_reach)
            reaches['rg_outlet'].append(outlet_reach)
        else:
            reaches['n_rch_up'].append(0)
            reaches['n_rch_down'].append(0)
            reaches['rch_id_up'].append('')
            reaches['rch_id_dn'].append('')
            reaches['fdir_set'].append(False)
            reaches['rg_us_nd'].append(None)
            reaches['rg_ds_nd'].append(None)
            reaches['rg_inlet'].append(False)
            reaches['rg_outlet'].append(False)

        reaches['rg_flux'].append(link_flux)
        reaches['rg_outflx'].append(link_flux if (link_flux is not None and outlet_reach) else None)
        reaches['conn_reach'].append(', '.join(str(x) for x in conn_reach))

    # Convert to GeoDataFrames and write to disk
    sword_nodes = gpd.GeoDataFrame(segs, crs=crs)
    sword_reaches = gpd.GeoDataFrame(reaches, crs=crs)

    # SWORD expects EPSG:4326
    sword_nodes = sword_nodes.to_crs(epsg=4326)
    sword_reaches = sword_reaches.to_crs(epsg=4326)

    # Add all the empty (non-RG-computed but existing SWORD properties.
    for segempty in sword_empty_segprops:
        sword_nodes[segempty] = None
    for reachempty in sword_empty_reachprops:
        sword_reaches[reachempty] = None

    # Append metadata last so user-provided values can intentionally override
    # placeholder SWORD fields such as `network`.
    if metadata:
        for k in metadata.keys():
            sword_reaches[k] = metadata[k]
            sword_nodes[k] = metadata[k]

    return sword_nodes, sword_reaches


def export_for_sword(links, nodes, imshape, gt, crs, paths, unit, metadata=None, flux_attr=None):
    """
    Export SWORD-style reaches and nodes files from a RivGraph network.

    The export always writes georeferenced fields in EPSG:4326. When flow
    directions have been assigned, connectivity is written using the standard
    SWORD upstream/downstream reach fields. When a link-level flux field is
    available (or explicitly requested), it is exported via RG-specific
    `rg_*` attributes so the core SWORD schema remains intact.
    """
    sword_nodes, sword_reaches = build_sword_geodataframes(
        links,
        nodes,
        imshape,
        gt,
        crs,
        unit,
        metadata=metadata,
        flux_attr=flux_attr,
    )

    _write_gdf(sword_nodes, paths['nodes_sword'])
    _write_gdf(sword_reaches, paths['reaches_sword'])
    return
