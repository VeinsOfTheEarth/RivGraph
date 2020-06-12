# -*- coding: utf-8 -*-
"""
delta_utils
===========

A collection of functions for pruning a delta channel network.

"""
import geopandas as gpd
from pyproj.crs import CRS
import numpy as np

import rivgraph.geo_utils as gu
import rivgraph.ln_utils as lnu
import rivgraph.im_utils as iu

#ir.load_network()
#links = ir.links.copy()
#nodes = ir.nodes.copy()
#shoreline_shp = ir.paths['shoreline']
#inlets_shp = ir.paths['inlet_nodes']
#gdobj = ir.gdobj

def prune_delta(links, nodes, shoreline_shp, inlets_shp, gdobj):
    """
    Clips a delta channel network given an inlet and shoreline shapefile and
    removes spurious links.

    Parameters
    ----------
    links : dict
        stores the network's links and their properties
    nodes : dict
        stores the network's nodes and their properties
    shoreline_shp : str
        path to the shoreline shapefile (polyline)
    inlets_shp : str
        path to the shapefile of inlet locations (point shapefile)
    gdobj : osgeo.gdal.Dataset
        gdal object corresponding to the georeferenced input binary channel mask

    Returns
    -------
    links : dict
        updated links dictionary
    nodes : dict
        updated nodes dictionary
    """

    # Get inlet nodes
    nodes = find_inlet_nodes(nodes, inlets_shp, gdobj)

    # Remove spurs from network (this includes valid inlets and outlets)
    links, nodes = lnu.remove_all_spurs(links, nodes, dontremove=list(nodes['inlets']))

    # Clip the network with a shoreline polyline, adding outlet nodes
    links, nodes = clip_by_shoreline(links, nodes, shoreline_shp, gdobj)

    # Remove sets of links that are disconnected from inlets/outlets except for a single bridge link (effectively re-pruning the network)
    links, nodes = lnu.remove_disconnected_bridge_links(links, nodes)

    # # Add artificial nodes where necessary
    # links, nodes = lnu.add_artificial_nodes(links, nodes, gdobj)

    # Remove one-node links
    links, nodes = lnu.remove_single_pixel_links(links, nodes)

    # Find parallel links
    links, nodes = lnu.find_parallel_links(links, nodes)

    return links, nodes


def find_inlet_nodes(nodes, inlets_shp, gdobj):
    """
    Loads the user-defined inlet nodes point shapefile and uses it to identify
    the inlet nodes within the network.

    Parameters
    ----------
    links : dict
        stores the network's links and their properties
    inlets_shp : str
        path to the shapefile of inlet locations (point shapefile)
    gdobj : osgeo.gdal.Dataset
        gdal object corresponding to the georeferenced input binary channel mask

    Returns
    -------
    nodes : dict
        nodes dictionary with 'inlets' key containing list of inlet node ids

    """

    # Check that CRSs match; reproject inlet points if not
    inlets_gpd = gpd.read_file(inlets_shp)
    mask_crs = CRS(gdobj.GetProjection())
    if inlets_gpd.crs != mask_crs:
        inlets_gpd = inlets_gpd.to_crs(mask_crs)
        print('Provided inlet points file does not have the same CRS as provided mask. Reprojecting.')

    # Convert all nodes to xy coordinates for distance search
    nodes_xy = gu.idx_to_coords(nodes['idx'], gdobj)

    # Map provided inlet nodes to actual network nodes
    inlets = []
    for inlet_geom in inlets_gpd.geometry.values:
        # Distances between inlet node and all nodes in network
        xy = inlet_geom.xy
        dists = np.sqrt((xy[0][0]-nodes_xy[0])**2 + (xy[1][0]-nodes_xy[1])**2)
        inlets.append(nodes['id'][np.argmin(dists)])

    # Append inlets to nodes dict
    nodes['inlets'] = inlets

    return nodes


def clip_by_shoreline(links, nodes, shoreline_shp, gdobj):
    """
    Clips links by a provided shoreline shapefile. The largest network is
    presumed to be the delta network and is thus retained. The network should
    have been de-spurred before running this function.

    Parameters
    ----------
    links : dict
        stores the network's links and their properties
    nodes : dict
        stores the network's nodes and their properties
    shoreline_shp : str
        path to the shapefile of shoreline polyline
    gdobj : osgeo.gdal.Dataset
        gdal object corresponding to the georeferenced input binary channel mask

    Returns
    -------
    links : dict
        links dictionary representing network clipped by the shoreline
    nodes : dict
        nodes dictionary representing network clipped by the shoreline. 'outlets'
        has been added to the dictionary to store a list of outlet node ids


    """

    # Get links as geopandas dataframe
    links_gdf = lnu.links_to_gpd(links, gdobj)

    # Load the coastline as a geopandas object
    shore_gdf = gpd.read_file(shoreline_shp)

    # Enusre we have consistent CRS before intersecting
    if links_gdf.crs != shore_gdf.crs:
        shore_gdf = shore_gdf.to_crs(links_gdf.crs)
        print('Provided shoreline file does not have the same CRS as provided mask. Reprojecting.')


    ## Remove the links beyond the shoreline
    # Intersect links with shoreline
    shore_int = gpd.sjoin(links_gdf, shore_gdf, op='intersects', lsuffix='left')

    # Get ids of intersecting links
    leftkey = [lid for lid in shore_int.columns if 'id' in lid.lower() and 'left' in lid.lower()][0]
    cut_link_ids = shore_int[leftkey].values

    # Loop through each cut link and truncate it near the intersection point;
    # add endpoint nodes; adjust connectivities
    for clid in cut_link_ids:

        # Remove the pixel that represents the intersection between the outlet links
        # and the shoreline. Gotta find it first.
        lidx = links['id'].index(clid)
        idcs = links['idx'][lidx][:]
        coords = gu.idx_to_coords(idcs, gdobj)

        # Intersection coordinates
        int_points = links_gdf['geometry'][list(links_gdf['id'].values).index(clid)].intersection(shore_gdf['geometry'][0])

        if int_points.type == 'Point':
            dists = np.sqrt((coords[0] - int_points.xy[0][0])**2 + (coords[1] - int_points.xy[1][0])**2)
            min_idx = np.argmin(dists)
            max_idx = min_idx
        elif int_points.type == 'MultiPoint':         # Handle multiple intersections by finding the first and last one so we can remove that section of the link
            cutidcs = []
            for pt in int_points:
                # Find index of closest pixel
                dists = np.sqrt((coords[0] - pt.xy[0][0])**2 + (coords[1] - pt.xy[1][0])**2)
                cutidcs.append(np.argmin(dists))
            min_idx = min(cutidcs)
            max_idx = max(cutidcs)

        # Delete the intersected link and add two new links corresponding to the
        # two parts of the (now broken) intersected link
        # First add the two new links
        conn = links['conn'][lidx]

        for c in conn:
            nidx = nodes['id'].index(c)
            nflatidx = nodes['idx'][nidx]
            if nflatidx == idcs[0]: # Link corresponds to beginning of idcs -> break (minus one to ensure the break is true)
                if min_idx == 0:
                    newlink_idcs = []
                else:
                    newlink_idcs = idcs[0:min_idx - 1]

            elif nflatidx == idcs[-1]: # Link corresponds to break (plus one to ensure the break is true) -> end of idcs
                if max_idx == 0:
                    newlink_idcs = idcs[2:]
                elif max_idx == len(idcs) - 1:
                    newlink_idcs = []
                else:
                    newlink_idcs = idcs[max_idx + 1:]
            else:
                RuntimeError('Check link-breaking.')

            # Only add new link if it contains and indices
            if len(newlink_idcs) > 0:
                links, nodes = lnu.add_link(links, nodes, newlink_idcs)

        # Now delete the old link
        links, nodes = lnu.delete_link(links, nodes, clid)

    # Now that the links have been clipped, remove the links that are not
    # part of the delta network
    shape = (gdobj.RasterYSize, gdobj.RasterXSize)

    # Burn links to grid where value is link ID
    I = np.ones(shape, dtype=np.int64) * -1
    # 2-pixel links can be overwritten and disappear, so redo them at the end
    twopix = [lid for lid, idcs in zip(links['id'], links['idx']) if len(idcs) < 3]
    for lidx, lid in zip(links['idx'], links['id']):
        xy = np.unravel_index(lidx, shape)
        I[xy[0], xy[1]] = lid
    if len(twopix) > 0:
        for tpl in twopix:
            lindex = links['id'].index(tpl)
            lidx = links['idx'][lindex]
            xy = np.unravel_index(lidx, shape)
            I[xy[0], xy[1]] = tpl

    # Binarize
    I_bin = np.array(I>-1, dtype=np.bool)
    # Keep the blob that contains the inlet nodes
    # Get the pixel indices of the different connected blobs
    blobidcs = iu.blob_idcs(I_bin)
    # Find the blob that contains the inlets
    inlet_coords = []
    for i in nodes['inlets']:
        inlet_coords.append(nodes['idx'][nodes['id'].index(i)])
    i_contains_inlets = []
    for i, bi in enumerate(blobidcs):
        if set(inlet_coords).issubset(bi):
            i_contains_inlets.append(i)
    # Error checking
    if len(i_contains_inlets) != 1:
        raise RuntimeError('Inlets not contained in any portion of the skeleton.')

    # Keep only the pixels in the blob containing the inlets
    keeppix = np.unravel_index(list(blobidcs[i_contains_inlets[0]]), I_bin.shape)
    Itemp = np.zeros(I.shape, dtype=np.bool)
    Itemp[keeppix[0], keeppix[1]] = True
    I[~Itemp] = -1
    keep_ids = set(np.unique(I))
    unwanted_ids = [lid for lid in links['id'] if lid not in keep_ids]

    # Delete all the unwanted links
    for b in unwanted_ids:
        links, nodes = lnu.delete_link(links, nodes, b)

    # Store outlets in nodes dict
    outlets = [nid for nid, ncon in zip(nodes['id'], nodes['conn']) if len(ncon) == 1 and nid not in nodes['inlets']]
    nodes['outlets'] = outlets

    return links, nodes
