# -*- coding: utf-8 -*-
"""
delta_utils
===========

A collection of functions for pruning a delta channel network.

"""
from loguru import logger
import geopandas as gpd
from pyproj.crs import CRS
import numpy as np
import networkx as nx
import rivgraph.geo_utils as gu
import rivgraph.ln_utils as lnu


def prune_delta(links, nodes, shoreline_shp, inlets_shp, gdobj,
                prune_less):
    """
    Prune a delta network.

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
        gdal object corresponding to the georeferenced input binary channel
        mask
    prune_less : bool
        Boolean to prune the network less... the first spur removal can
        create problems, especially for very small/simple networks.

    Returns
    -------
    links : dict
        updated links dictionary
    nodes : dict
        updated nodes dictionary

    """
    # Get inlet nodes
    nodes = find_inlet_nodes(nodes, inlets_shp, gdobj)

    if prune_less is False:
        # Remove spurs from network (this includes valid inlets and outlets)
        links, nodes = lnu.remove_all_spurs(links, nodes,
                                            dontremove=list(nodes['inlets']))

    # Clip the network with a shoreline polyline, adding outlet nodes
    links, nodes = clip_by_shoreline(links, nodes, shoreline_shp, gdobj)

    # Remove spurs from network (this includes valid inlets and outlets)
    links, nodes = lnu.remove_all_spurs(links, nodes,
                                        dontremove=list(nodes['inlets']) +
                                        list(nodes['outlets']))

    # Remove sets of links that are disconnected from inlets/outlets except for
    # a single bridge link (effectively re-pruning the network)
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
    Load inlets from a shapefile.

    Loads the user-defined inlet nodes point shapefile and uses it to identify
    the inlet nodes within the network.

    Parameters
    ----------
    links : dict
        stores the network's links and their properties
    inlets_shp : str
        path to the shapefile of inlet locations (point shapefile)
    gdobj : osgeo.gdal.Dataset
        gdal object corresponding to the georeferenced input binary channel
        mask

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
        logger.info('Provided inlet points file does not have the same CRS as provided mask. Reprojecting.')

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
        nodes dictionary representing network clipped by the shoreline.
        'outlets' has been added to the dictionary to store a list of outlet
        node ids

    """
    # Get links as geopandas dataframe
    links_gdf = lnu.links_to_gpd(links, gdobj)

    # Load the coastline as a geopandas object
    shore_gdf = gpd.read_file(shoreline_shp)

    # Enusre we have consistent CRS before intersecting
    if links_gdf.crs != shore_gdf.crs:
        shore_gdf = shore_gdf.to_crs(links_gdf.crs)
        logger.info('Provided shoreline file does not have the same CRS as provided mask. Reprojecting.')


    # Remove the links beyond the shoreline
    # Intersect links with shoreline
    shore_int = gpd.sjoin(links_gdf, shore_gdf, op='intersects',
                          lsuffix='left')

    # Get ids of intersecting links
    leftkey = [lid for lid in shore_int.columns if 'id' in lid.lower() and 'left' in lid.lower()][0]
    cut_link_ids = shore_int[leftkey].values

    # Loop through each cut link and truncate it near the intersection point;
    # add endpoint nodes; adjust connectivities
    newlink_ids = []
    for clid in cut_link_ids:

        # Remove the pixel that represents the intersection between the outlet
        # links and the shoreline. Gotta find it first.
        lidx = links['id'].index(clid)
        idcs = links['idx'][lidx][:]
        coords = gu.idx_to_coords(idcs, gdobj)

        # Intersection coordinates
        int_points = links_gdf['geometry'][list(links_gdf['id'].values).index(clid)].intersection(shore_gdf['geometry'][0])
        if int_points.type == 'Point':
            dists = np.sqrt((coords[0] - int_points.xy[0][0])**2 + (coords[1] - int_points.xy[1][0])**2)
            min_idx = np.argmin(dists)
            max_idx = min_idx
        elif int_points.type == 'MultiPoint':  # Handle multiple intersections by finding the first and last one so we can remove that section of the link
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
            if nflatidx == idcs[0]:  # Link corresponds to beginning of idcs -> break (minus one to ensure the break is true)
                if min_idx == 0:
                    newlink_idcs = []
                else:
                    newlink_idcs = idcs[0:min_idx - 1]

            elif nflatidx == idcs[-1]:  # Link corresponds to break (plus one to ensure the break is true) -> end of idcs
                if max_idx == 0:
                    newlink_idcs = idcs[2:]
                elif max_idx == len(idcs) - 1:
                    newlink_idcs = []
                else:
                    newlink_idcs = idcs[max_idx + 1:]
            else:
                RuntimeError('Check link-breaking.')

            # Only add new link if it contains indices
            if len(newlink_idcs) > 0:
                links, nodes = lnu.add_link(links, nodes, newlink_idcs)
                newlink_ids.append(links['id'][-1])

        # Now delete the old link
        links, nodes = lnu.delete_link(links, nodes, clid)

    # Now that the links have been clipped, remove the links that are not
    # part of the delta network

    # Use networkx graph to determine which links to keep
    G = nx.MultiGraph()
    G.add_nodes_from(nodes['id'])
    for lk, lc in zip(links['id'], links['conn']):
        G.add_edge(lc[0], lc[1], key=lk)

    # Find the network containing the inlet(s)
    main_net = nx.node_connected_component(G, nodes['inlets'][0])

    # Ensure all inlets are contained in this network
    for nid in nodes['inlets']:
        if len(main_net - nx.node_connected_component(G, nid)) > 0:
            logger.info('Not all inlets found in main connected component.')

    # Remove all nodes not in the main network
    remove_nodes = [n for n in G.nodes if n not in main_net]
    for rn in remove_nodes:
        G.remove_node(rn)

    # Get ids of the remaining links
    link_ids = [e[2] for e in G.edges]

    # Get ids to remove from network
    remove_links = [l for l in links['id'] if l not in link_ids]

    # Remove the links
    for rl in remove_links:
        links, nodes = lnu.delete_link(links, nodes, rl)

    # Identify the outlet nodes and add to nodes dictionary
    outlets = [nid for nid, ncon in zip(nodes['id'], nodes['conn']) if len(ncon)==1 and ncon[0] in newlink_ids]
    nodes['outlets'] = outlets

    return links, nodes
