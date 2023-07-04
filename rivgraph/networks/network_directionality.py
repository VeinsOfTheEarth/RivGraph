# -*- coding: utf-8 -*-
"""
river_directionality
====================

Created on Tue Nov  6 14:31:01 2018

@author: Jon
"""

import os
import itertools

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Polygon, Point
from tqdm import tqdm

import rivgraph.io_utils as io
import rivgraph.ln_utils as lnu
import rivgraph.geo_utils as gu
import rivgraph.directionality as dy


def set_directionality(links, nodes, Imask, exit_sides, gt, meshlines,
                       meshpolys, Idt, pixlen, manual_set_csv):
    """
    Set direction of each link.

    This function sets the direction of each link within the network. It
    calls a number of helping functions and uses a somewhat-complicated logic
    to achieve this. The algorithms and logic is described in this open-access
    paper: https://esurf.copernicus.org/articles/8/87/2020/esurf-8-87-2020.pdf

    Every time this is run, all directionality information is reset and
    recomputed. This includes checking for manually set links via the provided
    csv.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    Imask : np.array
        Binary mask of the network.
    exit_sides : str
        Two-character string of cardinal directions denoting the upstream and
        downsteram sides of the image that the network intersects (e.g. 'SW').
    gt : tuple
        gdal-type GeoTransform of the original binary mask.
    meshlines : list
        List of shapely.geometry.LineStrings that define the valleyline mesh.
    meshpolys : list
        List of shapely.geometry.Polygons that define the valleyline mesh.
    Idt : np.array()
        Distance transform of Imask.
    pixlen : float
        Length resolution of each pixel.
    manual_set_csv : str, optional
        Path to a user-provided csv file of known link directions. The default
        is None.

    Returns
    -------
    links : dict
        Network links and associated properties with all directions set.
    nodes : dict
        Network nodes and associated properties with all directions set.

    """
    imshape = Imask.shape

    # Add fields to links dict for tracking and setting directionality
    links, nodes = dy.add_directionality_trackers(links, nodes, 'river')

    # If a manual fix csv has been provided, set those links first
    links, nodes = dy.dir_set_manually(links, nodes, manual_set_csv)

    # Append morphological information used to set directionality to links dict
    print("Directional info...")
    links, nodes = directional_info(links, nodes, Imask, pixlen, exit_sides,
                                    gt, meshlines, meshpolys, Idt)

    # Begin setting link directionality
    # First, set inlet/outlet directions as they are always 100% accurate
    print("Inlet/outlet directions...")
    links, nodes = dy.set_inletoutlet(links, nodes)

    # the actual tricky directions
    links, nodes = set_inexact(links, nodes, imshape)

    # At this point, if any links remain unset, they are just set randomly
    if np.sum(links['certain']) != len(links['id']):
        print('{} links were randomly set.'.format(len(links['id']) -
                                                   np.sum(links['certain'])))
        links["certain_alg"][links['certain'] == 0] = dy.algmap("random")
        # random ones need to be set to certain for cycle resetting to work
        links['certain'][:] = 1

    # Check for and try to fix cycles in the graph
    links, nodes, cantfix_cyclelinks, cantfix_cyclenodes = fix_river_cycles(
        links, nodes, imshape, skip_threshold=200,
    )
    links["cycles"] = cantfix_cyclelinks
    nodes["cycles"] = cantfix_cyclenodes

    # Check for sources or sinks within the graph
    cont_violators = dy.check_continuity(links, nodes)
    nodes["continuity_violated"] = cont_violators

    # Summary of problems:
    manual_fix = 0
    if len(cantfix_cyclelinks) > 0:
        nc = sum([len(lc) for lc in cantfix_cyclelinks])
        print(f'Could not fix {nc} cycle links: {cantfix_cyclelinks}')
        manual_fix = 1
    else:
        print('All cycles were resolved.')
    if len(cont_violators) > 0:
        print(f'Continuity violated at {len(cont_violators)} nodes: {cont_violators}')
        manual_fix = 1

    # Create a csv to store manual edits to directionality if does not exist
    if manual_fix == 1:
        if os.path.isfile(manual_set_csv) is False:
            io.create_manual_dir_csv(manual_set_csv)
            print('A .csv file for manual fixes to link directions at {}.'.format(manual_set_csv))
        else:
            print('Use the csv file at {} to manually fix link directions.'.format(manual_set_csv))

    return links, nodes


def directional_info(links, nodes, Imask, pixlen, exit_sides, gt, meshlines,
                     meshpolys, Idt):
    """
    Compute information for link direction setting.

    Computes all the information required for link directions to be set for
    a river channel network.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    Imask : np.array
        Binary mask of the network.
    pixlen : float
        Length resolution of each pixel.
    exit_sides : str
        Two-character string of cardinal directions denoting the upstream and
        downsteram sides of the image that the network intersects (e.g. 'SW').
    gt : tuple
        gdal-type GeoTransform of the original binary mask.
    meshlines : list
        List of shapely.geometry.LineStrings that define the valleyline mesh.
    meshpolys : list
        List of shapely.geometry.Polygons that define the valleyline mesh.
    Idt : np.array()
        Distance transform of Imask.

    Returns
    -------
    links : dict
        Network links and associated properties including directional info.
    nodes : dict
        Network nodes and associated properties including directional info.

    """
    # Append pixel-based widths to links
    if 'wid_pix' not in links.keys():
        links = lnu.link_widths_and_lengths(links, Idt)

    # Compute all the information
    # links = dir_centerline(links, nodes, meshpolys, meshlines, Imask, gt,
    #                        pixlen)
    # print("Direction link width...")
    # links = dir_link_widths(links)

    #print("Direction network bridges...")
    #links, nodes = dy.dir_bridges(links, nodes)

    #for inl in tqdm(nodes['inlets'], 'Main channel direction'):
    #    links, nodes = dy.dir_main_channel(links, nodes, inlet=inl)
    #links, nodes = dy.dir_main_channel(links, nodes)
    return links, nodes


def fix_river_cycles(links, nodes, imshape, skip_threshold=200):
    """
    Attempt to resolve cycles in the network.

    Attempts to resolve all cycles within the river network. This function
    is essentially a wrapper for :func:`fix_river_cycle`, which is where the
    heavy lifting is actually done. This function finds cycles, calls
    :func:`fix_river_cycle` on each one, then aggregates the results.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with all possible cycles fixed.
    nodes : dict
        Network nodes and associated properties with all possible cycles fixed.
    cantfix_links : list of lists
        Contains link ids of unresolvable cycles. Length is equal to number of
        unresolvable cycles.
    cantfix_nodes : TYPE
        Contains node ids of unresolvable cycles. Length is equal to number of
        unresolvable cycles.

    """
    # Create networkx graph object
    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc in links['conn']:
        G.add_edge(lc[0], lc[1])

    # Check for cycles
    cantfix_nodes = []
    cantfix_links = []
    if nx.is_directed_acyclic_graph(G) is not True:

        # Get list of cycles to fix
        c_nodes, c_links = dy.get_cycles(links, nodes)
        # Remove any cycles that are subsets of larger cycles
        isin = np.empty((len(c_links), 1))
        isin[:] = np.nan
        for icn, cn in enumerate(c_nodes):
            for icn2, cn2 in enumerate(c_nodes):
                if cn2 == cn:
                    continue
                elif len(set(cn) - set(cn2)) == 0:
                    isin[icn] = icn2
                    break
        cfix_nodes = [cn for icn, cn in enumerate(c_nodes)
                      if np.isnan(isin[icn][0])]
        cfix_links = [cl for icl, cl in enumerate(c_links)
                      if np.isnan(isin[icl][0])]

        # start with smallest and respect skip threshold
        cfixnlsorted = zip(*sorted(zip(cfix_nodes, cfix_links),
                                   key=lambda x: len(x[1])))
        cfn_all, cfl_all = [list(c) for c in cfixnlsorted]
        cfix_nodes, cfix_links = [l[:min(skip_threshold, len(l))] for l in (cfn_all, cfl_all)]

        print('Attempting to fix {} cycles.'.format(len(cfix_nodes)))
        reset_functions = [
            re_set_linkdirs_flow_direction,
            re_set_linkdirs_darea_gradient,
            re_set_linkdirs_best_guess,
            re_set_linkdirs_main_channel_darea_grad,
            re_set_linkdirs_dir_shortest_paths,
        ]
        # Try to fix all the cycles
        for cnodes, clinks in tqdm(zip(cfix_nodes, cfix_links),
                                "Fixing cycles", total=len(cfix_nodes)):
            for func in reset_functions:
                links, nodes, fixed = fix_river_cycle(
                    links, nodes, clinks, cnodes, imshape, reset_function=func)
                if fixed != 0:
                    print(f"Fixed {clinks} with {func}")
                    break
            if fixed == 0:
                # try inlet outlet fix, update cycle in case the fix attempts have created adjacent cycles
                ncnd, nclk = dy.get_cycles(links, nodes, checknode=cnodes)
                links, nodes, fixed, reason = fix_cycle_inlet_outlet(links, nodes, nclk[0], ncnd[0])
            if fixed == 0:
                print(f"Cant fix cycle {clinks}, reason: {reason}")
                cantfix_nodes.append(cnodes)
                cantfix_links.append(clinks)
            # continues cycle checking in case any new ones appear
            ctfn, ctfl = dy.get_cycles(links, nodes)
            ctfl = [l for c in ctfl for l in c]
            print(f"Cycle (link) count: {len(ctfn)} {len(ctfl)}")
        # ignored cycles are not fixed
        cantfix_nodes.extend(cfn_all[min(skip_threshold, len(cfn_all)):])
        cantfix_links.extend(cfl_all[min(skip_threshold, len(cfl_all)):])
        # check cycles again as new ones might have been created
        cantfix_nodes, cantfix_links = dy.get_cycles(links, nodes)
        if cantfix_links:
            print(f"Failed to fix {len(cantfix_links)} cycles.")

    return links, nodes, cantfix_links, cantfix_nodes


def re_set_linkdirs_flow_direction(links, nodes, imshape):
    """
    Reset link directions.

    Resets the link directions for a braided river channel network. This
    function is called to reset directions of links that belong to a cycle.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with the directions re-set.
    nodes : dict
        Network nodes and associated properties with the directions re-set.

    """
    links, nodes = dy.set_continuity(links, nodes)

    # Set the directions of the links that are more certain via centerline angle method
    # alg = 23.1
    # alg = dy.algmap('cl_ang_rs')
    # cl_angthresh = np.percentile(links['clangs'][np.isnan(links['clangs']) == 0], 40)
    # for lid, cla, lg, lga, cert in zip(links['id'],  links['clangs'],
    #                                    links['guess'], links['guess_alg'],
    #                                    links['certain']):
    #     if cert == 1:
    #         continue
    #     if np.isnan(cla) == True:
    #         continue
    #     if cla <= cl_angthresh:
    #         linkidx = links['id'].index(lid)
    #         if dy.algmap('cl_ang_guess') in lga:
    #             usnode = lg[lga.index(dy.algmap('cl_ang_guess'))]
    #             links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)

    angthreshs = np.linspace(0, 1.3, 20)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a,
                                                       lenthresh=0,
                                                       alg=dy.algmap('known_fdr_rs'))
    if np.sum(links['certain']) != len(links['id']):
        links_notset = links['id'][np.where(links['certain'] == 0)[0][0]]
        print('Links {} were not set by re_set_linkdirs_flow_direction.'.format(links_notset))

    return links, nodes


def re_set_linkdirs_darea_gradient(links, nodes, imshape=None):
    """
    Reset link directions.

    Resets the link directions for a braided river channel network. This
    function is called to reset directions of links that belong to a cycle.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with the directions re-set.
    nodes : dict
        Network nodes and associated properties with the directions re-set.

    """
    links, nodes = dy.set_continuity(links, nodes)

    links, nodes = drainage_area_gradient(links, nodes)

    if np.sum(links['certain']) != len(links['id']):
        links_notset = links['id'][np.where(links['certain'] == 0)[0][0]]
        print('Links {} were not set by re_set_linkdirs_darea_gradient.'.format(links_notset))

    return links, nodes


def re_set_linkdirs_best_guess(links, nodes, imshape=None):
    """
    Reset link directions.

    Resets the link directions for a braided river channel network. This
    function is called to reset directions of links that belong to a cycle.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with the directions re-set.
    nodes : dict
        Network nodes and associated properties with the directions re-set.

    """
    links, nodes = dy.set_continuity(links, nodes)

    links, nodes = set_by_guess_agreement(links, nodes)

    if np.sum(links['certain']) != len(links['id']):
        links_notset = links['id'][np.where(links['certain'] == 0)[0][0]]
        print('Links {} were not set by re_set_linkdirs_best_guess.'.format(links_notset))

    return links, nodes


def re_set_linkdirs_main_channel_darea_grad(links, nodes, imshape=None):
    """
    Reset link directions.

    Resets the link directions for a braided river channel network. This
    function is called to reset directions of links that belong to a cycle.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with the directions re-set.
    nodes : dict
        Network nodes and associated properties with the directions re-set.

    """
    links, nodes = dy.set_continuity(links, nodes)

    # recalculate main channel gradient
    alg = dy.algmap('main_channel_darea_grad')
    links, nodes = guess_synthetic_slope(
            links, nodes, alg=alg,
            node_column="mainchannel_darea", ascending=True,
        )
    links, nodes = set_by_guess_agreement(links, nodes, guess_algs=[alg])

    if np.sum(links['certain']) != len(links['id']):
        links_notset = links['id'][np.where(links['certain'] == 0)[0][0]]
        print('Links {} were not set by re_set_linkdirs_main_channel_darea_grad.'.format(links_notset))

    return links, nodes


def re_set_linkdirs_dir_shortest_paths(links, nodes, imshape=None):
    """
    Reset link directions.

    Resets the link directions for a braided river channel network. This
    function is called to reset directions of links that belong to a cycle.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with the directions re-set.
    nodes : dict
        Network nodes and associated properties with the directions re-set.

    """
    links, nodes = dy.set_continuity(links, nodes)

    # recalculate main channel gradient
    alg_nodes = dy.algmap('sp_nodes')
    alg_links = dy.algmap('sp_links')

    links, nodes = dy.dir_shortest_paths_links(links, nodes)

    links, nodes = set_by_guess_agreement(links, nodes, min_guesses=1,
                           min_agree=1, alg=alg_links,
                           guess_algs=[alg_links])

    if np.sum(links['certain']) != len(links['id']):
        links_notset = links['id'][np.where(links['certain'] == 0)[0][0]]
        print('Links {} were not set by re_set_linkdirs_dir_shortest_paths.'.format(links_notset))

    return links, nodes


def fix_cycle_inlet_outlet(links, nodes, cycle_links, cycle_nodes):
    """Set directions of cycle by identifying the inlets and outlets of the
    cycle and if both exist connect each inlet node to its nearest outlet node.

    """
    fixed = 0
    G = nx.MultiDiGraph()
    allcon = set([l for n in cycle_nodes for l in nodes["conn"][nodes["id"].index(n)]])
    G.add_edges_from([tuple(links["conn"][links["id"].index(l)]) + ({"id": l},)
                      for l in allcon])
    # cycle nodes where outflows are connected
    outflows = [list(G.in_edges(n))[0][0] for n in G
                if len(G.out_edges(n)) == 0 and len(G.in_edges(n)) == 1]
    # cycle nodes where inflows are connected
    inflows = [list(G.out_edges(n))[0][1] for n in G
               if len(G.out_edges(n)) == 1 and len(G.in_edges(n)) == 0]
    # unsuccessful if either not found
    if not (outflows and inflows):
        reason = "no inflows" if outflows else "no outflows"
        return links, nodes, fixed, reason
    # find shortest paths between inflows and outflows and set lines accordingly
    Gund = G.to_undirected()
    shortest_paths = dict(nx.all_pairs_shortest_path(Gund))
    strong_links = []
    for inf in cycle_nodes:
        # nodes along shortest path to closest outlet
        shortest = sorted([shortest_paths[inf][o] for o in outflows], key=len)[0]
        # set directions
        for s, e in zip(shortest[:-1], shortest[1:]):
            lid = links["id"].index(Gund.edges[s, e, 0]['id'])  # ignores possible parallel edges
            if not (s, e, 0) in strong_links:
                strong_links.append((s, e, 0))
                links, nodes = dy.set_link(links, nodes, lid, s, alg=dy.algmap("sp_links"))
    # tag weak links of cycle
    Gund.remove_edges_from(strong_links)
    weak_links = set([i for _, _, i in Gund.edges(data="id")]) & set(cycle_links)
    links["certain_alg"][[links["id"].index(l) for l in weak_links]] = 9999
    # check again
    cn, cl = dy.get_cycles(links, nodes, checknode=cycle_nodes)
    if cn:
        print(f"Tried to resolve cycle {cycle_links} with nearest inlet-outlet, but failed.")
        reason = "in/outflows but failed"
    else:
        print(f"Fixed {cycle_links} with nearest inlet-outlet.")
        fixed = 1
        reason = "fixed"
    return links, nodes, fixed, reason


def fix_river_cycle(links, nodes, cyclelinks, cyclenodes, imshape,
                    reset_function=re_set_linkdirs_flow_direction):
    """
    Attempt to fix a single cycle.

    Attempts to resolve a single cycle within a river network. The general
    logic is that all link directions of the cycle are un-set except for those
    set by highly-reliable algorithms, and a modified direction-setting
    recipe is implemented to re-set these algorithms. This was developed
    according to the most commonly-encountered cases for real braided rivers,
    but could certainly be improved.


    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    cyclelinks : list
        List of link ids that comprise a cycle.
    cyclenodes : list
        List of node ids taht comprise a cycle.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with the cycle fixed if
        possible.
    nodes : dict
        Network nodes and associated properties with the cycle fixed if
        possible.
    fixed : int
        1 if the cycle was resolved, else 0.

    """

    # dont_reset_algs = [20, 21, 22, 23, 0, 5]
    dont_reset_algs = [dy.algmap(key) for key in ['manual_set',
                                                  'cl_dist_guess',
                                                  'cl_ang_guess',
                                                  'cl_dist_set',
                                                  'cl_ang_set',
                                                  'inletoutlet', 'bridges']]

    fixed = 1  # One if fix was successful, else zero
    reset = 0  # One if original orientation need to be reset

    # If an artifical node triad is present, flip its direction and see if the
    # cycle is resolved.
    # See if any links are part of an artificial triad
    clset = set(cyclelinks)
    all_pars = []
    for i, pl in enumerate(links['parallels']):
        if len(clset.intersection(set(pl))) > 0:
            all_pars.append(pl)

    # Get continuity violators before flipping
    pre_sourcesink = dy.check_continuity(links, nodes)

    if len(all_pars) == 1:  # There is one parallel link set, flip its direction and re-set other cycle links and see if cycle is resolved
        certzero = list(set(all_pars[0] + cyclelinks))
        orig_links = dy.cycle_get_original_orientation(links, certzero)  # Save the original orientations in case the cycle can't be fixed
        for cz in certzero:
            links['certain'][links['id'].index(cz)] = 0

        # Flip the links of the triad
        for l in all_pars[0]:
            links = lnu.flip_link(links, l)

    if len(all_pars) > 1:  # If there are multiple parallel pairs, more code needs to be written for these cases
        print('Multiple parallel pairs in the same cycle. Not implemented yet.')
        return links, nodes, 0

    elif len(all_pars) == 0:  # No aritifical node triads; just re-set all the cycle links and see if cycle is resolved
        certzero = cyclelinks
        orig_links = dy.cycle_get_original_orientation(links, certzero)
        for cz in certzero:
            lidx = links['id'].index(cz)
            if links['certain_alg'][lidx] not in dont_reset_algs:
                links['certain'][lidx] = 0

    # Resolve the unknown cycle links
    links, nodes = reset_function(links, nodes, imshape)

    # See if the fix violated continuity - if not, reset to original
    post_sourcesink = dy.check_continuity(links, nodes)
    if len(set(post_sourcesink) - set(pre_sourcesink)) > 0:
        reset = 1

    # See if the fix resolved the cycle - if not, reset to original
    cyc_n, cyc_l = dy.get_cycles(links, nodes, checknode=cyclenodes)
    if cyc_n is not None and cyclenodes[0] in cyc_n[0]:
        reset = 1

    # Return the links to their original orientations if cycle could not
    # be resolved
    if reset == 1:

        links = dy.cycle_return_to_original_orientation(links, orig_links)

        # Try a second method to fix the cycle: unset all the links of the
        # cycle AND the links connected to those links
        set_to_zero = set()
        for cn in cyclenodes:
            conn = nodes['conn'][nodes['id'].index(cn)]
            set_to_zero.update(conn)
        set_to_zero = list(set_to_zero)

        # Save original orientation in case cycle cannot be fixed
        orig_links = dy.cycle_get_original_orientation(links, set_to_zero)

        for s in set_to_zero:
            lidx = links['id'].index(s)
            if links['certain_alg'][lidx] not in dont_reset_algs:
                links['certain'][lidx] = 0

        links, nodes = reset_function(links, nodes, imshape)

        # See if the fix violated continuity - if not, reset to original
        post_sourcesink = dy.check_continuity(links, nodes)
        if len(set(post_sourcesink) - set(pre_sourcesink)) > 0:
            reset = 1

        # See if the fix resolved the cycle - if not, reset to original
        cyc_n, cyc_l = dy.get_cycles(links, nodes, checknode=cyclenodes)
        if cyc_n is not None and cyclenodes[0] in cyc_n[0]:
            reset = 1

        if reset == 1:
            links = dy.cycle_return_to_original_orientation(links, orig_links)
            fixed = 0

    return links, nodes, fixed


def dir_centerline(links, nodes, meshpolys, meshlines, Imask, gt, pixlen):
    """
    Guess flow directions of links in a braided river channel.

    Guesses the flow direction of links in a braided river channel network by
    exploiting a "valleyline" centerline. Two metrics are computed to help
    guess the correct direction. The first is the number of centerline
    transects (meshlines) that the link crosses. The second is the local angle
    of the centerline compared to the link's angle. These metrics are appended
    to the links dictionary as links['cldist'] and links['clangs'].

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    meshpolys : list
        List of shapely.geometry.Polygons that define the valleyline mesh.
    meshlines : list
        List of shapely.geometry.LineStrings that define the valleyline mesh.
    Imask : np.array
        Binary mask of the network.
    gt : tuple
        gdal-type GeoTransform of the original binary mask.
    pixlen : float
        Length resolution of each pixel.

    Returns
    -------
    links : dict
        Network links and associated properties with 'cldists' and 'clangs'
        attributes appended.

    """
    # alg = 20
    alg = dy.algmap('cl_dist_guess')

    # Create geodataframes for intersecting meshpolys with nodes
    mp_gdf = gpd.GeoDataFrame(geometry=[Polygon(mp) for mp in meshpolys])
    rc = np.unravel_index(nodes['idx'], Imask.shape)
    nodecoords = gu.xy_to_coords(rc[1], rc[0], gt)
    node_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(nodecoords[0], nodecoords[1])], index=nodes['id'])

    # Determine which meshpoly each node lies within
    intersect = gpd.sjoin(node_gdf, mp_gdf, op='intersects', rsuffix='right')

    # Compute guess and certainty, where certainty is how many transects apart
    # the link endpoints are (longer=more certain)
    cldists = np.zeros((len(links['id']), 1))
    for i, lconn in enumerate(links['conn']):
        try:
            first = intersect.loc[lconn[0]].index_right
            second = intersect.loc[lconn[1]].index_right
            cldists[i] = second-first
        except KeyError:
            pass

    for i, c in enumerate(cldists):
        if c != 0:
            if c > 0:
                links['guess'][i].append(links['conn'][i][0])
                links['guess_alg'][i].append(alg)
            elif c < 0:
                links['guess'][i].append(links['conn'][i][-1])
                links['guess_alg'][i].append(alg)

    # Save the distances for certainty
    links['cldists'] = np.abs(cldists)

    # Compute guesses based on how the link aligns with the local centerline
    # direction
    # alg = 21
    alg = dy.algmap('cl_ang_guess')
    clangs = np.ones((len(links['id']), 1)) * np.nan
    for i, (lconn, lidx) in enumerate(zip(links['conn'], links['idx'])):
        # Get coordinates of link endpoints
        rc = np.unravel_index([lidx[0], lidx[-1]], Imask.shape)

        try:  # Try is because some points may not lie within the mesh polygons
            # Get coordinates of centerline midpoints
            first = intersect.loc[lconn[0]].index_right
            second = intersect.loc[lconn[1]].index_right
            if first > second:
                first, second = second, first
            first_mp = np.mean(np.array(meshlines[first]), axis=0)  # midpoint
            second_mp = np.mean(np.array(meshlines[second+1]), axis=0)  # midpoint
        except KeyError:
            continue

        # Centerline vector
        cl_vec = second_mp - first_mp
        cl_vec = cl_vec/np.sqrt(np.sum(cl_vec**2))

        # Link vectors - as-is and flipped (reversed)
        link_vec = dy.get_link_vector(links, nodes, links['id'][i],
                                      Imask.shape, pixlen=pixlen)
        link_vec_rev = -link_vec

        # Compute interior radians between centerline vector and link vector
        # (then again with link vector flipped)
        lva = np.math.atan2(np.linalg.det([cl_vec, link_vec]),
                            np.dot(cl_vec, link_vec))
        lvar = np.math.atan2(np.linalg.det([cl_vec, link_vec_rev]),
                             np.dot(cl_vec, link_vec_rev))

        # Save the maximum angle
        clangs[i] = np.min(np.abs([lva, lvar]))

        # Make a guess; smaller interior angle (i.e. link direction that aligns
        # best with local centerline direction) guesses the link orientation
        if np.abs(lvar) < np.abs(lva):
            links['guess'][i].append(links['conn'][i][1])
            links['guess_alg'][i].append(alg)
        else:
            links['guess'][i].append(links['conn'][i][0])
            links['guess_alg'][i].append(alg)
    links['clangs'] = clangs

    return links


def dir_link_widths(links):
    """
    Guess link directions based on link widths.

    Guesses each link's direction based on link widths at the endpoints. The
    node with the larger width is guessed to be the upstream node. The ratio
    of guessed upstream to guessed downstream node widths is appended to links.
    If the width at a link's endpoint is 0, the percent is set to 1000. This
    ratio is appended to the links dictionary as links['wid_pctdiff'].


    Parameters
    ----------
    links : dict
        Network links and associated properties.

    Returns
    -------
    links : dict
        Network links and associated properties with 'wid_pctdiff' property
        appended.

    """
    # alg = 26
    alg = dy.algmap('wid_pctdiff')

    widpcts = np.zeros((len(links['id']), 1))
    for i in tqdm(range(len(links['id'])), "Dir link width"):

        lw = links['wid_pix'][i]

        if lw[0] > lw[-1]:
            links['guess'][i].append(links['conn'][i][0])
            links['guess_alg'][i].append(alg)
            if lw[-1] == 0 and lw[0] > 1:
                widpcts[i] = 10
            elif lw[-1] == 0:
                widpcts[i] = 0
            else:
                widpcts[i] = (lw[0] - lw[-1]) / lw[-1]
        else:
            links['guess'][i].append(links['conn'][i][1])
            links['guess_alg'][i].append(alg)
            if lw[0] == 0 and lw[-1] > 1:
                widpcts[i] = 10
            elif lw[0] == 0:
                widpcts[i] = 0
            else:
                widpcts[i] = (lw[-1] - lw[0]) / lw[0]

    # Convert to percent and store in links dict
    links['wid_pctdiff'] = widpcts * 100

    return links


def merge_shortest_path_guesses(links, nodes):
    spl, spn = (dy.algmap(a) for a in ('sp_links', 'sp_nodes'))
    for lix, (gu, ga) in enumerate(zip(links["guess"], links["guess_alg"])):
        sli, sni = (ga.index(i) if i in ga else None for i in (spl, spn))
        # has both guesses
        if sli and sni:
            # if they disagree remove both
            if gu[sli] != gu[sni]:
                links["guess"][lix].pop(sli)
                links["guess_alg"][lix].pop(sli)
                # new index after removal
                sni = links["guess_alg"].index(spn)
            links["guess"][lix].pop(sni)
            links["guess_alg"][lix].pop(sni)
    return links, nodes


def set_inexact(links, nodes, imshape):
    """Set links by inexact methods."""

    # compute neighbour independent guesses
    links, nodes = guess_synthetic_slope(links, nodes, min_slope=0.1)
    links, nodes = guess_synthetic_slope(
        links, nodes, alg=dy.algmap('main_channel_darea_grad'),
        node_column="mainchannel_darea", ascending=True,
        min_slope=0.001, max_slope=1,
    )
    links, nodes = drainage_area_gradient(links, nodes, guess=True)
    # shortest network path and merge guesses
    links, nodes = dy.dir_shortest_paths_nodes(links, nodes)
    links, nodes = dy.dir_shortest_paths_links(links, nodes)
    links, nodes = merge_shortest_path_guesses(links, nodes)

    # set very certain darea gradients
    certda = filter_ids(links, nodes, network_bridge=True, min_darea=1000)
    if len(certda) > 0:
        links, nodes = drainage_area_gradient(links, nodes, idx=certda)
    certda = filter_ids(links, nodes, network_bridge=True, coastal=False,
                        min_length=500, min_darea=50)
    if len(certda) > 0:
        links, nodes = drainage_area_gradient(links, nodes, idx=certda)

    links, nodes = dy.set_width_continuity(links, nodes)

    if np.all(links["certain"] == 1):
        return links, nodes

    # Set directions by most-certain angles
    angthreshs = np.linspace(0, 0.7, 20)
    fdr = dy.algmap("known_fdr")
    for a in tqdm(angthreshs, "Directions by shallow angles"):
        validlin = filter_ids(links, nodes, min_length=500)
        links, nodes = dy.set_by_known_flow_directions(
            links, nodes, imshape, idx=validlin, angthresh=a, guess=True,
        )
        # the direction must at least agree with two other guesses
        links, nodes = set_by_guess_agreement(
            links, nodes, min_guesses=3, min_agree=3,
            must_agree=fdr, alg=fdr,
        )
        links, nodes = dy.set_width_continuity(links, nodes)

    # set long coastal links where mainchannel_darea_grad and dist_coast_grad guesses agree
    # this will seed messy directions in deltas
    mcalg, cda = (dy.algmap(a) for a in ('main_channel_darea_grad', 'syn_dem'))
    cstuns = filter_ids(links, nodes, coastal=True, min_length=5000)
    guess = [(links["guess"][li], links["guess_alg"][li]) for li in cstuns]
    setln = []
    for li, (g, ga) in zip(cstuns, guess):
        if (cda in ga and mcalg in ga):
            if (g[ga.index(cda)] == g[ga.index(mcalg)]):
                setln.append((li, g[ga.index(cda)]))
    for li, usnd in setln:
        links, nodes = dy.set_link(links, nodes, li, usnd, alg=cda, checkcontinuity=True)
        links, nodes = dy.set_width_continuity(links, nodes, checknodes=links["conn"][li])

    # update flow direction guess to less certain angles
    links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape, guess=True)
    links, nodes = dy.set_width_continuity(links, nodes, guess=True, factor=2)

    """
    Guesses:
        - darea gradient
        - main channel darea gradient
        - outlet distance gradient
        - nodes + links shortest path (combine, merge if missing or same, remove if disagree)
    Bonus guesses:
        - flow direction (neighbour dependent)
        - width continuity (neighbour dependent)
    """

    # first set at least 3 agreeing with flow direction
    must_alg = [fdr, None]
    criteria = list(itertools.product(must_alg, [0, 1, 2], [5, 4, 3]))
    for malg, ndisagg, nguess in tqdm(criteria, "Set by best guesses"):
        # dont ever go below 3 guesses
        if (nguess - ndisagg) < 3:
            continue
        # iterate until no more links can be set with the (ndisagg, nguess) criteria
        n_unset = (links["certain"] == 0).sum()
        while n_unset:
            unset = links["certain"] == 0
            links, nodes = set_by_guess_agreement(
                links, nodes, min_guesses=nguess, min_agree=nguess - ndisagg, must_agree=malg,
            )
            n_unset = (unset & (links["certain"] == 1)).sum()
            # make sure newly set links also inform the guesses of less certain links
            #links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape, angthresh=0.4)
            links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape, guess=True)

    # set the rest by shortest path to outlet
    spalg = dy.algmap("sp_links")
    links, nodes = set_by_guess_agreement(links, nodes, alg=spalg, guess_algs=[spalg], min_agree=1)
    # # Set using direction of nearest main channel
    # print("Set directions by nearest main channel...")
    # links, nodes = dy.set_by_nearest_main_channel(links, nodes, imshape,
    #                                           nodethresh=1)

    # # Set directions by less-certain angles
    # angthreshs = np.linspace(0, 1.5, 20)
    # for a in tqdm(angthreshs, "Directions by steep angles"):
    #     links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
    #                                                    angthresh=a)
    return links, nodes


def set_by_guess_agreement(links, nodes, idx=None, min_guesses=None,
                           min_agree=None, must_agree=None, alg=None,
                           guess_algs=None):
    """Set links by guesses with various methods."""
    nguess = np.array([len(g) for g in links["guess"]])
    nmin = min_guesses or (len(guess_algs) if guess_algs else np.min(nguess))
    alg = alg or dy.algmap("n_agree") + nmin / 10
    con = (nguess >= nmin) & (links["certain"] == 0)
    # all must agree if no level of agreement is given
    if min_agree is None:
        nagree = np.array([len(set(g)) for g in links["guess"]])
        con = con & (nagree == 1)
    # filter and get indeces
    if idx is not None:
        idx = np.array(list(set(np.where(con)[0]) & set(idx)))
    else:
        idx = np.where(con)[0]
    # only continue if we have selected links
    if len(idx) == 0:
        return links, nodes
    # sort by descending length
    lenidx = np.argsort(np.array(links["len"])[idx])[::-1]
    # get back to full indeces
    lenidx = np.arange(len(links["id"]))[idx][lenidx]
    for lix in lenidx:
        # check if already certain in previous iteration
        if links["certain"][lix] == 1:
            continue
        # get best agreement
        gualgs, guesses = links["guess_alg"][lix], links["guess"][lix]
        if guess_algs is not None:
            guesses = [g for a, g in zip(links["guess_alg"][lix], links["guess"][lix]) if a in guess_algs]
            gualgs = [a for a in links["guess_alg"][lix] if a in guess_algs]
        if len(guesses) == 0:
            continue
        usns, counts = np.unique(guesses, return_counts=True)
        bestix = np.argmax(counts)
        usnode = usns[bestix]
        # check if min_agree and must_agree fulfilled
        als = [a for a, n in zip(gualgs, guesses) if n == usnode]
        if (min_agree and counts[bestix] < min_agree) or (must_agree and must_agree not in als):
            continue
        links, nodes = dy.set_link(
            links, nodes, lix, usnode, alg=alg, checkcontinuity=True,
        )
        links, nodes = dy.set_width_continuity(links, nodes, checknodes=links["conn"][lix])
    return links, nodes


def filter_ids(links, nodes, ids=None, idx=None, min_darea=None, dangle=None,
               min_length=None, min_width=None, min_darea_gradient=None,
               min_len_width_ratio=None, network_bridge=None, coastal=None, min_guesses=None):
    """Filter uncertain ids by link/node attributes and return link indeces (not ides)."""
    if idx is None:
        idx = np.arange(len(links['id'])) if ids is None else np.array([links["id"].index(i) for i in ids])
    else:
        idx = np.array(idx)
    # apply constraints
    idx = idx[links["certain"][idx] == 0]
    if min_length:
        idx = idx[np.array(links["len"])[idx] >= min_length]
    if min_width:
        idx = idx[np.array(links["wid_adj"])[idx] >= min_width]
    if min_darea_gradient is not None:
        idx = idx[np.abs(links["darea_gradient"][idx]) > min_darea_gradient]
    if min_len_width_ratio:
        lwr = np.array(links["len"])[idx] / np.array(links["wid_adj"])[idx]
        idx = idx[lwr > min_len_width_ratio]
    if network_bridge is not None:
        idx = idx[links["bridge"][idx] == network_bridge]
    if dangle is not None:
        idx = idx[links["dangle"][idx] == dangle]
    if coastal is not None:
        lnds = np.array(links["conn"])[idx]
        hascoast = [any([nodes["coastal"][nodes["id"].index(n)] for n in cn]) for cn in lnds]
        idx = idx[np.array(hascoast) == coastal]
    if min_darea:
        mindarea = np.min((links["darea_start"][idx], links["darea_end"][idx]), axis=0)
        idx = idx[mindarea >= min_darea]
    if min_guesses:
        nguess = np.array([len(g) for g in links["guess"]])[idx]
        idx = idx[nguess >= min_guesses]


    return idx


def drainage_area_gradient(links, nodes, idx=None, alg=dy.algmap("darea_grad"),
                           guess=False):
    """
    Set links by drainage area gradient iteratively starting with the ones with the largest darea.
    """
    idx = np.arange(len(links['id'])) if idx is None else np.array(idx)
    mindarea = np.min((links["darea_start"], links["darea_end"]), axis=0)
    
    while len(idx):
        lix = idx[np.argmax(mindarea[idx])]
        grad = links["darea_gradient"][lix]
        usnode = links["conn"][lix][int(grad < 0)]
        if guess:
            links['guess'][lix].append(usnode)
            links['guess_alg'][lix].append(alg)
        elif links["certain"][lix] == 0:
            links, nodes = dy.set_link(links, nodes, lix, usnode, alg)
        idx = filter_ids(links, nodes, idx=idx[idx != lix])
    return links, nodes


def guess_synthetic_slope(links, nodes, node_column="synthetic_elevation",
                          alg=dy.algmap('syn_dem'), ascending=False, min_slope=0, max_slope=1):
    """Guess directions by the slope of a synthetic elevation, e.g. distance to outlets.

    Links are not flipped, only guessed upstream node is attached to links[guess].
    """
    slopes = np.zeros(len(links["id"]), dtype=float)
    for linkidx, lid in enumerate(links['id']):
        conn = links["conn"][linkidx]
        elev = [nodes[node_column][nodes["id"].index(i)] for i in conn]
        slope = (elev[-1] - elev[0]) / (links["len"][linkidx] / links["sinuosity"][linkidx])
        # Make sure slope is negative, else flip direction
        di = slope < 0 if ascending else slope > 0
        if np.abs(slope) > min_slope and np.abs(slope) <= max_slope:
            usnode = conn[int(di)]
            # Store guess
            links['guess'][linkidx].append(usnode)
            links['guess_alg'][linkidx].append(alg)
        # Store slope
        slopes[linkidx] = slope * (-1 if di else 1)

    links[node_column+'_slope'] = slopes
    return links, nodes
