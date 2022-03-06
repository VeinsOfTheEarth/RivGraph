# -*- coding: utf-8 -*-
"""
river_directionality
====================

Created on Tue Nov  6 14:31:01 2018

@author: Jon
"""

import os
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

    # # Set the directions of the links that are more certain via centerline
    # # distance method
    # # alg = 22
    # alg = dy.algmap('cl_dist_set')
    # cl_distthresh = np.percentile(links['cldists'], 85)
    # for lid, cld, lg, lga, cert in zip(links['id'],  links['cldists'],
    #                                    links['guess'], links['guess_alg'],
    #                                    links['certain']):
    #     if cert == 1:
    #         continue
    #     if cld >= cl_distthresh:
    #         linkidx = links['id'].index(lid)
    #         if dy.algmap('cl_dist_guess') in lga:
    #             usnode = lg[lga.index(dy.algmap('cl_dist_guess'))]
    #             links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)

    # # Set the directions of the links that are more certain via centerline
    # # angle method
    # # alg = 23
    # alg = dy.algmap('cl_ang_set')
    # cl_angthresh = np.percentile(links['clangs'][np.isnan(links['clangs'])==0], 25)
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

    # # Set the directions of the links that are more certain via centerline
    # # distance AND centerline angle methods
    # # alg = 24
    # alg = dy.algmap('cl_dist_and_ang')
    # cl_distthresh = np.percentile(links['cldists'], 70)
    # ang_thresh = np.percentile(links['clangs'][np.isnan(links['clangs']) == 0],
    #                            35)
    # for lid, cld, cla, lg, lga, cert in zip(links['id'],  links['cldists'],
    #                                         links['clangs'], links['guess'],
    #                                         links['guess_alg'],
    #                                         links['certain']):
    #     if cert == 1:
    #         continue
    #     if cld >= cl_distthresh and cla < ang_thresh:
    #         linkidx = links['id'].index(lid)
    #         if dy.algmap('cl_dist_guess') in lga and dy.algmap('cl_ang_guess') in lga:
    #             if lg[lga.index(dy.algmap('cl_dist_guess'))] == lg[lga.index(dy.algmap('cl_ang_guess'))]:
    #                 usnode = lg[lga.index(dy.algmap('cl_dist_guess'))]
    #                 links, nodes = dy.set_link(links, nodes, linkidx, usnode,
    #                                            alg)

    # Set directions by most-certain angles
    angthreshs = np.linspace(0, 0.4, 10)
    for a in tqdm(angthreshs, "Directions by shallow angles"):
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a,
                                                       lenthresh=3)

    # Set using direction of nearest main channel
    print("Set directions by nearest main channel...")
    links, nodes = dy.set_by_nearest_main_channel(links, nodes, imshape,
                                                  nodethresh=1)

    # Set directions by less-certain angles
    angthreshs = np.linspace(0, 1.5, 20)
    for a in tqdm(angthreshs, "Directions by steep angles"):
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a)

    # At this point, if any links remain unset, they are just set randomly
    if np.sum(links['certain']) != len(links['id']):
        print('{} links were randomly set.'.format(len(links['id']) -
                                                   np.sum(links['certain'])))
        links['certain'] = np.ones((len(links['id']), 1))

    # Check for and try to fix cycles in the graph
    links, nodes, cantfix_cyclelinks, cantfix_cyclenodes = fix_river_cycles(links, nodes, imshape)
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
    print("Direction link width...")
    links = dir_link_widths(links)

    #print("Direction network bridges...")
    #links, nodes = dy.dir_bridges(links, nodes)

    #for inl in tqdm(nodes['inlets'], 'Main channel direction'):
    #    links, nodes = dy.dir_main_channel(links, nodes, inlet=inl)
    links, nodes = dy.dir_main_channel(links, nodes)
    return links, nodes


def fix_river_cycles(links, nodes, imshape):
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
        cfix_nodes = [cn for icn, cn in enumerate(c_nodes) if np.isnan(isin[icn][0])]
        cfix_links = [cl for icl, cl in enumerate(c_links) if np.isnan(isin[icl][0])]

        print('Attempting to fix {} cycles.'.format(len(cfix_nodes)))

        # Try to fix all the cycles
        for cnodes, clinks in tqdm(zip(cfix_nodes, cfix_links),
                                   "Fixing cycles", total=len(cfix_nodes)):
            links, nodes, fixed = fix_river_cycle(links, nodes, clinks,
                                                  cnodes, imshape)
            if fixed == 0:
                cantfix_nodes.append(cnodes)
                cantfix_links.append(clinks)

        if len(cantfix_links) > 0:
            print(f"Failed to fix {len(cantfix_links)} cycles.")

    return links, nodes, cantfix_links, cantfix_nodes


def fix_river_cycle(links, nodes, cyclelinks, cyclenodes, imshape):
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
    links, nodes = re_set_linkdirs(links, nodes, imshape)

    # See if the fix violated continuity - if not, reset to original
    post_sourcesink = dy.check_continuity(links, nodes)
    if len(set(post_sourcesink) - set(pre_sourcesink)) > 0:
        reset = 1

    # See if the fix resolved the cycle - if not, reset to original
    cyc_n, cyc_l = dy.get_cycles(links, nodes, checknode=cyclenodes[0])
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

        links, nodes = re_set_linkdirs(links, nodes, imshape)

        # See if the fix violated continuity - if not, reset to original
        post_sourcesink = dy.check_continuity(links, nodes)
        if len(set(post_sourcesink) - set(pre_sourcesink)) > 0:
            reset = 1

        # See if the fix resolved the cycle - if not, reset to original
        cyc_n, cyc_l = dy.get_cycles(links, nodes, checknode=cyclenodes[0])
        if cyc_n is not None and cyclenodes[0] in cyc_n[0]:
            reset = 1

        if reset == 1:
            links = dy.cycle_return_to_original_orientation(links, orig_links)
            fixed = 0

    return links, nodes, fixed


def re_set_linkdirs(links, nodes, imshape):
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
        print('Links {} were not set by re_set_linkdirs.'.format(links_notset))

    return links, nodes


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


""" Functions below here are deprecated or unused, but might be useful later """
""" No testing performed """


def set_unknown_cluster_by_widthpct(links, nodes):
    """
    Set unknown links based on width differences at endpoints.

    (flow goes wide->narrow)

    """
    alg = 26

    # Get indices of uncertain links
    uc_idx = np.where(links['certain'] == 0)[0].tolist()

    # Create graph to find clusters of uncertains
    G = nx.Graph()
    for idx in uc_idx:
        lc = links['conn'][idx]
        G.add_edge(lc[0], lc[1])

    # Compute connected components (nodes)
    cc = nx.connected_components(G)
    ccs = [c for c in cc if len(c) > 1]

    # Convert cc nodes to cc edges
    cc_edges = []
    for ccnodes in ccs:
        ccedge = []
        for idx in uc_idx:
            lc = links['conn'][idx]
            if lc[0] in ccnodes and lc[1] in ccnodes:
                ccedge.append(links['id'][idx])
        cc_edges.append(ccedge)

    # Loop through all the unknown clusters, setting the most-certain-by-width
    for lclust in cc_edges:
        widpcts = [links['wid_pctdiff'][links['id'].index(l)][0] for l in lclust]
        link_toset = lclust[widpcts.index(max(widpcts))]
        linkidx = links['id'].index(link_toset)
        usnode = links['guess'][linkidx][links['guess_alg'][linkidx].index(26)]

        links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg=alg)

    return links, nodes
