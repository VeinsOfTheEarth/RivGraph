# -*- coding: utf-8 -*-
"""
delta_directionality
====================

Created on Sun Nov 18 19:26:01 2018

@author: Jon
"""

from loguru import logger
import os
import numpy as np
import networkx as nx
from scipy.stats import mode, linregress
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt
import rivgraph.io_utils as io
import rivgraph.directionality as dy

# Todo: create the manual fix csv no matter what; allow user to input values
# before running directionality.


def set_link_directions(links, nodes, imshape, manual_set_csv=None):
    """
    Set each link direction in a network.

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
    imshape : tuple
        Shape of binary mask as (nrows, ncols).
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
    # Add fields to links dict for tracking and setting directionality
    links, nodes = dy.add_directionality_trackers(links, nodes, 'delta')

    # If a manual fix csv has been provided, set those links first
    links, nodes = dy.dir_set_manually(links, nodes, manual_set_csv)

    # Initial attempt to set directions
    links, nodes = set_initial_directionality(links, nodes, imshape)

    # At this point, all links have been set. Check for nodes that violate
    # continuity
    cont_violators = dy.check_continuity(links, nodes)

    # Attempt to fix any sources or sinks within the network
    if len(cont_violators) > 0:
        links, nodes = dy.fix_sources_and_sinks(links, nodes)

    # Check that continuity problems are resolved
    cont_violators = dy.check_continuity(links, nodes)
    if len(cont_violators) > 0:
        logger.info('Nodes {} violate continuity. Check connected links and fix manually.'.format(cont_violators))

    # Attempt to fix any cycles in the network (reports unfixable within function)
    links, nodes, allcyclesfixed = fix_delta_cycles(links, nodes, imshape)

    # The following is done automatically now, regardless of if cycles or sinks exist
    # # Create a csv to store manual edits to directionality if does not exist
    # if os.path.isfile(manual_set_csv) is False:
    #     if len(cont_violators) > 0 or allcyclesfixed == 0:
    #         io.create_manual_dir_csv(manual_set_csv)
    #         print('A .csv file for manual fixes to link directions at {}.'.format(manual_set_csv))

    if allcyclesfixed == 2:
        logger.info('No cycles were found in network.')

    return links, nodes


def set_initial_directionality(links, nodes, imshape):
    """
    Make initial attempt to set flow directions.

    Makes an initial attempt to set all flow directions within the network.
    This represents the core of the "delta recipe" described in the following
    open access paper:
    https://esurf.copernicus.org/articles/8/87/2020/esurf-8-87-2020.pdf
    However, note that as RivGraph develops, this recipe may no longer match
    the one presented in that paper. The recipe chains together a number of
    exploitative algorithms to iteratively set flow directions for the most
    certain links first.


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
        Network links and associated properties with initial directions set.
    nodes : dict
        Network nodes and associated properties with initial directions set.

    """
    # Compute all the "guesses"
    links, nodes = dy.dir_main_channel(links, nodes)
    links, nodes = dir_synthetic_DEM(links, nodes, imshape)
    links, nodes = dy.dir_shortest_paths_nodes(links, nodes)
    links, nodes = dy.dir_shortest_paths_links(links, nodes)
    links, nodes = dy.dir_bridges(links, nodes)

    # Set link directions
    # First, set inlet/outlet directions as they are always 100% accurate
    links, nodes = dy.set_inletoutlet(links, nodes)

    # Use bridges to set links as they are always 100% accurate
    # alg = 5
    alg = dy.algmap('bridges')
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'],
                                        links['guess'], links['guess_alg'],
                                        links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links that are known from bridge links
        if alg in lga:
            links, nodes = dy.set_link(links, nodes, linkidx,
                                       lg[lga.index(alg)], alg)

    # Use main channels (4) to set links
    # alg = 4
    alg = dy.algmap('main_chans')
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'],
                                        links['guess'], links['guess_alg'],
                                        links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links that are known from main_channel
        if alg in lga:
            links, nodes = dy.set_link(links, nodes, linkidx,
                                       lg[lga.index(alg)], alg)

    # Set the longest, steepest links according to io_surface
    # (these are those we are most certain of)
    # alg = 13
    alg = dy.algmap('longest_steepest')
    len75 = np.percentile(links['len_adj'], 75)
    slope50 = np.percentile(np.abs(links['slope']), 50)
    for lid, llen, lg, lga, cert, lslope in zip(links['id'], links['len_adj'],
                                                links['guess'],
                                                links['guess_alg'],
                                                links['certain'],
                                                links['slope']):
        if cert == 1:
            continue
        if llen > len75 and abs(lslope) > slope50:
            linkidx = links['id'].index(lid)
            if dy.algmap('syn_dem') in lga:
                usnode = lg[lga.index(dy.algmap('syn_dem'))]
                links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)

    # Set the most certain angles
    angthreshs = np.linspace(0, 0.5, 10)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a)

    # Set using direction of nearest main channel
    links, nodes = dy.set_by_nearest_main_channel(links, nodes, imshape,
                                                  nodethresh=2)

    # Set the most certain angles
    angthreshs = np.linspace(0, 0.7, 10)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a)

    # Set using direction of nearest main channel
    links, nodes = dy.set_by_nearest_main_channel(links, nodes, imshape,
                                                  nodethresh=1)

    angthreshs = np.linspace(0, 0.8, 10)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a,
                                                       lenthresh=3)

    # Use io_surface (3) to set links that are longer
    # than the median link length
    alg = dy.algmap('syn_dem')
    medlinklen = np.median(links['len'])
    for lid, llen, lg, lga, cert in zip(links['id'], links['len'],
                                        links['guess'], links['guess_alg'],
                                        links['certain']):
        if cert == 1:
            continue
        if llen > medlinklen and dy.algmap('syn_dem') in lga:
            linkidx = links['id'].index(lid)
            usnode = lg[lga.index(dy.algmap('syn_dem'))]
            links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)

    # Set again by angles, but reduce the lenthresh
    # (shorter links will be set that were not previously)
    angthreshs = np.linspace(0, 0.6, 10)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a,
                                                       lenthresh=0)

    # If any three methods agree, set that link to whatever they agree on
    # alg = 15
    alg = dy.algmap('three_agree')
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'],
                                        links['guess'], links['guess_alg'],
                                        links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links with 3 or more guesses that agree
        m = mode(lg)
        if m.count[0] > 2:
            links, nodes = dy.set_link(links, nodes, linkidx, m.mode[0], alg)

    # Set again by angles, but reduce the lenthresh
    # (shorter links will be set that were not previously)
    angthreshs = np.linspace(0, 0.8, 10)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,
                                                       angthresh=a,
                                                       lenthresh=0)

    # If artificial DEM and at least one shortest path method agree,
    # set link to be their agreement
    # alg = 16
    alg = dy.algmap('syn_dem_and_sp')
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'],
                                        links['guess'], links['guess_alg'],
                                        links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links with 2 or more same guesses that are not
        # shortest path (one may be shortest path)
        if dy.algmap('syn_dem') in lga and dy.algmap('sp_links') in lga:
            if lg[lga.index(dy.algmap('syn_dem'))] == lg[lga.index(dy.algmap('sp_links'))]:
                links, nodes = dy.set_link(links, nodes, linkidx,
                                           lg[lga.index(dy.algmap('syn_dem'))],
                                           alg)
        elif dy.algmap('syn_dem') in lga and dy.algmap('sp_nodes') in lga:
            if lg[lga.index(dy.algmap('syn_dem'))] == lg[lga.index(dy.algmap('sp_nodes'))]:
                links, nodes = dy.set_link(links, nodes, linkidx,
                                           lg[lga.index(dy.algmap('syn_dem'))],
                                           alg)

    # Find remaining uncertain links
    uncertain = [l for l, lc in zip(links['id'], links['certain']) if lc != 1]

    # Set remaining uncertains according to io_surface (3)
    # alg = 10 # change this one!
    alg = dy.algmap('syn_dem')
    for lid in uncertain:
        linkidx = links['id'].index(lid)
        if alg in links['guess_alg'][linkidx]:
            usnode = links['guess'][linkidx][links['guess_alg'][linkidx].index(alg)]
            links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)

    return links, nodes


def fix_delta_cycles(links, nodes, imshape):
    """
    Attempt to resolve cycles within the network.

    Attempts to resolve all cycles within the delta network. This function
    is essentially a wrapper for :func:`fix_delta_cycle`, which is where the
    heavy lifting is actually done. This function finds cycles, calls
    :func:`fix_delta_cycle` on each one, then aggregates the results.

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
    allfixed : bool
        True if all cycles were resolved, else False.

    """
    # Tracks if all cycles were fixed
    allfixed = 1

    # Create networkx graph object
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes['id'])
    for lc in links['conn']:
        G.add_edge(lc[0], lc[1])

    # Check for cycles
    cantfix_links = []
    fixed_links = []
    if nx.is_directed_acyclic_graph(G) is not True:

        # Get list of cycles to fix
        c_nodes, c_links = dy.get_cycles(links, nodes)

        # Combine all cycles that share links
        c_links = dy.merge_list_of_lists(c_links)

        # Fix the cycles
        for ic, cfix_links in enumerate(c_links):
            links, nodes, fixed = fix_delta_cycle(links, nodes, cfix_links,
                                                  imshape)
            if fixed == 0:
                cantfix_links.append(ic)
            elif fixed == 1:
                fixed_links.append(ic)

        # Report
        if len(cantfix_links) > 0:
            allfixed = 0
            logger.info('Could not fix the following cycles (links): {}'.format([c_links[i] for i in cantfix_links]))

        if len(c_links) > 0:
            allfixed = 0
            logger.info('The following cycles (links) were fixed, but should be manually checked: {}'.format([c_links[i] for i in fixed_links]))

    else:
        allfixed = 2  # Indicates there were no cycles to fix

    return links, nodes, allfixed


def fix_delta_cycle(links, nodes, cyc_links, imshape):
    """
    Attempt to resolve a single cycle.

    Attempts to resolve a single cycle within a delta network. The general
    logic is that all link directions of the cycle are un-set except for those
    set by highly-reliable algorithms, and a modified direction-setting
    recipe is implemented to re-set these algorithms. This was developed
    according to the most commonly-encountered cases for real deltas, but could
    certainly be improved.


    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    cyc_links : list
        Link ids comprising the cycle to fix.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).

    Returns
    -------
    links : dict
        Network links and associated properties with cycle fixed, if possible.
    nodes : dict
        Network nodes and associated properties with cycle fixed, if possible.
    fixed : bool
        True if the cycle was resolved, else False.

    """

    def re_set_linkdirs(links, nodes, imshape):
        # Cycle links are attempted to be reset according to algorithms here.
        angthreshs = np.linspace(0, 1.2, 20)
        for a in angthreshs:
            links, nodes = dy.set_by_known_flow_directions(links, nodes,
                                                           imshape,
                                                           angthresh=a,
                                                           lenthresh=0,
                                                           alg=dy.algmap('known_fdr_rs'))

        return links, nodes

    # Track if fix was successful (1:yes, 0:no)
    fixed = 1

    # List of algorithm ids that should not be reset if previously used to
    # determine direction
    # dont_reset_algs = [-1, 0, 4, 5, 13]
    dont_reset_algs = [dy.algmap(key) for key in ['manual_set', 'inletoutlet',
                                                  'main_chans', 'bridges',
                                                  'longest_steepest']]

    # Simplest method: unset the cycle links and reset them according to angles
    # Get resettale links
    toreset = [l for l in cyc_links if links['certain_alg'][links['id'].index(l)] not in dont_reset_algs]

    # Get original link orientations in case fix does not work
    orig = dy.cycle_get_original_orientation(links, toreset)

    # Set certainty of cycle links to zero
    for tr in toreset:
        links['certain'][links['id'].index(tr)] = 0

    links, nodes = re_set_linkdirs(links, nodes, imshape)

    # Check that all links were reset
    if sum([links['certain'][links['id'].index(l)] for l in toreset]) != len(toreset):
        fixed = 0

    # Check that cycle was resolved
    cyclenode = links['conn'][links['id'].index(toreset[0])][0]
    cyc_n, cyc_l = dy.get_cycles(links, nodes, checknode=cyclenode)

    # If the cycle was not fixed, try again, but set the cycle links AND the
    # links connected to the cycle to unknown
    if cyc_n is not None and cyclenode in cyc_n[0]:
        # First return to original orientation
        links = dy.cycle_return_to_original_orientation(links, orig)

        # Get all cycle links and those connected to cycle
        toreset = set()
        for cn in cyc_n[0]:
            conn = nodes['conn'][nodes['id'].index(cn)]
            toreset.update(conn)
        toreset = list(toreset)

        # Save original orientation in case cycle cannot be fixed
        orig_links = dy.cycle_get_original_orientation(links, toreset)

        # Un-set the cycle+connected links
        for tr in toreset:
            lidx = links['id'].index(tr)
            if links['certain_alg'][lidx] not in dont_reset_algs:
                links['certain'][lidx] = 0

        links, nodes = re_set_linkdirs(links, nodes, imshape)

        # See if the fix resolved the cycle - if not, reset to original
        cyc_n, cyc_l = dy.get_cycles(links, nodes, checknode=cyclenode)
        if cyc_n is not None and cyclenode in cyc_n[0]:
            links = dy.cycle_return_to_original_orientation(links, orig_links)
            fixed = 0

    return links, nodes, fixed


def hull_coords(xy):
    """
    Compute convex hull of a set of input points.

    Computes the convex hull of a set of input points. Arranges the convex
    hull coordinates in a clockwise manner and removes the longest edge.
    This function is required by :func:`dir_synthetic_dem`.

    Parameters
    ----------
    xy : np.array
        Two element array. First element contains x coordinates, second
        contains y coordinates of points to compute a convex hull around.

    Returns
    -------
    hull_coords : np.array
        Nx2 array of coordinates defining the convex hull of the input points.

    """
    # Find the convex hull of a set of coordinates, then order them clockwisely
    # and remove the longest edge
    hull_verts = ConvexHull(np.transpose(np.vstack((xy[0], xy[1])))).vertices
    hull_coords = np.transpose(np.vstack((xy[0][hull_verts], xy[1][hull_verts])))
    hull_coords = np.reshape(np.append(hull_coords, [hull_coords[0, :]]),
                             (int((hull_coords.size+2)/2), 2))

    # Find the biggest gap between hull points
    dists = np.sqrt((np.diff(hull_coords[:, 0]))**2 + \
                     np.diff(hull_coords[:, 1])**2)
    maxdist = np.argmax(dists) + 1
    first_part = hull_coords[maxdist:, :]
    second_part = hull_coords[0:maxdist, :]
    if first_part.size == 0:
        hull_coords = second_part
    elif second_part.size == 0:
        hull_coords = first_part
    else:
        hull_coords = np.concatenate((first_part, second_part))

    return hull_coords


def dir_synthetic_DEM(links, nodes, imshape):
    """
    Build a synthetic DEM using inlet/outlet locations.

    Builds a synthetic DEM by considering inlets as "hills" and outlets as
    "depressions." This synthetic is then used to compute the "slope"
    of each link, which is added to the links dictionary. Additionally,
    direction guesses for each link's flow are computed.

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
        Network links and associated properties including a 'slope'.
    nodes : dict
        Network nodes and associated properties.

    """
    alg = dy.algmap('syn_dem')

    # Create empty image to store surface
    I = np.ones(imshape, dtype=float)

    # Get row,col coordinates of outlet nodes, arrange them in a
    # clockwise order
    outs = [nodes['idx'][nodes['id'].index(o)] for o in nodes['outlets']]
    outsxy = np.unravel_index(outs, I.shape)
    if len(outsxy[0]) < 3:
        hco = np.transpose(np.vstack((outsxy[0], outsxy[1])))
    else:
        hco = hull_coords(outsxy)

    # Burn the hull into the Iout surface
    for i in range(len(hco)-1):
        linterp = interp1d(hco[i:i+2, 0], hco[i:i+2, 1])
        xinterp = np.arange(np.min(hco[i:i+2, 0]), np.max(hco[i:i+2, 0]), .1)
        yinterp = linterp(xinterp)
        for x,y in zip(xinterp, yinterp):
            I[int(round(x)), int(round(y))] = 0

    Iout = distance_transform_edt(I)
    Iout = (Iout - np.min(Iout)) / (np.max(Iout) - np.min(Iout))

    # Get coordinates of inlet nodes; use only the widest inlet and any#
    # inlets within 25% of its width
    ins = [nodes['idx'][nodes['id'].index(i)] for i in nodes['inlets']]
    in_wids = []
    for i in nodes['inlets']:
        linkid = nodes['conn'][nodes['id'].index(i)][0]
        linkidx = links['id'].index(linkid)
        in_wids.append(links['wid_adj'][linkidx])
    maxwid = max(in_wids)
    keep = [ii for ii, iw in enumerate(in_wids) if
            abs((iw - maxwid)/maxwid) < .25]
    ins_wide_enough = [ins[k] for k in keep]
    insxy = np.unravel_index(ins_wide_enough, imshape)
    if len(insxy[0]) < 3:
        hci = np.transpose(np.vstack((insxy[0], insxy[1])))
    else:
        hci = hull_coords(insxy)

    # Burn the hull into the Iout surface
    I = np.zeros(imshape, dtype=float) + 1
    if hci.shape[0] == 1:
        I[hci[0][0], hci[0][1]] = 0
    else:
        for i in range(len(hci)-1):
            linterp = interp1d(hci[i:i+2, 0], hci[i:i+2, 1])
            xinterp = np.arange(np.min(hci[i:i+2, 0]),
                                np.max(hci[i:i+2, 0]), .1)
            yinterp = linterp(xinterp)
            for x, y in zip(xinterp, yinterp):
                I[int(round(x)), int(round(y))] = 0

    Iin = distance_transform_edt(I)
    Iin = np.max(Iin) - Iin
    Iin = (Iin - np.min(Iin)) / (np.max(Iin) - np.min(Iin))

    # Compute the final surface by adding the inlet and outlet images
    Isurf = Iout + Iin

    # Guess the flow direction of each link
    slopes = []
    for lid in links['id']:

        linkidx = links['id'].index(lid)
        lidcs = links['idx'][linkidx][:]
        rc = np.unravel_index(lidcs, imshape)

        dists_temp = np.cumsum(np.sqrt(np.diff(rc[0])**2 + np.diff(rc[1])**2))
        dists_temp = np.insert(dists_temp, 0, 0)

        elevs = Isurf[rc[0], rc[1]]

        linreg = linregress(dists_temp, elevs)

        # Make sure slope is negative, else flip direction
        if linreg.slope > 0:
            usnode = nodes['id'][nodes['idx'].index(lidcs[-1])]
        else:
            usnode = nodes['id'][nodes['idx'].index(lidcs[0])]

        # Store guess
        links['guess'][linkidx].append(usnode)
        links['guess_alg'][linkidx].append(alg)

        # Store slope
        slopes.append(linreg.slope)

    links['slope'] = slopes

    return links, nodes
