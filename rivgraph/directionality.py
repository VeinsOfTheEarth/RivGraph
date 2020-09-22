# -*- coding: utf-8 -*-
"""
Directionality Utils (directionality.py)
========================================

Created on Wed Nov  7 11:38:16 2018
@author: Jon
"""
import os
import numpy as np
import networkx as nx
import itertools
import pandas as pd
from scipy.stats import mode
from rivgraph import ln_utils as lnu


def add_directionality_trackers(links, nodes, ntype):
    """
    Adds fields to the links and nodes dictionaries that are required for
    setting and diagnosing directionality. Nodes are not altered in this
    function, but passed for consistency and generalizability.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    ntype : str
        Network type. Choose either 'delta' or 'river'.

    Returns
    -------
    links : dict
        Network links with added directionality properties.
    nodes : dict
        Network nodes and associated properties.
    """

    # Add a 'certain' entry to the links dict to keep track of links whose directions have been set
    links['certain'] = np.zeros(len(links['id']))  # tracks whether a link's directinoality is certain or not
    links['certain_order'] = np.zeros(len(links['id']))  # tracks the order in which links certainty is set
    links['certain_alg'] = np.zeros(len(links['id']))  # tracks the algorithm used to set certainty

    # Add a "guess" entry to keep track of the different algorithms' guesses for flow directionality
    links['guess'] = [[] for a in range(len(links['id']))]  # contains guess at upstream ndoe
    links['guess_alg'] = [[] for a in range(len(links['id']))]  # contains algorithm that made guess

    # Add network-type-specific properties
    if ntype == 'river':
        links['maxang'] = np.ones(len(links['id'])) * np.nan  # saves the angle used in set_by_flow_directions, diagnostic only

    return links, nodes


def algmap(key):
    """
    Returns a numeric key corresponding to each algorithm used by RivGraph to
    set link directionality. These numbers are found in links['guess_alg'] and
    links['certain_alg'] and are useful for diagnosing issues in setting
    directionality.

    Parameters
    ----------
    key : str
        See the mapper below for the possible keywords.

    Returns
    -------
    algno : float or int
        Numeric representation of direction-setting algorithm.
    """

    mapper = {'sourcesinkfix' : -2,
              'manual_set' : -1,
              'inletoutlet' : 0,
              'continuity' : 1,
              'parallels' : 2,
              'artificials' : 2.1,
              'main_chans' : 4,
              'bridges' : 5,
              'known_fdr' : 6,
              'known_fdr_rs' : 6.1,
              'syn_dem' : 10,
              'syn_dem_med' : 10.1,
              'sym_dem_leftover' : 10.2,
              'sp_links' : 11,
              'sp_nodes' : 12,
              'longest_steepest' : 13,
              'three_agree' : 15,
              'syn_dem_and_sp' : 16,
              'cl_dist_guess' : 20,
              'cl_ang_guess' : 21,
              'cl_dist_set' : 22,
              'cl_ang_set' : 23,
              'cl_ang_rs' : 23.1,
              'cl_dist_and_ang' : 24,
              'short_no_bktrck' : 25,
              'wid_pctdiff' : 26}

    algno = mapper[key]

    return algno


def set_by_nearest_main_channel(links, nodes, imshape, nodethresh=0):
    """
    Sets all possible links' directions using the nearest main channel. The
    main channels are found as the widest, shortest paths from each inlet to
    each outlet. For each unknown link, the distance is computed from its
    endpoints to the nearest nodes of the nearest main channel. If the nearest
    nodes of the main channel include more than 2 + nodethresh nodes, the
    direction of the main channel is used to set the unknown link.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of binary mask as (nrows, ncols).
    nodethresh : int, optional
        Threshold for the number of nodes along the nearest main channel that
        must be encompassed by the unknown link's endnodes to confidently
        set the unknown link's direction. The default is 0.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.
    """

    # alg = 3 # identifier for diagnostics
    alg = algmap('parallels')

    # Find widest inlet node
    inlet_idx = widest_inlet_index(links, nodes)

    # More weight given to longer and narrower channels
    Aweight = (np.max(links['wid_adj']) - links['wid_adj'])
    weights = Aweight * links['len']

    # Create networkX graph, adding weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], weights):
        G.add_edge(lc[0], lc[1], weight=wt)

    # Get all paths from inlet(s) to outlets
    all_pathnodes = []
    for inl in nodes['inlets']:
        for o in nodes['outlets']:
            all_pathnodes.append(nx.dijkstra_path(G, nodes['inlets'][inlet_idx], o, weight='weight'))

    # Reduce all pathnodes to smallest set, saving associated path for each node
    pathnode_set = set()
    for ap in all_pathnodes:
        pathnode_set.update(ap)
    pathnode_set = list(pathnode_set)
    belongs_to = [[] for i in range(len(pathnode_set))]
    for i, p in enumerate(pathnode_set):
        for j, ap in enumerate(all_pathnodes):
            if p in ap:
                belongs_to[i].append(j)

    # Get node coordinates of all path nodes
    pathnode_set_idcs = [nodes['idx'][nodes['id'].index(n)] for n in pathnode_set]
    rc_pathnodes = np.unravel_index(pathnode_set_idcs, imshape)

    # Find the nearest path to all uncertain links
    uncertains = np.where(links['certain'] == 0)[0]
    for u in uncertains:

        # Since we're not updating uncertains as links are being set, need to
        # recheck that the link hasn't been set by continuity/aritifical node
        # due to setting a previous uncertain link
        if links['certain'][u] == 1:
            continue

        # Find nearest path to each centerline endpoint
        nconn = links['conn'][u]
        nidxs = [nodes['idx'][nodes['id'].index(n)] for n in nconn]
        lrc = np.unravel_index(nidxs, imshape)

        nearest_paths = []
        for r, c in zip(lrc[0], lrc[1]):
            nearest_pathnode = np.argmin(np.sqrt((r-rc_pathnodes[0])**2 + (c-rc_pathnodes[1])**2))
            nearest_paths.append(belongs_to[nearest_pathnode])

        # Choose any path that is nearest to both nodes
        closest_to_link = set(nearest_paths[0]).intersection(set(nearest_paths[1]))
        if len(closest_to_link) == 0:  # The links's endpoints are closer to two different paths
            continue

        use_path = closest_to_link.pop()

        # Now that path to compare against is known, determine the flow direction
        path = all_pathnodes[use_path]
        path_rc = np.unravel_index([nodes['idx'][nodes['id'].index(n)] for n in path], imshape)

        # Closest index of each end node of link:
        nearest_nodes = []
        for r, c in zip(lrc[0], lrc[1]):
            nearest_nodes.append(np.argmin(np.sqrt((r-path_rc[0])**2 + (c-path_rc[1])**2)))

        # If threshold is surpassed, set link
        if abs(nearest_nodes[0] - nearest_nodes[1]) <= nodethresh:
            continue
        elif nearest_nodes[0] > nearest_nodes[1]:
            links, nodes = set_link(links, nodes, u, nconn[1], alg=alg)
        elif nearest_nodes[1] > nearest_nodes[0]:
            links, nodes = set_link(links, nodes, u, nconn[0], alg=alg)

    return links, nodes


def nodepath_to_links(path, links, nodes):
    """
    Converts a path defined by node ids to a path defined by link ids.

    Parameters
    ----------
    path : list
        Node ids comprising a path within the network.
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    linkpath : list
        Link ids comprising the path defined by the given node ids.
    """

    linkpath = []
    for u, v in zip(path[0:-1], path[1:]):
        ulinks = nodes['conn'][nodes['id'].index(u)]
        vlinks = nodes['conn'][nodes['id'].index(v)]
        linkpath.append([ul for ul in ulinks if ul in vlinks][0])

    return linkpath


def widest_inlet_index(links, nodes):
    """
    Finds the index of the inlet occurring at the widest link. Index is with
    respect to the nodes['id'] list.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    inlet_idx : int
        Index of the inlet node of the widest link.
    """

    # Find apex node, assuming it's connected to the widest channel(s)
    inletW = []
    for nid in nodes['inlets']:
        nidx = nodes['id'].index(nid)
        lids = nodes['conn'][nidx]
        inletW.append(np.sum([links['wid_adj'][links['id'].index(li)] for li in lids]))

    W = max(inletW)
    inlet_idx = inletW.index(W)

    return inlet_idx


def dir_main_channel(links, nodes):
    """
    Guesses directionality of links based on shortest paths from widest inlet
    link to all the outlet links. Links are also weighted by width, such that
    deviations from the "main channel" width cost more to traverse.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.
    """
    # alg = 4 # identifier for diagnostics
    alg = algmap('main_chans')

    inlet_idx = widest_inlet_index(links, nodes)

    # More weight given to longer and narrower channels
    Aweight = (np.max(links['wid_adj']) - links['wid_adj'])
    weights = Aweight * links['len']

    # Create networkX graph, adding weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], weights):
        G.add_edge(lc[0], lc[1], weight=wt)

    for o in nodes['outlets']:

        pathnodes = nx.dijkstra_path(G, nodes['inlets'][inlet_idx], o, weight='weight')
        pathlinks = nodepath_to_links(pathnodes, links, nodes)

        # Set the directionality of each of the links
        for usnode, pl in zip(pathnodes, pathlinks):

            linkidx = links['id'].index(pl)

            # Don't set if already set
            if alg in links['guess_alg'][linkidx]:
                continue
            else:
                # Store guess
                links['guess'][linkidx].append(usnode)
                links['guess_alg'][linkidx].append(alg)

    return links, nodes


def dir_shortest_paths_nodes(links, nodes):
    """
    Guesses link directionality based on the shortest path from its end
    nodes to the nearest outlet (or pre-outlet). For each unknown link, if the
    path flows through the link attached to the node (as opposed to immediately
    away from the link), its directionality is set; otherwise nothing
    is done. Note that this will not set all links' directionalities.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.
    """

    # alg = 12 # identifier for diagnostics
    alg = algmap('sp_nodes')

    # Create networkX graph, adding weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], links['len']):
        G.add_edge(lc[0], lc[1], weight=wt)

    # Get all "pre-outlet", i.e. nodes one link upstream of outlets. Use these so that decision of where to chop off outlet links doesn't play a role in shortest path.
    preoutlets = []
    for o in nodes['outlets']:
        linkconn = links['conn'][links['id'].index(nodes['conn'][nodes['id'].index(o)][0])]
        othernode = linkconn[:]
        othernode.remove(o)
        preoutlets.append(othernode[0])

    # All shortest paths
    msdpl = nx.multi_source_dijkstra_path(G, preoutlets)

    # Loop through all nodes; the directionality of the first link
    # flowed through to reach the nearest outlet (or preoutlet) node is set
    for nid, nidx, nconn in zip(nodes['id'], nodes['idx'], nodes['conn']):

        if nid in nodes['outlets'] or nid in nodes['inlets'] or nid in preoutlets:
            continue

        # Get the shortest path from nid to the nearest outlet or preoutlet
        shortpath = msdpl[nid][::-1]

        # Set first link of the shortest path
        # Find the link first
        for ip, posslink in enumerate(nconn):
            if set(links['conn'][links['id'].index(posslink)]) == set(shortpath[:2]):
                linkid = nconn[ip]

        linkidx = links['id'].index(linkid)

        if alg in links['guess_alg'][linkidx]:
            # If the guess agrees with previously guessed for this algorithm, move on
            if links['guess'][linkidx][links['guess_alg'].index(alg)] == nid:
                continue
            else:
                links['guess'][linkidx].remove(nid)
        else:
            # Update certainty
            links['guess'][linkidx].append(nid)
            links['guess_alg'][linkidx].append(alg)

    return links, nodes


def dir_shortest_paths_links(links, nodes, difthresh=0):
    """
    Guesses a link's directionality by which of its endpoint nodes is
    closer to the nearest outlet (or preoutlet). "difthresh" refers to the
    difference in distances between endpoint nodes; higher means directionality
    is less likely to be ascertained.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    difthresh : int, optional
        Threshold for the length (in number of nodes) that must be surpassed
        between an unknown link's endpoints for directionality to be set.
        Lengths are measured from the link's endpoints to the nearest
        outlet (or preoutlet). The default is 0.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.
    """
    # alg = 11 # identifier for diagnostics
    alg = algmap('sp_links')

    # Create networkX graph, adding weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], links['len']):
        G.add_edge(lc[0], lc[1], weight=wt)

    # Get all "pre-outlet", i.e. nodes one link upstream of outlets. Use these so that decision of where to chop off outlet links doesn't play a role in shortest path.
    preoutlets = []
    for o in nodes['outlets']:
        linkconn = links['conn'][links['id'].index(nodes['conn'][nodes['id'].index(o)][0])]
        othernode = linkconn[:]
        othernode.remove(o)
        preoutlets.append(othernode[0])

    # NetworkX's multi-source dijkstra path length iterator
    msdpl = nx.multi_source_dijkstra_path_length(G, preoutlets)

    # METHOD 2: Direction ascertained based on the nearness of a link's endnodes
    # to the nearest outlet [closer is downstream]
    for il,lid in enumerate(links['id']):

        linkidx = links['id'].index(lid)
        lconn = links['conn'][linkidx][:]

        # Compute shortest distance from each end node to nearest pre-outlet
        node_min_len = []
        for endnode in lconn:
            node_min_len.append(msdpl[endnode])

        # Skip if the minima are too similar
        if abs(node_min_len[0] - node_min_len[1]) < difthresh:
            continue
        else:  # DS node is the closer one
            dsnode = lconn[node_min_len.index(min(node_min_len))]
            usnode = list(set(lconn) - set([dsnode]))[0]

            # Update certainty
            links['guess'][linkidx].append(usnode)
            links['guess_alg'][linkidx].append(alg)

    return links, nodes


def dir_known_link_angles(links, nodes, dims, checklinks='all'):
    """
    Guesses directionality for unknown links whose immediately adjacent links'
    directionalities are known. Computes the angle of "flow" between outlet and
    inlet end nodes for each neighboring link, then sets the unknown link
    such that its orientation minimizes the error between its neighbors and
    itself.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    dims : tuple
        Shape of the input mask as (nrows, ncols).
    checklinks : list OR str, optional
        A list of link ids can be provided to check only specific links.
        Otherwise, the default is 'all' which checks all links.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.
    """

    def angle_between(v1, v2):
        """
        Returns the angle in degrees between vectors v1 and v2. Angle is computed
        in the clockwise direction from v1.
        """

        ang1 = np.arctan2(*v1[::-1])
        ang2 = np.arctan2(*v2[::-1])

        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    # alg = 10 # identifier for diagnostics
    alg = algmap('syn_dem')

    if checklinks == 'all':
        checklinks = links['id']

    linkangs = np.ones((len(links['id']), 1)) * np.nan
    for lid in checklinks:
        linkidx = links['id'].index(lid)
        conn = links['conn'][linkidx]
        lidcs = links['idx'][linkidx]

        # Ensure that all the directionalities of links connected to this one are known
        connlinks = set()
        for c in conn:
            linkconn = nodes['conn'][nodes['id'].index(c)][:]
            linkconn.remove(lid)
            connlinks.update(linkconn)

        certs = []
        for cl in connlinks:
            certs.append(links['certain'][links['id'].index(cl)])

        if sum(certs) != len(certs):
            continue

        # Coordinates of unknown link
        rc_idcs = np.unravel_index(lidcs, dims)

        # Get angles of all connected links (oriented up-to-downstream)
        angs = []
        horiz_vec = (1, 0)
        for l in connlinks:
            linkidxt = links['id'].index(l)
            lidcst = links['idx'][linkidxt]
            rc = np.unravel_index(lidcst, dims)
            # Vector is downstream node minus upstream node
            ep_vec = (rc[1][-1]-rc[1][0], rc[0][0] - rc[0][-1])
            angs.append(angle_between(ep_vec, horiz_vec))

        # Compute angle for both orientations of unknown link
        poss_angs = []
        for orient in [0, 1]:
            if orient == 0:
                ep_vec = (rc_idcs[1][-1]-rc_idcs[1][0], rc_idcs[0][0] - rc_idcs[0][-1])
            else:
                ep_vec = (rc_idcs[1][0]-rc_idcs[1][-1], rc_idcs[0][-1] - rc_idcs[0][0])
            poss_angs.append(angle_between(ep_vec, horiz_vec))

        # Compute the error
        err = [np.sqrt(np.sum((pa-angs)**2)) for pa in poss_angs]

        # Choose orientation with smallest error
        usnode = conn[err.index(min(err))]

        linkangs[linkidx] = min(err)
        links['guess'][linkidx].append(usnode)
        links['guess_alg'][linkidx].append(alg)

    return links, nodes


def dir_bridges(links, nodes):
    """
    Guesses directionality of bridge links. Bridge links are those which flow
    must pass through to reach the outlet; in other words, if a bridge link is
    removed from the network, there will no longer be just one connected
    network. Directionality is inferred by consideration of which subnetwork
    (after removing the bridge link) contains inlet and outlet nodes.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.
    """
    # alg = 5 # identifier for diagnostic
    alg = algmap('bridges')

    # Create networkX graph object
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, l in zip(links['conn'], links['len']):
        G.add_edge(lc[0], lc[1], weight=l)

    # Find bridge links; we don't want to count inlet and outlet links
    preoutlets = []
    for o in nodes['outlets']:
        linkconn = links['conn'][links['id'].index(nodes['conn'][nodes['id'].index(o)][0])]
        othernode = linkconn[:]
        othernode.remove(o)
        preoutlets.append(othernode[0])

    bridges = list(nx.bridges(G))
    bridgenodes = []
    endnodes = set(nodes['outlets']) | set(nodes['inlets'])
    for b in bridges:
        bset = set(b) - endnodes
        if len(bset) == 2:
            bridgenodes.append(b)

    bridgelinks = []
    for bn in bridgenodes:
        conn = nodes['conn'][nodes['id'].index(bn[0])]
        for c in conn:
            if bn[1] in links['conn'][links['id'].index(c)]:
                bridgelinks.append(c)
                break

    # Guess the bridge link
    for bl, bn in zip(bridgelinks, bridgenodes):
        # Remove edge from graph
        G.remove_edge(bn[0], bn[1])
        # See which bridgenode connects to upstream node(s) -
        # Must account for possibility of multiple inlets, so must check both nodes
        bn0_up, bn0_down, bn1_up, bn1_down = False, False, False, False
        for o in nodes['outlets']:
            if nx.has_path(G, bn[0], o) is True:
                bn0_down = True
            if nx.has_path(G, bn[1], o) is True:
                bn1_down = True
        for i in nodes['inlets']:
            if nx.has_path(G, bn[0], i) is True:
                bn0_up = True
            if nx.has_path(G, bn[1], i) is True:
                bn1_up = True

        # There are rare cases where the bridge link cannot be set (if both sides of the bridge have an inlet and outlet)
        # In these cases, do nothing.
        if bn0_down is True and bn1_up is True and bn0_up is True and bn1_down is True:
            # Add the edge back to graph
            G.add_edge(bn[0], bn[1])
            # Skip
            continue

        if bn0_down is False or bn1_up is False:
            usnode = bn[0]
        elif bn0_up is False or bn1_down is False:
            usnode = bn[1]

        blidx = links['id'].index(bl)
        links['guess_alg'][blidx].append(alg)
        links['guess'][blidx].append(usnode)

        # Add the edge back to graph
        G.add_edge(bn[0], bn[1])

    return links, nodes


def cycle_get_original_orientation(links, lids):
    """
    Saves the orientation of a set of links. Used before attempting to fix
    cycles. Properties besides 'conn' must also be reversed, so their original
    orientations are also stored.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    lids : list
        Link ids whose orientations should be saved.

    Returns
    -------
    orig : dict
        A subset of the links dictionary containing the orientations of each
        link in lids.
    """

    lidx = [links['id'].index(l) for l in lids]
    orig = dict()
    orig['id'] = lids
    orig['conn'] = [links['conn'][l][:] for l in lidx]
    orig['idx'] = [links['idx'][l][:] for l in lidx]
    orig['wid_pix'] = [links['wid_pix'][l][:] for l in lidx]
    orig['certain_alg'] = [links['certain_alg'][l] for l in lidx]
    orig['certain_order'] = [links['certain_order'][l] for l in lidx]

    return orig


def cycle_return_to_original_orientation(links, orig):
    """
    Returns a set of links to its original orientation.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    orig : dict
        Network links and associated properties to return to original
        orientations.

    Returns
    -------
    links : dict
        Network links and associated properties in their original orientations.
    """

    for i, oid in enumerate(orig['id']):
        lidx = links['id'].index(oid)
        links['conn'][lidx] = orig['conn'][i][:]
        links['idx'][lidx] = orig['idx'][i][:]
        links['wid_pix'][lidx] = orig['wid_pix'][i][:]
        links['certain_alg'][lidx] = orig['certain_alg'][i]
        links['certain_order'][lidx] = orig['certain_order'][i]

    return links


def merge_list_of_lists(inlist):
    """
    Merges a list of lists into a single list, but removes duplicates.
    Uses networkx and itertools for a fun solution.

    Parameters
    ----------
    inlist : list of lists
        List of lists to merge without dulplicates.

    Returns
    -------
    merged : list
        List of length equal to number of unique entries across all lists of
        inlist.
    """

    # Combine overlapping cycles (where cycles share the same nodes, join them)
    from itertools import combinations_with_replacement, chain
    import networkx as nx

    edges = chain.from_iterable(combinations_with_replacement(set(n),2) for n in inlist)
    G = nx.Graph(edges)
    merged = [list(n) for n in nx.connected_components(G)]

    return merged


def set_link(links, nodes, linkidx, usnode, alg=9999, checkcontinuity=True):
    """
    Sets a link directionality. Option to check for continuity (including
    parallel links) after setting.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    linkidx : int
        Index of link within links['id'] to set. This index can be found from
        the link id via links['id'].index(linkid).
    usnode : int
        Node id of the upstream node.
    alg : int, optional
        The algorithm id used to set the link. Used for diagnostic purposes and
        fixing cycles. The default is 9999.
    checkcontinuity : bool, optional
        If True, checks nearby links for continuity and parallel links after
        setting the link's direction. The default is True.

    Returns
    -------
    links : dict
        Network links and associated properties with updated link direction.
    nodes : dict
        Network nodes and associated properties with updated link direction.

    """
    links['conn'][linkidx].remove(usnode)
    links['conn'][linkidx].insert(0, usnode)
    if links['idx'][linkidx][0] != nodes['idx'][nodes['id'].index(usnode)]:
        links['idx'][linkidx] = links['idx'][linkidx][::-1]
        if 'wid_pix' in links.keys():
            links['wid_pix'][linkidx] = links['wid_pix'][linkidx][::-1]

    links['certain'][linkidx] = 1
    links['certain_alg'][linkidx] = alg
    links['certain_order'][linkidx] = max(links['certain_order']) + 1

    if checkcontinuity is True:
        # Set any other possible links via continuity
        links, nodes = set_continuity(links, nodes, checknodes=links['conn'][linkidx][:])

        # Also set parallel links connected to this one, or if this link is part
        # of a parallel set, set the others in the set
        links, nodes = set_parallel_links(links, nodes, knownlink=links['id'][linkidx])

    return links, nodes


def fix_sources_and_sinks(links, nodes):
    """
    Fixes sources and sinks within the network by flipping link directionality
    near the problem node(s). The links to flip are chosen by ensuring they do
    not create a cycle; if multiple links can be flipped, the shortest one is
    chosen. If no solution is found, links are returned to their original
    orientation.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    links : dict
        Network links and associated properties with sources/sinks fixed if
        possible.
    nodes : dict
        Network links and associated properties with sources/sinks fixed if
        possible.

    """
    badnodes = check_continuity(links, nodes)

    for bn in badnodes:

        linkidx = None
        n_bn = len(check_continuity(links, nodes))

        # Get all the connected links
        lconn = nodes['conn'][nodes['id'].index(bn)]

        # Reverse their order and see if we've violated continuity or created a cycle
        bn_linkflip = []  # number of bad nodes after flipping link
        cycle_linkflip = []
        for l in lconn:
            lidx = links['id'].index(l)
            links = lnu.flip_link(links, l)

            # See if we've violated continuity after flipping the link
            badnodes_temp = check_continuity(links, nodes)
            bn_linkflip.append(len(badnodes_temp))

            # See if we've created a cycle after flipping the link
            endnodes = links['conn'][lidx][:]
            c_nodes, _ = get_cycles(links, nodes, endnodes[0])
            c_nodes2, _ = get_cycles(links, nodes, endnodes[1])

            if c_nodes or c_nodes2:
                cycle_linkflip.append(1)
            else:
                cycle_linkflip.append(0)

            # Re-flip links to original position
            links = lnu.flip_link(links, l)

        # Now check if any of the flipped links solves the bad nodes AND doesn't
        # create a cycle--use that orientation if so
        poss_bn = [l for l, bnlf in zip(lconn, bn_linkflip) if bnlf + 1 == n_bn]
        poss_cy = [l for l, clf in zip(lconn, cycle_linkflip) if clf == 0]
        poss_links = list(set(poss_bn).intersection(set(poss_cy)))

        if len(poss_links) == 0:  # No possible links to flip, move on
            continue
        elif len(poss_links) == 1:  # Only one possible link we can flip, so do it
            linkidx = links['id'].index(poss_links[0])
        else:  # More than one link meets the criteria; choose the shortest
            linklens = [links['len'][links['id'].index(l)] for l in poss_links]
            linkidx = links['id'].index(poss_links[linklens.index(min(linklens))])

        if linkidx:
            set_link(links, nodes, linkidx, links['conn'][linkidx][1], alg=algmap('sourcesinkfix'))

    return links, nodes


def check_continuity(links, nodes):
    """
    Finds all sinks or sources within the network, excluding inlets and outlets.
    Returns any nodes where continuity is violated. Only checks nodes for whom
    all attached links are certain.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    problem_nodes : list
        Node ids corresponding to nodes['id'] where continuity is violated.

    """
    problem_nodes = []
    for nid, nidx, nconn in zip(nodes['id'], nodes['idx'], nodes['conn']):

        if nid in nodes['outlets'] or nid in nodes['inlets']:
            continue

        certains = [links['certain'][links['id'].index(lid)] for lid in nconn]
        if np.sum(certains) != len(nconn):
            continue

        firstidx = []
        lastidx = []
        for linkid in nconn:
            linkidx = links['id'].index(linkid)
            firstidx.append(links['idx'][linkidx][0])
            lastidx.append(links['idx'][linkidx][-1])

        if firstidx[1:] == firstidx[:-1]:
            problem_nodes.append(nid)
        elif lastidx[1:] == lastidx[:-1]:
            problem_nodes.append(nid)

    return problem_nodes


def find_a_cycle(links, nodes):
    """
    Finds a single cycle in the network. Multiple cycles may exist; this
    function returns only the first one encountered.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    cycle_nodes : list
        Node ids comprising the cycle. Arranged in order of cycle.
    cycle_links : list
        Link ids comprising the cycle. Arranged in order of cycle.

    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc in links['conn']:
        G.add_edge(lc[0], lc[1])

    cycle_nodes = nx.find_cycle(G)

    us = [c[0] for c in cycle_nodes]
    vs = [c[1] for c in cycle_nodes]
    cycle_links = []
    for u, v in zip(us, vs):
            ulinks = nodes['conn'][nodes['id'].index(u)]
            vlinks = nodes['conn'][nodes['id'].index(v)]
            cycle_links.append([ul for ul in ulinks if ul in vlinks][0])

    cycle_nodes = [c[0] for c in cycle_nodes]

    return cycle_nodes, cycle_links


def get_cycles(links, nodes, checknode='all'):
    """
    Finds either all cycles in the network or cycles containing the checknode'th
    node. Cycles are returned as both nodes and links.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    checknode : int OR str, optional
        ID of node to check for its inclusion in any cycles. If the default
        value 'all' is selected, all cycles will be returned.

    Returns
    -------
    cycle_nodes : list of lists
        List containing lists of all cycles found in the network; cycles are
        returned in cyclic order and in terms of node ids.
    cycles_links : TYPE
        List containing lists of all cycles found in the network; cycles are
        returned in cyclic order and in terms of link ids.

    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc in links['conn']:
        G.add_edge(lc[0], lc[1])

    if checknode == 'all':
        cycle_nodes = nx.simple_cycles(G)
        # Unpack the iterator
        cycle_nodes = list(cycle_nodes)
    else:
        try:
            single_cycle = nx.find_cycle(G, source=checknode)
            single_cycle = list(single_cycle)
            cycle_nodes = []
            for cn in single_cycle:
                cycle_nodes.append(cn[0])
            cycle_nodes = [cycle_nodes]
        except Exception:
            cycle_nodes = None

    # Get links of cycles
    cycles_links = []
    if cycle_nodes is not None:
        for c in cycle_nodes:
            pathlinks = []
            for us, vs in zip(c, c[1:] + [c[0]]):
                ulinks = nodes['conn'][nodes['id'].index(us)]
                vlinks = nodes['conn'][nodes['id'].index(vs)]
                pathlinks.append([ul for ul in ulinks if ul in vlinks][0])
            cycles_links.append(pathlinks)
    else:
        cycles_links = cycle_nodes

    return cycle_nodes, cycles_links


def flip_links_in_G(G, links2flip):
    """
    Flips the directionality of links in a networkx graph object.

    Parameters
    ----------
    G : networkx.Graph or similar
        Graph object representing the network.
    links2flip : tuple or
        An Nx2 tuple containing the US and DS node ids of the edge (link) to
        flip.

    Returns
    -------
    G : networkx.Graph or similar
        Graph object representing the network with the requested links flipped.

    """
    if links2flip == 'all':
        links2flip = list(G.edges)

    for lf in links2flip:
        # Remove the link
        G.remove_edge(lf[0], lf[1])
        # Add it, but reversed
        G.add_edge(lf[1], lf[0])

    return G


def fix_cycles(links, nodes):
    """
    Attempts to remove all cycles in the directed graph. A number of approaches
    are tried, and if any of them result in a source/sink node or a cycle,
    another approach is attempted until the cycle is resolved or we run out
    of approaches.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    links : dict
        Network links and associated properties with all possible cycles
        resolved.
    nodes : dict
        Network nodes and associated properties with all possible cycles
        resolved
    n_cycles_remaining : int
        Number of cycles that were unresolvable.

    """
    # Create networkx graph object
    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc in links['conn']:
        G.add_edge(lc[0], lc[1])

    # Check for cycles
    if nx.is_directed_acyclic_graph(G) is not True:

        # Get list of cycles to fix
        c_nodes, c_links = get_cycles(links, nodes)

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

        # Fix all the cycles
        for cycle_n, cycle_l in zip(cfix_nodes, cfix_links):

            # We will try every combination of flow directions and check each
            # combination for continuity violation and cycles, subsetting possibilities
            # to only those configurations that don't violate either.

            # First, subset our network to make computations faster; put subset
            # into networkX graph
            acl = []
            for nid in cycle_n:
                acl.extend(nodes['conn'][nodes['id'].index(nid)])
            all_cycle_links = set(acl)
            acn = []
            for lid in all_cycle_links:
                acn.extend(links['conn'][links['id'].index(lid)])
            all_cycle_nodes = set(acn)
            G = nx.DiGraph()
            G.add_nodes_from(all_cycle_nodes)
            for lid in all_cycle_links:
                lidx = links['id'].index(lid)
                lc = links['conn'][lidx]
                G.add_edge(lc[0], lc[1])

            # Determine our "inlet" and "outlet" links/nodes - these we will not flip
            dangle_nodes = all_cycle_nodes - set(cycle_n)
            dangle_links = [l for l in all_cycle_links if len(set(links['conn'][links['id'].index(l)]) - dangle_nodes) == 1]
            ins_l = []
            outs_l = []
            ins_n = []
            outs_n = []
            dns = []
            for dl in dangle_links:
                lidx = links['id'].index(dl)
                lconn = links['conn'][lidx]
                dn = list(set(lconn).intersection(dangle_nodes))[0]
                dns.append(dn)
                if dn == lconn[0]:
                    ins_l.append(dl)
                    ins_n.append(dn)
                else:
                    outs_l.append(dl)
                    outs_n.append(dn)

            # Try all configurations of links and count the number of cycles/continuity violations (don't change dangle links)
            # First, find all combinations
            fliplinks = list(all_cycle_links - set(dangle_links))
            all_combos = []
            for L in range(1, len(fliplinks)+1):
                for subset in itertools.combinations(fliplinks, L):
                    all_combos.append(subset)

            # If a cycle is too big, there will be too many combinations to
            # feasibly check all possibilities on a single processor. For now,
            # we just skip these and report them so they can be manually
            # corrected.
            if len(all_combos) > 1024:
                print('The cycle links {} is too large to attempt to fix automatically.'.format(cycle_l))
                continue

            # Iterate through each combination and determine violations: there are four conditions that must be met:
            # 1) no cycles,
            # 2) no sources/sinks
            # 3) all inlets must be able to drain to an outlet, and all outlets must be reachable from at least one inlet (inlets and outlets refer to those of the subset, not the entire graph)
            # 4) links cannot flow in opposition to manually set directions
            cont_violators = []
            len_cycle = []
            has_path = []
            manually_set = []
            for flink in all_combos:

                # Flip all the links in flink
                links2flip = []
                for fl in flink:
                    links2flip.append(links['conn'][links['id'].index(fl)][:])
                G = flip_links_in_G(G, links2flip)

                # Check if a cycle exists
                try:
                    cycles_temp = nx.find_cycle(G)
                    len_cycle.append(len(cycles_temp))
                except:
                    len_cycle.append(0)

                # Check if continuity is violated
                sink_nodes = set([node for node, outdegree in list(G.out_degree) if outdegree == 0]) - dangle_nodes
                source_nodes = set([node for node, indegree in list(G.in_degree) if indegree == 0]) - dangle_nodes
                cont_violators.append(len(sink_nodes) + len(source_nodes))

                # Check if each inflow can reach an outflow and each outflow is reached by an inflow
                hp_ins = []
                for ii in ins_n:
                    for oo in outs_n:
                        if nx.has_path(G, ii, oo) is True:
                            hp_ins.append(1)
                            break
                # Flip links to test outlets
                G = flip_links_in_G(G, links2flip='all')
                hp_outs = []
                for oo in outs_n:
                    for ii in ins_n:
                        if nx.has_path(G, oo, ii) is True:
                            hp_outs.append(1)
                            break
                # Flip em back
                G = flip_links_in_G(G, links2flip='all')

                # Assign a "1" where inlet/outlet criteria are met
                if len(ins_n) == sum(hp_ins) and len(outs_n) == sum(hp_outs):
                    has_path.append(1)
                else:
                    has_path.append(0)

                # Check that we're not flipping any links that have been set manually
                set_by_alg = []
                for fl in flink:
                    set_by_alg.append(links['certain_alg'][links['id'].index(fl)])
                if algmap('manual_set') in set_by_alg:
                    manually_set.append(1)
                else:
                    manually_set.append(0)

                # Flip links back to original
                links2flipback = []
                for fl in flink:
                    c = links['conn'][links['id'].index(fl)][:]
                    links2flipback.append([c[1], c[0]])
                G = flip_links_in_G(G, links2flipback)

            # Find configurations that don't violate continuity and have no cycles
            poss_configs = [i for i, (nv, nc, hp, ms) in enumerate(zip(cont_violators, len_cycle, has_path, manually_set)) if nv == 0 and nc == 0 and hp == 1 and ms == 0]

            if len(poss_configs) == 0:
                print('Unfixable cycle found at links: {}.'.format(cycle_l))
                continue

            # Choose the configuration that flips the fewest links
            pc_lens = [len(all_combos[pc]) for pc in poss_configs]
            links_to_flip = all_combos[poss_configs[pc_lens.index(min(pc_lens))]]

            # Flip the links to fix the cycle
            for l in links_to_flip:
                links = lnu.flip_link(links, l)

    # Check if any cycles remain
    c_nodes, _ = get_cycles(links, nodes)
    if c_nodes is None:
        n_cycles_remaining = 0
    else:
        n_cycles_remaining = len(c_nodes)

    return links, nodes, n_cycles_remaining


def dir_set_manually(links, nodes, manual_set_csv):
    """
    Sets link directions based on a user-provided csv-file. The csv file has
    exactly two columns; one called 'link_id', and one called 'usnode'.
    This file can be created automatically by the function
    io_utils.create_manual_dir_csv().

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    manual_set_csv : str
        Path to csv defining which links have which upstream nodes.

    Returns
    -------
    links : dict
        Network links and associated properties with specified links set
        manually.
    nodes : dict
        Network nodes and associated properties with specified links set
        manually.

    """
    # alg = -1
    alg = algmap('manual_set')

    # Read the csv file for fixing link directions.
    if os.path.isfile(manual_set_csv) is False:
        print('No file found for manually setting link directions.')
        return links, nodes
    else:
        print('Using {} to manually set flow directions.'.format(manual_set_csv))

    df = pd.read_csv(manual_set_csv)

    # Check if any links have been manually corrected and correct them
    if len(df) != 0:
        usnodes = df['usnode'].values
        links_to_set = df['link_id'].values

        for lid, usn in zip(links_to_set, usnodes):
            links, nodes = set_link(links, nodes, links['id'].index(lid), usn,
                                    alg=alg)

    return links, nodes


def set_inletoutlet(links, nodes):
    """
    Sets directions of links that are connected to inlets and outlets.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.

    """
    # alg = 0
    alg = algmap('inletoutlet')

    # Set directionality of inlet links
    for i in nodes['inlets']:

        # Get links attached to inlets
        conn = nodes['conn'][nodes['id'].index(i)]

        for c in conn:
            linkidx = links['id'].index(c)

            # Set link directionality
            links, nodes = set_link(links, nodes, linkidx, i, alg=alg,
                                    checkcontinuity=True)

    # Set directionality of outlet links
    for o in nodes['outlets']:

        # Get links attached to outlets
        conn = nodes['conn'][nodes['id'].index(o)]

        for c in conn:
            linkidx = links['id'].index(c)

            # Set link directionality
            usnode = links['conn'][linkidx][:]
            usnode.remove(o)
            links, nodes = set_link(links, nodes, linkidx, usnode[0], alg=alg,
                                    checkcontinuity=True)

    return links, nodes


def set_continuity(links, nodes, checknodes='all'):
    """
    Sets link directions by enforce continuity at each node such that there
    are no sources or sinks within the network. Iterates until no more links
    directions can be set.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    checknodes : list OR str, optional
        Each link connected to each node id in this list will be checked for
        continuity. All nodes will be checked if using the default value 'all'.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.

    """
    # alg = 1
    alg = algmap('continuity')

    if checknodes == 'all':
        checknodes = nodes['id'][:]

    for nid in checknodes:
        nindex = nodes['id'].index(nid)
        nidx = nodes['idx'][nindex]
        conn = nodes['conn'][nindex]

        # Initialize bookkeeping for all the links connected to this node
        linkdir = np.zeros((len(conn), 1), dtype=np.int)  # 0 if uncertain, 1 if into, 2 if out of

        if linkdir.shape[0] < 2:
            continue

        # Populate linkdir
        for il, lid in enumerate(conn):
            lidx = links['id'].index(lid)

            # Determine if link is flowing into node or out of node
            # Skip if we're uncertain about the link's direction
            if links['certain'][lidx] == 0:
                continue
            elif links['idx'][lidx][0] == nidx:  # out of
                linkdir[il] = 2

            elif links['idx'][lidx][-1] == nidx:  # into
                linkdir[il] = 1

        if np.sum(linkdir == 0) == 1: # If there is only a single unknown link
            unknown_link_id = conn[np.where(linkdir == 0)[0][0]]
            unknown_link_idx = links['id'].index(unknown_link_id)
            m = mode(linkdir[linkdir > 0])

            lconn = links['conn'][unknown_link_idx][:]

            if m.count[0] == linkdir.shape[0]-1:  # if the non-zero elements are all the same (either all 1s or 2s)

                if m.mode[0] == 1:  # The unknown link must be out of the node
                    links, nodes = set_link(links, nodes, unknown_link_idx, nid, alg=alg)

                elif m.mode[0] == 2:  # The unknown link must be into the node
                    usnode = [n for n in lconn if n != nid][0]
                    links, nodes = set_link(links, nodes, unknown_link_idx, usnode, alg=alg)

    return links, nodes


def set_parallel_links(links, nodes, knownlink):
    """
    Sets directionality of parallel links. If two links are parallel, they
    share the same end nodes. If the direction of one of the links is known,
    the other must be set in a direction to avoide creating a cycle within
    the graph. This function thus prevents cycles from being created and in
    general should be run after any link direction is set.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    knownlink : int
        Link id of link whose direction has been set to check for parallel
        links.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.

    """
    # alg = 2
    alg = algmap('parallels')

    if 'parallels' not in links.keys():
        return links, nodes
    lidx = links['id'].index(knownlink)
    docheck = False
    for parpairs in links['parallels']:
        docheck = False

        # Determine the upstream and downstream nodes for the parallel set.
        if knownlink in parpairs:  # If the knownlink is part of the set
            dsnode = links['conn'][lidx][1]
            usnode = links['conn'][lidx][0]
        else:  # If the knownlink is not part of the set
            dsnode = links['conn'][lidx][0]
            usnode = links['conn'][lidx][1]

        # Check if any parallel sets are connected to the known link
        ppnodes = links['conn'][links['id'].index(parpairs[0])][:]
        for parlink in parpairs:
            if links['certain'][links['id'].index(parlink)] == 0:
                if dsnode in ppnodes:
                    usnode_set = [n for n in ppnodes if n != dsnode][0]
                    links, nodes = set_link(links, nodes, links['id'].index(parlink), usnode_set, alg=alg, checkcontinuity=False)
                    docheck = True
                if usnode in ppnodes:
                    links, nodes = set_link(links, nodes, links['id'].index(parlink), usnode, alg=alg, checkcontinuity=False)
                    docheck = True
        if docheck is True:
            links, nodes = set_continuity(links, nodes, checknodes=ppnodes)

    return links, nodes


def get_link_vector(links, nodes, linkid, imshape, pixlen=1, normalized=True,
                    trim=False):
    """
    Computes the normalized vector of a link that indicates the direction of
    flow through a link based on its connectivity. This function will first trim
    the link by its endpoint (half)width values to mitigate cases where where wide
    channels are connected to narrow channels.

    Requires a link contain at least min_len_for_trim pixels post-trimming.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    linkid : int
        ID of the link correspoinding to those found in links['id'].
    imshape : tuple
        Shape of the binary mask as (nrows, ncols).
    pixlen : float, optional
        Length resolution of each pixel. The default is 1.
    normalized : bool, optional
        If True, will return a vector on [-1, 1]. The default is True.
    trim : bool, optional
        If True, links will be trimmed to remove possibly direction-biasing
        end segments. If False, the untrimmed link endpoints will define the
        vector direction. The default is False.

    Returns
    -------
    link_vec : np.array
        Two-element array describing the x and y components of the direction
        vector describing the link.

    """
    min_len_for_trim = 3  # number of pixels remaining after trim to use trimmed version; else defaults to using endpoints

    # Get pixel coordinates of link
    lidx = links['id'].index(linkid)
    rc = np.unravel_index(links['idx'][lidx], imshape)

    # Try to trim the link if desired and possible
    if trim is True and 'wid_pix' in links.keys():

        # Get the half-widths at the endpoints
        ep_wids = [round(links['wid_pix'][lidx][i]/pixlen/2) for i in [0, -1]]

        # Get the cumulative length of the link (forwards and reversed)
        s_for = np.cumsum(np.sqrt((np.diff(rc[1]))**2 + (np.diff(rc[0]))**2))
        s_for = np.insert(s_for, 0, 0)
        s_rev = np.cumsum(np.sqrt((np.diff(np.flipud(rc[1])))**2 + (np.diff(np.flipud(rc[0])))**2))
        s_rev = np.insert(s_rev, 0, 0)

        # Find the index along the link corresponding to each width's endpoint
        f_idx = np.argmax(s_for > ep_wids[0])
        r_idx = len(rc[0]) - np.argmax(s_rev > ep_wids[1]) - 1

        # Trim both ends if possible
        if f_idx < r_idx:
            if r_idx - f_idx >= min_len_for_trim:
                rc = [rc[0][f_idx:r_idx], rc[1][f_idx:r_idx]]
            elif f_idx > len(rc[0]) - r_idx and len(rc[0]) - r_idx >= min_len_for_trim:  # Trim only the beginning
                rc = [rc[0][f_idx:], rc[1][f_idx:]]
            elif f_idx >= min_len_for_trim:  # Trim only the end
                rc = [rc[0][0:r_idx], rc[1][0:r_idx]]

        # Trim both ends if possible; else don't trim
        if f_idx < r_idx and r_idx - f_idx >= min_len_for_trim:
            rc = [rc[0][f_idx:r_idx], rc[1][f_idx:r_idx]]

    # Use the endpoints of the trimmed (or not trimmed) pixel coordinates to
    # compute direction vector
    link_vec = np.array([rc[1][-1]-rc[1][0], rc[0][0]-rc[0][-1]])

    if normalized is True:
        link_vec = link_vec/np.sqrt(np.sum(link_vec**2))

    return link_vec


def set_by_known_flow_directions(links, nodes, imshape, angthresh=2,
                                 lenthresh=0, nknown_thresh=1,
                                 alg=algmap('known_fdr')):
    """
    Sets unknown link directions by determining which flow direction through
    the link minimizes the overall change in flow direction at the connecting
    nodes. For a set of links connected to a node, at least one of the links'
    directions must be known for this algorithm to be effective. Links are
    set from longest to shortest, as longer links generally have less
    uncertainty associated with their directions.

    A number of thresholds are provided as options to control the power of
    the algorithm. This function should generally be applied iteratively,
    beginning with the most restrictive thresholds and gradually relaxing
    them.

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    imshape : tuple
        Shape of the binary mask as (nrows, ncols).
    angthresh : float, optional
        Angle in radians to determine if an unknown link is parallel enough to
        its connected known links to set its direction. The default is 2.
    lenthresh : int, optional
        Length in pixels of the minimum link length required to use this
        algorithm to set its directionality. The default is 0.
    nknown_thresh : int, optional
        Number of known connected links required to use this algorithm to
        set the unknown link's directionality. The default is 1.
    alg : float, optional
        Algorithm ID to assign to the set link. The default is 6.

    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.

    """

    def get_candidates(links, nodes, lenthresh, nknown_thresh):
        """
        Returns unknown links that have at least one known connected link
        for setting by its direction.
        """

        unknown_links = np.array(links['id'])[np.where(links['certain'] == 0)[0]]

        if len(unknown_links) == 0:
            return []

        dolinks = []
        n_known = []
        for lid in unknown_links:

            lidx = links['id'].index(lid)
            conn = links['conn'][lidx]

            # Check both endos of the link; if for either the link in question
            # is the only unknown link; it is a candidate
            nknown = 0
            for nid in conn:
                conn_links = nodes['conn'][nodes['id'].index(nid)]
                nknown = nknown + sum([1 for c in conn_links if links['certain'][links['id'].index(c)] == 1])

            if nknown > 0:
                dolinks.append(lid)
                n_known.append(nknown)

        # Ensure there are any candidate links
        if len(dolinks) == 0:
            return []

        # Get link lengths
        lengths = [links['len'][links['id'].index(dl)] for dl in dolinks]
        # Get link lengths in pixels
        lengths_pix = [len(links['wid_pix'][links['id'].index(dl)]) for dl in dolinks]

        # Sort all dolinks by number of known connected links (max to min)
        forsort = np.transpose(np.array([dolinks, n_known, lengths, lengths_pix]))

        # Apply nknown threshold
        forsort = forsort[forsort[:, 1] >= nknown_thresh]

        # Apply (pixel) length threshold
        forsort = forsort[forsort[:, 3] >= lenthresh]

        # Do the sorting
        forsort = np.flipud(forsort[forsort[:, 1].argsort()])

        # Now sort dolinks from longer to shorter
        ns = np.flip(np.sort(list(set(forsort[:, 1]))))
        dolinks_sorted = []
        for n in ns:
            tosort = forsort[np.where(forsort[:, 1] == n)[0], :]
            tosort = np.flipud(tosort[tosort[:, 2].argsort()])
            dolinks_sorted.extend(tosort[:, 0].tolist())

        # Convert to int
        dolinks_sorted = [int(dls) for dls in dolinks_sorted]

        return dolinks_sorted

    dolinks = get_candidates(links, nodes, lenthresh, nknown_thresh)

    while len(dolinks) > 0:

        # Get the first unknown link to set
        dolink = dolinks.pop(0)
        lidx = links['id'].index(dolink)

        # In case the link was set by continuity
        if links['certain'][lidx] == 1:
            continue

        nconn = links['conn'][lidx]

        ang_guess = []
        us_node_guess = []
        for n in nconn:

            # Ensure unknown link flows from node
            if links['idx'][lidx][0] != nodes['idx'][nodes['id'].index(n)]:
                lnu.flip_link(links, dolink)

            ul_vec = get_link_vector(links, nodes, dolink, imshape)

            lconn = nodes['conn'][nodes['id'].index(n)]

            # Find the connected links that are known
            known_links = [l for l in lconn if links['certain'][links['id'].index(l)]==1]

            # Determine which set links are into/out of the node
            in_links = []
            out_links = []
            for sl in known_links:
                lidxt = links['id'].index(sl)
                if links['conn'][lidxt][0] == n:
                    out_links.append(sl)
                else:
                    in_links.append(sl)

            # Compute the vectors for into-and out-of link. If there are
            # multiple vectors, their average is used.
            in_dvec = []
            out_dvec = []
            if len(in_links) > 0:
                in_dvec = np.zeros((len(in_links), 2))
                for i, il in enumerate(in_links):
                    in_dvec[i, :] = -get_link_vector(links, nodes, il, imshape)
                in_dvec = np.mean(in_dvec, axis=0)

            if len(out_links) > 0:
                out_dvec = np.zeros((len(out_links), 2))
                for i, ol in enumerate(out_links):
                    out_dvec[i, :] = get_link_vector(links, nodes, ol, imshape)
                out_dvec = np.mean(out_dvec, axis=0)

            # Try both orientations of unknown link and compute the angles between
            # it and the known nodes
            in_angs, out_angs = [], []
            for m in [1, -1]:  # 1 is into node, -1 is out of node
                ul_vec = m * ul_vec
                if len(in_dvec) > 0:
                    in_angs.append(np.abs(np.math.atan2(np.linalg.det([in_dvec,ul_vec]),np.dot(in_dvec,ul_vec))))
                if len(out_dvec) > 0:
                    out_angs.append(np.abs(np.math.atan2(np.linalg.det([out_dvec,ul_vec]),np.dot(out_dvec,ul_vec))))

            # "Most parallel" known link is that with the smallest interior angle
            # Smallest interior angle is determined as min of both orientations of unknown link
            in_flip, out_flip = False, False
            if len(in_angs) > 0:
                in_par_angle = min(in_angs)
                if in_angs[0] > in_angs[1]:
                    in_flip = True
            else:
                in_par_angle = 100
            if len(out_angs) > 0:
                out_par_angle = min(out_angs)
                if out_angs[0] > out_angs[1]:
                    out_flip = True
            else:
                out_par_angle = 100

            # Determine us node based on most-parallel known link
            # Need to know if the minimum angle came from the flipped orientation of the unknown link to set direction
            if out_par_angle < in_par_angle:
                ang_guess.append(out_par_angle)
                if out_flip is True:
                    usng = set(nconn) - set([n])
                    us_node_guess.append(usng.pop())
                else:
                    us_node_guess.append(n)
            else:  # Unknown link flows into node
                ang_guess.append(in_par_angle)
                if in_flip is True:
                    us_node_guess.append(n)
                else:
                    usng = set(nconn) - set([n])
                    us_node_guess.append(usng.pop())

        # Set link based on best guess from all known connected links
        min_ang = min(ang_guess)
        # Ensure threshold is met
        if min_ang < angthresh:
            usnode = us_node_guess[ang_guess.index(min_ang)]
            links, nodes = set_link(links, nodes, lidx, usnode,
                                    alg = alg, checkcontinuity=True)

    return links, nodes

""" Functions below here are deprecated or unused, but might be useful later """
""" No testing performed """


def set_no_backtrack(links, nodes):

    for lid, nconn, cert in zip(links['id'], links['conn'], links['certain']):

        if cert != 1:
            continue

        if nconn[0] in nodes['inlets'] or nconn[1] in nodes['outlets']:
            continue

        uslinks = nodes['conn'][nodes['id'].index(nconn[0])][:]
        uslinks.remove(lid)
        dslinks = nodes['conn'][nodes['id'].index(nconn[1])][:]
        dslinks.remove(lid)

        us_certains = [l for l in uslinks if links['certain'][links['id'].index(l)] == 1]
        ds_certains = [l for l in dslinks if links['certain'][links['id'].index(l)] == 1]

        if len(us_certains) == len(uslinks) and len(ds_certains) != len(dslinks):
            startnode = nconn[-1]
            removelink = lid
            links, nodes = set_shortest_no_backtrack(links, nodes, startnode, removelink, 'ds')
        elif len(ds_certains) == len(dslinks) and len(us_certains) != len(uslinks):
            startnode = nconn[0]
            removelink = lid
            links, nodes = set_shortest_no_backtrack(links, nodes, startnode, removelink, 'us')

    return links, nodes


def set_shortest_no_backtrack(links, nodes, startnode, removelink, usds):

    alg = 25

    # Create networkX graph, adding weighted edges
    weights = links['len']
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], weights):
        G.add_edge(lc[0], lc[1], weight=wt)

    # Remove the immediately upstream (or downstream) known link so flow cannot
    # travel the wrong direction
    rem_edge = links['conn'][links['id'].index(removelink)]
    G.remove_edge(rem_edge[0], rem_edge[1])

    # Find the endnode to travel to
    if usds == 'ds':
        endpoints = nodes['outlets']
    else:
        endpoints = nodes['inlets']

    if len(endpoints) == 1:
        endnode = endpoints[0]
    else:
        len_to_ep = []
        for ep in endpoints:
            len_to_ep.append(nx.dijkstra_path_length(G, startnode, ep))
        endnode = endpoints[len_to_ep.index(min(len_to_ep))]

    # Get shortest path nodes
    pathnodes = nx.dijkstra_path(G, startnode, endnode, weight='weight')

    # Convert to link-to-link path
    pathlinks = []
    for u,v in zip(pathnodes[0:-1], pathnodes[1:]):
        ulinks = nodes['conn'][nodes['id'].index(u)]
        vlinks = nodes['conn'][nodes['id'].index(v)]
        pathlinks.append([ul for ul in ulinks if ul in vlinks][0])

    # Set the directionality of each of the links
    if usds == 'ds':
        pathnodes = pathnodes[0:-1]
    else:
        pathnodes = pathnodes[1:]

    for usnode, pl in zip(pathnodes, pathlinks):
        linkidx = links['id'].index(pl)
        if links['certain'][linkidx] == 1:
            break
        else:
            links, nodes = set_link(links, nodes, linkidx, usnode, alg = alg)

    return links, nodes


def set_artificial_nodes(links, nodes, checknodes='all'):
    """
    This function is deprecated as of v0.3 and is no longer called.

    Set the directionality of links where aritificial nodes were added. For such
    loops, flow will travel the same way through both sides of the loop (to avoid
    cycles). Therefore, if one side is known, we can set the other side.
    Method 1 sets a broken link if its counterpart is known.
    Method 2 sets a side of the loop if the other side is known.
    Method 3 sets both sides if the input to one of the end nodes is known.
    Parameters
    ----------
    links : dict
        Network links and associated properties.
    nodes : dict
        Network nodes and associated properties.
    checknodes : int or str, optional
        Node ids to check for presence of settable artificial links. If 'all',
        all nodes in the network are checked.
    Returns
    -------
    links : dict
        Network links and associated properties updated to set directionality
        according to this algorithm.
    nodes : dict
        Network nodes and associated properties updated to set directionality
        according to this algorithm.
    """

    alg = 2.1

    for n in checknodes:

        if n in nodes['inlets'] or n in nodes['outlets']:
            continue

        # Determine if we're at a head node of an artificial loop
        nidx = nodes['id'].index(n)
        linkconn = nodes['conn'][nidx][:]
        for lc in linkconn:
            nodecheck = links['conn'][links['id'].index(lc)][:]
            nodecheck.remove(n)
            # If a neighboring node is an aritifical one, we're at a head node
            if nodecheck[0] in nodes['arts']:
                a_node = nodecheck[0]

        # If there is no artificial node, move to next
        try:
            a_node
        except NameError:
            continue

        # Ensure the non-artificial links are known and all flow either into
        # or out of nead node
        artlinks = links['arts'][nodes['arts'].index(a_node)]
        nonartlinks = [l for l in linkconn if l not in artlinks]

        # Ensure nonartificial links are known
        certs = [links['certain'][links['id'].index(nal)] for nal in nonartlinks]
        if sum(certs) != len(certs):
            continue

        # Check that non-artificial links are same directionality wrt head node
        firstidcs = set([links['idx'][links['id'].index(nal)][0] for nal in nonartlinks])
        lastidcs = set([links['idx'][links['id'].index(nal)][-1] for nal in nonartlinks])
        if len(firstidcs) == 1 and list(firstidcs)[0] == nodes['idx'][nidx]:# Links are leaving head node
            inout = 'out'
        elif len(lastidcs) == 1 and list(lastidcs)[0] == nodes['idx'][nidx]:
            inout = 'in'
        else:
            continue

        # Determine the short links of the artificial link triad
        shortlinks = nodes['conn'][nodes['id'].index(a_node)][:]
        endnodes = []
        for sl in shortlinks:
            forappend = links['conn'][links['id'].index(sl)][:]
            forappend.remove(a_node)
            endnodes.append(forappend[0])

        # Find corresponding long link of the triad
        posslinks = nodes['conn'][nodes['id'].index(endnodes[0])]
        for p in posslinks:
            linkidx = links['id'].index(p)
            if set(links['conn'][linkidx]) == set(endnodes):
                longlink = p

        # Set the longlink
        ll_idx = links['id'].index(longlink)
        ll_conn = links['conn'][ll_idx][:]
        if inout == 'out':
            ll_conn.remove(n)
            usnode = ll_conn[0]
        elif inout == 'in':
            usnode = n
        if links['certain'][ll_idx] == 0:
            links, nodes = set_link(links, nodes, ll_idx, usnode, alg=alg, checkcontinuity=False)

        # Set short link that shares a node with longlink and link_into
        # The final shortlink of the triad will be set by continuity
        shortlink = [l for l in shortlinks if n in links['conn'][links['id'].index(l)]][0]
        slidx = links['id'].index(shortlink)
        sl_conn = links['conn'][slidx][:]
        if inout == 'out':
            sl_conn.remove(n)
            usnode = sl_conn[0]
        elif inout == 'in':
            usnode = n
        if links['certain'][slidx] == 0:
            links, nodes = set_link(links, nodes, slidx, usnode, alg=alg)

        # Now check continuity at all the artificial nodes
        links, nodes = set_continuity(links, nodes, checknodes=endnodes)

    return links, nodes
