# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:31:01 2018

@author: Jon
"""

from rivgraph.rivers import river_utils as ru
import rivgraph.ln_utils as lnu
import rivgraph.geo_utils as gu
import rivgraph.io_utils as io
import rivgraph.directionality as dy
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import os
import networkx as nx

#brahma = b
#links = brahma.links
#nodes = brahma.nodes
#Imask = brahma.Imask
#exit_sides = brahma.exit_sides
#gt = brahma.gt
#meshlines = brahma.meshlines
#meshpolys = brahma.meshpolys
#Idt = brahma.Idist
#path_csv='blah'


def set_directionality(links, nodes, Imask, exit_sides, gt, meshlines, meshpolys, Idt, pixlen, path_csv):
    
    imshape = Imask.shape

    # Add fields to links dict for tracking and setting directionality
    links['certain'] = np.zeros(len(links['id'])) # tracks whether a link's directionality is certain or not
    links['certain_order'] = np.zeros(len(links['id'])) # tracks the order in which links certainty is set
    links['certain_alg'] = np.zeros(len(links['id'])) # tracks the algorithm used to set certainty
    links['maxang'] = np.ones(len(links['id'])) * np.nan # saves the angle used in set_by_flow_directions, diagnostic only

    # Append morphological information used to set directionality to links dict
    links, nodes = directional_info(links, nodes, Imask, pixlen, exit_sides, gt, meshlines, meshpolys, Idt)
 
    # Begin setting link directionality
    # First, set inlet/outlet directions as they are always 100% accurate    
    links, nodes = dy.set_inletoutlet(links, nodes)
            
    # Set the directions of the links that are more certain via centerline distance method
    alg = 22
    cl_distthresh = np.percentile(links['cldists'], 85)
    for lid, cld, lg, lga, cert in zip(links['id'],  links['cldists'], links['guess'], links['guess_alg'], links['certain']):
        if cert == 1:
            continue
        if cld >= cl_distthresh:
            linkidx = links['id'].index(lid)
            if 20 in lga:
                usnode = lg[lga.index(20)]
                links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)

    # Set the directions of the links that are more certain via centerline angle method
    alg = 23
    cl_angthresh = np.percentile(links['clangs'][np.isnan(links['clangs'])==0], 25)
    for lid, cla, lg, lga, cert in zip(links['id'],  links['clangs'], links['guess'], links['guess_alg'], links['certain']):
        if cert == 1:
            continue
        if np.isnan(cla) == True:
            continue
        if cla <= cl_angthresh:
            linkidx = links['id'].index(lid)
            if 21 in lga:
                usnode = lg[lga.index(21)]
                links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)
    
    # Set the directions of the links that are more certain via centerline distance AND centerline angle methods
    alg = 24
    cl_distthresh = np.percentile(links['cldists'], 70)
    ang_thresh = np.percentile(links['clangs'][np.isnan(links['clangs'])==0], 35)
    for lid, cld, cla, lg, lga, cert in zip(links['id'],  links['cldists'], links['clangs'], links['guess'], links['guess_alg'], links['certain']):
        if cert == 1:
            continue
        if cld >= cl_distthresh and cla < ang_thresh:
            linkidx = links['id'].index(lid)
            if 20 in lga and 21 in lga:
                if lg[lga.index(20)] == lg[lga.index(21)]:
                    usnode = lg[lga.index(20)]
                    links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)
    
    # Set directions by most-certain angles
    angthreshs = np.linspace(0, 0.4, 10)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape, angthresh=a, lenthresh=3)
                
    # Set using direction of nearest main channel
    links, nodes = dy.set_by_nearest_main_channel(links, nodes, imshape, nodethresh=1)
   
    angthreshs = np.linspace(0, 1.5, 20)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape, angthresh=a)
        
    # At this point, if any links remain unset, they are just set randomly
    if np.sum(links['certain']) != len(links['id']):
        print('{} links were randomly set.'.format(len(links['id']) - np.sum(links['certain'])))
        links['certain'] = np.ones((len(links['id']), 1))
        
    # Check for and try to fix cycles in the graph
    links, nodes, cantfix_cyclelinks, cantfix_cyclenodes = fix_river_cycles(links, nodes, imshape)

    # Check for sources or sinks within the graph
    cont_violators = dy.check_continuity(links, nodes)
    
    # Summray of problems:
    manual_fix = 0
    if len(cantfix_cyclelinks) > 0:
        print('Could not fix cycle links: {}.'.format(cantfix_cyclelinks))
        manual_fix = 1
    else:
        print('All cycles were resolved.')
    if len(cont_violators) > 0:
        print('Continuity violated at nodes {}.'.format(cont_violators))
        manual_fix = 1
        
    # Create a csv to store manual edits to directionality if does not exist
    if manual_fix == 1:
        if os.path.isfile(path_csv) is False:
            io.create_manual_dir_csv(path_csv)
            print('A .csv file for manual fixes to link directions at {}.'.format(path_csv))
        else:
            print('Use the csv file at {} to manually fix link directions.'.format(path_csv))
            
    return links, nodes
    

def directional_info(links, nodes, Imask, pixlen, exit_sides, gt, meshlines, meshpolys, Idt):
    """
    Using the links['guess'] dictionary, sets the directionality of all links.
    Directions are set in order of certainty--with the more certain being set
    first.
    """
    
    # Add a "guess" entry to keep track of the different information used for flow directionality
    links['guess'] = [[] for a in range(len(links['id']))]
    links['guess_alg'] = [[] for a in range(len(links['id']))]
            
    # Append pixel-based widths to links
    if 'wid_pix' not in links.keys():
        links = lnu.link_widths_and_lengths(links, Idt)
    
    # Compute all the information
    links = dir_centerline(links, nodes, meshpolys, meshlines, Imask, gt, pixlen)
    links = dir_link_widths(links)
    links, nodes = dy.dir_bridges(links, nodes)
    links, nodes = dy.dir_main_channel(links, nodes)
                
    return links, nodes
    

def fix_river_cycles(links, nodes, imshape):
    """
    This algorithm attempts to fix all cycles in the directed river graph.
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
        isin = np.empty((len(c_links),1))
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
        for cnodes, clinks in zip(cfix_nodes, cfix_links):
            links, nodes, fixed = fix_river_cycle(links, nodes, clinks, cnodes, imshape)
            if fixed == 0:
                cantfix_nodes.append(cnodes)
                cantfix_links.append(clinks)
                                
    return links, nodes, cantfix_links, cantfix_nodes


def fix_river_cycle(links, nodes, cyclelinks, cyclenodes, imshape):
    """
    Attempts to fix a cycle in a directed river network. All directions should
    be set before running this.
    """
    
    dont_reset_algs = [20, 21, 22, 23, 0, 5]    
    
    fixed = 1 # One if fix was successful, else zero
    reset = 0 # One if original orientation need to be reset
           
    # If an artifical node triad is present, flip its direction and see if the
    # cycle is resolved.
    # See if any links are part of an artificial triad
    clset = set(cyclelinks)
    all_triads = []
    triadnodes = []
    for i, atl in enumerate(links['arts']):
        artlinks = clset.intersection(set(atl))
        if len(artlinks) > 0:
            all_triads.append(atl)
            triadnodes.append(nodes['arts'][i])

    pre_sourcesink = dy.check_continuity(links, nodes) # Get continuity violators before flipping
        
    if len(all_triads) == 1: # There is one aritificial node triad, flip its direction and re-set other cycle links and see if cycle is resolved
        # Set all cycle + triad links to unknown
        certzero = list(set(all_triads[0] + cyclelinks))
        orig_links = dy.cycle_get_original_orientation(links, certzero) # Save the original orientations in case the cycle can't be fixed
        for cz in certzero:
            links['certain'][links['id'].index(cz)] = 0
        
        # Flip the links of the triad
        for l in all_triads[0]:
            links = lnu.flip_link(links, l)

    if len(all_triads) > 1: # If there are multiple triads, more code needs to be written for these cases
        print('Multiple artifical node triads in the same cycle. Not implemented yet.')
        return links, nodes, 0
        
    elif len(all_triads) == 0: # No aritifical node triads; just re-set all the cycle links and see if cycle is resolved
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

    # Return the links to their original orientations if cycle could not be resolved
    if reset == 1:

        links = dy.cycle_return_to_original_orientation(links, orig_links)
        
        # Try a second method to fix the cycle: unset all the links of the cycle
        # AND the links connected to those links
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
       
#[links['certain'][links['id'].index(l)] for l in set_to_zero]
#[links['certain_alg'][links['id'].index(l)] for l in set_to_zero]
#links['conn'][links['id'].index(3661)]

def re_set_linkdirs(links, nodes, imshape):
        
    links, nodes = dy.set_continuity(links, nodes)
    
    # Set the directions of the links that are more certain via centerline angle method
    alg = 23.1
    cl_angthresh = np.percentile(links['clangs'][np.isnan(links['clangs'])==0], 40)
    for lid, cla, lg, lga, cert in zip(links['id'],  links['clangs'], links['guess'], links['guess_alg'], links['certain']):
        if cert == 1:
            continue
        if np.isnan(cla) == True:
            continue
        if cla <= cl_angthresh:
            linkidx = links['id'].index(lid)
            if 21 in lga:
                usnode = lg[lga.index(21)]
                links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg)

    
    angthreshs = np.linspace(0, 1.3, 20)
    for a in angthreshs:
        links, nodes = dy.set_by_known_flow_directions(links, nodes, imshape,  angthresh=a, lenthresh=0, alg=6.1)


    if np.sum(links['certain']) != len(links['id']):
        links_notset = links['id'][np.where(links['certain']==0)[0][0]]
        print('Links {} were not set by re_set_linkdirs.'.format(links_notset))
        
    return links, nodes


def set_unknown_cluster_by_widthpct(links, nodes):
    """
    Set unknown links based on width differences at endpoints (flow goes wide->narrow)
    """
    
    alg = 26
    
    # Get indices of uncertain links
    uc_idx = np.where(links['certain']==0)[0].tolist()

    # Create graph to find clusters of uncertains    
    G = nx.Graph()
    for idx in uc_idx:
        lc = links['conn'][idx]
        G.add_edge(lc[0], lc[1])
 
    # Compute connected components (nodes)
    cc = nx.connected_components(G)
    ccs = [c for c in cc if len(c)>1]
    
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
        
        links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg = alg)
        
    return links, nodes


def get_centerline(links, nodes, Imask, exit_sides, gt):
    """
    Guesses link direction by generating a mesh following the centerline,
    then checking if links' endpoints are within different mesh cells. If so,
    since the us->ds order of the mesh cells are known, the link direction
    can be guessed.
    """
    
    # Get centerline
    cl = ru.mask_to_centerline(Imask, exit_sides)
    
    # Transform centerline to coordinates
    clx, cly = gu.xy_to_coords(cl[:,0], cl[:,1], gt)
    clcoords = [(x,y) for x,y in zip(clx, cly)]
    
    # Get widths for parameterizing centerline mesh
    pixel_area = gt[1]*-gt[5]
    width_channels, width_extent = ru.chan_width(clcoords, Imask, pixarea=pixel_area)

    # Generate centerline mesh
    meshlines, meshpolys = ru.centerline_mesh(clcoords, width_channels, width_extent, n_widths_grid_spacing=.1, n_widths_initial_smooth=50, n_widths_buffer=5)
    
    return meshlines, meshpolys, cl


def dir_centerline(links, nodes, meshpolys, meshlines, Imask, gt, pixlen):
           
    alg = 20

    # Create geodataframes for intersecting meshpolys with nodes
    mp_gdf = gpd.GeoDataFrame(geometry=[Polygon(mp) for mp in meshpolys])
    rc = np.unravel_index(nodes['idx'], Imask.shape)
    nodecoords = gu.xy_to_coords(rc[1],rc[0], gt)
    node_gdf = gpd.GeoDataFrame(geometry=[Point(x,y) for x,y in zip(nodecoords[0], nodecoords[1])], index=nodes['id'])
    
    # Determine which meshpoly each node lies within
    intersect = gpd.sjoin(node_gdf, mp_gdf, op='intersects', rsuffix='right')
    
    # Compute guess and certainty, where certainty is how many transects apart
    # the link endpoints are (longer=more certain)
    cldists = np.zeros((len(links['id']),1))
    for i, lconn in enumerate(links['conn']):
        try:
            first = intersect.loc[lconn[0]].index_right
            second = intersect.loc[lconn[1]].index_right
            cldists[i] = second-first
        except KeyError:
            pass
    
    for i, c in enumerate(cldists):
        if c !=0:
            if c > 0:
                links['guess'][i].append(links['conn'][i][0])
                links['guess_alg'][i].append(alg)
            elif c < 0:
                links['guess'][i].append(links['conn'][i][-1])
                links['guess_alg'][i].append(alg)

    # Save the distances for certainty
    links['cldists'] = np.abs(cldists)
    
    # Compute guesses based on how the link aligns with the local centerline direction
    alg = 21
    clangs = np.ones((len(links['id']),1)) * np.nan
    for i, (lconn, lidx) in enumerate(zip(links['conn'], links['idx'])):
        
#        if links['id'][i] == 5795:
#            break
        
        # Get coordinates of link endpoints
        rc = np.unravel_index([lidx[0], lidx[-1]], Imask.shape)
        
        try: # Try is because some points may not lie within the mesh polygons
            # Get coordinates of centerline midpoints
            first = intersect.loc[lconn[0]].index_right
            second = intersect.loc[lconn[1]].index_right
            if first > second:
                first, second = second, first
            first_mp = np.mean(np.array(meshlines[first]), axis=0) # midpoint
            second_mp = np.mean(np.array(meshlines[second+1]), axis=0) # midpoint
        except KeyError:
            continue
        
        # Centerline vector
        cl_vec = second_mp - first_mp
        cl_vec = cl_vec/np.sqrt(np.sum(cl_vec**2))

        # Link vectors - as-is and flipped (reversed)
        link_vec = dy.get_link_vector(links, nodes, links['id'][i], Imask.shape, pixlen=pixlen)
        link_vec_rev = -link_vec
                
        # Compute interior radians between centerline vector and link vector (then again with link vector flipped)
        lva = np.math.atan2(np.linalg.det([cl_vec,link_vec]),np.dot(cl_vec,link_vec))
        lvar = np.math.atan2(np.linalg.det([cl_vec,link_vec_rev]),np.dot(cl_vec,link_vec_rev))
      
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
            links, nodes = dy.set_link(links, nodes, linkidx, usnode, alg = alg)

    return links, nodes


def dir_link_widths(links):
    """
    Guesses direction based on link widths at the endpoints. The node with the
    larger width is guessed to be the upstream node. The ratio of guessed 
    upstream to guessed downstream node widths is appended to links.
    If the width at a link's endpoint is 0, the percent is set to 1000.
    """
    
    alg = 26
    
    widpcts = np.zeros((len(links['id']),1))
    for i in range(len(links['id'])):
        
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


#def linkcheck(links):
#    import pandas as pd
#    nc = pd.read_csv(r"X:\RivGraph\Results\Brahma\nodecheck.csv")
#    wrong = []
#    ncert = 0
#    for lid, usn in zip(nc['linkid'], nc['usnode']):
#        lidx = links['id'].index(lid)
#        if links['certain'][lidx] == 1:
#            ncert = ncert + 1
#            if links['conn'][lidx][0] != usn:
#                wrong.append(lid)
#            
#    print('# certain: {}, frac wrong: {}.'.format(ncert, len(wrong)/ncert))
#    return wrong


#    # Create a "minigraph" that only inlcudes the cycle and the 
#    # links connected to it, as well as any identified aritifical links/nodes
#    links_to_use = []
#    for cn in list(set(cyclenodes + triadnodes)):
#        links_to_use.extend(nodes['conn'][nodes['id'].index(cn)])
#    nodes_to_use = []
#    for lm in links_to_use:
#        nodes_to_use.extend(links['conn'][links['id'].index(lm)])
#    links_to_use = list(set(links_to_use))
#    nodes_to_use = list(set(nodes_to_use))
#        
#    keys = ['id', 'conn', 'idx']
#    minilinks = dict([(key, []) for key in keys])
#    mininodes = dict([(key, []) for key in keys])
#    minilinks['id'] = links_to_use
#    mininodes['id'] = nodes_to_use
#    minilinks['conn'] = [links['conn'][links['id'].index(l)] for l in links_to_use]
#    mininodes['conn'] = [nodes['conn'][nodes['id'].index(n)] for n in nodes_to_use]
#    minilinks['idx'] = [links['idx'][links['id'].index(l)] for l in links_to_use]
#    mininodes['idx'] = [nodes['idx'][nodes['id'].index(n)] for n in nodes_to_use]
#    # Remove outside link connections to endnodes of minigraph
#    for i, nconn in enumerate(mininodes['conn']):
#        mininodes['conn'][i] = [n for n in nconn if n in minilinks['id']]
#    mininodes['inlets'] = [n for n, nc in zip(mininodes['id'], mininodes['conn']) if len(nc)==1]
#    mininodes['outlets'] = []
#    minilinks['certain'] = np.ones((len(minilinks['id']), 1))
#    
#    # Flip the link triad in the minigraph
#    for l in all_triads[0]:
#        lidx = minilinks['id'].index(l)
#        minilinks['conn'][lidx] = minilinks['conn'][lidx][::-1]
#        minilinks['idx'][lidx] = minilinks['idx'][lidx][::-1]
#    # Did flipping violate continuity anywhere?
#    sourcesink = dy.check_continuity(minilinks, mininodes)
   

