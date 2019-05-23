# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:29:10 2018

@author: Jon
"""
import numpy as np
import networkx as nx
from ordered_set import OrderedSet
from scipy.ndimage.morphology import distance_transform_edt
import shapely
from shapely.geometry import LineString
import scipy.interpolate as si
from scipy import signal
from scipy.spatial.distance import cdist

import rivgraph.im_utils as iu
import rivgraph.mask_to_graph as m2g
import rivgraph.ln_utils as lnu

#links = indus.links
#nodes = indus.nodes
#exit_sides= indus.exit_sides
#Iskel = indus.Iskel
#gdobj = indus.gdobj

def prune_river(links, nodes, exit_sides, Iskel, gdobj):
                        
    # Get inlet nodes
    nodes = find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel)    
        
    # Remove spurs from network (this includes valid inlets and outlets)
    links, nodes = lnu.remove_all_spurs(links, nodes, dontremove=list(nodes['inlets'] + nodes['outlets']))
        
    # Add artificial nodes where necessary
    links, nodes = lnu.add_artificial_nodes(links, nodes, gdobj)
    
    # Remove sets of links that are disconnected from inlets/outlets except for a single bridge link (effectively re-pruning the network)
    links, nodes = lnu.remove_disconnected_bridge_links(links, nodes)    
    
    # Remove one-pixel links
    links, nodes = lnu.remove_single_pixel_links(links, nodes)
    
    return links, nodes


def find_inlet_outlet_nodes(links, nodes, exit_sides, Iskel):
    """ 
    Appends the inlet and outlet nodes to the nodes dictionary. Only works for
    rivers; deltas must be treated differently.
    """
    
    # Find possible inlet/outlet link candidates as those attached to a node
    # of degree-1.
    poss_endlinks = []
    for nconn in nodes['conn']:
        if len(nconn) == 1:
            poss_endlinks.append(nconn[0])
            
    # Find the row(s)/column(s) corresponding to extent of the river at the 
    # exit sides
    pixy, pixx = np.where(Iskel==True)
    e, w, n, s = np.max(pixx), np.min(pixx), np.min(pixy), np.max(pixy)
    
    # Get row, column coordinates of all nodes endpoints
    n_r, n_c = np.unravel_index(nodes['idx'], Iskel.shape)
    
    # Find inlets and outlets by searching for nodes that intersect the first exit_side of
    # the image
    ins_outs = []
    for j in [0,1]:
        if exit_sides[j] == 'n':
            idcs = np.where(n_r==n)[0]
        elif exit_sides[j] == 's':
            idcs = np.where(n_r==s)[0]
        elif exit_sides[j] == 'e':
            idcs = np.where(n_c==e)[0]
        elif exit_sides[j] == 'w':
            idcs = np.where(n_c==w)[0]
            
        ins_outs.append([nodes['id'][i] for i in idcs])
    
    # Append inlets and outlets to nodes dictionary
    nodes['inlets'] = ins_outs[0]
    nodes['outlets'] = ins_outs[1]
    
    if len(nodes['inlets']) == 0:
        print('No inlet nodes found.')
    if len(nodes['outlets']) == 0:
        print('No outlet nodes found.')
        
    ## TODO: handle special cases where the link intersects the edge of the 
    ## image but the node does not because the link is a loop. This might be
    ## "fixable" by adjusting the padding multiplier; I don't have any test
    ## cases to work on currently so leaving this unimplemented for now.
    
    return nodes    


def mask_to_centerline(Imask, es):
    """
    This function takes an input binary mask of a river and extracts its centerline.
    If there are multiple channels (and therefore islands) in the river, they
    will be filled before the centerline is computed.
    
    The input mask should have the following properties:
        1) There should be only one "blob" (connected component)
        2) Where the blob intersects the image edges, there should be only
           one channel. This avoids ambiguity in identifying inlet/outlet links
    
    INPUTS:
        Imask: the mask image (numpy array)
        es: two-character string comprinsed of "n", "e", "s", or "w". Exit sides
            correspond to the sides of the image that the river intersects. 
            Upstream should be first, followed by downstream.
    OUTPUTS:
        dt.tif: geotiff of the distance transform of the binary mask
        skel.tif: geotiff of the skeletonized binary mask
        centerline.shp: shapefile of the centerline, arranged upstream to downstream
        cl.pkl: pickle file containing centerline coords, EPSG, and paths dictionary
    """    
                  
    # Lowercase the exit sides
    es = es.lower()
    
    # Keep only largest connected blob
    I = iu.largest_blobs(Imask, nlargest=1, action='keep')
    
    # Fill holes in mask
    Ihf = iu.fill_holes(I)
    
    # Skeletonize holes-filled river image
    Ihf_skel = m2g.skeletonize_river_mask(Ihf, es)
    
    # In some cases, skeleton spurs can prevent the creation of an endpoint
    # at the edge of the image. This next block of code tries to condition
    # the skeleton to prevent this from happening.
    # Find skeleton border pixels
    skel_rows, skel_cols = np.where(Ihf_skel)
    idcs_top = np.where(skel_rows==0)
    idcs_bottom = np.where(skel_rows==Ihf_skel.shape[0]-1)
    idcs_right = np.where(skel_cols==Ihf_skel.shape[1]-1)
    idcs_left = np.where(skel_cols==0)
    # Remove skeleton border pixels
    Ihf_skel[skel_rows[idcs_top], skel_cols[idcs_top]] = 0
    Ihf_skel[skel_rows[idcs_bottom], skel_cols[idcs_bottom]] = 0
    Ihf_skel[skel_rows[idcs_right], skel_cols[idcs_right]] = 0
    Ihf_skel[skel_rows[idcs_left], skel_cols[idcs_left]] = 0
    # Remove all pixels now disconnected from the main skeleton
    Ihf_skel = iu.largest_blobs(Ihf_skel, nlargest=1, action='keep')
    # Add the border pixels back
    Ihf_skel[skel_rows[idcs_top], skel_cols[idcs_top]] = 1
    Ihf_skel[skel_rows[idcs_bottom], skel_cols[idcs_bottom]] = 1
    Ihf_skel[skel_rows[idcs_right], skel_cols[idcs_right]] = 1
    Ihf_skel[skel_rows[idcs_left], skel_cols[idcs_left]] = 1  
    
    # Keep only the largest connected skeleton
    Ihf_skel = iu.largest_blobs(Ihf_skel, nlargest=1, action='keep')
            
    # Convert skeleton to graph
    hf_links, hf_nodes = m2g.skel_to_graph(Ihf_skel)
    
    # Compute holes-filled distance transform
    Ihf_dist = distance_transform_edt(Ihf) # distance transform
    
    # Append link widths and lengths
    hf_links = lnu.link_widths_and_lengths(hf_links, Ihf_dist)
    
    """ Find shortest path between inlet/outlet centerline nodes"""
    # Put skeleton into networkX graph object
    G = nx.Graph()
    G.add_nodes_from(hf_nodes['id'])
    for lc, wt in zip(hf_links['conn'], hf_links['len']):
        G.add_edge(lc[0], lc[1], weight=wt)
    
    # Get endpoints of graph
    endpoints = [nid for nid, nconn in zip(hf_nodes['id'], hf_nodes['conn']) if len(nconn) == 1]
                       
    # Filter endpoints if we have too many--shortest path compute time scales as a power of len(endpoints)
    while len(endpoints) > 100:
        ep_r, ep_c = np.unravel_index([hf_nodes['idx'][hf_nodes['id'].index(ep)] for ep in endpoints], Ihf_skel.shape)
        pct = 10
        ep_keep = set()
        for esi in [0,1]:
            if es[esi] == 'n':
                n_pct = int(np.percentile(ep_r, pct))
                ep_keep.update(np.where(ep_r <= n_pct)[0])
            elif es[esi] == 's':
                s_pct = int(np.percentile(ep_r, 100-pct))
                ep_keep.update(np.where(ep_r >= s_pct)[0])
            elif es[esi] == 'e':
                e_pct = int(np.percentile(ep_c, 100-pct))
                ep_keep.update(np.where(ep_c > e_pct)[0])
            elif es[esi] == 'w':
                w_pct = int(np.percentile(ep_c, pct))
                ep_keep.update(np.where(ep_c < w_pct)[0])
            
        endpoints = [endpoints[ek] for ek in ep_keep]
        
    # Get all paths from inlet(s) to outlets
    longest_shortest_paths = []
    for inl in endpoints:
        temp_lens = []
        for o in endpoints:
            temp_lens.append(nx.dijkstra_path_length(G, inl, o, weight='weight'))
        longest_shortest_paths.append(max(temp_lens))

    # The two end nodes with the longest shortest path are the centerline's endnodes
    end_nodes_idx = np.where(np.isclose(np.max(longest_shortest_paths),longest_shortest_paths))[0]
    end_nodes = [endpoints[i] for i in end_nodes_idx]
    
    # It is possible that more than two endnodes were identified; in these cases,
    # choose the nodes that are farthest apart in Euclidean space
    en_r, en_c = np.unravel_index([hf_nodes['idx'][hf_nodes['id'].index(en)] for en in end_nodes], Ihf_skel.shape)
    ep_coords = np.r_['1,2,0', en_r, en_c]
    ep_dists = cdist(ep_coords, ep_coords, 'euclidean')
    en_idcs_to_use = np.unravel_index(np.argmax(ep_dists), ep_dists.shape)
    end_nodes = [end_nodes[eitu] for eitu in en_idcs_to_use]
    
    # Ensure that exactly two end nodes are identified
    if len(end_nodes) != 2:
        raise RuntimeError('{} endpoints were found for the centerline. (Need exactly two).'.format(len(end_nodes)))
   
    # Find upstream node
    en_r, en_c = np.unravel_index([hf_nodes['idx'][hf_nodes['id'].index(n)] for n in end_nodes], Ihf_skel.shape)
    
    # Compute error for each end node given the exit sides
    errors = []
    for orientation in [0,1]:
        if orientation == 0:
            er = en_r
            ec = en_c
        elif orientation == 1:
            er = en_r[::-1]
            ec = en_c[::-1]
    
        err = 0
        for ot in [0,1]:
            if es[ot].lower() == 'n':
                err = err + er[ot]
            elif es[ot].lower() == 's':
                err = err + Ihf_dist.shape[0] - er[ot]
            elif es[ot].lower() == 'w':
                err = err + ec[ot]
            elif es[ot].lower() == 'e':
                err = err + Ihf_dist.shape[1] - ec[ot]
        errors.append(err)
    # Flip end node orientation to get US->DS arrangement
    if errors[0] > errors[1]:
        end_nodes = end_nodes[::-1]
        
    # Create centerline from links along shortest path
    nodespath = nx.dijkstra_path(G, end_nodes[0], end_nodes[1]) # nodes shortest path
    # Find the links along the shortest node path
    cl_link_ids = []
    for u,v in zip(nodespath[0:-1], nodespath[1:]):
        ulinks = hf_nodes['conn'][hf_nodes['id'].index(u)]
        vlinks = hf_nodes['conn'][hf_nodes['id'].index(v)]
        cl_link_ids.append([ul for ul in ulinks if ul in vlinks][0])
    
    # Create a shortest-path links dict
    cl_links = dict.fromkeys(hf_links.keys())
    dokeys = list(hf_links.keys())
    dokeys.remove('n_networks') # Don't need n_networks
    for clid in cl_link_ids:
        for k in dokeys:
            if cl_links[k] is None:
                cl_links[k] = []
            cl_links[k].append(hf_links[k][hf_links['id'].index(clid)])
    
    # Save centerline as shapefile
#    lnu.links_to_shapefile(cl_links, igd, rmh.get_EPSG(paths['skel']), paths['cl_temp_shp'])
    
    # Get and save coordinates of centerline
    cl = []
    for ic, cll in enumerate(cl_link_ids):
        if ic == 0:
            if hf_links['idx'][hf_links['id'].index(cll)][0] != hf_nodes['idx'][hf_nodes['id'].index(end_nodes[0])]:
                hf_links['idx'][hf_links['id'].index(cll)] = hf_links['idx'][hf_links['id'].index(cll)][::-1]
        else:
            if hf_links['idx'][hf_links['id'].index(cll)][0] != cl[-1]:
                hf_links['idx'][hf_links['id'].index(cll)] = hf_links['idx'][hf_links['id'].index(cll)][::-1]
    
        cl.extend(hf_links['idx'][hf_links['id'].index(cll)][:])
    
    # Uniquify points, preserving order
    cl = list(OrderedSet(cl))
    
    # Convert back to coordinates
    cly, clx = np.unravel_index(cl, Ihf_skel.shape)
    
    # Get width at each pixel of centerline
    pix_width = [Ihf_dist[y,x]*2 for x,y in zip(clx,cly)]
        
    coords = np.transpose(np.vstack((clx,cly)))
        
    return coords, pix_width


def resample_line(xs, ys, npts=None, k=3):
    """
    Resamples a line defined by x,y coordinates such that coordinates are
    evenly-spaced. 
    If the optional npts parameter is not specified, the 
    line will be resampled with the same number of points as the input
    coordinates.
    k refers to order of spline that is fit to coordinates for resampling.
    """
        
    # FitPack in si.splprep can't handle duplicate points in the line, so remove them
    no_dupes = np.array(np.where(np.abs(np.diff(xs)) + np.abs(np.diff(ys)) > 0)[0], dtype=np.int)
    xs_nd = np.r_[xs[no_dupes], xs[-1]]
    ys_nd = np.r_[ys[no_dupes], ys[-1]]
        
    # Create spline
    spline, _ = si.splprep([xs_nd, ys_nd], k=k)
    
    # Evaluate spline
    if npts is None:
        npts = len(xs)
    t = np.linspace(0, 1, npts)
    resampled_coords = si.splev(t, spline)
    
    return resampled_coords
    

def offset_linestring(linestring, distance, side):
    
    """
    INPUTS:
        linestring: shapely linestring
        distance: distance to offset linestring
        side: 'left' or 'right' to specify which side to compute offset
    """
    
    # Perform the offset
    offset = linestring.parallel_offset(distance, side)
    
    # Ensure that offset is not a MultiLineString by deleting all but the longest linestring
    if type(offset) is shapely.geometry.multilinestring.MultiLineString:
        ls_lengths = [ls.length for ls in offset]
        offset = offset[ls_lengths.index(max(ls_lengths))]
        print('Multilinestring returned in offset_linestring; clipped to longest but check output.')
    
    # Ensure offset linestring is oriented the same as the input linestring
    xy_orig_start = (linestring.coords.xy[0][0], linestring.coords.xy[1][0])
    
    # Get endpoint coordinates of offset linestring
    xy_offset_start = (offset.coords.xy[0][0], offset.coords.xy[1][0])
    xy_offset_end = (offset.coords.xy[0][-1], offset.coords.xy[1][-1])

    if np.sum(np.sqrt((xy_orig_start[0]-xy_offset_start[0])**2 + (xy_orig_start[1]-xy_offset_start[1])**2)) > np.sum(np.sqrt((xy_orig_start[0]-xy_offset_end[0])**2 + (xy_orig_start[1]-xy_offset_end[1])**2)):
        offset = LineString(offset.coords[::-1])

    return offset


def inflection_pts_oversmooth(xs, ys, n_infs):
    """
    Computes inflection points as the intersection of a line given by 
    xs, ys and its smoothed version.
    INPUTS:
        xs: np-type array of x-coordinates
        ys: np-type array of y-coordinates
        n_infs: approx how many inflection points are expected? This sets the
                degree of smoothing. For meandering rivers, n_infs can be
                approximated by the relationship: wavelength = 10W, for average
                width W.
    """
    def generate_smoothing_windows(start, stop, nbreaks, polyorder=3):
        """
        Generate an array of window sizes for smoothing. Each value must be 
        greater than the polyorder of the smoother and be odd.
        """
        
        smoothwins = np.linspace(start, stop, nbreaks, dtype=np.int)
        
        # Window must be greater than polyorder
        smoothwins[smoothwins<polyorder] = polyorder + 1
        # Window must be odd
        smoothwins[smoothwins % 2 == 0] = smoothwins[smoothwins % 2 == 0] + 1
        smoothwins = np.unique(smoothwins)
        
        return smoothwins


    def smoothing_iterator(xs, ys, smoothing_windows, n_infs, polyorder):
       
        for i, sw in enumerate(smoothing_windows):
            
            # Smooth the line's coordinates
            xs_sm = signal.savgol_filter(xs, window_length=sw, polyorder=polyorder, mode='interp')
            ys_sm = signal.savgol_filter(ys, window_length=sw, polyorder=polyorder, mode='interp')
            
            # Conver to shapely objects for intersection detection
            ls = LineString([(x,y) for x,y in zip(xs, ys)])
            ls_sm = LineString([(x,y) for x,y in zip(xs_sm, ys_sm)]) 
            
            n_ints = len(ls.intersection(ls_sm) )
            
            if n_ints < n_infs:
                break
            
        return smoothing_windows[i-1], smoothing_windows[i]

    # Set polyorder for smoother; could be included as a parameter
    polyorder = 3

    # Find the smoothing window that provides a number of intersections closest
    # to the provided n_infs    
    prepost_tolerance = 10 # difference between smoothing window sizes to stop iterating
    prev = 0 # initial minimum smoothing window size
    post = len(xs)/5 # initial maximum smoothing window size
    while post - prev > prepost_tolerance:
        s_windows = generate_smoothing_windows(prev, post, 25, polyorder=3)
        prev, post = smoothing_iterator(xs, ys, s_windows, n_infs, polyorder)
        
        ## TODO: should add a counter/tracker to ensure n_infs can actually be
        # obtained and avoid an infinite loop
    
    # Use the optimized smoothing window to smooth the signal
    window = post
    xs_sm = signal.savgol_filter(xs, window_length=window, polyorder=polyorder, mode='interp')
    ys_sm = signal.savgol_filter(ys, window_length=window, polyorder=polyorder, mode='interp')
    
    # Cast coordinates as shapely LineString objects
    ls = LineString([(x,y) for x,y in zip(xs, ys)])
    ls_sm = LineString([(x,y) for x,y in zip(xs_sm, ys_sm)]) 

    # Compute intersection points between original and oversmoothed centerline
    int_pts = ls.intersection(ls_sm)
    int_coords = np.array([(int_pts[i].coords.xy[0][0], int_pts[i].coords.xy[1][0]) for i in range(len(int_pts))])
    
    # Map the intersecting coordinates to the indices of the original signal
    idx = []
    for ic in int_coords:
        idx.append(np.argmin(np.sqrt((ic[0]-xs)**2+(ic[1]-ys)**2)))
    idx = np.sort(idx)

#    plt.close('all')
##    plt.plot(xs_sm,ys_sm)
#    plt.plot(xs,ys)
#    plt.plot(xs[idx], ys[idx], '.')
#    plt.axis('equal')
    
    # Package oversmoothed coordinates for export
    smoothline = np.array([xs_sm, ys_sm])
    
    return idx, smoothline



#coords = brahma.centerline
#width_chan = brahma.width_chans
#width_ext = brahma.width_extent
#n_widths_grid_spacing = np.percentile(brahma.links['len'],25) / brahma.width_chans 
#n_widths_initial_smooth=100
#n_widths_buffer=20
    

def centerline_mesh(coords, width_chan, meshwidth, grid_spacing, smoothing_param=1):
    """
    Generates a centerline mesh. Differs from valleyline_mesh() in that it draws
    perpendiculars rather than offsetting the valley line to compute mesh
    polygons. This method is more effective for narrower channels that don't
    require an exceptionally wide mesh (i.e. not much change).
    INPUTS: 
        coords: 2xN list, tuple, np.array (xs, ys) of coordinates defining centerline
        width_chan: width of the river in same units of coords
        mesh_dist: how wide should the mesh be, in same units of coords
        mesh_spacing: how far apart should mesh cells be, in same units of coords
    """
#    coords = coords_gt
#    width_chan = avg_w
#    meshwidth = avg_w*8
#    grid_spacing = avg_w
#    grid_spacing = width_chan / 2
#    meshwidth = width_chan * 2
#    smoothing_param = 1

    
    if np.shape(coords)[0] == 2 and np.size(coords) != 4:
        coords = np.transpose(coords)
      
    # Get lengths along centerline          
    s, ds = s_ds(coords[:,0], coords[:,1])
            
    # Mirror centerline manually since scipy fucks it up - only flip the axis that has the largest displacement
    # Mirroring done to avoid edge effects when smoothing
    npad = int(width_chan / np.mean(ds) * 10) # Padding fixed at 10 channel widths
    xs_m, ys_m = mirror_line_ends(coords[:,0], coords[:,1], npad)
    
    # A smoothing filter of one-channel width will be passed over the centerline coordinates
    window_len = int(width_chan / np.mean(ds) * smoothing_param)
    if window_len % 2 == 0: # Window must be odd
        window_len = window_len + 1
    
    # Smooth
    xs_sm = signal.savgol_filter(xs_m, window_length=window_len, polyorder=3, mode='interp')
    ys_sm = signal.savgol_filter(ys_m, window_length=window_len, polyorder=3, mode='interp')

    # Re-sample centerline to even spacing
    s, _ = s_ds(xs_sm, ys_sm)
    npts = int(s[-1]/grid_spacing)
    xs_rs, ys_rs = resample_line(xs_sm, ys_sm, npts)
    
    # Get angles at each point along centerline
    C, A, s = curvars(xs_rs,ys_rs, unwrap=True)

    # Draw perpendiculars at each centerline point
    mesh_hwidth = meshwidth/2 
    
    # Compute slope of perpendicular (w/ref to dx/dy and dy/dx)       
    m_inv_xy = -1/(np.diff(xs_rs) / np.diff(ys_rs))
    m_inv_yx = -1/(np.diff(ys_rs) / np.diff(xs_rs))
    # For storing perpendicular points
    perps = []
    for ic in range(len(m_inv_xy)):

        # Compute perpendicular lines based on largest of dx, dy (reduces distortion)        
        if m_inv_yx[ic] > m_inv_xy[ic]:
            dx = np.sqrt(mesh_hwidth**2/(1+m_inv_yx[ic]**2))
            dy = dx * m_inv_yx[ic]            
            
        else:
            dy = np.sqrt(mesh_hwidth**2/(1+m_inv_xy[ic]**2))
            dx = dy * m_inv_xy[ic]
        
        upper_pt = (xs_rs[ic] + dx, ys_rs[ic] + dy)
        lower_pt = (xs_rs[ic] - dx, ys_rs[ic] - dy)

        perps.append((upper_pt, lower_pt))

    # Now orient perpendiculars so that both sides are continuous
    # NOTE: this method is not guaranteed to work when the grid spacing is much 
    # larger than the buffer width (it likely will be fine, but for highly-
    # curved bends failure is possible). There are more robust ways to separate
    # points into left/right bank, but this is quick, dirty, and works for most
    # applications.
    perp_aligned = [perps[0]]
    for ip in range(1,len(perps)):
        
        left_pre, right_pre = perp_aligned[ip-1]
                
        p0 = perps[ip][0]
        p1 = perps[ip][1]
        
        if np.sqrt((p0[0]-left_pre[0])**2 + (p0[1]-left_pre[1])**2) <  np.sqrt((p1[0]-left_pre[0])**2 + (p1[1]-left_pre[1])**2):
            perp_aligned.append((p0, p1))
        else:
            perp_aligned.append((p1, p0))

#    plt.close('all')
#    plt.plot(xs_rs, ys_rs,'.')
#    plt.axis('equal')
#    for p in perp_aligned:
#        plt.plot(p[0][0], p[0][1], 'k.')
#        plt.plot(p[1][0], p[1][1], 'r.')
    

    # Trim the centerline to remove the mirrored portions
    start_idx = np.argmin(np.sqrt((coords[0,0]-xs_rs)**2+(coords[0,1]-ys_rs)**2)) - 1
    end_idx = np.argmin(np.sqrt((coords[-1,0]-xs_rs)**2+(coords[-1,1]-ys_rs)**2)) + 1
    
    # Build the polygon mesh
    polys = []
    for i in range(start_idx, end_idx+1):
        polys.append([perp_aligned[i][0], perp_aligned[i][1], perp_aligned[i+1][1], perp_aligned[i+1][0], perp_aligned[i][0]])
   
    
    perps_out = perp_aligned[start_idx:end_idx+1]
    cl_resampled = np.r_['1,2,0', xs_rs, ys_rs] 
    s_out = s[start_idx:end_idx+1] - s[start_idx]
    
    return perps_out, polys, cl_resampled, s_out 
        
    

def mirror_line_ends(xs, ys, npad):
    """
    Reflects both ends of a line defined by x and y coordinates. The mirrored
    distance is set by npad, which refers to the number of vertices along the
    line to mirror.
    """
    
    # Mirror the beginning of the line
    diff_x = np.diff(xs[0:npad])
    xs_m = np.concatenate((np.flipud(xs[1] - np.cumsum(diff_x)), xs))
    diff_y = np.diff(ys[0:npad])
    ys_m = np.concatenate((np.flipud(ys[1] - np.cumsum(diff_y)), ys))
       
    # Mirror the end of the line
    diff_x = np.diff(xs[-npad:][::-1])
    xs_m = np.concatenate((xs_m, xs_m[-1] - np.cumsum(diff_x)))
    diff_y = np.diff(ys[-npad:][::-1])
    ys_m = np.concatenate((ys_m, ys_m[-1] - np.cumsum(diff_y)))
    
    return xs_m, ys_m
    



def valleyline_mesh(coords, width_chan, bufferdist, grid_spacing, smoothing=0.15):

    """ 
    This function generates a mesh over an input river centerline. The mesh
    is generated across the valley, not just the channel width, in order to 
    perform larger-scale spatial analyses. With the correct parameter 
    combinations, it can also be used to generate a mesh for smaller-scale
    analysis, but it is optimized for larger and strange behavior may occur.
    
    Many plotting commands are commented out throughout this script as it's
    still somewhat in beta mode.
    
    INPUTS:
        coords: Nx2 list, tuple, or np.array of x,y coordinates. Coordinates MUST be in projected CRS for viable results.
        width_chan: estimated width. Units MUST correspond to those of the input coordinates
        bufferdist: distance between centerline and left or right bufferline, in units of coords
        grid_spacing: fraction of input centerline length that should be used for smoothing to create the valley centerline  (between 0 and 1)
        smoothing: fraction of centerline length to use for smoothing window
    
    OUTPUTS:
        lines - the "perpendiculars" to the centerline used to generate the mesh
        polys - coordinates of the polygons representing the grid cells of the mesh
    """
    
    if np.shape(coords)[0] == 2 and np.size(coords) != 4:
        coords = np.transpose(coords)
                
    # Separate coordinates into xs and ys (o indicates original coordinates)
    xs_o = coords[:,0]
    ys_o = coords[:,1]
    
    s, ds = s_ds(xs_o, ys_o)
    
    # Set smoothing window size based on smoothing parameter and centerline length
    window_len = int(smoothing * s[-1] / np.mean(ds))
    window_len = int(min(len(xs_o)/5, window_len)) # Smoothing window cannot be longer than 1/5 the centerline
    if window_len % 2 == 0: # Window must be odd
        window_len = window_len + 1
    
    # Mirror centerline manually since scipy fucks it up - only flip the axis that has the largest displacement
    # Mirroring done to avoid edge effects when smoothing
    npad = window_len
    xs_o2, ys_o2 = mirror_line_ends(xs_o, ys_o, npad)
    npad = window_len  
    diff_x = np.diff(xs_o[0:npad])
    xs_o2 = np.concatenate((np.flipud(xs_o[1] - np.cumsum(diff_x)), xs_o))
    diff_y = np.diff(ys_o[0:npad])
    ys_o2 = np.concatenate((np.flipud(ys_o[1] - np.cumsum(diff_y)), ys_o))
       
    diff_x = np.diff(xs_o[-npad:][::-1])
    xs_o2 = np.concatenate((xs_o2, xs_o2[-1] - np.cumsum(diff_x)))
    diff_y = np.diff(ys_o[-npad:][::-1])
    ys_o2 = np.concatenate((ys_o2, ys_o2[-1] - np.cumsum(diff_y)))
    
    # Smooth the coordinates before buffering 
    xs_sm = signal.savgol_filter(xs_o2, window_length=window_len, polyorder=3, mode='interp')
    ys_sm = signal.savgol_filter(ys_o2, window_length=window_len, polyorder=3, mode='interp')
    
    # Create left and right valleylines from oversmoothed centerline
    cl = []
    for x, y in zip(xs_sm, ys_sm):
        cl.append((x,y))
    # Shapely bug: if the spacing of nodes near the end of the centerline is too small,
    # the parallel offset will produce "hooks" on the end. Remove the second and second-to-last
    # points to help prevent this -- does not prevent all cases!
    del(cl[1])
    del(cl[-2])
    cl = LineString(cl)
    
    # Offset valley centerline for left and right valleylines
    left = offset_linestring(cl, bufferdist, 'left')
    right = offset_linestring(cl, bufferdist, 'right')
    
    # Resample center, left, and right lines to have evenly spaced points
    rs_spacing = width_chan/8 # eight points per channel width
    npts_c, npts_l, npts_r = int(cl.length/rs_spacing), int(left.length/rs_spacing), int(right.length/rs_spacing)
    c_es = resample_line(cl.coords.xy[0].tolist(), cl.coords.xy[1].tolist(), npts_c)
    l_es = resample_line(left.coords.xy[0].tolist(), left.coords.xy[1].tolist(), npts_l)
    r_es = resample_line(right.coords.xy[0].tolist(), right.coords.xy[1].tolist(), npts_r)

    # Put resampled lines in shapely format for intersecting
    l_shapely = LineString([(x,y) for x,y in zip(l_es[0], l_es[1])])
    r_shapely = LineString([(x,y) for x,y in zip(r_es[0], r_es[1])])
    
    
    """ Get centerline inflection points for mapping to corresponding left/right bank points """
    # Compute angles, curvatures for each signal
    Ccl,Acl,scl = curvars(c_es[0], c_es[1])
    Cl,Al,sl = curvars(l_es[0],  l_es[1])
    Cr,Ar,sr = curvars(r_es[0], r_es[1])
    
    # Estimate the smoothing window size
    widths_per_bend = 10 # How many channel widths make up a bend?
    expected_num_infs = cl.length/widths_per_bend/width_chan/2 # Divide by 2 since we're operating on buffered, smoothed versions of the centerline, not the centerline itself
    
    # Try different size windows; use the smallest one that gives the expected number of inflection points
    posswindows = np.arange(25, len(Ccl)/2, 101, dtype=np.int)
    # Ensure all windows are odd
    for ip, pw in enumerate(posswindows):
        if pw % 2 == 0:
            posswindows[ip] = pw + 1
    ninfs = []
    for w in posswindows:
        if w % 2 == 0:
            w = w + 1
        Ctemp = signal.savgol_filter(Ccl, window_length=w, polyorder=0, mode='interp')
        ninfs.append(len(inflection_points(Ctemp)))
    C_smoothing_window = posswindows[np.where(expected_num_infs- ninfs > 0)[0][0]]
    
    # Perform the smoothing on all three curvature signals
    Ccls = signal.savgol_filter(Ccl, window_length=C_smoothing_window, polyorder=3, mode='interp')
    Cls = signal.savgol_filter(Cl, window_length=C_smoothing_window, polyorder=3, mode='interp')
    Crs = signal.savgol_filter(Cr, window_length=C_smoothing_window, polyorder=3, mode='interp')
        
    # Get and filter inflection points for centerline - must be buffer width apart
    n_infs = scl[-1]/10/width_chan
    infs_c, _ = inflection_pts_oversmooth(c_es[0], c_es[1], n_infs)
    infs_l, _ = inflection_pts_oversmooth(l_es[0], l_es[1], n_infs)
    infs_r, _ = inflection_pts_oversmooth(r_es[0], r_es[1], n_infs)
    
#    infs_c = inflection_points(Ccls)
#    infs_r = inflection_points(Crs)
#    infs_l = inflection_points(Cls)
    
    # Prune centerline inflection points to include only those in original centerline
    idcs_to_add = int(grid_spacing/rs_spacing) + 1 # Add an extra meshpoly to the upstream and downstream ends to ensure full coverage
    startidx = np.argmin(np.sqrt((xs_o[0] - c_es[0])**2 + (ys_o[0] - c_es[1])**2)) - idcs_to_add
    endidx = np.argmin(np.sqrt((xs_o[-1] - c_es[0])**2 + (ys_o[-1] - c_es[1])**2)) + idcs_to_add
    infs_c = infs_c[np.logical_and(infs_c > startidx, infs_c < endidx)]
    
    # Insert "inflection points" at beginning and end of list to ensure we cover
    # the full domain. 
    infs_c = np.concatenate(([startidx], infs_c, [endidx]))
    
    # Working upstream->downstream, remove inflection points that are within a 
    # buffer width of their downstream neighbor (only for centerline)
    cs, cds = s_ds(c_es[0], c_es[1]) 
    keep_pts = [infs_c[0]]
    check_pts = np.ndarray.tolist(infs_c[1:])
    while len(check_pts) > 0:
        if cs[check_pts[0]] - cs[keep_pts[-1]] > bufferdist:
            keep_pts.append(check_pts.pop(0))
        else:
            check_pts.pop(0)
    # Ensures the endpoint is retained
    if endidx not in keep_pts:
        keep_pts.pop(-1)
        keep_pts.append(endidx)
    infs_c = keep_pts
    
    plt.close('all')
    plt.plot(c_es[0], c_es[1])
    plt.plot(c_es[0][infs_c], c_es[1][infs_c], 'o')
    plt.plot(l_es[0], l_es[1])
    plt.plot(l_es[0][infs_l], l_es[1][infs_l], 'o')
    plt.plot(r_es[0], r_es[1])
    plt.plot(r_es[0][infs_r], r_es[1][infs_r], 'o')
    plt.axis('equal')
    
    """ Map each centerline inflection point to its counterpart on the left and right valleylines """
    # Draw perpendiculars at each centerline point, intersect them with the left/right
    # valleylines, and find the nearest point on those lines
    intidx_l = []
    intidx_r = []
    for i, ic in enumerate(infs_c):
        
        # Slope of line from currenct centerline inflection point
        m = (c_es[1][ic] - c_es[1][ic-1]) / (c_es[0][ic] - c_es[0][ic-1])
        # Slope of perpendicular line
        minv = -1/m
        # Construct perpendicular lines
        upper_pt = (c_es[0][ic] + bufferdist*2, c_es[1][ic] + bufferdist*2*minv )
        lower_pt = (c_es[0][ic] - bufferdist*2, c_es[1][ic] - bufferdist*2*minv )
#        perpx = [upper_pt[0], lower_pt[0]]
#        perpy = [upper_pt[1], lower_pt[1]]
#        plt.close('all')
#        plt.plot(c_es[0], c_es[1])
#        plt.plot(l_es[0], l_es[1])
#        plt.plot(r_es[0], r_es[1])
#        plt.plot(perpx, perpy)
#        plt.axis('equal')
    
        perpline = LineString([upper_pt, lower_pt])
        
        for lorr in ['l','r']:
            
            if lorr == 'l':
                val_line_shply = l_shapely
                val_line = l_es
            elif lorr == 'r':
                val_line_shply = r_shapely
                val_line = r_es
    
            # Find intersection point
            # At the ends, it's possible that the perpendicular does not intersect the valleylines
            if perpline.intersects(val_line_shply) is False:
                if i == 0:
                    int_pt = (val_line_shply.coords.xy[0][0], val_line_shply.coords.xy[1][0])
                elif i == len(infs_c) - 1:
                    int_pt = (val_line_shply.coords.xy[0][-1], val_line_shply.coords.xy[1][-1])
                else:
                    raise RuntimeError('Perpendicular lines not intersecting valleylines somewhere; try increasing smoothing parameter.')
            else:
                int_pt = perpline.intersection(val_line_shply).coords.xy
            
            # Find corresponding index along valleyline
            int_idx = np.argmin(np.sqrt((val_line[0]-int_pt[0])**2 + (val_line[1]-int_pt[1])**2))
        
            if lorr == 'l':
                intidx_l.append(int_idx)
            elif lorr == 'r':
                intidx_r.append(int_idx)
    
    # Map the intersection points to the nearest inflection point; if there is no
    # inflection point within a threshold, just use the intersection point.
    ii_dist_thresh = width_chan * 3 # threshold for locating matching inflection point
    # Left
    lidx = []
    for il, inti in enumerate(intidx_l):
        if il == 0 or il == len(intidx_l) - 1: # For the first and last points, we would rather map to the nearest point, not necessarily nearest inflection point
            lidx.append(inti)
        else:
            # Closest inflection point
            int_inf_dist = np.abs(sl[inti] - sl[infs_l])
            if np.min(int_inf_dist) < ii_dist_thresh:
                lidx.append(infs_l[np.argmin(int_inf_dist)])
            else:
                lidx.append(inti)
    # Right
    ridx = []
    for ir, inti in enumerate(intidx_r):
        if ir == 0 or ir == len(intidx_r) - 1: # For the first and last points, we would rather map to the nearest point, not necessarily nearest inflection point
            ridx.append(inti)
        else:
            # Closest inflection point
            int_inf_dist = np.abs(sr[inti] - sr[infs_r])
            if np.min(int_inf_dist) < ii_dist_thresh:
                ridx.append(infs_r[np.argmin(int_inf_dist)])
            else:
                ridx.append(inti)
                
    
#    # Plot "perpendiculars" to check
#    plt.close('all')
#    plt.plot(c_es[0], c_es[1])
#    plt.plot(l_es[0], l_es[1])
#    plt.plot(r_es[0], r_es[1])
#    for clid, leftpt, rightpt in zip(infs_c, lidx, ridx):
#        plt.plot([c_es[0][clid], l_es[0][leftpt]], [c_es[1][clid], l_es[1][leftpt]],'k')
#        plt.plot([c_es[0][clid], r_es[0][rightpt]], [c_es[1][clid], r_es[1][rightpt]],'k')
#    plt.axis('equal')
    
    """ Now that "perpendiculars" are known at inflection points, the mesh can be
        generated. Parameterize each segment of line between break points and sample
        along the line at the same interval for left, right, and center. """
    # Can re-use the evenly-spaced spline, just need to determine parameterization 
    # between each pair of break points.
        
#    # Determine starting/ending indices to draw perps--we resampled the centerline so need new index
#    startidx = np.argmin(np.sqrt((xs_o[0] - c_es[0])**2 + (ys_o[0] - c_es[1])**2))
#    endidx = np.argmin(np.sqrt((xs_o[-1] - c_es[0])**2 + (ys_o[-1] - c_es[1])**2))
#    
#    # Prune the inflection indices to the original signal extents
#    toolow = np.where(infs_c < startidx)[0]
#    toohigh = np.where(infs_c > endidx)[0]
#    infs_c = np.delete(infs_c, np.concatenate((toolow, toohigh)))
#    lidx = np.delete(lidx, np.concatenate((toolow, toohigh)))
#    ridx = np.delete(ridx, np.concatenate((toolow, toohigh)))
        
    # Get the u-parameterization corresponding to the grid perps
    u_c = np.arange(scl[startidx], scl[endidx], grid_spacing)/scl[-1]
    u_breaks_c = scl[infs_c]/scl[-1]
    u_breaks_l = sl[lidx]/sl[-1]
    u_breaks_r = sr[ridx]/sr[-1]
    
    # Find the u-breaks for left and right lines correspoinding to those in u_breaks
    u_l = []
    u_r = []
    for i in range(len(u_breaks_c)-1):
                
        du_c = u_breaks_c[i+1] - u_breaks_c[i]
        breaks_within = u_c[np.where(np.logical_and(u_c >= u_breaks_c[i], u_c < u_breaks_c[i+1]))[0]]
        start_u_frac = (breaks_within[0] - u_breaks_c[i]) / du_c
        end_u_frac = (u_breaks_c[i+1] - breaks_within[-1]) / du_c
        # Left
        du_l = u_breaks_l[i+1] - u_breaks_l[i]
        start_u = du_l*start_u_frac + u_breaks_l[i]
        end_u = u_breaks_l[i+1] - du_l * end_u_frac
        uvals = np.linspace(start_u, end_u, len(breaks_within))
        u_l.extend(uvals)
    
        # Right
        du_r = u_breaks_r[i+1] - u_breaks_r[i]
        start_u = du_r*start_u_frac + u_breaks_r[i]
        end_u = u_breaks_r[i+1] - du_r * end_u_frac
        uvals = np.linspace(start_u, end_u, len(breaks_within))
        u_r.extend(uvals)
                        
    # Now resample at the new u-values to get mesh coordinates
    c_mesh = si.splev(u_c, c_spline)
    l_mesh = si.splev(u_l, left_spline)
    r_mesh = si.splev(u_r, right_spline)
    
#    plt.close('all')
#    plt.plot(c_es[0], c_es[1])
#    plt.plot(l_es[0], l_es[1])
#    plt.plot(r_es[0], r_es[1])
#    for i in range(len(l_mesh[0])): 
#        plt.plot([l_mesh[0][i], r_mesh[0][i]], [l_mesh[1][i], r_mesh[1][i]])
#    for i in range(len(c_mesh[0])):
#        plt.plot(c_mesh[0][i], c_mesh[1][i], 'o')
#    plt.axis('equal')
    
    # Mesh is generated; export perpendiculars and grid as shapefile
    lines = []
    for lx, ly, rx, ry in zip(l_mesh[0], l_mesh[1], r_mesh[0], r_mesh[1]):
        lines.append([(lx, ly), (rx, ry)])
    
    # Mesh polygons
    polys = []
    for i in range(len(c_mesh[0])-1):
        polys.append([(l_mesh[0][i], l_mesh[1][i]), (r_mesh[0][i], r_mesh[1][i]), 
                      (r_mesh[0][i+1], r_mesh[1][i+1]), (l_mesh[0][i+1], l_mesh[1][i+1]),
                      (l_mesh[0][i], l_mesh[1][i])])
        
    return lines, polys, [c_es[0][startidx:endidx], c_es[1][startidx:endidx]]


def chan_width(coords, Imask, pixarea=1):

    """
    Returns two estimates of channel width: width_channels is the average
    width of just the channels, and width_extent is the average width of the
    extent of the river (includes islands).
    
    Inputs:
        coords - Nx2 list of (x,y) coordinates defining the centerline of the input mask
        Imask - binary mask on which the centerline was computed
        pixarea - (Optional, float) area of each pixel in the mask. If none is
                  provided, widths will be in units of pixels.
    """
    
    # Estimate length from coodinates
    s, _ = s_ds(coords[0], coords[1])
    len_est = s[-1]
    
    # Estimate unfilled channel width (average widths of actual channels)
    Imask = iu.largest_blobs(Imask, nlargest=1, action='keep')
    area_est = np.sum(np.array(Imask, dtype=np.bool)) * pixarea
    width_channels = area_est/len_est
    
    # Estimate filled channel width (average width of entire channel extents i.e. including islands)
    Imask = iu.fill_holes(Imask)
    area_est = np.sum(np.array(Imask, dtype=np.bool)) * pixarea
    width_extent = area_est/len_est
    
    return width_channels, width_extent


def curvars(xs, ys, unwrap=True):

    """
    Compute curvature (and intermediate variables) for a given set of x,y
    coordinates.
    """
    xs = np.array(xs)
    ys = np.array(ys)
    
    xAi0  = xs[:-1]
    xAi1 = xs[1:]
    yAi0 = ys[:-1]
    yAi1 = ys[1:]
    
    # Compute angles between x,y nodes
#    A = np.arctan(np.divide(yAi1-yAi0,xAi1-xAi0))
    A = np.arctan2(yAi1-yAi0,xAi1-xAi0)
    
    if unwrap is not True:
        Acopy = A.copy()
        
    # Fix phase jumps in angle larger than pi
    A = np.unwrap(A)

    # Compute distance and cumulative distance between nodes
    s, ds = s_ds(xs, ys)
    s = np.delete(s, 0)
    
    # Compute curvature via central differencing
    # See: http://terpconnect.umd.edu/~toh/spectrum/Differentiation.html
    sd = np.zeros([len(s)])
    sd[0] = s[2]
    sd[1:] = s[0:-1]
    
    su = np.zeros([len(s)])
    su[0:-1] = s[1:]
    su[-1] = s[-3]
    
    Ad = np.zeros([len(s)])
    Ad[0] = A[2]
    Ad[1:] = A[0:-1]
    
    Au = np.zeros([len(s)])
    Au[0:-1] = A[1:]
    Au[-1] = A[-3]
    
    # Curvatures - checked against Matlab implementation, OK
    C = -np.divide((np.divide(Au-A,su-s)*(s-sd)+np.divide(A-Ad,s-sd)*(su-s)),(su-sd))
    
    if unwrap is not True:
        Areturn = Acopy
    else:
        Areturn = A

    return C, Areturn, s


def s_ds(xs, ys):
    
    ds = np.sqrt((np.diff(xs))**2+(np.diff(ys))**2)
    s = np.insert(np.cumsum(ds), 0, 0)

    return s, ds


def smooth_curvatures(C, cvtarget, tolerance=10):
    """
    Smoothes a curvature signal until the coefficient of variation of its differenced
    curvatures is within tolerance percent.
    """    
    from scipy.stats import variation

    smoothstep = int(len(C)/100)  
    window = smoothstep
    if window % 2 == 0: # Window must be odd
        window = window + 1
    
    cv = 10000
    while abs((cv-cvtarget)/cv*100) > tolerance:
        Cs = signal.savgol_filter(C, window_length=window, polyorder=3, mode='interp')
        cv = variation(np.diff(Cs))
        
        window = window + smoothstep
        if window % 2 == 0: # Window must be odd
            window = window + 1
    
        if window > len(C)/2:
            print('Could not find solution.')
            return Cs
    
    return Cs


def inflection_points(C):
    """
    Returns the inflection points for an input curvature signal.
    """
    infs1 = np.where(np.logical_and(C[1:] > 0, C[:-1] < 0))[0] + 1
    infs2 = np.where(np.logical_and(C[1:] < 0, C[:-1] > 0))[0] + 1
    infs = np.sort(np.concatenate((infs1, infs2)))
    
    return infs


