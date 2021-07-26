# -*- coding: utf-8 -*-
"""
Mask Filtering Utils (mask_utils.py)
====================================

Functions for filtering islands from channel masks. Currently in beta.
Also see :mod:`im_utils` for morphologic operations that could be useful.

Created on Mon Jul 6 18:29:23 2020

@author: Jon
"""
import numpy as np
import geopandas as gpd
import rivgraph.im_utils as iu
import rivgraph.geo_utils as gu
from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from scipy import stats
import rivgraph.im_utils as im
import networkx as nx


def pixagon(c_cent, r_cent, pixlen):
    """
    Returns a shapely polygon
    of a provided center coordinate given the pixel resolution (pixlen).
    """
    halflen = pixlen/2
    c_corner = np.array([c_cent-halflen, c_cent+halflen, c_cent+halflen,
                         c_cent-halflen, c_cent-halflen])
    r_corner = np.array([r_cent+halflen, r_cent+halflen, r_cent-halflen,
                         r_cent-halflen, r_cent+halflen])

    pixgon = Polygon(zip(c_corner, r_corner))

    return pixgon


def get_island_properties(Imask, pixlen, pixarea, crs, gt, props, connectivity=2):
    """Get island properties."""

    # maxwidth is an additional property
    if 'maxwidth' in props:
        props.remove('maxwidth')
        do_maxwidth = True
    else:
        do_maxwidth = False

    # Need perimeter to make island polygons
    if 'perimeter' not in props:
        props.append('perimeter')

    # Pad by one pixel to help identify and remove the outer portion of
    # the channel netowrk
    Imaskpad = np.array(np.pad(Imask, 1, mode='constant'), dtype=bool)
    Imp_invert = np.invert(Imaskpad)

    rp_islands, Ilabeled = iu.regionprops(Imp_invert, props=props, connectivity=connectivity)

    # Make polygons of the island perimeters
    # Also get ids to match the labeled image
    pgons = []
    ids = []
    for ip, p in enumerate(rp_islands['perimeter']):
        ids.append(Ilabeled[p[0][0], p[0][1]])  # Store the index

        p = np.vstack((p, p[0]))  # Close the polygon
        # Adjust for the single-pixel padding we added to the image
        cr = gu.xy_to_coords(p[:, 1] - 1, p[:, 0] - 1, gt)

        # Special cases: where the island is two pixels or less, we use the
        # corner coordinates rather than the center coordinates to define
        # the polygon.
        if len(cr[0]) <= 2:
            pixgon = [pixagon(cc, rc, pixlen) for cc, rc, in zip(cr[0], cr[1])]
            if len(pixgon) > 1:
                pixgon = cascaded_union(pixgon)
            pgons.append(pixgon)
        else:
            pgons.append(Polygon(zip(cr[0], cr[1])))

    # Do maximum width if requested
    if do_maxwidth:
        Idist = distance_transform_edt(Imp_invert)
        maxwids = []
        for i in ids:
            maxwids.append(np.max(Idist[Ilabeled == i])*2*pixlen)

    # Convert requested properties to proper units
    if 'area' in props:
        rp_islands['area'] = rp_islands['area'] * pixarea
    if 'major_axis_length' in props:
        rp_islands['major_axis_length'] = rp_islands['major_axis_length'] * pixlen
    if 'minor_axis_length' in props:
        rp_islands['minor_axis_length'] = rp_islands['minor_axis_length'] * pixlen
    if 'perim_len' in props:
        rp_islands['perim_len'] = rp_islands['perim_len'] * pixlen
    if 'convex_area' in props:
        rp_islands['convex_area'] = rp_islands['convex_area'] * pixarea

    # Need to change 'area' key as it's a function in geopandas
    if 'area' in rp_islands:
        rp_islands['Area'] = rp_islands.pop('area')

    # Create islands geodataframe
    gdf_dict = {k:rp_islands[k] for k in rp_islands if k not in ['coords', 'perimeter', 'centroid']}
    gdf_dict['geometry'] = pgons
    gdf_dict['id'] = ids
    if do_maxwidth:
        gdf_dict['maxwid'] = maxwids
    gdf = gpd.GeoDataFrame(gdf_dict)
    gdf.crs = crs

    # Identify and remove the border blob
    border_id = Ilabeled[0][0]
    Ilabeled[Ilabeled == border_id] = 0
    gdf = gdf[gdf.id.values != border_id]

    # Put 'id' column in front
    colnames = [k for k in gdf.keys()]
    colnames.remove('id')
    colnames.insert(0, 'id')
    gdf = gdf[colnames]

    return gdf, Ilabeled[1:-1, 1:-1]


def surrounding_link_properties(links, nodes, Imask, islands, Iislands,
                                pixlen, pixarea):
    """
    Find the links surrounding each island and computes their properties. This
    function is useful for filtering; e.g. when it is desired to remove islands
    surrounded by very large channels.

    Parameters
    ----------
    links : dict
        Network links.
    nodes : dict
        Network nodes.
    Imask : np.array
        Binary mask of the channel network.
    islands : geopandas.GeoDataframe
        Contains island boundaries and associated properties. Created by
        get_island_properties().
    Iislands : np.array
        Image wherein each island has a unique integer ID.
    pixlen : numeric
        Nominal length of a pixel (i.e. its resolution).
    pixarea : numeric
        Nominal area of a pixel.

    Returns
    -------
    islands : geopandas.GeoDataframe
        DESCRIPTION.

    """
    # obj = d
    # links = obj.links
    # nodes = obj.nodes
    # Imask = obj.Imask
    # pixlen = obj.pixlen
    # pixarea = obj.pixarea
    # gt = obj.gt
    # crs = obj.crs
    # props=['area', 'maxwidth', 'major_axis_length', 'minor_axis_length']
    # islands, Iislands = get_island_properties(obj.Imask, pixlen, pixarea, obj.crs, obj.gt, props)
    # islands.to_file(r"C:\Users\Jon\Desktop\Research\John Shaw\Deltas\GBM\GBM_islands.shp")
    # np.save(r'C:\Users\Jon\Desktop\Research\John Shaw\Deltas\GBM\GBM_Iislands.npy', Iislands)
    # # islands = gpd.read_file(r"C:\Users\Jon\Desktop\Research\eBI\Results\Indus\Indus_islands.shp")
    # # Iislands = np.load(r'C:\Users\Jon\Desktop\Research\eBI\Results\Indus\Indus_Iislands.npy')

    # Rasterize the links and nodes
    Iln = np.zeros(Imask.shape, dtype=int)

    # Burn links into raster
    for lidcs in links['idx']:
        rcidcs = np.unravel_index(lidcs, Iln.shape)
        Iln[rcidcs] = 1
    # Burn nodes into raster, but use their negative so we can find them later
    for nid, nidx in zip(nodes['id'], nodes['idx']):
        rc = np.unravel_index(nidx, Iln.shape)
        Iln[rc] = -nid

    # Pad Ilids and Imask to avoid edge effects later
    npad = 8
    Iln = np.pad(Iln, npad, mode='constant')
    Imask = np.array(np.pad(Imask, npad, mode='constant'), dtype=bool)
    Iislands = np.pad(Iislands, npad, mode='constant')

    # Make a binary version of the network skeleton
    Iskel = np.array(Iln, dtype=bool)
    # Invert the skeleton
    Iskel = np.invert(Iskel)

    # Find the regions of the inverted map
    regions, Ireg = im.regionprops(Iskel, props=['coords', 'area', 'label'],
                                   connectivity=1)
    regions['area'] = regions['area'] * pixarea

    # Dilate each region and get the link ids that encompass it
    # Ensure the set of link ids forms a closed loop; remove link ids that don't
    # Use the loop links to compute the average river width around the island
    # Finally, map the region to its corresponding island and compute the island
    # properties to determine whether or not to fill it
    keys = ['sur_area', 'sur_avg_wid', 'sur_max_wid', 'sur_min_wid']
    for k in keys:
        islands[k] = [np.nan for r in range(len(islands))]
    islands['sur_link_ids'] = ['' for r in range(len(islands))]

    # # Can speed up the calculation by skipping huge regions
    # max_area = np.mean(links['wid_adj'])**2 * 20

    imshape = Ireg.shape
    for idx in range(len(islands)):
        print(idx)

        # Identify the region associated with the island
        i_id = islands.id.values[idx]
        r_id = stats.mode(Ireg[Iislands == i_id])[0][0]

        # It is possible that the corresponding region is a 0 pixel, or one
        # that comprises the network. This usually happens only when the island
        # is one or two pixels. Skip these islands
        if r_id == 0:
            continue
        r_idx = np.where(regions['label'] == r_id)[0][0]

        # Get the region's properties
        ra = regions['area'][r_idx]
        rc = regions['coords'][r_idx]

        # if ra > max_area:
        #     continue

        # Make region blob
        Irblob, cropped = im.crop_binary_coords(rc)

        # Pad and dilate the blob
        Irblob = np.pad(Irblob, npad, mode='constant')
        Irblob = np.array(im.dilate(Irblob, n=2, strel='disk'), dtype=bool)

        # Adjust padded image in case pads extend beyond original image boundary
        if cropped[0] - npad < 0:
            remove = npad - cropped[0]
            Irblob = Irblob[:, remove:]
            cropped[0] = 0
        else:
            cropped[0] = cropped[0] - npad
        if cropped[1] - npad < 0:
            remove = npad - cropped[1]
            Irblob = Irblob[abs(remove):, :]
            cropped[1] = 0
        else:
            cropped[1] = cropped[1] - npad
        if cropped[2] + npad > imshape[1]:
            remove = (cropped[2] + npad) - imshape[1]
            Irblob = Irblob[:, :(-remove-1)]
            cropped[2] = imshape[1]
        else:
            cropped[2] = cropped[2] + npad
        if cropped[3] + npad > imshape[0]:
            remove = (cropped[3] + npad) - imshape[0]
            Irblob = Irblob[:(-remove-1), :]
            cropped[3] = imshape[0]
        else:
            cropped[3] = cropped[3] + npad

        # Get node ids that overlap the dilated blob
        Iln_crop = Iln[cropped[1]:cropped[3]+1, cropped[0]:cropped[2]+1]
        lids = Iln_crop[Irblob]
        overlap_nodes = -np.unique(lids[lids < 0])

        # Get the links connected to the overlap nodes so we can construct the
        # mini-graph
        overlap_links = [li for l in [nodes['conn'][nodes['id'].index(nid)] for nid in overlap_nodes] for li in l]

        # Try to find a loop using the identified link ids
        G = nx.Graph()
        G.add_nodes_from(overlap_nodes)
        lconn = [links['conn'][links['id'].index(lid)] for lid in overlap_links]
        for lc in lconn:
            G.add_edge(lc[0], lc[1])
        surrounding_nodes = nx.cycle_basis(G)

        # Check if we're dealing with a parallel loop
        if len(surrounding_nodes) == 0:
            if len(overlap_nodes) == 2:
                if sum([l in nodes['conn'][nodes['id'].index(overlap_nodes[1])] for l in nodes['conn'][nodes['id'].index(overlap_nodes[0])]]) > 1:
                    surrounding_nodes = [[o for o in overlap_nodes]]
            else:  # We assume that if no loops were found, this must be a parallel loop
                for on in overlap_nodes:
                    conn = nodes['conn'][nodes['id'].index(on)]
                    for on2 in overlap_nodes:
                        if on2 == on:
                            continue
                        else:
                            conn2 = nodes['conn'][nodes['id'].index(on2)]
                            if sum([c in conn2 for c in conn]) == 2:
                                surrounding_nodes = [[on, on2]]
                                break

        # # Check if links are at outlet or inlet
        # if len(surrounding_nodes) == 0:
        #     poss_nodes = np.array([links['conn'][links['id'].index(lid)] for lid in lids]).flatten()
        #     if any(np.in1d(poss_nodes, nodes['inlets'])) or any(np.in1d(poss_nodes, nodes['outlets'])):
        #         # Only keep link ids that have 3 or more occurrences
        #         surrounding_nodes = [[lid for lid, ct in zip(cts[0], cts[1]) if ct > 2]]
        #         print('io:{}'.format(ic))

        if len(surrounding_nodes) == 0:
            Warning('Cant find surrounding links for region {}.'.format(idx))

        # If multiple loops were found
        if len(surrounding_nodes) > 1:
            # Choose the surrounding nodes that contain the highest
            # fraction of overlap with the overlap_nodes
            fracs = []
            for sn in surrounding_nodes:
                in_or_out = [s in overlap_nodes for s in sn]
                fracs.append(sum(in_or_out)/len(overlap_nodes))
            surrounding_nodes = [surrounding_nodes[fracs.index(max(fracs))]]

        # At this point, only one loop should be present
        # if len(surrounding_nodes) != 1:
        #     import pdb
        #     pdb.set_trace()
        assert(len(surrounding_nodes)==1)
        surrounding_nodes = surrounding_nodes[0]
        surrounding_nodes.append(surrounding_nodes[0])

        # Get the links of the loop
        surrounding_links = []
        for i in range(len(surrounding_nodes)-1):
            n1 = surrounding_nodes[i]
            n2 = surrounding_nodes[i+1]
            for lid in overlap_links:
                lconn = links['conn'][links['id'].index(lid)]
                if n1 in lconn and n2 in lconn:
                    surrounding_links.append(lid)
        surrounding_links = list(set(surrounding_links))
        islands.sur_link_ids.values[idx] = str(surrounding_links)

        # Now that links surrounding the island are known, can compute some
        # of their morphologic metrics.
        # Use a length-weighted width. Could alternatively use the 'wid_pix' but
        # that includes the misleading connector pixels
        wids = np.array([links['wid_adj'][links['id'].index(lid)] for lid in surrounding_links])
        lens = np.array([links['len_adj'][links['id'].index(lid)] for lid in surrounding_links])
        avg_wid = np.sum(wids * lens) / np.sum(lens)
        islands.sur_avg_wid.values[idx] = avg_wid
        islands.sur_max_wid.values[idx] = np.max(wids)
        islands.sur_min_wid.values[idx] = np.min(wids)
        islands.sur_area.values[idx] = ra  # already converted to pixarea

    return islands


def thresholding_set1(islands, apex_width):

    # Thresholding
    remove = set()

    # Global thresholding -- islands smaller than 1/10 the apex_wid^2
    area_thresh = (1/10 * apex_width)**2
    remove.update(np.where(islands.Area.values < area_thresh)[0].tolist())

    # Threshold islands whose major axis length is less than 1/3 of the apex width
    maj_axis_thresh = apex_width/4
    remove.update(np.where((islands.major_axis_length.values < maj_axis_thresh))[0].tolist())

    # Threshold island area/surrounding area
    area_rat_thresh = 0.01
    remove.update(np.where(islands.Area.values/islands.sur_area.values < area_rat_thresh)[0].tolist())

    # # Threshold average island width as a fraction of surrounding channel widths
    avgwid_ratio_thresh = 0.1
    imal = islands.major_axis_length.values
    imal[imal == 0] = np.nan
    avg_island_wid = islands.Area.values / imal
    remove.update(np.where(avg_island_wid/islands.sur_avg_wid.values < avgwid_ratio_thresh)[0].tolist())

    # Keep islands with a major axis length greater than the apex width
    keep = set()
    keep.update(np.where(islands.major_axis_length.values > apex_width)[0].tolist())

    # Do the thresholding
    remove = remove - keep

    return remove
