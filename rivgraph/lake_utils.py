# -*- coding: utf-8 -*-
"""
lake_utils
==========

A collection of functions for handling channel networks with lake objects.

"""
import warnings
import rivgraph.deltas.delta_utils as du
import rivgraph.ln_utils as lnu
import rivgraph.im_utils as imu
from shapely.geometry import Polygon
from skimage.graph import route_through_array
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np


def prune_deltalakes(links, nodes, path_shoreline, path_inletnodes,
                     gdobj):
    """Pruning functionality that preserves lakes."""
    nodes = du.find_inlet_nodes(nodes, path_inletnodes, gdobj)

    links, nodes = du.clip_by_shoreline(links, nodes, path_shoreline, gdobj)

    links, nodes = lnu.remove_all_spurs(links, nodes,
                                        dontremove=list(nodes['inlets']) +
                                        list(nodes['outlets']) +
                                        list(nodes['lakes']) +
                                        list(nodes['lake_edges']))

    # bridge link removal still buggy
    # links, nodes = lnu.remove_disconnected_bridge_links(links, nodes)

    links, nodes = lnu.remove_single_pixel_links(links, nodes)

    links, nodes = lnu.find_parallel_links(links, nodes)

    return links, nodes


def make_lakenodes(links, nodes, Ilakes, proplist=None, min_dil=1, max_dil=2,
                   verbose=True):
    """
    Routine to define the lake nodes.

    Function currently relies on two parameters controlling the range of
    morphological dilations that occur. This is less than ideal, but without
    this type of manual control, it is unclear how we will be able to
    automatically handle all cases (lakes with 1, 2, 3+ connections). So some
    manual inspection is needed after running this function (and later after
    pruning) to make sure that all of the lakes are connected. Or if you have
    an idea about the distance from the lake to the perimeter nodes you should
    use that to set the minimum dilation (min_dil) parameter.

    Parameters
    ----------
    links: dict
        RivGraph dictionary of links and their attributes
    nodes : dict
        RivGraph dictionary of nodes and their attributes
    Ilakes : np.ndarray
        2-D binary array of lakes
    proplist : list, optional
        List of lake parameters to store in the lakes dictionary. These can
        be any property available in the :obj:`~rivgraph.im_utils.regionprops`
        function. Perimeter and centroid are always calculated.
    min_dil : int, optional
        Minimum number of dilations to perform for detecting nodes adjacent to
        lakes. Default is 1, although visual inspection should be used to make
        sure that all lakes are connected.
    max_dil : int, optional
        Maximum number of dilations you are willing to perform to try and
        connect lake centers to nodes along the perimeter. Default value is
        2 because that is one value above the default minimum of 1. If this
        default value is below min_dil, it is set to be greater than min_dil
        by 1.
    verbose : bool, optional
        Controls verbosity of the method, True means print-statements are
        output to the console, False is suppresses output. Default is True.

    Returns
    -------
    lakes : dict
        New RivGraph dictionary of lakes and their properties/attributes
    links : dict
        RivGraph dictionary with new links after accounting for the lakes
    nodes : dict
        RivGraph dictionary of nodes with lake information now provided

    """
    # handle proplist
    if proplist is None:
        proplist = ['perimeter', 'centroid']
    if (type(proplist) is list) is False:
        raise ValueError('proplist was not actually a list, it must be.')
    if 'perimeter' not in proplist:
        proplist.append('perimeter')
    if 'centroid' not in proplist:
        proplist.append('centroid')

    # handle dilations
    if max_dil <= min_dil:
        max_dil = min_dil + 1

    # init node attributes for lakes and their centroids
    nodes['lakes'] = []
    nodes['lake_centroids'] = []
    nodes['lake_edges'] = []  # nodes along lake perimeters

    # define new lakes dictionary w/ id and conn keys
    lakes = {'id': [], 'conn': []}

    # get properties associated w/ lakes
    props, labeled = imu.regionprops(Ilakes, props=proplist)
    if verbose is True:
        print(str(np.max(labeled)) + ' lakes identified.')

    # add and set keys-value pairs for all properties in proplist
    for i in proplist:
        lakes[i] = props[i]

    # identify the number of lakes
    nlakes = len(props['perimeter'])

    # Loop through each lake, connect it to the network
    # also store some properties
    for i in range(nlakes):
        # set lake id
        lakes['id'].append(i)
        # Get representative point within lake
        # similar to centroid, but ensures point is within the polygon
        perim = props['perimeter'][i]
        perim = np.vstack((perim, perim[0]))  # Close the perimeter
        lakepoly = Polygon(perim)  # Make polygon
        rep_pt = lakepoly.representative_point().coords.xy
        # Check that integer-ed rep_pt is within the blob
        rpx, rpy = round(rep_pt[1][0]), round(rep_pt[0][0])
        if Ilakes[rpy, rpx] is False:
            raise Exception('Found a case where rounded representative '
                            'point is not within polygon :(')

        # Find all links connected to this lake
        # First, create a cropped image of the lake
        xmax, xmin, ymax, ymin = np.max(perim[:, 1]) + 1, \
            np.min(perim[:, 1]), np.max(perim[:, 0]) + 1, \
            np.min(perim[:, 0])
        Ilakec = np.array(labeled[ymin:ymax, xmin:xmax], dtype=bool)

        # Pad the image so there is room to dilate
        npad = 3
        Ilakec = np.pad(Ilakec, npad, mode='constant')

        # Find nodes connected to the lake
        conn_nodes = find_conn_nodes(Ilakec, min_dil, ymin, xmin, npad, nodes,
                                     Ilakes)

        # Assume at least one node connected to each lake
        dil_iter = 1
        while len(conn_nodes) == 0 and dil_iter <= max_dil:
            if verbose is True:
                print('No connecting nodes were found for lake {L}; '
                      'over-dilating with dilation = {D}'.format(L=i,
                                                                 D=dil_iter))
            # I have found one case in our test image where the skeleton
            # doesn't actually reach the edge of the blob, but is instead an
            # extra pixel away. Not sure the best approach to handle these, but
            # we can dilate the lake blob a little more to capture them. Don't
            # like this approach because it **could** capture other non-link
            # endpoints that aren't actually connected...but I also don't see
            # another option immediately.

            # Find the nodes connected to the lake, dilate 1 extra time
            conn_nodes = find_conn_nodes(Ilakec, dil_iter, ymin, xmin, npad,
                                         nodes, Ilakes)

            dil_iter += 1  # increment amount of dilation

        # If we still can't find one, give up for now?
        if len(conn_nodes) == 0 and dil_iter == max_dil and verbose is True:
            print('No connecting nodes were found for lake {L} '
                  'and over-dilating did not fix. Debug it'.format(L=i))
        elif len(conn_nodes) != 0 and dil_iter > 1 and verbose is True:
            print('Found a connecting node with a dilation of '
                  '{D}.'.format(D=dil_iter-1))

        # record these connecting nodes as lake edge nodes
        nodes['lake_edges'] = nodes['lake_edges'] + conn_nodes

        # Now we connect all the adjacent link nodes to the representative point within the lake;
        # Ensure we stay within lake pixels by using shortest path
        for cn in conn_nodes:
            # Adjust the representative point to be inside the cropped bounds
            reppt = (rpy - ymin + npad, rpx - xmin + npad)

            # Make a cost image to find a shortest path from node to rep point
            Icosts = np.ones(Ilakec.shape)
            Icosts[reppt] = 0
            Icosts = distance_transform_edt(Icosts)
            Icosts[~Ilakec] = np.max((100000, np.max(Icosts)**2))  # Set values outside the lake to be very high so path will not traverse them

            # Put the link's node in the cropped coordinates
            node_rc = np.unravel_index(nodes['idx'][nodes['id'].index(cn)],
                                       Ilakes.shape)
            node_rc = (node_rc[0] - ymin + npad, node_rc[1] - xmin + npad)
            # Compute the shortest path
            path, cost = route_through_array(
                Icosts, start=(node_rc), end=reppt, fully_connected=True)

            # Convert the pixels of the shortest path back into global coordinates
            path = np.array(path)
            path[:, 0] = path[:, 0] + ymin - npad
            path[:, 1] = path[:, 1] + xmin - npad
            p_idcs = np.ravel_multi_index((path[:, 0], path[:, 1]),
                                          Ilakes.shape)

            # set of node ids before creating a new one
            old_ids = set(nodes['id'])

            # Add the link to the links dictionary
            links, nodes = lnu.add_link(links, nodes, p_idcs)

            # use old id list to identify the new node that has been created
            new_node_id = list(set(nodes['id']).difference(old_ids))

            # add this to the lake info
            if len(new_node_id) == 1:
                nodes['lakes'].append(new_node_id[0])
                nodes['lake_centroids'].append(props['centroid'][i])
                lakes['conn'].append(new_node_id[0])

    # check that number of lake nodes == number of lakes
    if np.max(labeled) != len(nodes['lakes']):
        warnings.warn('Prototype lake identification has failed. '
                      'The number of lakes from the mask does not '
                      'equal the number of lake nodes identified!')

    return lakes, links, nodes


def find_conn_nodes(Ilakec, num_dil, ymin, xmin, npad, nodes, Ilakes):
    """
    Find nodes that connect to a lake.

    This is really an internal function that a user should not need to call.
    Basically the lake footprint is morphologically dilated to find any
    connecting nodes present in the graph. The number of dilations is expected
    to be very low for this process, if this exceeds 2, we suggest manual
    inspection to take a closer look at the issue.

    Parameters
    ----------
    Ilakec : np.ndarray
        Padded boolean array of the lake.
    num_dil : int
        Number of times to perform the dilation. When this function is called
        internally, 1 is used, and 2 is only used as a fall-back. Anything
        beyond 2 we consider anomalous and worth human inspection of the issue.
    ymin : int
        Minimum y-coordinate for this lake
    xmin : int
        Minimum x-coordinate for this lake
    npad : int
        Padding used to make Ilakec
    nodes : dict
        RivGraph nodes dictionary
    Ilakes : np.ndarray
        2-D array of the lakes, used here for its shape really

    Returns
    -------
    conn_nodes : list
        List of nodes connected to the lake

    """
    # Dilate the lake and get the newly-added pixels
    Ilakec_d = imu.dilate(Ilakec, n=num_dil, strel='square')
    Ilakec_d[Ilakec] = False
    # Get the indices of the dilated pixels
    dpix_rc = np.where(Ilakec_d)
    # Adjust the rows and columns to account image cropping
    dpix_rc = (dpix_rc[0] + ymin - npad, dpix_rc[1] + xmin - npad)
    # Convert to index
    dpix_idx = np.ravel_multi_index(dpix_rc, Ilakes.shape)

    # Find the nodes connected to the lake
    conn_nodes = []
    for nidx, nid in zip(nodes['idx'], nodes['id']):
        if nidx in dpix_idx:
            conn_nodes.append(nid)

    return conn_nodes
