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
                                        list(nodes['lakes']))

    # need to modify bridge check for lakes
    # links, nodes = lnu.remove_disconnected_bridge_links(links, nodes)

    links, nodes = lnu.remove_single_pixel_links(links, nodes)

    links, nodes = lnu.find_parallel_links(links, nodes)

    return links, nodes


def make_lakenodes(links, nodes, Ilakes, verbose=True):
    """
    Routine to define the lake nodes.

    Parameters
    ----------
    links: dict
        RivGraph dictionary of links and their attributes
    nodes : dict
        RivGraph dictionary of nodes and their attributes
    Ilakes : np.ndarray
        2-D binary array of lakes
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
    # init node attributes for lakes and their centroids
    nodes['lakes'] = []
    nodes['lake_centroids'] = []

    # define new lakes dictionary
    lakes = dict()

    # get properties associated w/ lakes
    props, labeled = imu.regionprops(Ilakes, props=['perimeter', 'centroid'])
    if verbose is True:
        print(str(np.max(labeled)) + ' lakes identified.')

    # identify the number of lakes
    nlakes = len(props['perimeter'])

    # Loop through each lake, connect it to the network
    # also store some properties
    for i in range(nlakes):
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
        # Dilate the lake and get the newly-added pixels
        Ilakec_d = imu.dilate(Ilakec, n=1, strel='square')
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

        # Assume at least one node connected to each lake
        if len(conn_nodes) == 0:
            print('No connecting nodes were found for lake {}; over-dilating...'.format(i))
            # I have found one case in our test image where the skeleton doesn't actually reach the
            # edge of the blob, but is instead an extra pixel away. Not sure the best approach
            # to handle these, but we can dilate the lake blob a little more to capture them.
            # Don't like this approach because it **could** capture other non-link endpoints
            # that aren't actually connected...but I also don't see another option immediately.

            # Bad coding practice but I'm copy/pasting the above code here but with
            # a fatter dilation
            # Just dilate one more time
            Ilakec_d = imu.dilate(Ilakec, n=2, strel='square')
            Ilakec_d[Ilakec] = False
            dpix_rc = np.where(Ilakec_d)
            dpix_rc = (dpix_rc[0] + ymin - npad, dpix_rc[1] + xmin - npad)
            dpix_idx = np.ravel_multi_index(dpix_rc, Ilakes.shape)

            # Find the nodes connected to the lake
            conn_nodes = []
            for nidx, nid in zip(nodes['idx'], nodes['id']):
                if nidx in dpix_idx:
                    conn_nodes.append(nid)

            # If we still can't find one, give up for now?
            if len(conn_nodes) == 0:
                print('No connecting nodes were found for lake {} and over-dilating did not fix. Debug it'.format(i))

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

            # Add the link to the links dictionary
            links, nodes = lnu.add_link(links, nodes, p_idcs)

    # check that number of lake nodes == number of lakes
    if np.max(labeled) != len(nodes['lakes']):
        warnings.warn('Prototype lake identification has failed. '
                      'The number of lakes from the mask does not '
                      'equal the number of lake nodes identified!')

    return lakes, links, nodes
