"""
Mask to Graph Utilities (mask_to_graph.py)
==========================================

Functions for converting a binary channel mask to a graphical representation.
"""

import cv2
import numpy as np
from skimage import morphology, measure
from rivgraph import walk
import rivgraph.ln_utils as lnu
import rivgraph.im_utils as imu
from rivgraph.ordered_set import OrderedSet


def skel_to_graph(Iskel):
    """
    Resolves a skeleton into its consitutent links and nodes.
    This function finds a starting point to walk along a skeleton, then begins
    the walk. Rules are in place to ensure the network is fully resolved. One
    of the key algorithms called by this function involves the identfication of
    branchpoints in a way that eliminates unnecessary ones to create a parsimonious
    network. Rules are baked in for how to walk along the skeleton in cases
    where multiple branchpoints are clustered or there are multiple possible
    links to walk along.

    Note that some minor adjustments to the skeleton may be made in order to
    reduce the complexity of the network. For example, in the case of a "+"
    with a missing center pixel in the skeleton, this function will add the
    pixel to the center to enable the use of a single branchpoint as opposed to
    four.

    The takeaway is that there is no guarantee that the input skeleton will
    be perfectly preserved when network-ifying. One possible workaround, if
    perfect preservation is required, is to resample the skeleton to double the
    resolution.

    Parameters
    ----------
    Iskel : np.ndarray
        Binary image of a skeleton.

    Returns
    -------
    links : dict
        Links of the network with four properties:

        1. 'id' - a list of unique ids for each link in the network

        2. 'idx' - a list containing the pixel indices within Iskel that defines the link. These are ordered.

        3. 'conn' - a list of 2-element lists containing the node ids that the link is connected to.

        4. 'n_networks' - the number of disconnected networks found in the skeleton

    nodes : dict
        Nodes of the network with four properties:

        1. 'id' - a list of unique ids for each node in the network

        2. 'idx' - the index within Iskel of the node location

        3. 'conn' - a list of lists containing the link ids connected to this node

    """

    def check_startpoint(spidx, Iskel):
        """
        Returns True if a skeleton pixel's first neighbor is not a branchpoint
        (i.e. the start pixel is valid for a walk), else returns False.


        Parameters
        ----------
        spidx : int
            Index within Iskel of the point to check.
        Iskel : np.array
            Image of skeletonized mask.

        Returns
        -------
        chk_sp : bool
            True if the startpoint is valid; else False.

        """
        neighs = walk.walkable_neighbors([spidx], Iskel)
        isbp = walk.is_bp(neighs.pop(), Iskel)

        if isbp == 0:
            chk_sp = True
        else:
            chk_sp = False

        return chk_sp


    def find_starting_pixels(Iskel):
        """
        Finds an endpoint pixel to begin walking to resolve network.

        Parameters
        ----------
        Iskel : np.array
            Image of skeletonized mask.

        Returns
        -------
        startpoints : list
            Possible starting points for the walk.

        """
        # Get skeleton connectivity
        eps = imu.skel_endpoints(Iskel)
        eps = set(np.ravel_multi_index(eps, Iskel.shape))

        # Get one endpoint per connected component in network
        rp, _ = imu.regionprops(Iskel, ['coords'])
        startpoints = []
        for ni in rp['coords']:
            idcs = set(np.ravel_multi_index((ni[:,0], ni[:,1]), Iskel.shape))
            # Find a valid endpoint for each connected component network
            poss_id = idcs.intersection(eps)
            if len(poss_id) > 0:
                for pid in poss_id:
                    if check_startpoint(pid, Iskel) is True:
                        startpoints.append(pid)
                        break

        return startpoints

    # Pad the skeleton image to avoid edge problems when walking along skeleton
    initial_dims = Iskel.shape
    npad = 20
    Iskel = np.pad(Iskel, npad, mode='constant', constant_values=0)
    dims = Iskel.shape

    # Find starting points of all the networks in Iskel
    startpoints = find_starting_pixels(Iskel)

    # Initialize topology storage vars
    nodes = dict()
    nodes['idx'] = OrderedSet([])
    nodes['conn'] = []

    links = dict()
    links['idx'] = [[]]
    links['conn'] = [[]]
    links['id'] = OrderedSet([])

    # Initialize first links emanting from all starting points
    for i, sp in enumerate(startpoints):
        links = lnu.link_updater(links, len(links['id']), sp, i)
        nodes = lnu.node_updater(nodes, sp, i)
        first_step = walk.walkable_neighbors(links['idx'][i], Iskel)
        links = lnu.link_updater(links, i, first_step.pop())

    links['n_networks'] = i+1

    # Initialize set of links to be resolved
    links2do = OrderedSet(links['id'])

    while links2do:

        linkid = next(iter(links2do))
        linkidx = links['id'].index(linkid)

        walking = 1
        cantwalk = walk.cant_walk(links, linkidx, nodes, Iskel)

        while walking:

            # Get next possible steps
            poss_steps = walk.walkable_neighbors(links['idx'][linkidx], Iskel)

            # Now we have a few possible cases:
            # 1) endpoint reached,
            # 2) only one pixel to walk to: must check if it's a branchpoint so walk can terminate
            # 3) two pixels to walk to: if neither is branchpoint, problem in skeleton. If one is branchpoint, walk to it and terminate link. If both are branchpoints, walk to the one that is 4-connected.

            if len(poss_steps) == 0: # endpoint reached, update node, link connectivity
                nodes = lnu.node_updater(nodes, links['idx'][linkidx][-1], linkid)
                links = lnu.link_updater(links, linkid, conn=nodes['idx'].index(links['idx'][linkidx][-1]))
                links2do.remove(linkid)
                break # must break rather than set walking to 0 as we don't want to execute the rest of the code

            if len(links['idx'][linkidx]) < 4:
                poss_steps = list(poss_steps - cantwalk)
            else:
                poss_steps = list(poss_steps)

            if len(poss_steps) == 0: # We have reached an emanating link, so delete the current one we're working on
                links, nodes = walk.delete_link(linkid, links, nodes)
                links2do.remove(linkid)
                walking = 0

            elif len(poss_steps) == 1: # Only one option, so we'll take the step
                links = lnu.link_updater(links, linkid, poss_steps)

                # But check if it's a branchpoint, and if so, stop marching along this link and resolve all the branchpoint links
                if walk.is_bp(poss_steps[0], Iskel) == 1:
                    links, nodes, links2do = walk.handle_bp(linkid, poss_steps[0], nodes, links, links2do, Iskel)
                    links, nodes, links2do = walk.check_dup_links(linkid, links, nodes, links2do)
                    walking = 0 # on to next link

            elif len(poss_steps) > 1: # Check to see if either/both/none are branchpoints
                isbp = []
                for n in poss_steps:
                    isbp.append(walk.is_bp(n, Iskel))

                if sum(isbp) == 0:

                    # Compute 4-connected neighbors
                    isfourconn = []
                    for ps in  poss_steps:
                        checkfour = links['idx'][linkidx][-1] - ps
                        if checkfour in [-1, 1, -dims[1], dims[1]]:
                            isfourconn.append(1)
                        else:
                            isfourconn.append(0)

                    # Compute noturn neighbors
                    noturn = walk.idcs_no_turnaround(links['idx'][linkidx][-2:], Iskel)
                    noturnidx = [n for n in noturn if n in poss_steps]

                    # If we can walk to a 4-connected pixel, we will
                    if sum(isfourconn) == 1:
                        links = lnu.link_updater(links, linkid, poss_steps[isfourconn.index(1)])
                    # If we can't walk to a 4-connected, try to walk in a direction that does not turn us around
                    elif len(noturnidx) == 1:
                        links = lnu.link_updater(links, linkid, noturnidx)
                    # Else, shit. You've found a critical flaw in the algorithm.
                    else:
                        print('idx: {}, poss_steps: {}'.format(links['idx'][linkidx][-1], poss_steps))
                        raise RuntimeError('Ambiguous which step to take next :( Please raise issue at https://github.com/jonschwenk/RivGraph/issues.')

                elif sum(isbp) == 1:
                    # If we've already accounted for this branchpoint, delete the link and halt
                    links = lnu.link_updater(links, linkid, poss_steps[isbp.index(1)])
                    links, nodes, links2do = walk.handle_bp(linkid, poss_steps[isbp.index(1)], nodes, links, links2do, Iskel)
                    links, nodes, links2do = walk.check_dup_links(linkid, links, nodes, links2do)
                    walking = 0

                elif sum(isbp) > 1:
                        # In the case where we can walk to more than one branchpoint, choose the
                        # one that is 4-connected, as this is how we've designed branchpoint
                        # assignment for complete network resolution.
                        isfourconn = []
                        for ps in  poss_steps:
                            checkfour = links['idx'][linkidx][-1] - ps
                            if checkfour in [-1, 1, -dims[1], dims[1]]:
                                isfourconn.append(1)
                            else:
                                isfourconn.append(0)

                        # Find poss_step(s) that is both 4-connected and a branchpoint
                        isbp_and_fourconn_idx = [i for i in range(0,len(isbp)) if isbp[i]==1 and isfourconn[i]==1]

                        # If we don't have exactly one, shit.
                        if len(isbp_and_fourconn_idx) != 1:
                            print('idx: {}, poss_steps: {}'.format(links['idx'][linkidx][-1], poss_steps))
                            raise RuntimeError('There is not a unique branchpoint to step to.')
                        else:
                            links = lnu.link_updater(links, linkid, poss_steps[isbp_and_fourconn_idx[0]])
                            links, nodes, links2do = walk.handle_bp(linkid, poss_steps[isbp_and_fourconn_idx[0]], nodes, links, links2do, Iskel)
                            links, nodes, links2do = walk.check_dup_links(linkid, links, nodes, links2do)
                            walking = 0

    # Put the link and node coordinates back into the unpadded
    links, nodes = lnu.adjust_for_padding(links, nodes, npad, dims, initial_dims)

    # Add indices to nodes--this probably should've been done in network extraction
    # but since nodes have unique idx it was unnecessary.
    nodes['id'] = OrderedSet(range(0,len(nodes['idx'])))

    # Remove duplicate links if they exist; for some single-pixel links,
    # duplicates are formed. Ideally the walking code should ensure that this
    # doesn't happen, but for now removing duplicates suffices.
    links, nodes = lnu.remove_duplicate_links(links, nodes)

    return links, nodes


def skeletonize_mask(Imask):
    """
    Skeletonizes an input binary image, typically a mask. Also performs some
    skeleton simplification by (1) removing pixels that don't alter connectivity,
    and (2) filling small skeleton holes and reskeletonizing.

    Parameters
    ----------
    Imask : np.array
        Binary image to be skeletonized.

    Returns
    -------
    Iskel : np.array
        The skeletonization of Imask.

    """
    # Create copy of mask to skeletonize
    Iskel = np.array(Imask, copy=True, dtype='bool')

    # Perform skeletonization
    Iskel = morphology.skeletonize(Iskel)

    # Simplify the skeleton (i.e. remove pixels that don't alter connectivity)
    Iskel = simplify_skel(Iskel)

    # Fill small skeleton holes, re-skeletonize, and re-simplify
    Iskel = imu.fill_holes(Iskel, maxholesize=4)
    Iskel = morphology.skeletonize(Iskel)
    Iskel = simplify_skel(Iskel)

    # Fill single pixel holes
    Iskel = imu.fill_holes(Iskel, maxholesize=1)

    return Iskel


def skeletonize_river_mask(I, es, padscale=2):
    """
    Skeletonizes a binary mask of a river channel network. Differs from
    skeletonize mask above by using knowledge of the exit sides of the river
    with respect to the mask (I) to avoid edge effects of skeletonization by
    mirroring the mask at its ends, then trimming it after processing. As with
    skeletonize_mask, skeleton simplification is performed.

    Parameters
    ----------
    I : np.array
        Binary river mask to skeletonize.
    es : str
        A two-character string (from N, E, S, or W) that denotes which sides
        of the image the river intersects (upstream first) -- e.g. 'NS', 'EW',
        'NW', etc.
    padscale : int, optional
        Pad multiplier that sets the size of the padding. Multplies the blob
        size along the axis of the image that the blob intersect to determine
        the padding distance. The default is 2.

    Returns
    -------
    Iskel : np.array
        The skeletonization of I.

    """
    # Crop image
    Ic, crop_pads = imu.crop_binary_im(I)

    # Pad image (reflects channels at image edges)
    Ip, pads = pad_river_im(Ic, es, pm=padscale)

    # Skeletonize padded image
    Iskel = morphology.skeletonize(Ip)

    # Remove padding
    Iskel = Iskel[pads[0]:Iskel.shape[0]-pads[1], pads[3]:Iskel.shape[1]-pads[2]]
    # Add back what was cropped so skeleton image is original size
    crop_pads_add = ((crop_pads[1], crop_pads[3]),(crop_pads[0], crop_pads[2]))
    Iskel = np.pad(Iskel, crop_pads_add, mode='constant', constant_values=(0,0))

    # Ensure skeleton is prepared for analysis by RivGraph
    # Simplify the skeleton (i.e. remove pixels that don't alter connectivity)
    Iskel = simplify_skel(Iskel)

    # Fill small skeleton holes, re-skeletonize, and re-simplify
    Iskel = imu.fill_holes(Iskel, maxholesize=4)
    Iskel = morphology.skeletonize(Iskel)
    Iskel = simplify_skel(Iskel)

    # Fill single pixel holes
    Iskel = imu.fill_holes(Iskel, maxholesize=1)

    # The handling of edges can leave pieces of the skeleton stranded (i.e.
    # disconnected from the main skeleton). Remove those here by keeping the
    # largest blob.
    Iskel = imu.largest_blobs(Iskel, nlargest=1, action='keep')

    return Iskel


def simplify_skel(Iskel):
    """
    This function iterates through all skeleton pixels pixels that have
    connectivity > 2. It removes the pixel and checks if the
    number of blobs has changed after removal. If so, the pixel is necessary to
    maintain connectivity. Otherwise the pixel is not retained. It also adds
    pixels to the centers of "+" cases, as this reduces the number
    of branchpoints from 4 to 1.

    Parameters
    ----------
    Iskel : np.array
        Image of the skeleton to simplify.

    Returns
    -------
    Iskel : np.array
        The simplified skeleton.

    """
    Iskel = np.array(Iskel, dtype=np.uint8)
    Ic = imu.im_connectivity(Iskel)
    ypix, xpix = np.where(Ic > 2) # Get all pixels with connectivity > 2

    for y, x in zip(ypix, xpix):
        nv = imu.neighbor_vals(Iskel, x, y)

        # Skip edge cases
        if np.any(np.isnan(nv)) == True:
            continue

        # Create 3x3 array with center pixel removed
        Inv = np.array([[nv[0], nv[1], nv[2]], [nv[3], 0, nv[4]], [nv[5], nv[6], nv[7]]], dtype=np.bool)


        # Check the connectivity after removing the pixel, set to zero if unchanged
        Ilabeled = measure.label(Inv, background=0, connectivity=2)
        if np.max(Ilabeled) == 1:
            Iskel[y,x] = 0

    # We simplify the network if we actually add pixels to the centers of
    # "+" cases, so let's do that.
    kern = np.array([[1, 10, 1], [10, 1, 10], [1, 10, 1]], dtype=np.uint8)
    Iconv = cv2.filter2D(Iskel, -1, kern)
    Iskel[Iconv==40] = 1

    return Iskel


def pad_river_im(I, es, pm=2):
    """
    Pads the edges of a binary river image by extending the ends of the river.
    Different than mirrored padding in that the "mirror" here is just a
    rectangle. This simplifies skeletonization and interpretation in cases
    where the channel is complex near the boundaries.

    Parameters
    ----------
    I : np.array
        Binary image to pad; typically a river channel network.
    es : str
        A two-character string (from N, E, S, or W) that denotes which sides
        of the image the river intersects (upstream first) -- e.g. 'NS', 'EW',
        'NW', etc.
    pm : int, optional
        Pad multiplier that sets the size of the padding. Multplies the blob
        size along the axis of the image that the blob intersect to determine
        the padding distance. The default is 2.

    Returns
    -------
    Ip : np.array
        The padded image..
    pads : list
        4 entry list containing the number of pixels padded on the [n, s, e, w]
        edges of the image.

    """
    Ip = I.copy() # so original array is not modified
    pads = [0, 0, 0, 0] # saves the number of pixels padded to each [n,s,e,w] edge

    if 'n' in es.lower():
        rowidcs = np.where(Ip[0,:]==True)[0]
        st = np.min(rowidcs)
        en = np.max(rowidcs)
        pads[0] = (en-st) * pm

        # Make pad
        addpad = np.zeros((pads[0], Ip.shape[1]), dtype=np.bool)
        addpad[:,rowidcs] = True

        # Add pad to image
        Ip = np.concatenate((addpad, Ip))

    if 's' in es.lower():

        rowidcs = np.where(Ip[-1,:]==True)[0]
        st = np.min(rowidcs)
        en = np.max(rowidcs)
        pads[1] = (en-st) * pm

        # Make pad
        addpad = np.zeros((pads[1], Ip.shape[1]), dtype=np.bool)
        addpad[:,rowidcs] = True

        # Add pad to image
        Ip = np.concatenate((Ip, addpad))

    if 'e' in es.lower():

        colidcs = np.where(Ip[:,-1]==True)[0]
        st = np.min(colidcs)
        en = np.max(colidcs)
        pads[2] = (en-st) * pm

        # Make pad
        addpad = np.zeros((Ip.shape[0], pads[2]), dtype=np.bool)
        addpad[colidcs,:] = True
        Ip = np.concatenate((Ip, addpad), axis=1)

    if 'w' in es.lower():

        colidcs = np.where(Ip[:,0]==True)[0]
        st = np.min(colidcs)
        en = np.max(colidcs)
        pads[3] = (en-st) * pm

        # Make pad
        addpad = np.zeros((Ip.shape[0], pads[3]), dtype=np.bool)
        addpad[colidcs,:] = True
        Ip = np.concatenate((addpad, Ip), axis=1)

    return Ip, pads
