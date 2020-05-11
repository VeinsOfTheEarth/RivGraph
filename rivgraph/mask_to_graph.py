# -*- coding: utf-8 -*-
"""
mask_to_graph
=============

Created on Tue Apr 10 14:29:01 2018

@author: Jon
"""


from rivgraph import walk
from rivgraph import ln_utils as lnu
from rivgraph import im_utils as iu
from skimage import morphology, measure
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from ordered_set import OrderedSet

#Iskel = rg.Iskel

def skel_to_graph(Iskel):
    """
    Breaks a skeletonized image into links and nodes; exports if desired.
    """

    def check_startpoint(spidx, Iskel):
        """
        Returns True if a skeleton pixel's first neighbor is not a branchpoint (i.e.
        the start pixel is valid), else returns False.
        """

        neighs = walk.walkable_neighbors([spidx], Iskel)
        isbp = walk.is_bp(neighs.pop(), Iskel)

        if isbp == 0:
            return True
        else:
            return False


    def find_starting_pixels(Iskel):
        """
        Finds an endpoint pixel to begin network resolution
        """

        # Get skeleton connectivity
        eps = iu.skel_endpoints(Iskel)
        eps = set(np.ravel_multi_index(eps, Iskel.shape))

        # Get one endpoint per connected component in network
        rp = iu.regionprops(Iskel, ['coords'])
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
    nodes['conn'] = [] #[[] for i in range(3)]

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
                    # Else, fuck. You've found a critical flaw in the algorithm.
                    else:
                        print('idx: {}, poss_steps: {}'.format(links['idx'][linkidx][-1], poss_steps))
                        raise RuntimeError('Ambiguous which step to take next :(')

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
    # but since nodes have unique idx it was unnecessary. IDs may be required
    # for further processing, though.
    nodes['id'] = OrderedSet(range(0,len(nodes['idx'])))

    # Remove duplicate links if they exist; for some single-pixel links,
    # duplicates are formed. Ideally the walking code should ensure that this
    # doesn't happen, but for now removing duplicates suffices.
    links, nodes = lnu.remove_duplicate_links(links, nodes)

    return links, nodes


def skeletonize_mask(Imask):
    """
    Skeletonize any input binary mask.
    """

    # Create copy of mask to skeletonize
    Iskel = np.array(Imask, copy=True, dtype='bool')

    # Perform skeletonization
    Iskel = morphology.skeletonize(Iskel)

    # Simplify the skeleton (i.e. remove pixels that don't alter connectivity)
    Iskel = simplify_skel(Iskel)

    # Fill small skeleton holes, re-skeletonize, and re-simplify
    Iskel = iu.fill_holes(Iskel, maxholesize=4)
    Iskel = morphology.skeletonize(Iskel)
    Iskel = simplify_skel(Iskel)

    # Fill single pixel holes
    Iskel = iu.fill_holes(Iskel, maxholesize=1)

    return Iskel


def skeletonize_river_mask(I, es, padscale=2):
    """
    Skeletonize a binary river mask. Crops mask to river extents, reflects the
    mask at the appropriate borders, skeletonizes, then un-crops the mask.

    INPUTS:
        I - binary river mask to skeletonize
       es - NSEW "exit sides" corresponding to the upstream and downstream
             sides of the image that intersect the river. e.g. 'NS', 'EN', 'WS'
 padscale - (Optional) number of pixels to reflect I before skeletonization
               to remove edge effects of skeletonization
    """

    # Crop image
    Ic, crop_pads = iu.crop_binary_im(I)

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
    Iskel = iu.fill_holes(Iskel, maxholesize=4)
    Iskel = morphology.skeletonize(Iskel)
    Iskel = simplify_skel(Iskel)

    # Fill single pixel holes
    Iskel = iu.fill_holes(Iskel, maxholesize=1)

    # The handling of edges can leave pieces of the skeleton stranded (i.e.
    # disconnected from the main skeleton). Remove those here by keeping the
    # largest blob.
    Iskel = iu.largest_blobs(Iskel, nlargest=1, action='keep')

    return Iskel



def simplify_skel(Iskel):
    """
    This function iterates through all skeleton pixels
    pixels that have connectivity > 2. It removes the pixel and checks if the
    number of blobs has changed after removal. If so, the pixel is necessary to
    maintain connectivity. Otherwise the pixel is not retained.
    It also adds pixels to the centers of "+" cases, as this reduces the number
    of branchpoints from 4 to 1.
    """

    Iskel = np.array(Iskel, dtype=np.uint8)

    Ic = iu.im_connectivity(Iskel)
    ypix, xpix = np.where(Ic > 2) # Get all pixels with connectivity > 2


    for y, x in zip(ypix, xpix):
        nv = iu.neighbor_vals(Iskel, x, y)

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

    INPUTS:
        I - binary river mask to pad
        es - NSEW "exit sides" corresponding to the upstream and downstream
             sides of the image that intersect the river. e.g. 'NS', 'EN', 'WS'
        pm - (Optional) "pad multiplier" that sets the size of the padding.
    OUTPUTS:
        Ip - padded version of I
        pads - [n,s,e,w] padding values (number of pixels)
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
