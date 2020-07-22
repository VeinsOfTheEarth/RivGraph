# -*- coding: utf-8 -*-
"""
im_utils
========

Created on Mon Sep 10 10:21:35 2018

@author: Jon
"""
import cv2
import numpy as np
from scipy import ndimage as nd
import geopandas as gpd
from shapely.geometry import Polygon
from skimage import morphology, measure, util
import rivgraph.geo_utils as gu


def get_array(idx, I, size):

    """
    Size is (nrows, ncols) corresponding to the number of rows, columns to return with
    idx at the center. idx can be flattend index or [row, col]

    ## Should add check for border cases so we don't query to raster
    ## beyond its bounds

    """

    dims = I.shape

    try:
        lidx = len(idx)
    except:
        lidx = len([idx])

    if lidx == 1:
        row, col = np.unravel_index(idx, dims)

    else:
        row = idx[0]
        col = idx[1]

    # Get row, column of top-left most pixel in array
    row = int(row - (size[0]-1)/2)
    col = int(col - (size[1]-1)/2)

    array = I[row:row+size[0], col:col+size[1]].copy()

    return array, row, col


def neighbors_flat(idx, imflat, ncols, filt='nonzero'):
    '''
    Given a flattened image (np.ravel) and an index, returns all the
    neighboring indices and their values. Uses native python datatypes.
    Set filt='nonzero' for returning only nonzero indices; otherwise set
    filt='none' to return all indices and values.
    '''

    if isinstance(idx, np.generic):
        idx = idx.item()
    if isinstance(ncols, np.generic):
        idx = idx.item()

    dy = ncols
    dx = 1

    possneighs = np.array([idx-dy-dx,
                  idx-dy,
                  idx-dy+dx,
                  idx-dx,
                  idx+dx,
                  idx+dy-dx,
                  idx+dy,
                  idx+dy+dx])

    # For handling edge cases
    if idx % ncols == 0: # first column of image

        if idx < ncols: # top row of image
            pullvals = [4,6,7]
        elif idx >= len(imflat) - ncols: # bottom row of image
            pullvals = [1,2,4]
        else:
            pullvals = [1,2,4,6,7]

    elif (idx + 1) % ncols == 0: # last column of image

        if idx < ncols: # top row
            pullvals = [3,5,6]
        elif idx >= len(imflat) - ncols: # bottom row of image
            pullvals = [0,1,3]
        else:
            pullvals = [0,1,3,5,6]

    elif idx < ncols:
        pullvals = [3,4,5,6,7]

    elif idx >= len(imflat) - ncols: # bottom row of image
        pullvals = [0,1,2,3,4]

    else: # Regular cases (non-edges)
        pullvals = [0,1,2,3,4,5,6,7]

    idcs = possneighs[pullvals]
    vals = imflat[idcs]

    if filt == 'nonzero':
        keepidcs = vals != 0
        idcs = idcs[keepidcs]
        vals = vals[keepidcs]

    return idcs, vals


def reglobalize_flat_idx(idxlist, idxlistdims, row_offset, col_offset, globaldims):
    """
    If idxlist is (x,y), then idxlistdims should be (dim_x, dim_y), etc.
    """
    convertflag = 0

    if type(idxlist) is int:
        idxlist = [idxlist]
        convertflag = 'int'

    if type(idxlist) is set:
        idxlist = list(idxlist)
        convertflag = 'set'

    idcsrowcol = np.unravel_index(idxlist, idxlistdims)
    rows = idcsrowcol[0] + row_offset
    cols = idcsrowcol[1] + col_offset
    idcsflat = list(np.ravel_multi_index((rows, cols), globaldims))

    if convertflag == 'int':
        idcsflat = int(idcsflat[0])
    elif convertflag == 'set':
        idcsflat = set(idcsflat)

    return idcsflat


def nfour_connectivity(I):

    """
    Returns an image of four-connectivity for each pixel, where the pixel value
    is the number of 4-connected neighbors.
    """

    Ir = np.ravel(I)
    edgeidcs = edge_coords(I.shape, dtype='flat')

    allpix = set(np.where(Ir==1)[0])

    dopix = allpix - edgeidcs
    savepix = list(dopix)

    fourconnidcs = four_conn(list(dopix), I)
    n_fourconn = [len(fc) for fc in fourconnidcs]

    Infc = np.zeros_like(Ir, dtype=np.uint8)
    Infc[savepix] = n_fourconn
    Infc = np.reshape(Infc, I.shape)

    return Infc


def four_conn(idcs, I):
    """
    Counts the number of 4-connected neighbors for a given flat index in I.
    idcs must be a list, even if a single value.
    """

    Iflat = np.ravel(np.array(I, dtype=np.bool))

    fourconn = []
    for i in idcs:
        neigh_idcs = neighbors_flat(i, Iflat, I.shape[1])[0]
        ni_check = abs(neigh_idcs - i)
        fourconn.append([n for i, n in enumerate(neigh_idcs) if ni_check[i] == 1 or ni_check[i]==I.shape[1]])

    return fourconn


def edge_coords(sizeI, dtype='flat'):
    """
    Given an image size, returns the coordinates of all the edge pixels (4 edges).
    Can return as indices (dtype=flat) or x,y coordinates (dtyep='xy').
    """
    # xs
    x_l = np.zeros(sizeI[0], dtype=np.int64)
    x_b = np.arange(0,sizeI[1], dtype=np.int64)
    x_r = x_l + sizeI[1] - 1
    x_t = np.flip(x_b, axis=0)

    # ys
    y_l = np.arange(0, sizeI[0], dtype=np.int64)
    y_r = np.flip(y_l, axis=0)
    y_t = np.zeros(sizeI[1], dtype=np.int64)
    y_b = y_t + sizeI[0] - 1

    edgeptx = np.concatenate([x_l, x_b, x_r, x_t])
    edgepty = np.concatenate([y_l, y_b, y_r, y_t])
    if dtype == 'flat':
        edgepts = set(np.ravel_multi_index((edgepty, edgeptx), sizeI))
    elif dtype == 'xy':
        edgepts = [edgeptx, edgepty]

    return edgepts


def neighbor_idcs(x, y):
    """
    Input x,y coordinates and return all the neighbor indices.
    """
    xidcs = [x-1, x, x+1, x-1, x+1, x-1, x, x+1]
    yidcs = [y-1, y-1, y-1, y, y, y+1, y+1, y+1]

    return xidcs, yidcs


def neighbor_vals(im, x, y):

    vals = np.empty((8,1))
    vals[:] = np.NaN

    if x == 0:

        if y == 0:
            vals[4] = im[y,x+1]
            vals[6] = im[y+1,x]
            vals[7] = im[y+1,x+1]
        elif y == np.shape(im)[0]-1:
            vals[1] = im[y-1,x]
            vals[2] = im[y-1,x+1]
            vals[4] = im[y,x+1]
        else:
            vals[1] = im[y-1,x]
            vals[2] = im[y-1,x+1]
            vals[4] = im[y,x+1]
            vals[6] = im[y+1,x]
            vals[7] = im[y+1,x+1]

    elif x == np.shape(im)[1]-1:

        if y == 0:
            vals[3] = im[y,x-1]
            vals[5] = im[y+1,x-1]
            vals[6] = im[y+1,x]
        elif y == np.shape(im)[0]-1:
            vals[0] = im[y-1,x-1]
            vals[1] = im[y-1,x]
            vals[3] = im[y,x-1]
        else:
            vals[0] = im[y-1,x-1]
            vals[1] = im[y-1,x]
            vals[3] = im[y,x-1]
            vals[5] = im[y+1,x-1]
            vals[6] = im[y+1,x]

    elif y == 0:
        vals[3] = im[y,x-1]
        vals[4] = im[y,x+1]
        vals[5] = im[y+1,x-1]
        vals[6] = im[y+1,x]
        vals[7] = im[y+1,x+1]

    elif y == np.shape(im)[0]-1:
        vals[0] = im[y-1,x-1]
        vals[1] = im[y-1,x]
        vals[2] = im[y-1,x+1]
        vals[3] = im[y,x-1]
        vals[4] = im[y,x+1]

    else:
        vals[0] = im[y-1,x-1]
        vals[1] = im[y-1,x]
        vals[2] = im[y-1,x+1]
        vals[3] = im[y,x-1]
        vals[4] = im[y,x+1]
        vals[5] = im[y+1,x-1]
        vals[6] = im[y+1,x]
        vals[7] = im[y+1,x+1]

    return np.ndarray.flatten(vals)


def neighbor_xy(x, y, idx):
    # Feed in x, y location and a neighbor index
    # Return the x, y location of the neighbor
    xs = np.array([-1, 0, 1, -1, 1 ,-1, 0, 1], dtype=np.int)
    ys = np.array([-1, -1, -1, 0, 0 ,1, 1, 1], dtype=np.int)

    x = x + xs[idx]
    y = y + ys[idx]

    return x, y


def remove_blobs(I, blobthresh, connectivity=2):
    """
    Returns a binary image with blobs less than size blobthresh removed from
    the input binary image.
    """

    props = ['area', 'coords']
    rp, _ = regionprops(I, props, connectivity=connectivity)
    areas = np.array(rp['area'])
    coords = rp['coords']

    remove_idcs = np.where(areas < blobthresh)[0]

    Ic =  np.copy(I)
    for r in remove_idcs:
        Ic[coords[r][:,0],coords[r][:,1]] = False

    return Ic


def imshowpair(I1, I2):

    from matplotlib import colors
    from matplotlib import pyplot as plt
    Ip  = np.zeros(np.shape(I1),dtype='uint8')
    Ip[I1>0] = 2
    Ip[I2>0] = 3
    Ip[np.bitwise_and(I1, I2)==True] = 1

    cmap = colors.ListedColormap(['black','white', 'magenta', 'lime'])
    bounds=[-1,0.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    plt.imshow(Ip, origin='upper', cmap=cmap, norm=norm)

#    img = plt.imshow(Ip, origin='upper',
#                        cmap=cmap, norm=norm)
#    # make a color bar
#    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 5, 10])


#def fill_islands(I, min_n_island_pixels=20):
#
#    # Invert binary image
#    Icomp = util.invert(I)
#
#    # Compute region props
#    props = ['coords','area']
#    rp = regionprops(Icomp, props, connectivity=1)
#    for a, c in zip(rp['area'], rp['coords']):
#        if a < min_n_island_pixels:
#            I[c[:,0], c[:,1]] = True
#    return I


def largest_blobs(I, nlargest=1, action='remove', connectivity=2):
    """
    Returns a binary image with the nlargest blobs removed from the input
    binary image.
    """
    props = ['area', 'coords']
    rp, _ = regionprops(I, props, connectivity=connectivity)
    areas = np.array(rp['area'])
    coords = rp['coords']
    # Sorts the areas array and keeps the nlargest indices
    maxidcs = areas.argsort()[-nlargest:][::-1]

    if action == 'remove':
        Ic =  np.copy(I)
        for m in maxidcs:
            Ic[coords[m][:,0],coords[m][:,1]] = False
    elif action == 'keep':
        Ic = np.zeros_like(I)
        for m in maxidcs:
            Ic[coords[m][:,0],coords[m][:,1]] = True
    else:
        print('Improper action specified: either choose remove or keep')
        Ic = I

    return Ic


def blob_idcs(I, connectivity=2):
    """
    Returns a list where each entry contains a set of all indices within each
    connected blob. Indices are returned as single-index coordinates, rather
    than x,y.
    """
    props = ['coords']
    rp, _ = regionprops(I, props, connectivity=connectivity)
    coords = rp['coords']
    idcs = []
    for c in coords:
        idcs.append(set(np.ravel_multi_index([c[:,0], c[:,1]], I.shape)))

    return idcs


def regionprops(I, props, connectivity=2):
    
    ### TODO: Add a check that appropriate props are requested

    Ilabeled = measure.label(I, background=0, connectivity=connectivity)
    properties = measure.regionprops(Ilabeled, intensity_image=I)

    out = {}
    # Get the coordinates of each blob in case we need them later
    if 'coords' in props or 'perimeter' in props:
        coords = [p.coords for p in properties]

    for prop in props:
        if prop == 'area':
            allprop = [p.area for p in properties]
        elif prop == 'coords':
            allprop = coords
        elif prop == 'centroid':
            allprop = [p.centroid for p in properties]
        elif prop == 'mean':
            allprop = [p.mean_intensity for p in properties]
        elif prop == 'perim_len':
            allprop = [p.perimeter for p in properties]
        elif prop == 'perimeter':
            perim = []
            for blob in coords:
                # Crop to blob to reduce cv2 computation time and save memory
                Ip, pads = crop_binary_coords(blob, npad=1)
                Ip = np.array(Ip, dtype='uint8')

                contours, _ = cv2.findContours(Ip, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # IMPORTANT: findContours returns points as (x,y) rather than (row, col)
                contours = contours[0]
                crows = []
                ccols = []
                for c in contours:
                    crows.append(c[0][1] + pads[1]) # must add back the cropped rows and columns
                    ccols.append(c[0][0] + pads[0])
                cont_np = np.transpose(np.array((crows,ccols))) # format the output
                perim.append(cont_np)
            allprop = perim
        elif prop == 'convex_area':
            allprop = [p.convex_area for p in properties]
        elif prop == 'eccentricity':
            allprop = [p.eccentricity for p in properties]
        elif prop == 'equivalent_diameter':
            allprop = [p.equivalent_diameter for p in properties]
        elif prop == 'major_axis_length':
            allprop = [p.major_axis_length for p in properties]
        elif prop == 'minor_axis_length':
            allprop = [p.minor_axis_length for p in properties]
        elif prop == 'label':
            allprop = [p.label for p in properties]
        else:
            print('{} is not a valid property.'.format(prop))

        out[prop] = np.array(allprop)

    return out, Ilabeled


def erode(I, n=1, strel='square'):

    if n == 0:
        return I

    if strel == 'square':
        selem = morphology.square(3)
    elif strel == 'plus':
        selem = morphology.diamond(1)
    elif strel == 'disk':
        selem = morphology.disk(3)

    for i in np.arange(0,n):
        I = morphology.erosion(I, selem)

    return I


def dilate(I, n=1, strel='square'):

    if n == 0:
        return I

    if strel == 'square':
        selem = morphology.square(3)
    elif strel == 'plus':
        selem = morphology.diamond(1)
    elif strel == 'disk':
        selem = morphology.disk(3)

    for i in np.arange(0,n):

        I = morphology.dilation(I, selem)

    return I



def trim_idcs(imshape, idcs):
    """
    Trims a list of x,y indices by removing rows containing indices that cannot
    fit within a raster of imshape
    """

    idcs = idcs[idcs[:, 0] < imshape[0], :]
    idcs = idcs[idcs[:, 1] < imshape[1], :]
    idcs = idcs[idcs[:, 0] >= 0, :]
    idcs = idcs[idcs[:, 1] >= 0, :]

    return idcs


def crop_binary_im(I, connectivity=2):
    """
    Crops a binary image to the smallest bounding box containing all the blobs
    in the image.
    """

    coords = np.where(I==1)
    uly = np.min(coords[0])
    ulx = np.min(coords[1])
    lry = np.max(coords[0]) + 1
    lrx = np.max(coords[1]) + 1

    Icrop = I[uly:lry, ulx:lrx]
    pads = [ulx, uly, I.shape[1]-lrx, I.shape[0] - lry]

    return Icrop, pads


def crop_binary_coords(coords, npad=0):

    # Coords are of format [row, col]

    uly = np.min(coords[:,0]) - npad
    ulx = np.min(coords[:,1]) - npad
    lry = np.max(coords[:,0]) + npad
    lrx = np.max(coords[:,1]) + npad

    I = np.zeros((lry-uly+1,lrx-ulx+1))
    I[coords[:,0]-uly,coords[:,1]-ulx] = True

    pads = [ulx, uly, lrx, lry]
    return I, pads


def fill_holes(I, maxholesize=0):

    I = np.array(I, dtype=np.bool)

    if maxholesize == 0:
        I = nd.morphology.binary_fill_holes(I)
        return I
    else:
        # Fill only holes less than maxholesize
        Icomp = util.invert(I)

        # Remove boundary pixels so holes created by image boundary are not considered blobs
        Icomp[:,0] = 0
        Icomp[:,-1] = 0
        Icomp[0,:] = 0
        Icomp[-1,:]= 0

        # Get blob properties of complement image
        props = ['coords','area']
        rp, _ = regionprops(Icomp, props, connectivity=1)

        # Blob indices less than specified threshold
        keepidcs = [i for i, x in enumerate(rp['area']) if x <= maxholesize]

        # Fill 'dem holes!
        for k in keepidcs:
            I[rp['coords'][k][:,0],rp['coords'][k][:,1]] = 1

        return I


def im_connectivity(im):
    """
    Returns an image of 8-connectivity for an input image of all pixels in a
    binary image.
    """
    # Fix input
    im = im.copy()
    im[im!=0] = 1
    im = np.uint8(im)

    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])

    src_depth = -1
    filtered = cv2.filter2D(im, src_depth, kernel)
    Iret = np.zeros(np.shape(filtered), dtype ='uint8')
    Iret[filtered>10] = filtered[filtered>10]-10
    return Iret


def downsample_binary_image(I, newsize, thresh=0.05):
    """
    Given an input binary image and a new size, this downsamples (i.e. reduces
    resolution) the input image to the new size. A pixel is considered "on"
    in the new image if 'thresh' fraction of its area is covered by the
    higher-res image. E.g. set 'thresh' to zero if you want the output image
    to be "on" everywhere at least a single pixel is "on" in the original
    image.
    """

    # Get locations of all smaller pixels
    row,col = np.where(I>0)

    # Get the scaling factor in both directions
    rowfact = newsize[0]/I.shape[0]
    colfact = newsize[1]/I.shape[1]

    # Scale the row,col coordinates and turn them into integers
    rowcol = np.vstack((np.array(row * rowfact, dtype=np.uint16),np.array(col * colfact, dtype=np.uint16)))

    # Get the number of smaller pixels within each larger pixel
    rc_unique, rc_counts = np.unique(rowcol, axis=1, return_counts=True)

    # Filter out the large-pixel coordinates that don't contain enough area
    area_ratio = rowfact * colfact
    area_fracs = rc_counts * area_ratio
    rc_unique = rc_unique[:,np.where(area_fracs>=thresh)[0]]

    # Create the downsampled image
    Iout = np.zeros(newsize)
    Iout[rc_unique[0,:], rc_unique[1,:]] = 1

    return Iout


def skel_endpoints(skel):
    '''
    For a given input skeleton image/array, returns the x,y coordinates of
    the endpoints (eps).
    '''

    Ic = im_connectivity(skel)
    eps = np.where(Ic==1)

    return eps


def skel_branchpoints(Iskel):
    '''
    For a given input skeleton image/array, returns the x,y coordinates of the
    branchpoints.
    '''
    Ibps = np.uint16(im_connectivity(Iskel))

    # Initial branchpoints are defined by pixels with conn > 2
    Ibps[Ibps<3] = 0
    Ibps[Ibps>0] = 1

#    Ibps_O = np.copy(Ibps)

    # Filter branchpoints using convolution kernel that results in a unique
    # value for each possible configuration

    # Create kernel
    kern = np.array([[256, 32, 4], [128, 16, 2], [64, 8, 1]], dtype=np.uint16)

    # Patterns whose center branchpoints we want to remove
    basepat = []
    basepat.append(np.array([ [0, 0, 0], [0, 1, 0], [1, 1, 0] ])) # 3.1
    basepat.append(np.array([ [1, 0, 0], [0, 1, 0], [1, 1, 0] ])) # 4.1
    basepat.append(np.array([ [0, 1, 0], [0, 1, 0], [1, 1, 0] ])) # 4.2
    basepat.append(np.array([ [0, 0, 1], [0, 1, 0], [1, 1, 0] ])) # 4.3
#    basepat.append(np.array([ [0, 0, 0], [0, 1, 1], [1, 1, 0] ])) # 4.4
    basepat.append(np.array([ [0, 0, 0], [0, 1, 0], [1, 1, 1] ])) # 4.5
#    basepat.append(np.array([ [0, 1, 0], [1, 1, 1], [0, 0, 0] ])) # 4.6
    basepat.append(np.array([ [0, 0, 1], [1, 1, 1], [1, 0, 0] ])) # 5.1
    basepat.append(np.array([ [1, 0, 1], [1, 1, 1], [0, 0, 0] ])) # 5.2
    basepat.append(np.array([ [0, 1, 0], [0, 1, 0], [1, 1, 1] ])) # 5.3

    rmvals = set()
    for bp in basepat:
        for i in range(0,4):
            rmvals.update([int(np.sum(kern[bp==1]))])
            bp = np.rot90(bp, 1)
        bp = np.flipud(bp)
        for i in range(0,4):
            rmvals.update([int(np.sum(kern[bp==1]))])
            bp = np.rot90(bp, 1)
        bp = np.fliplr(bp)
        for i in range(0,4):
            rmvals.update([int(np.sum(kern[bp==1]))])
            bp = np.rot90(bp, 1)

    # Add three more cases where a 2x2 block of branchpoints exists. In these
    # cases, we choose the top-right one to be the only branchpoint.
    rmvals.update([27, 54, 432, 283, 433, 118])

    # Convolve
    src_depth = -1
    Iconv = cv2.filter2D(Ibps, src_depth, kern)

    # Remove unwanted branchpoints based on patterns
    Irm_flat = np.in1d(Iconv, list(rmvals))
    rmy, rmx = np.unravel_index(np.where(Irm_flat==1), Iconv.shape)
    Ibps[rmy, rmx] = 0

    # Filter remaining branchpoints: cases where there are two 4-connected branchpoints
    # Count the neighbors, remove the one with fewer neighbors. Neighbors are
    # counted distinctly, i.e. none are shared between the two branchpoints when counting.
    # Find all the cases
    rp, _ = regionprops(Ibps, ['area','coords'], connectivity=1)
    idcs = np.where(rp['area']==2)[0]
    for idx in idcs:
        c = rp['coords'][idx] # coordinates of both 4-connected branchpoint pixels

        if c[0,0] == c[1,0]: # Left-right orientation
            if c[0,1] < c[1,1]: # First pixel is left-most
                lneigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                lneighsum = lneigh[0] + lneigh[1] + lneigh[3] + lneigh[5] + lneigh[6]
                rneigh = neighbor_vals(Iskel, c[1,1], c[1,0])
                rneighsum = rneigh[1] + rneigh[2] + rneigh[4] + rneigh[6] + rneigh[7]
                if lneighsum > rneighsum:
                    Ibps[c[1,0], c[1,1]] = 0
                elif rneighsum > lneighsum:
                    Ibps[c[0,0], c[0,1]] = 0
            else: # Second pixel is left-most
                lneigh = neighbor_vals(Iskel, c[1,1], c[1,0])
                lneighsum = lneigh[0] + lneigh[1] + lneigh[3] + lneigh[5] + lneigh[6]
                rneigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                rneighsum = rneigh[1] + rneigh[2] + rneigh[4] + rneigh[6] + rneigh[7]
                if lneighsum > rneighsum:
                    Ibps[c[0,0], c[0,1]] = 0
                elif rneighsum > lneighsum:
                    Ibps[c[1,0], c[1,1]] = 0

        else: # Up-down orientation
            if c[0,0] < c[1,0]: # First pixel is up-most
                uneigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                uneighsum = uneigh[0] + uneigh[1] + uneigh[2] + uneigh[3] + uneigh[5]
                dneigh = neighbor_vals(Iskel, c[1,1], c[1,0])
                dneighsum = dneigh[3] + dneigh[4] + dneigh[5] + dneigh[6] + dneigh[7]
                if uneighsum > dneighsum:
                    Ibps[c[1,0], c[1,1]] = 0
                elif dneighsum > uneighsum:
                    Ibps[c[0,0], c[0,1]] = 0
            else: # Second pixel is up-most
                uneigh = neighbor_vals(Iskel, c[1,1], c[1,0])
                uneighsum = uneigh[0] + uneigh[1] + uneigh[2] + uneigh[3] + uneigh[5]
                dneigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                dneighsum = dneigh[3] + dneigh[4] + dneigh[5] + dneigh[6] + dneigh[7]
                if uneighsum > dneighsum:
                    Ibps[c[0,0], c[0,1]] = 0
                elif dneighsum > uneighsum:
                    Ibps[c[1,0], c[1,1]] = 0

    # Now handle the cases where there are three branchpoints in a row/column
    idcs = np.where(rp['area']>2)[0]
    for idx in idcs:
        c = rp['coords'][idx]

        if np.sum(np.abs(np.diff(c[:,0]))) == c.shape[0]-1: # Vertical
            if c[0,0] < c[1,0]: # First pixel is up-most
                # Check if end pixels can be removed
                e1_neigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                e1_sum = e1_neigh[0] + e1_neigh[1] + e1_neigh[2] + e1_neigh[3] + e1_neigh[4]
                if e1_sum < 2:
                    Ibps[c[0,0],c[0,1]] = 0
                e2_neigh = neighbor_vals(Iskel, c[-1,1], c[-1,0])
                e2_sum = e2_neigh[3] + e2_neigh[4] + e2_neigh[5] + e2_neigh[6] + e2_neigh[7]
                if e2_sum < 2:
                    Ibps[c[-1,0],c[-1,0]] = 0
            else: # First pixel is right-most
                e1_neigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                e1_sum = e1_neigh[3] + e1_neigh[4] + e1_neigh[5] + e1_neigh[6] + e1_neigh[7]
                if e1_sum < 2:
                    Ibps[c[0,0],c[0,1]] = 0
                e2_neigh = neighbor_vals(Iskel, c[-1,1], c[-1,0])
                e2_sum = e2_neigh[3] + e2_neigh[4] + e2_neigh[5] + e2_neigh[6] + e2_neigh[7]
                if e2_sum < 2:
                    Ibps[c[-1,0],c[-1,0]] = 0
            # Check if middle pixels can be removed
            for p in c[1:-1]:
                pneigh = neighbor_vals(Iskel, p[1], p[0])
                psum = pneigh[3] + pneigh[4]
                if psum == 0:
                    Ibps[p[0], p[1]] = 0

        elif np.sum(np.abs(np.diff(c[:,1]))) == c.shape[0]-1: # Horizontal
            if c[0,1] < c[-1,1]: # First pixel is left-most
                # Check if end pixels can be removed
                e1_neigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                e1_sum = e1_neigh[0] + e1_neigh[1] + e1_neigh[3] + e1_neigh[5] + e1_neigh[6]
                if e1_sum < 2:
                    Ibps[c[0,0],c[0,1]] = 0
                e2_neigh = neighbor_vals(Iskel, c[-1,1], c[-1,0])
                e2_sum = e2_neigh[1] + e2_neigh[2] + e2_neigh[4] + e2_neigh[6] + e2_neigh[7]
                if e2_sum < 2:
                    Ibps[c[-1,0],c[-1,0]] = 0
            else: # First pixel is right-most
                e1_neigh = neighbor_vals(Iskel, c[0,1], c[0,0])
                e1_sum = e1_neigh[1] + e1_neigh[2] + e1_neigh[4] + e1_neigh[6] + e1_neigh[7]
                if e1_sum < 2:
                    Ibps[c[0,0],c[0,1]] = 0
                e2_neigh = neighbor_vals(Iskel, c[-1,1], c[-1,0])
                e2_sum = e2_neigh[0] + e2_neigh[1] + e2_neigh[3] + e2_neigh[5] + e2_neigh[6]
                if e2_sum < 2:
                    Ibps[c[-1,0],c[-1,0]] = 0
            # Check if middle pixels can be removed
            for p in c[1:-1]:
                pneigh = neighbor_vals(Iskel, p[1], p[0])
                psum = pneigh[1] + pneigh[6]
                if psum == 0:
                    Ibps[p[0], p[1]] = 0

#    # This dangerously assumes that coordinates are returned in order...was the case for all tests run. If there are problems, revisit this!
#    idcs = np.where(rp['area']==3)[0]
#    for idx in idcs:
#        c = rp['coords'][idx] # coordinates of both 4-connected branchpoint pixels
#        Ibps[c[1,0], c[1,1]] = 0
#
#    # Finally, handle the cases where there are five branchpoints in a row/column,
#    # same method as for three in a row/column
#    idcs = np.where(rp['area']==5)[0]
#    for idx in idcs:
#        c = rp['coords'][idx] # coordinates of both 4-connected branchpoint pixels
#        Ibps[c[1,0], c[1,1]] = 0
#        Ibps[c[3,0], c[3,1]] = 0

    # To wrap up, reconvolve with the updated branchpoint image to filter
    # branchpoints made removable by the previous filtering
    Iconv = cv2.filter2D(Ibps, src_depth, kern)
    # Remove unwanted branchpoints based on patterns
    Irm_flat = np.in1d(Iconv, list(rmvals))
    rmy, rmx = np.unravel_index(np.where(Irm_flat==1), Iconv.shape)
    Ibps[rmy, rmx] = 0


#    plt.close('all')
#    rgh.imshowpair(Iskel,Ibps)
#    idcs = np.where(rp['area']>2)
#    for idx in range(0,len(rp['area'])):
#        plt.scatter(rp['coords'][idcs[0][idx]][:,1],rp['coords'][idcs[0][idx]][:,0])


#    Iremove = np.zeros(np.shape(Ibps), dtype='uint8')
#    src_depth = -1
#    for caseno in np.arange(1,9):
#        kern = rgh.bp_kernels(caseno)
#        filtered = cv2.filter2D(Ibps,src_depth,kern)
##        print(sum(sum(filtered)))
#        Iremove[np.uint8(filtered)>11] = 1
#
#    Ibps[Iremove > 0] = 0
#    bps = np.where(Ibps>0)

    return Ibps


def bp_kernels(caseno):
    '''
    Provides kernels for convolving with branchpoints image to remove special
    cases (where three branchpoints make an "L" shape). There are 8 possible
    orientations (or caseno's).
    '''

    kernel = np.zeros(9, np.uint8)
    kernel[4] = 10
    if caseno == 1:
        kernel[[0,1]] = 1
    elif caseno == 2:
        kernel[[1,2]] = 1
    elif caseno == 3:
        kernel[[2,5]] = 1
    elif caseno == 4:
        kernel[[5,8]] = 1
    elif caseno == 5:
        kernel[[7,8]] = 1
    elif caseno == 6:
        kernel[[7,6]] = 1
    elif caseno == 7:
        kernel[[6,3]] = 1
    elif caseno == 8:
        kernel[[0,3]] = 1

    return np.int8(np.reshape(kernel,(3,3)))


def skel_kernels(caseno):
    '''
    Provides kernels for convolving with skeleton image to remove the following case:

    0 0 0
    1 1 1
    0 1 0

    and its rotations. The center pixel would be removed as doing so
    does not break connectivity of the skeleton.
    '''

    kernel = np.zeros(9, np.uint8)
    kernel[4] = 1
    # "T" cases
    if caseno == 1: # up
        kernel[[1,3,5]] = 1
    elif caseno == 2: # right
        kernel[[1,5,7]] = 1
    elif caseno == 3: # down
        kernel[[3,5,7]] = 1
    elif caseno == 4: # left
        kernel[[1,3,7]] = 1

    return np.int8(np.reshape(kernel,(3,3)))


def skel_pixel_curvature(Iskel, nwalk = 4):

# Takes an input skeleton and provides an estimate of the curvature at each
# pixel. Not all pixels are considered; some boundary pixels (i.e. those
# near the end of the skeleton) will not be computed. The more pruned and
# thinned the input skeleton, the less likely that any strange values will
# be returned. This code was written so that strange cases are merely
# skipped.
#
# Given an input skeleton image, the code looks at each pixel and fits a
# trendline through npix pixels "upstream" and "downstream" of the pixel in
# question. Scale is therefore set by npixels; larger will reduce
# variability in the curvature estimates.
#
# INPUTS:      Iskel - binary image of skeleton whose pixels you want
#                      curvature values
#
# OUTPUTS:         I - image of skeleton with curvature values at pixels



    # Pad the image to avoid boundary problems
    # NOT DONE HERE

    # Get coordinates of skeleton pixels
    skelpix = np.argwhere(Iskel==1)
    py = np.ndarray.tolist(skelpix[:,0])
    px = np.ndarray.tolist(skelpix[:,1])

    # Initialize storage image
    I = np.zeros_like(Iskel, dtype=np.float)
    I.fill(np.nan)

    # Loop through each pixel to compute its curvature
    for x, y in zip(px,py):

#        j = 500
#        x = px[j]
#        y = py[j]

        nvals = neighbor_vals(Iskel, x, y)

        if sum(nvals) != 2: # Only compute curvature if there are only two directions to walk
            continue

        walkdirs = [p for p, q in enumerate(nvals) if q == 1]

        walks = []
        for i in [0,1]: # Walk in both directions
            wdir = {walkdirs[i]}
            xwalk = [x]
            ywalk = [y]
            for ii in np.arange(0,nwalk):
                if next(iter(wdir)) == 0:
                    xwalk.append(xwalk[-1]-1)
                    ywalk.append(ywalk[-1]-1)
                    rmdir = {7}
                elif next(iter(wdir)) == 1:
                    xwalk.append(xwalk[-1])
                    ywalk.append(ywalk[-1]-1)
                    rmdir = {5, 6, 7}
                elif next(iter(wdir)) == 2:
                    xwalk.append(xwalk[-1]+1)
                    ywalk.append(ywalk[-1]-1)
                    rmdir = {5}
                elif next(iter(wdir)) == 3:
                    xwalk.append(xwalk[-1]-1)
                    ywalk.append(ywalk[-1])
                    rmdir = {2, 4, 7}
                elif next(iter(wdir)) == 4:
                    xwalk.append(xwalk[-1]+1)
                    ywalk.append(ywalk[-1])
                    rmdir = {0, 3, 5}
                elif next(iter(wdir)) == 5:
                    xwalk.append(xwalk[-1]+1)
                    ywalk.append(ywalk[-1]+1)
                    rmdir = {2}
                elif next(iter(wdir)) == 6:
                    xwalk.append(xwalk[-1])
                    ywalk.append(ywalk[-1]+1)
                    rmdir = {0, 1, 2}
                elif next(iter(wdir)) == 7:
                    xwalk.append(xwalk[-1]+1)
                    ywalk.append(ywalk[-1]+1)
                    rmdir = {0}

                nxtvals = neighbor_vals(Iskel, xwalk[-1], ywalk[-1])
                wdir = {n for n, m in enumerate(nxtvals) if m == 1}
                wdir = wdir - rmdir

                if len(wdir) > 1 or len(wdir) == 0:
                    walks.append([xwalk, ywalk])
                    break

                if ii == nwalk - 1:
                    walks.append([xwalk, ywalk])


        # Now compute curvature using angle between lines fit through pixel walks
        if len(walks) < 2 or len(walks[0][0]) < 2 or len(walks[1][0]) < 2:
            continue

        # Initialize values
        x1 = np.nan
        y1 = np.nan
        x2 = np.nan
        y2 = np.nan

        # Make vectors emanting from origin
        xs1 = [x - walks[0][0][0] for x in walks[0][0]]
        ys1 = [y - walks[0][1][0] for y in walks[0][1]]
        xs2 = [x - walks[1][0][0] for x in walks[1][0]]
        ys2 = [y - walks[1][1][0] for y in walks[1][1]]

        # Check for straight-line segments (can't polyfit to them)
        if all(xcheck == 0 for xcheck in xs1):
            x1 = 0
            if np.mean(ys1) < 0:
                y1 = 1
            else:
                y1 = -1
        elif all(ycheck == 0 for ycheck in ys1):
            y1 = 0
            if np.mean(xs1) > 0:
                x1 = 1
            else:
                x1 = -1

        if all(xcheck == 0 for xcheck in xs2):
            x2 = 0
            if np.mean(ys2) < 0:
                y2 = 1
            else:
                y2 = -1
        elif all(ycheck == 0 for ycheck in ys2):
            y2 = 0
            if np.mean(xs2) > 0:
                x2 = 1
            else:
                x2 = -1

        # Fit line to walked paths; no intercept as lines must intersect at the origin

        # Reduce to vectors with base pixel as origin
        if np.isnan(x1):
            if np.ptp(xs1) >= np.ptp(ys1):
                w1fit = [np.array(xs1, ndmin=2), np.array(ys1, ndmin=2)]
                m1 = np.linalg.lstsq(w1fit[0].T,w1fit[1].T)[0]
                dx = 1 * np.sign(walks[0][0][-1]-walks[0][0][0])
                if dx == 0:
                    dx = 1
                x1 = dx
                y1 = -m1 * dx
            else:
                w1fit = [np.array(ys1, ndmin=2), np.array(xs1, ndmin=2)]
                m1 = np.linalg.lstsq(w1fit[0].T,w1fit[1].T)[0]
                dy = -(1 * np.sign(walks[0][1][-1] - walks[0][1][0]))
                y1 = dy
                x1 = -m1 * dy

        if np.isnan(x2):
            if np.ptp(xs2) >= np.ptp(ys2):
                w2fit = [np.array(xs2, ndmin=2), np.array(ys2, ndmin=2)]
                m2 = np.linalg.lstsq(w2fit[0].T,w2fit[1].T)[0]
                dx = 1 * np.sign(walks[1][0][-1]-walks[1][0][0])
                if dx == 0:
                    dx = 1
                x2 = dx
                y2 = -m2 * dx
            else:
                w2fit = [np.array(ys2, ndmin=2), np.array(xs2, ndmin=2)]
                m2 = np.linalg.lstsq(w2fit[0].T,w2fit[1].T)[0]
                dy = -(1 * np.sign(walks[1][1][-1] - walks[1][1][0]))
                y2 = dy
                x2 = -m2 * dy

        # Flip second vector so it emanates from origin
        x2 = -x2
        y2 = -y2

        dot = x1 * x2 + y1 * y2
        det = x1 * y2 - y1 * x2

        # Store the curvature value
        I[y,x] = np.abs(np.arctan2(det,dot)*360/(2*np.pi))

    return I


def hand_clean(I, action='erase'):
    """
    Allows user to hand-draw regions of interest on a binary mask that can
    either be filled (set to True) or erased (set to False).

    Interact with plot via the following:
        left-click: save a vertex of the polygon
        right-click: remove the last point (useful when zooming/panning)
        enter key: stop recording points

    Possible actions are 'erase' (default) and 'fill'.
    """

    from matplotlib import pyplot as plt

    def make_mask(Ishape, coords):

        from PIL import Image, ImageDraw
        Imask = Image.new("1", [Ishape[1], Ishape[0]], 0)
        ImageDraw.Draw(Imask).polygon(coords, outline=1, fill=1)

        return np.array(Imask, dtype=np.bool)


    plt.close('all')
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.imshow(I)

    coordssave = plt.ginput(n=-1, timeout=0, show_clicks=True)

    # Create polygon mask from selected coordinates
    Imask = make_mask(I.shape, coordssave)

    # Apply polygon mask
    if action == 'erase':
        I[Imask==True] = 0
    elif action == 'fill':
        I[Imask==True] = 1

    return I
