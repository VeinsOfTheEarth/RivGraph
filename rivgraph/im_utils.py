# -*- coding: utf-8 -*-
"""
Image Utilities (im_utils.py)
========
Almost all the functions that manipulate images within this file require that
the images be binary.

Many of these functions are wrappers around functionality found in skimage or
opencv that either add additonal functionality or provide convenience. 
"""
import cv2
import numpy as np
from scipy import ndimage as nd
from skimage import morphology, measure, util

## TODO: add checks for inputs to ensure proper types

def get_array(idx, I, size):
    """
    Returns a sub-image of an input image, centered at a specified index within
    the input image and a specified size.

    Parameters
    ----------
    idx : int
        Index within I on which to center the returned array. Can also be a
        [row, col] list.
    I : np.array
        Image to pull from.
    size : list
        Two-entry list specifying the number of [rows, cols] to return from
        I centered at idx.

    Returns
    -------
    subimage : np.array
        The sub-image of I, centered at idx with shape specified by size.
    row : int
        The first row within I that the sub-image is drawn from.
    col : int
        The first column within I that the sub-image is drawn from.

    """
    ## TODO: Should add check for border cases so we don't query to raster 
    # beyond its bounds. Or add error handling for those cases.

    try:
        lidx = len(idx)
    except:
        lidx = len([idx])

    if lidx == 1:
        row, col = np.unravel_index(idx, I.shape)

    else:
        row = idx[0]
        col = idx[1]

    # Get row, column of top-left most pixel in array
    row = int(row - (size[0]-1)/2)
    col = int(col - (size[1]-1)/2)

    subimage = I[row:row+size[0], col:col+size[1]].copy()

    return subimage, row, col


def neighbors_flat(idx, imflat, ncols, filt='nonzero'):
    """
    Returns all 8-neighbor pixel indices and values from an index of a 
    flattened image. Can filter out zero values with the filt keyword. 

    Parameters
    ----------
    idx : int
        Index within imflat to return neighbors.
    imflat : np.array
        Flattened image (using np.ravel).
    ncols : int
        Number of columns in the un-flattened image.
    filt : str, optional
        If 'nonzero', only nonzero indices and values are returned. Else all
        neighbors are returned.

    Returns
    -------
    idcs : np.array
        Indices within the flattened image of neighboring pixels.
    vals : np.array
        Values within the flattened image of neighboring pixels; matches idcs.

    """
    
    if isinstance(idx, np.generic):
        idx = idx.item()
    if isinstance(ncols, np.generic):
        idx = idx.item()

    dy = ncols
    dx = 1
    possneighs = np.array([idx-dy-dx, idx-dy, idx-dy+dx,
                           idx-dx,            idx+dx,
                           idx+dy-dx, idx+dy, idx+dy+dx])

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
    Takes a list of indices from a subimage and returns their indices within
    their parent image.
    
    If idxlist is (x,y), then idxlistdims should be (dim_x, dim_y), etc.

    Parameters
    ----------
    idxlist : np.array or list or set
        List-like array of indices within the subimage.
    idxlistdims : tuple
        (nrows, ncols) of the subimage.
    row_offset : int
        The row in the parent image corresponding to the lowest row in the 
        subimage.
    col_offset : int
        The column in the parent image corresponding to the left-most column
        in the subimage.
    globaldims : tuple
        (nrows, ncols) of the parent image.

    Returns
    -------
    idcsflat : list
        The indices in indexlist in terms of the parent image.

    """    
    # Handle input
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

    Parameters
    ----------
    I : np.array
        Binary image.

    Returns
    -------
    Infc : np.array
        Image of 4-connectivity for each pixel. Same shape as I.

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
    Returns the number of 4-connected neighbors for a given flat index in I.
    idcs must be a list, even if a single value.


    Parameters
    ----------
    idcs : list
        Indices within I to find 4-connected neighbors.
    I : np.array
        Binary image.

    Returns
    -------
    fourconn : list
        Number of four-connected neighbors for each index in idcs.
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
    Can return as indices (dtype=flat) or x,y coordinates (dtype='xy').

    Parameters
    ----------
    sizeI : tuple
        Shape of the image.
    dtype : str, optional
        If 'flat', returns output as indices within I. If 'xy', returns (col, row)
        of edge pixels.
    Returns
    -------
    edgepts : set or list
        Coordinates of the edge pixels. If dtype=='flat', returns a set. If
        dtype=='xy', returns a list of [columns, rows].

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


def neighbor_idcs(c, r):
    """
    Returns the column, row coordinats of all eight neighbors of a given
    column and row.
    
    Returns are ordered as
    [0 1 2
     3   4
     5 6 7]

    Parameters
    ----------
    c : int
        Column.
    r : int
        Row.

    Returns
    -------
    cidcs : list
        Columns of the eight neighbors.
    ridcs : TYPE
        Rows of the eight neighbors.

    """
    cidcs = [c-1, c, c+1, c-1, c+1, c-1, c, c+1]
    ridcs = [r-1, r-1, r-1, r, r, r+1, r+1, r+1]

    return cidcs, ridcs


def neighbor_vals(I, c, r):
    """
    Returns the neighbor values in I of a specified pixel coordinate. Handles
    edge cases.

    Parameters
    ----------
    I : np.array
        Image to draw values from.
    c : int
        Column defining pixel to find neighbor values.
    r : int
        Row defining pixel to find neighbor values.

    Returns
    -------
    vals : np.array
        A flattened array of all the neighboring pixel values.

    """
    vals = np.empty((8,1))
    vals[:] = np.NaN

    if c == 0:

        if r == 0:
            vals[4] = I[r,c+1]
            vals[6] = I[r+1,c]
            vals[7] = I[r+1,c+1]
        elif r == np.shape(I)[0]-1:
            vals[1] = I[r-1,c]
            vals[2] = I[r-1,c+1]
            vals[4] = I[r,c+1]
        else:
            vals[1] = I[r-1,c]
            vals[2] = I[r-1,c+1]
            vals[4] = I[r,c+1]
            vals[6] = I[r+1,c]
            vals[7] = I[r+1,c+1]

    elif c == I.shape[1]-1:

        if r == 0:
            vals[3] = I[r,c-1]
            vals[5] = I[r+1,c-1]
            vals[6] = I[r+1,c]
        elif r == I.shape[0]-1:
            vals[0] = I[r-1,c-1]
            vals[1] = I[r-1,c]
            vals[3] = I[r,c-1]
        else:
            vals[0] = I[r-1,c-1]
            vals[1] = I[r-1,c]
            vals[3] = I[r,c-1]
            vals[5] = I[r+1,c-1]
            vals[6] = I[r+1,c]

    elif r == 0:
        vals[3] = I[r,c-1]
        vals[4] = I[r,c+1]
        vals[5] = I[r+1,c-1]
        vals[6] = I[r+1,c]
        vals[7] = I[r+1,c+1]

    elif r == I.shape[0]-1:
        vals[0] = I[r-1,c-1]
        vals[1] = I[r-1,c]
        vals[2] = I[r-1,c+1]
        vals[3] = I[r,c-1]
        vals[4] = I[r,c+1]

    else:
        vals[0] = I[r-1,c-1]
        vals[1] = I[r-1,c]
        vals[2] = I[r-1,c+1]
        vals[3] = I[r,c-1]
        vals[4] = I[r,c+1]
        vals[5] = I[r+1,c-1]
        vals[6] = I[r+1,c]
        vals[7] = I[r+1,c+1]
        
    vals = np.ndarray.flatten(vals)

    return vals


def neighbor_xy(c, r, idx):
    """
    Returns the coordinates of a neighbor of a pixel given the index of the
    desired neighbor. Indices should be provided according to 
    [0 1 2
     3   4
     5 6 7].

    Parameters
    ----------
    c : int
        Column of the pixel to find the neighbor.
    r : int
        Row of the pixel to find the neighbor.
    idx : int
        Index of the neighbor position.

    Returns
    -------
    c : int
        Column of the neighbor pixel.
    r : int
        Row of the neighbor pixel.

    """
    cs = np.array([-1, 0, 1, -1, 1 ,-1, 0, 1], dtype=np.int)
    rs = np.array([-1, -1, -1, 0, 0 ,1, 1, 1], dtype=np.int)

    c = c + cs[idx]
    r = r + rs[idx]

    return c, r


def remove_blobs(I, blobthresh, connectivity=2):
    """
    Remove blobs of a binary image that are smaller than blobthresh. A blob is
    simply a set of connected "on" pixels.

    Parameters
    ----------
    I : np.array
        Binary image to remove blobs from.
    blobthresh : int
        Minimum number of pixels a blob must contain for it to be kept.
    connectivity : int, optional
        If 1, 4-connectivity will be used to determine connected blobs. If 
        2, 8-connectivity will be used. The default is 2. 

    Returns
    -------
    Ic : np.array
        Binary image with blobs filetered. Same shape as I.

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
    """
    Overlays two binary images with a color scheme that shows their differences
    and overlaps. Provides similar functionality to matlab's imshowpair. Useful
    for quick diagnostic of differences between two binary images.
    
    This function will plot on current figure if it exists, else it will create
    a new one.

    Parameters
    ----------
    I1 : np.array
        The first binary image.
    I2 : np.array
        The second binary image.

    Returns
    -------
    None.

    """

    from matplotlib import colors
    from matplotlib import pyplot as plt
    Ip  = np.zeros(np.shape(I1),dtype='uint8')
    Ip[I1>0] = 2
    Ip[I2>0] = 3
    Ip[np.bitwise_and(I1, I2)==True] = 1

    cmap = colors.ListedColormap(['black','white', 'magenta', 'lime'])
    bounds=[-1,0.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(Ip, origin='upper', cmap=cmap, norm=norm)
    

def largest_blobs(I, nlargest=1, action='remove', connectivity=2):
    """
    Provides filtering for the largest blobs in a binary image. Can choose to 
    either keep or remove them.

    Parameters
    ----------
    I : np.array
        Binary image to filter.
    nlargest : int, optional
        Number of blobs to filter. The default is 1.
    action : str, optional
        If 'keep', will keep the nlargest blobs. If 'remove', will remove the 
        nlargest blobs. The default is 'remove'.
    connectivity : int, optional
        If 1, 4-connectivity will be used to determine connected blobs. If 
        2, 8-connectivity will be used. The default is 2. 

    Returns
    -------
    Ic : np.array
        The filtered image. Same shape as I.

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
    Finds all the indices for each blob within I.

    Parameters
    ----------
    I : np.array
        Binary image containing blobs.
    connectivity : int, optional
        If 1, 4-connectivity will be used to determine connected blobs. If 
        2, 8-connectivity will be used. The default is 2. 

    Returns
    -------
    idcs : list
        An n-element list of sets, where n is the number of blobs in I, and
        each set contains the pixel indices within I of a blob.

    """
    props = ['coords']
    rp, _ = regionprops(I, props, connectivity=connectivity)
    coords = rp['coords']
    idcs = []
    for c in coords:
        idcs.append(set(np.ravel_multi_index([c[:,0], c[:,1]], I.shape)))

    return idcs


def regionprops(I, props, connectivity=2):
    """
    Finds blobs within a binary image and returns requested properties of
    each blob.
    
    This function was modeled after matlab's regionprops and is essentially
    a wrapper for skimage's regionprops. Not all of skimage's available blob
    properties are available here, but they can easily be added.

    Parameters
    ----------
    I : np.array
        Binary image containing blobs.
    props : list
        Properties to compute for each blob. Can include 'area', 'coords', 
        'perimeter', 'centroid', 'mean', 'perimeter', 'perim_len', 'convex_area',
        'eccentricity', 'convex_area', 'major_axis_length', 'minor_axis_length',
        'label'.
    connectivity : int, optional
        If 1, 4-connectivity will be used to determine connected blobs. If 
        2, 8-connectivity will be used. The default is 2. 

    Returns
    -------
    out : dict
        Keys of the dictionary correspond to the requested properties. Values
        for each key are lists of that property, in order such that, e.g., the
        first entry of each property's list corresponds to the same blob.
    Ilabeled : np.array
        Image where each pixel's value corresponds to its blob label. Labels
        can be returned by specifying 'label' as a property.

    """    
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
                Ip, cropped = crop_binary_coords(blob)
                
                # Pad cropped image to avoid edge effects
                Ip = np.pad(Ip, 1, mode='constant')
                
                # Convert to cv2-ingestable data type
                Ip = np.array(Ip, dtype='uint8')

                contours, _ = cv2.findContours(Ip, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # IMPORTANT: findContours returns points as (x,y) rather than (row, col)
                contours = contours[0]
                crows = []
                ccols = []
                for c in contours:
                    crows.append(c[0][1] + cropped[1] - 1) # must add back the cropped rows and columns, as well as the single-pixel pad
                    ccols.append(c[0][0] + cropped[0] - 1)
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
    """
    Erodes a binary image using a specified kernel shape.

    Parameters
    ----------
    I : np.array
        Binary image to erode.
    n : int, optional
        Number of times to apply the eroding kernel. The default is 1.
    strel : str, optional
        Kernel shape. Follows skimage's options, but only 'square', 'plus', and
        'disk' are supported. The default is 'square'.

    Returns
    -------
    Ie : np.array
        Eroded binary image. Same shape as I.

    """

    if n == 0:
        return I

    if strel == 'square':
        selem = morphology.square(3)
    elif strel == 'plus':
        selem = morphology.diamond(1)
    elif strel == 'disk':
        selem = morphology.disk(3)

    Ie = I.copy()
    for i in np.arange(0,n):
        Ie = morphology.erosion(Ie, selem)

    return Ie


def dilate(I, n=1, strel='square'):
    """
    Dilates a binary image using a specified kernel shape.

    Parameters
    ----------
    I : np.array
        Binary image to dilate.
    n : int, optional
        Number of times to apply the dilating kernel. The default is 1.
    strel : str, optional
        Kernel shape. Follows skimage's options, but only 'square', 'plus', and
        'disk' are supported. The default is 'square'.

    Returns
    -------
    Id : np.array
        Dilated binary image. Same shape as I.

    """

    if n == 0:
        return I

    if strel == 'square':
        selem = morphology.square(3)
    elif strel == 'plus':
        selem = morphology.diamond(1)
    elif strel == 'disk':
        selem = morphology.disk(3)

    Id = I.copy()
    for i in np.arange(0,n):
        Id = morphology.dilation(Id, selem)

    return Id


def trim_idcs(imshape, idcs):
    """
    Trims a list of indices by removing the indices that cannot fit within a 
    raster of imshape.

    Parameters
    ----------
    imshape : tuple
        Shape of image to filter idcs against.
    idcs : np.array
        Indices to ensure fit within an image of shape imshape.

    Returns
    -------
    idcs : np.array
        Indices that can fit within imshape.

    """
    
    idcs = idcs[idcs[:, 0] < imshape[0], :]
    idcs = idcs[idcs[:, 1] < imshape[1], :]
    idcs = idcs[idcs[:, 0] >= 0, :]
    idcs = idcs[idcs[:, 1] >= 0, :]

    return idcs


def crop_binary_im(I):
    """
    Crops a binary image to the smallest bounding box containing all the "on"
    pixels in the image.

    Parameters
    ----------
    I : np.array
        Binary image to crop.

    Returns
    -------
    Icrop : np.array
        The cropped binary image.
    pads : list
        Four element list containing the number of pixels that were cropped
        from the [left, top, right, bottom] of I.

    """
    coords = np.where(I==1)
    uly = np.min(coords[0])
    ulx = np.min(coords[1])
    lry = np.max(coords[0]) + 1
    lrx = np.max(coords[1]) + 1

    Icrop = I[uly:lry, ulx:lrx]
    pads = [ulx, uly, I.shape[1]-lrx, I.shape[0] - lry]

    return Icrop, pads


def crop_binary_coords(coords):
    """
    Crops an array of (row, col) coordinates (e.g. blob indices) to the smallest
    possible array.

    Parameters
    ----------
    coords : np.array
        N x 2 array. First column are rows, second are columns of pixel coordinates.

    Returns
    -------
    I : np.array
        Image of the cropped coordinates, plus padding if desired.
    clipped : list
        Number of pixels in [left, top, right, bottom] direction that were
        clipped.  Clipped returns the indices within the original coords image 
        that define where I should be positioned within the original image.

    """
    top = np.min(coords[:,0])
    bottom = np.max(coords[:,0])
    left = np.min(coords[:,1])
    right = np.max(coords[:,1])

    I = np.zeros((bottom-top+1,right-left+1))
    I[coords[:,0]-top,coords[:,1]-left] = True
    
    clipped = [left, top, right, bottom]

    return I, clipped


def fill_holes(I, maxholesize=0):
    """
    Fills holes up to a specified size in a binary image. The boundary pixels
    of the image are turned off before identifying holes so that holes created 
    by the edge of the image are not considered holes.

    Parameters
    ----------
    I : np.array
        Binary image.
    maxholesize : int, optional
        The maximum allowed hole size in pixels. The default is 0.

    Returns
    -------
    I : np.array
        The holes-filled image.

    """
    
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


def im_connectivity(I):
    """
    Returns an image of 8-connectivity for an input image of all pixels in a
    binary image.

    Parameters
    ----------
    I : np.array
        Binary image.

    Returns
    -------
    Iret : np.array
        Image of 8-connectivity count for each pixel in the input image.

    """
    # Fix input
    I = I.copy()
    I[I!=0] = 1
    I = np.uint8(I)

    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])

    src_depth = -1
    filtered = cv2.filter2D(I, src_depth, kernel)
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

    Parameters
    ----------
    I : np.array
        Binary image to downsample.
    newsize : tuple
        Two entry tuple (nrows, ncols) specifying the desired shape of the 
        image.
    thresh : float, optional
        The fraction of filled area that downsampled pixels must have to 
        consider them on. Setting to 0 is the equialent of "all touched". 
        The default is 0.05.

    Returns
    -------
    Iout : np.array
        The downsampled image.

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


def skel_endpoints(Iskel):
    """
    Finds all the endpoints of a skeleton.

    Parameters
    ----------
    Iskel : np.array
        Binary skeleton image.

    Returns
    -------
    eps : np.array
        Skeleton endpoint positions in (row, col) format.

    """

    Ic = im_connectivity(Iskel)
    eps = np.where(Ic==1)

    return eps


def skel_branchpoints(Iskel):
    """
    Finds the branchpoints in a skeletonized image. Branchpoints are not simply
    those with more than two neighbors; they are identified in a way that 
    minimizes the number of branchpoints required to resolve the skeleton
    fully with the fewest number of branchpoints.

    Parameters
    ----------
    Iskel : np.array
        Skeletonized image.

    Returns
    -------
    Ibps : np.array.
        Binary image of shape Iskel where only branchpoints are on.

    """
    Ibps = np.uint16(im_connectivity(Iskel))

    # Initial branchpoints are defined by pixels with conn > 2
    Ibps[Ibps<3] = 0
    Ibps[Ibps>0] = 1

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
    
    return Ibps


def bp_kernels(caseno):
    """
    Provides kernels for convolving with branchpoints image to remove special
    cases (where three branchpoints make an "L" shape). There are 8 possible
    orientations (caseno's).

    Parameters
    ----------
    caseno : int
         Identifier for specific kernel cases. Can be 1-8.

    Returns
    -------
    kernel : np.array
        3x3 kernel corresponding to the caseno.

    """
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

    kernel = np.int8(np.reshape(kernel,(3,3)))
    return kernel


def skel_kernels(caseno):
    """
    Provides kernels for convolving with skeleton image to remove the following case:

    0 0 0
    1 1 1
    0 1 0

    and its rotations. The center pixel would be removed as doing so
    does not break connectivity of the skeleton.  

    Parameters
    ----------
    caseno : int
         Identifier for specific kernel cases. Can be 1-4.

    Returns
    -------
    kernel : np.array
        3x3 kernel corresponding to the caseno.

    """
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

    kernel = np.int8(np.reshape(kernel,(3,3)))
    
    return kernel


def skel_pixel_curvature(Iskel, nwalk = 4):
    """
    Provides an estimate of the curvature at most pixels within a skeleton image. 
    Not all pixels are considered; some boundary pixels (i.e. those
    near the end of the skeleton) will not be computed. The more pruned and
    thinned the input skeleton, the less likely that any strange values will
    be returned. This code was written so that strange cases are merely
    skipped.
    
    Given an input skeleton image, the code looks at each pixel and fits a
    trendline through npix pixels "upstream" and "downstream" of the pixel in
    question. Scale is therefore set by npixels; larger will reduce
    variability in the curvature estimates.
    
    INPUTS:      Iskel - binary image of skeleton whose pixels you want
                          curvature values
    
    OUTPUTS:         I - image of skeleton with curvature values at pixels


    Parameters
    ----------
    Iskel : np.array
        Binary skeleton image to compute curvatures.
    nwalk : int, optional
        Number of pixels to walk away, in each direction, to define the 
        curve to compute a pixel's curvature. Higher results in smoother
        curvature values. The default is 4.

    Returns
    -------
    I : np.array
        Image wherein pixel values represent the curvature of the skeleton.

    """

    ## TODO: Pad the image to avoid boundary problems
    
    # Get coordinates of skeleton pixels
    skelpix = np.argwhere(Iskel==1)
    py = np.ndarray.tolist(skelpix[:,0])
    px = np.ndarray.tolist(skelpix[:,1])

    # Initialize storage image
    I = np.zeros_like(Iskel, dtype=np.float)
    I.fill(np.nan)

    # Loop through each pixel to compute its curvature
    for x, y in zip(px,py):

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
    either be filled (set to True) or erased (set to False). Only one polygon
    can be drawn per call.

    Interact with plot via the following:
        left-click: save a vertex of the polygon
        right-click: remove the last point (useful when zooming/panning)
        enter key: stop recording points

    Possible actions are 'erase' (default) and 'fill'.
    
    Requires matplotlib and Python Image Library (PIL), but they are imported
    here so that this function can be imported independently of the script.

    Parameters
    ----------
    I : np.array
        Binary image to fill or erase with hand-drawn polygons.
    action : str, optional
        If =='fill', will fill the drawn region. If =='erase', will erase the
        drawn region. The default is 'erase'.

    Returns
    -------
    I : np.array
        The image with the hand-drawn polygon either filled or erased.

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
