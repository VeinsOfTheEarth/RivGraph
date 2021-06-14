# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:51:38 2021

@author: Jon
"""
# code: C:\Users\Jon\Anaconda3\envs\rglakes\Lib\site-packages\rivgraph-0.4-py3.8.egg\rivgraph
# data: X:\RivGraph\develop\lakes\test1
import numpy as np
import matplotlib.pyplot as plt
from rivgraph.classes import deltalakes

path_mask = r'X:\RivGraph\develop\lakes\test1\mod_mask.tif'
path_results = r'X:\RivGraph\develop\lakes\test1\Results'
path_shoreline = r"X:\RivGraph\develop\lakes\test1\shoreline.shp"
path_inlets = r"X:\RivGraph\develop\lakes\test1\inlet.shp"
path_lakemask = r'X:\RivGraph\develop\lakes\test1\lake_mask.tif'

# # Prepare lake mask - only need to run once to create a lake mask geotiff - Note that you can also just pass in the lakemask array!
# from osgeo import gdal
# gdobj = gdal.Open(path_mask)
# Imask = gdobj.ReadAsArray()
# Ilakes = Imask == 2
# from rivgraph.io_utils import write_geotiff as wg
# wg(Ilakes, gdobj.GetGeoTransform(), gdobj.GetProjection(), path_lakemask, dtype=gdal.GDT_Byte)

DL = deltalakes('test1', path_mask, path_lakemask, path_results, verbose=True)

# Mask out the lake pixels from the watermask before computing network
DL.Imask[DL.Ilakes] = False
DL.skeletonize()
DL.compute_network()

""" Here is new connect_lakes functionality """
import rivgraph.im_utils as imu
from shapely.geometry import Polygon
import skimage.graph
from scipy.ndimage.morphology import distance_transform_edt
from rivgraph import ln_utils as lnu

props, labeled = imu.regionprops(DL.Ilakes,
                                 props=['perimeter', 'centroid'])

nlakes = len(props['perimeter'])
# Loop through each lake, connect it to the network, and store some properties
for i in range(nlakes):    
    # Get representative point within lake (similar to centroid, but ensures
    # point is within the polygon)
    perim = props['perimeter'][i]
    perim = np.vstack((perim, perim[0])) # Close the perimeter
    lakepoly = Polygon(perim) # Make polygon
    rep_pt = lakepoly.representative_point().coords.xy
    # Check that integer-ed rep_pt is within the blob
    rpx, rpy = round(rep_pt[1][0]), round(rep_pt[0][0])
    if DL.Ilakes[rpy, rpx] is False:
        raise Exception('Found a case where rounded representative point is not within polygon :(')
        
    # Find all links connected to this lake
    # First, create a cropped image of the lake
    xmax, xmin, ymax, ymin = np.max(perim[:,1]) + 1, np.min(perim[:,1]), np.max(perim[:,0]) + 1, np.min(perim[:,0])
    Ilakec = np.array(labeled[ymin:ymax, xmin:xmax], dtype=bool)
    # Pad the image so there is room to dilate
    npad = 3
    Ilakec = np.pad(Ilakec, npad, mode='constant')
    # Dilate the lake and get the newly-added pixels
    Ilakec_d = imu.dilate(Ilakec, n=1, strel='square')
    Ilakec_d[Ilakec] = False
    # Get the indices of the dilated pixels
    dpix_rc = np.where(Ilakec_d)
    # Adjust the rows and columns to account for the fact that we cropped the image
    dpix_rc = (dpix_rc[0] + ymin - npad, dpix_rc[1] + xmin - npad)
    # Convert to index
    dpix_idx = np.ravel_multi_index(dpix_rc, DL.Ilakes.shape)
    
    # # Debugging plotting
    # imu.imshowpair(Ilakec_d, Ilakec)
    
    # Find the nodes connected to the lake
    conn_nodes = []
    for nidx, nid in zip(DL.nodes['idx'], DL.nodes['id']):
        if nidx in dpix_idx:
            conn_nodes.append(nid)
            
    # We better have at least one node connected to each lake (assumes mask is made correctly)
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
        dpix_idx = np.ravel_multi_index(dpix_rc, DL.Ilakes.shape)
       
        # Find the nodes connected to the lake
        conn_nodes = []
        for nidx, nid in zip(DL.nodes['idx'], DL.nodes['id']):
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
        Icosts[~Ilakec] = np.max((100000, np.max(Icosts)**2)) # Set values outside the lake to be very high so path will not traverse them

        # Put the link's node in the cropped coordinates
        node_rc = np.unravel_index(DL.nodes['idx'][DL.nodes['id'].index(cn)], DL.Ilakes.shape)
        node_rc = (node_rc[0] - ymin + npad, node_rc[1] - xmin + npad)
        # Compute the shortest path
        path, cost = skimage.graph.route_through_array(Icosts, start=(node_rc), end=reppt, fully_connected=True)
        
        # Plotting shortest path
        # plt.close('all')
        # plt.imshow(Icosts)
        # ys = [p[0] for p in path]
        # xs = [p[1] for p in path]
        # plt.plot(xs, ys)

        # Convert the pixels of the shortest path back into global coordinates
        path = np.array(path)
        path[:,0] = path[:,0] + ymin - npad
        path[:,1] = path[:,1] + xmin - npad
        p_idcs = np.ravel_multi_index((path[:,0], path[:,1]), DL.Ilakes.shape)
        
        # Add the link to the links dictionary
        DL.links, DL.nodes = lnu.add_link(DL.links, DL.nodes, p_idcs)

DL.to_geovectors()


        
    


# DL.compute_lakes()
# DL.prune_network(path_shoreline, path_inlets)
# DL.compute_link_width_and_length()
# DL.assign_flow_directions()



