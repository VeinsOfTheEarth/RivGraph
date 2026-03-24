.. _issues:

************
Known issues 
************

There are several known issues that new, and experienced, users may encounter. These may arise when some of the assumptions of RivGraph are violated, and this will serve as a resource to troubleshoot what is wrong. Many of these isues have been identified in the `issues page <https://github.com/VeinsOfTheEarth/RivGraph/issues>`_ and found during regular use of RivGraph. If you encounter more issues, please add them to the issues page.

Shoreline clipping issues
=========================
See :doc:`/shoreline/index` further information on methods to generate a shoreline. 

Left id failure
---------------

Shoreline features should not contain any attributes named left_fid, LEFT_ID, etc. which popular GIS software such as ArcGIS & QGIS can add. It is recommended to enter a shoreline with a single attribute, id, with a null value inside to avoid an `error <https://github.com/VeinsOfTheEarth/RivGraph/issues/9>`_.

Non-point intersection
----------------------

An assumption of :obj:`rivgraph.deltas.delta_utils.clip_by_shoreline` is that the channel skeleton and the shoreline intersect at a point, not at a line segment. If a shoreline was extracted using an automatic method such as the `Opening Angle Method <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2008GL033963>`_, a line intersection can arise. It's recommended to smooth the shoreline, e.g. using `spline interpolation <https://gis.stackexchange.com/questions/24827/smoothing-polygons-in-contour-map>`_ or manually adjust the shoreline in that case to ensure point intersection. 

Mask issues
===========

The basics of maskmaking and mask assumptions are covered in the maskmaking documentation (:doc:`/maskmaking/index`), and this section details what happens when something goes awry. 

Skeleton around the image forming a cyclic graph
------------------------------------------------

As noted in the maskmaking documentation (:doc:`/maskmaking/index`), there should be no no-data pixels in the mask. This is because no-data pixels are treated as 1 (i.e. water), and `this can result <https://github.com/VeinsOfTheEarth/RivGraph/issues/34>`_ in a cyclical network which loops back to the delta apex. Here's an example of the resulting skeleton where the teal is no data, blue is water, grey is land, yellow is the shoreline, and orange is the channel network skeleton after pruning. This issue can arise when projecting a mask from WGS84 (EPSG:4326) imagery to a local UTM zone, which results in the generation of triangular bands of no-data on the edge of the image. The NA values in the projected mask should be set as land (0) prior to running channel network skeletonization to prevent any issues. 

.. image:: https://user-images.githubusercontent.com/18738680/103107918-c4725d00-45f7-11eb-990f-c6b49bebeba9.png

Multiple inlet nodes after assigning directionality
---------------------------------------------------

After successfully extracting the channel network and assigning flow directionality, an `error <https://github.com/VeinsOfTheEarth/RivGraph/issues/52>`_ that may arise when computing the delta metrics is an error that there are multiple inlet nodes found. This arises from an assumption built into RivGraph as of v0.4: the skeletonization of the mask will fill any islands 4 pixels or less, in a 4-neighbor sense, in the network. If these islands exist in the mask the skeleton can intersect one of these islands, which may in some instances cause an error. Here is an example of where this occurs and an issue arises. In the image below the link intersecting the island will have an `wid_adj` of zero, as the adjusted vector `wid_pix` will have a value of essentially zero. 

.. image:: https://user-images.githubusercontent.com/14874485/118341935-709cdd80-b4de-11eb-87a6-bbbc24fc8aec.png

Without this hole-filling assumption the following skeleton would be generated:

.. image:: https://user-images.githubusercontent.com/14874485/118342093-236d3b80-b4df-11eb-81b1-ec032d841e0e.png

Depending on the analysis you are performing, it may or may not be appropriate to have the skeleton shown in the second case. This issue can arise in both the ``delta`` and the ``river`` skeletonization methods.

As changing this assumption may lead to unknown downstream changes in RivGraph's functionality, there are no plans at this time to remove it from the skeletonization procedure. 

Therefore there are several options available to treat this issue if it arises. If the island is relevant to the problem at hand, i.e. represents a significant feature in the network being analyzed: 

1) Comment out the following three lines in :obj:`rivgraph.mask_to_graph.skeletonize_mask()` for your relevant class:

.. code:: python

    Iskel = imu.fill_holes(Iskel, maxholesize=4)
    Iskel = morphology.skeletonize(Iskel)
    Iskel = simplify_skel(Iskel)

2. Downscale the mask such that the island becomes larger than 4 pixels. This will increase processing time, but for relatively small masks may not be significant. 

If the island doesn't represent a significant feature in the network and could be removed: 

3. Remove islands than 4 pixels during mask processing. The function :obj:`rivgraph.im_utils.fill_holes()` provides this functionality.
