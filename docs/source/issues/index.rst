.. _issues:

************
Known issues
************

This page collects a few recurring problems that users may encounter when RivGraph's mask or shoreline assumptions are violated. Many of these cases are discussed in more detail on the `issue tracker <https://github.com/VeinsOfTheEarth/RivGraph/issues>`_. If you encounter a new one, please open an issue.

Shoreline clipping issues
=========================

See :doc:`/shoreline/index` for guidance on generating a shoreline.

Left-id failure
---------------

Shoreline features should not contain attributes such as ``left_fid`` or ``LEFT_ID``, which some GIS tools can add automatically. A clean shoreline layer with a simple ``id`` field avoids `issue #9 <https://github.com/VeinsOfTheEarth/RivGraph/issues/9>`_.

Non-point intersection
----------------------

An assumption of :obj:`rivgraph.deltas.delta_utils.clip_by_shoreline` is that the channel skeleton and the shoreline intersect at points, not along line segments. If a shoreline was extracted automatically, for example with the `Opening Angle Method <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2008GL033963>`_, a line intersection can arise. In that case it is usually best to smooth or manually adjust the shoreline so the crossings are clean point intersections.

Mask issues
===========

The basic mask assumptions are covered in :doc:`/maskmaking/index`. This section focuses on what happens when those assumptions are violated.

Skeleton around the image forming a cyclic graph
------------------------------------------------

As noted in :doc:`/maskmaking/index`, masks should not contain no-data pixels. No-data is treated as water, and `this can result <https://github.com/VeinsOfTheEarth/RivGraph/issues/34>`_ in a cyclic network that loops around the image edge and reconnects to the delta apex. This often appears when a mask is projected from WGS84 imagery into a local UTM zone, leaving triangular no-data wedges at the image boundary.

Set those no-data values to land (0) before skeletonization.

.. image:: https://user-images.githubusercontent.com/18738680/103107918-c4725d00-45f7-11eb-990f-c6b49bebeba9.png

Multiple inlet nodes after assigning directionality
---------------------------------------------------

`Issue #52 <https://github.com/VeinsOfTheEarth/RivGraph/issues/52>`_ can arise when a very small island is filled during skeleton preparation and a skeleton branch ends up intersecting the island. In those cases the affected link may acquire a near-zero adjusted width, and later delta-metric calculations can fail with a multiple-inlet error.

.. image:: https://user-images.githubusercontent.com/14874485/118341935-709cdd80-b4de-11eb-87a6-bbbc24fc8aec.png

Without that small-hole filling step, the skeleton would instead look like this:

.. image:: https://user-images.githubusercontent.com/14874485/118342093-236d3b80-b4df-11eb-81b1-ec032d841e0e.png

Whether that alternative is preferable depends on the application. In practice, the safest options are usually:

1. increase the effective mask resolution so the island spans more than four pixels
2. remove tiny islands during mask preprocessing if they are not important to the analysis
3. for advanced source users, modify the small-hole-filling behavior in ``rivgraph.mask_to_graph.skeletonize_mask()``

The third option is intentionally listed last because it changes a core preprocessing assumption and should be done only when you understand the downstream consequences.
