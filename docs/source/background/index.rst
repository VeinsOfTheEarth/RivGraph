.. _background:

==========
Background
==========

.. todo:: add background information 

**The birth of RivGraph**

In 2016, I walked into the postdoc office at the Saint Anthony Falls Laboratory to see deltas on display as 8.5 x 11 sheets of paper 
taped together with hand-drawn markings all over. My colleagues Alejandro Tejedor and Anthony Longjas were trying to whip these deltas
into shape for a series of [papers]() introducing a new way to think about deltas: through the lens of their channel networks. As I 
looked at their handiwork, I couldn't help thinking that there must be a better way. My own research at the time had led me to develop 
[RivMAP](), a Matlab toolbox for analyzing the morphodynamics of meandering rivers using binary masks. And so RivGraph was born as a small
set of Matlab scripts with very limited functionality.

I finished my PhD with RivGraph in a barely-formed state, and followed my adviser, [Efi](), to UC-Irvine for a short postdoc. Alex and
Anthony were pushing the deltas work even farther, and I kept adding scripts to RivGraph. The inevitable idea was hatched to analyze 
dozens of deltas worldwide, and eventually became a [successful proposal](). In the meantime, I took a postdoc position at Los Alamos
National Laboratory with an eye toward global river morphodynamics. My arrival at LANL marked the beginning of my Python journey, and
soon I had converted all the RivGraph scripts to Python. My research needs led to a fuller development of RivGraph and brought braided
rivers into the mix, and in 2017 sometime, RivGraph officially became RivGraph.

I cut my Python teeth while developing RivGraph, so there is a degree of inherited clunkiness and inefficiency baked in. Initially, there
were also some implements (such as automatically adding artificial nodes to parallel edges) that were tailored to somewhat particular use
cases. However, as the user base continues to grow, improvements and upgrades have been implemented to meet their needs. Please add your requests
to the mix and report any bugs you find using Github's [issue tracker]().

**What's different about RivGraph?**

RivGraph fills a void in coding space by providing a full-featured package for working with masks of river channels. 
In my opinion, most of RivGraph is just a convenient collection of already-existing functionality like CRS handling, path management, 
etc. That said, there are three somewhat novel components of RivGraph that are not available elsewhere, or at least not available in
Python. The first of these is the walking algorithm that breaks a skeleton into its constituent links and nodes. While there is similar
functionality available in GIS packages, RivGraph ensures that the resulting skeleton is parsimonious--i.e. contains as few nodes as possible
while fully preserving the topology of the mask. This is achieved with help from a double-convolution, where the first convolution identifies possible
branchpoints and the second reduces branchpoint clusters to a minimum-required set. The second novelty RivGraph offers is its ability
to automatically set flow directions. This is trivial for some networks (like Wax Lake Delta), but many channel networks are wild beasts. [RivGraph's
solution]() is correspondingly complicated, but does a pretty good job. We published the method, its validation, and its implications 
[here](). The third novel component of RivGraph is its abilty to generate an along-channel mesh that approximately follows a river's centerline
while transecting the centerline approximately perpendicularly. This was a surprisingly tricky function to get right, and I'm not even
sure it's *there* yet. The first iteration of this appeared in RivMAP, and there have probably been 3-4 method changes before settling on
the current version, which uses Dynamic Time Warping (thanks [Zoltan et. al]() for introducing this to me, although I still contend that
it's inappropriate to use for measuring channel migration rates) to iteratively map vertices on buffered centerlines away from the original.

**What else?**

While RivGraph is designed around channel networks, it contains a smattering of tools that can be useful across a broad range of analyses.
For instance, the [mask_to_graph.py]() script contains tools that will convert *any* binary mask to a vectorized skeleton, not just river
channel networks. There are a number of image processing tools in [im_utils.py]() that I use frequently in other projects, like a Matlab-like
implementation of regionprops(). I have found one of the most broadly useful tools in RivGraph is [write_geotiff](), which does what it says.
 