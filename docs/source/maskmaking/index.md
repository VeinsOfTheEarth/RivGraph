# Maskmaker, Maskmaker Make Me a Mask

![](C:\Users\Jon\Documents\GitHub\RivGraph\examples\images\lena_mask.PNG)

*RivGraph* requires that you provide a mask of your channel network. A mask is simply a binary image (only ones and zeros) where pixels belonging to the channel network are "on", like the right panel of the Lena Delta above. In this document, we'll cover the following:

- What should a mask capture?
- Where do I get a mask?
- How can I prepare my mask?
- Does my mask need to be georeferenced?
- Can my mask represent something that isn't a river?
- What filetypes are supported for my mask?

One thing to keep in mind: although *RivGraph* contains functions for pruning and otherwise modifying your channel network, it will always honor the mask you provide. You may need to iterate between altering your mask and *RivGraph*ing it to achieve your desired results.

## What should a mask capture?

Above, we defined a mask as all the pixels "belonging to the channel network." But which pixels are part of the channel network? The obvious starting point is to consider all *surface water* pixels as defining the channel network. Is it important that your mask shows bankfull channels or includes small streams that may only be active during flood conditions? 

The bottom line here is that your mask should reflect your analysis goals. For example, if you're only interested in counting the number of loops in a delta channel network, then you might only care about ensuring all the channels are represented. If you want to route fluxes through your extracted network, then you probably should try to obtain a mask that captures the full width of the channels during a representative stage. 

## Where do I get a mask?

Masks

Masks from satellite imagery, also from modeling outputs.

In order of complexity:

Published masks. If you know of more, please mention them in the Issue Tracker!

Global Extent of Rivers and Streams mask at mean annual discharge. Tile boundary problems, feathery along braided rivers.

Using Google Earth Engine to export Global Surface Water masks.

Using Google Earth Engine to train your own classifier.

DeepWaterMap



## How should I prepare my mask?

The quality of your mask directly translates to the quality of your channel network. 

## Does my mask need to be georeferenced?

Most masks are already produced in a GIS context and are already geographically referenced. However,*RivGraph* does not require that your mask image be georeferenced (e.g. a GeoTIFF). If you provide a mask without any georeference information, *RivGraph* will assign it a "dummy" projection in order to proceed. This has no effect on the network extraction. However, it is strongly advised that you provide a georeferenced mask. There are three primary reasons for this:

1) The coordinate reference system (CRS) of your mask will be carried through all your analysis, meaning that shapefiles and GeoTIFFs you export using *RivGraph* will align perfectly with your mask. Additionally, your results will be easily importable into a GIS for further analysis or fusion with other geospatial data.

2) *RivGraph* computes morphologic metrics (length and width) using pixel coordinates. A georeferenced mask contains information about the units of the mask, and thus any metrics of physical distance will inherit these units. If your CRS is meters-based, your results will be in meters.

3) Some of *RivGraph*'s functionality under the hood requires some heuristic thresholds or parameters. While these were designed to be as CRS-agnostic as possible, these functions will undoubtedly perform better when pixels have known units. As an example, generating a mesh along a braided river corridor requires some parameters defining the size and smoothness of the mesh. Having a mask with physically-meaningful units makes this parameterization much simpler and more intuitive. 

It is also highly advisable that you **avoid** degree-based CRSs (like EPSG:4326). This is because the length of a degree is not uniform, but varies with latitude. For example, at the equator, a degree of longitude is roughly 111 km. In Anchorage, Alaska, a degree of longitude is approximately 55 km. Effectively, degrees are meaningless units of physical measurements. A more prudent approach would be to first project your mask into a meters-based CRS (e.g. the appropriate [UTM zone](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system)) before analysis with *RivGraph*.

## Can my mask represent something that isn't a river?



## What filetypes are supported for my mask?

