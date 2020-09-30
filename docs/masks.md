# Maskmaker, Maskmaker Make Me a Mask

![](C:\Users\Jon\Documents\GitHub\RivGraph\examples\images\lena_mask.PNG)

RivGraph requires that you bring a mask to the table. A mask is simply a binary image of ones and zeros where pixels belonging to the channel network are "on", like the right panel of the Lena Delta above. In this document, we'll cover the following:

- What should a mask capture?
- Where do I get a mask?
- How can I prepare my mask?
- Does my mask need georeferencing information?

## What should a mask capture?

Above, we defined a mask as all the pixels "belonging to the channel network." But which pixels are part of the channel network? The obvious starting point is to consider all *surface water* pixels as defining the channel network. What about bankfull?

The bottom line here is that your mask should reflect your analysis goals. For example, if you're only interested in counting the number of loops in a delta channel network, then you might only care about ensuring all the channels are represented. If you want to route fluxes through your extracted network, then you probably should try to obtain a mask that captures the full width of the channels during a representative stage. 

## Where do I get a mask?

Masks from satellite imagery, also from modeling outputs.

In order of complexity:

Published masks. If you know of more, please mention them in the Issue Tracker!

Global Extent of Rivers and Streams mask at mean annual discharge. Tile boundary problems, feathery along braided rivers.

Using Google Earth Engine to export Global Surface Water masks.

Using Google Earth Engine to train your own classifier.

DeepWaterMap



## How should I prepare my mask?

The quality of your mask directly translates to the quality of your channel network. 