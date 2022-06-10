[![build](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/build.yml/badge.svg)](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/build.yml)
[![Coverage Status](https://coveralls.io/repos/github/jonschwenk/RivGraph/badge.svg)](https://coveralls.io/github/jonschwenk/RivGraph)
![docs](https://github.com/VeinsOfTheEarth/RivGraph/workflows/docs/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02952/status.svg)](https://doi.org/10.21105/joss.02952)
<br />

[![RivGraph logo](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/docs/logos/rg_logo_full.png)](https://VeinsOfTheEarth.github.io/RivGraph/ "Go to documentation.")

About
-----

RivGraph is a Python package that provides tools for converting a binary mask of a channel network into a directed, weighted graph (i.e. a set of connected links and nodes).

![Core functionality of RivGraph.\label{fig:corefunctions}](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/examples/images/rivgraph_overview_white.PNG)

The figure above demonstrates the core components of RivGraph, but many other features are provided, including:

- Morphologic metrics (lengths, widths, branching angles, braiding indices)
- Algebraic representations of the channel network graph
- Topologic metrics (both topologic and dynamic such as alternative paths, flux sharing, entropies, mutual information, etc.)
- Tools for cleaning and preparing your binary channel network mask
- Island detection, metrics, and filtering
- Mesh generation for characterizing along-river characteristics
- (beta) Tools for centerline migration analysis

All of RivGraph's functionality maintains and respects georeferencing information. If you start with a georeferenced mask (e.g. a GeoTIFF), RivGraph exports your results in the CRS (coordinate reference system) of your mask for convenient mapping, analysis, and fusion with other datasets in a GIS.

You can see some description of RivGraph's functionality via this [AGU poster](https://www.researchgate.net/publication/329845073_Automatic_Extraction_of_Channel_Network_Topology_RivGraph), and the flow directionality logic and validation is described in our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html). Examples demonstrating the basic RivGraph features are available for a [delta channel network](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/examples/delta_example.ipynb) and a [braided river](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/examples/braided_river_example.ipynb).

Installing
-----
RivGraph v0.4 is hosted on the anaconda channel [jschwenk](https://anaconda.org/jschwenk/rivgraph). We recommend installing into a fresh conda environment to minimize the risk of dependency clashes. The easiest way to do this is to download the [environment.yml](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/environment.yml) file, then open Terminal (Mac/Unix) or Anaconda Prompt (Windows) and type:

<pre><code>conda env create --file /path/to/environment.yml  # the environment name will be 'rivgraph', but you can change the environment file to name it anything</code></pre>

You may then want to install Spyder or your preferred IDE. Conda should fetch all the required dependencies and handle versioning.

If you want to install RivGraph into an already-existing environment, you can run <pre><code>conda activate myenv
conda install rivgraph -c jschwenk</code></pre>

You may also [install RivGraph from this Github repo](https://VeinsOfTheEarth.github.io/RivGraph/install/index.html#installation-from-source).

Instructions for testing your installation are available [here](https://VeinsOfTheEarth.github.io/RivGraph/install/index.html#installation-from-source).

How to use?
-----
Please see the [documentation](https://VeinsOfTheEarth.github.io/RivGraph/) for more detailed instructions.

RivGraph requires that you provide a binary mask of your network. [This page](https://VeinsOfTheEarth.github.io/RivGraph/maskmaking/index.html) provides some help, hints, and tools for finding or creating your mask.

To see what RivGraph does and how to operate it, you can work through the [Colville Delta example](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/examples/delta_example.ipynb) or the [Brahmaputra River example](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/examples/braided_river_example.ipynb). Both examples include sample masks.

RivGraph contains two primary classes (`delta` and `river`) that provide convenient methods for creating a processing workflow for a channel network. As the examples demonstrate, you can instantiate a delta or river class, then apply associated methods for each. After looking at the examples, take a look at [classes.py](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/rivgraph/classes.py) to understand what methods are available.

**Note**: there are many functions under the hood that may be useful to you. Check out the [im_utils script](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/rivgraph/im_utils.py) (image utilities) in particular for functions to help whip your mask into shape!


Contributing
------------
If you think you're not skilled or experienced enough to contribute, think again! We agree wholeheartedly with the sentiments expressed by this [Imposter syndrome disclaimer](https://github.com/Unidata/MetPy#contributing). We welcome all forms of user contributions including feature requests, bug reports, code, documentation requests, and code. Simply open an issue in the [tracker](https://github.com/VeinsOfTheEarth/RivGraph/issues). For code development contributions, please contact us via email to be added to our slack channel where we can hash out a plan for your contribution.

Citing RivGraph
------------

Citations help us justify the effort that goes into building and maintaining this project. If you used RivGraph for your research, please consider citing us.

If you use RivGraph's flow directionality algorithms, please cite our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html). Additionally, if you publish work wherein RivGraph was used to process your data, please cite our [JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.02952).

Contacting us
-------------

The best way to get in touch is to [open an issue](https://github.com/VeinsOfTheEarth/rivgraph/issues/new) or comment on any open issue or pull request. Otherwise, send an email to j.........k@gmail.com


License
------------

This is free software: you can redistribute it and/or modify it under the terms of the **BSD 3-clause License**. A copy of this license is provided in [LICENSE.txt](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/LICENSE.txt).

RivGraph has been assigned number C19049 by the Feynman Center for Innovation.
