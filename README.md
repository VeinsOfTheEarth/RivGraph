[![Build Status](https://api.travis-ci.org/jonschwenk/rivgraph.svg?branch=master)](https://api.travis-ci.org/jonschwenk/rivgraph)
[![Coverage Status](https://coveralls.io/repos/github/jonschwenk/RivGraph/badge.svg?branch=master)](https://coveralls.io/github/jonschwenk/RivGraph?branch=master)
![docs](https://github.com/jonschwenk/RivGraph/workflows/docs/badge.svg)
<br />

![RivGraph logo](https://github.com/jonschwenk/RivGraph/blob/master/docs/logos/rg_logo_full.png)

About
-----

RivGraph is a Python package that provides tools for converting a binary mask of a channel network into a directed, weighted graph (i.e. a set of connected links and nodes). 

![Core functionality of RivGraph.\label{fig:corefunctions}](https://github.com/jonschwenk/RivGraph/blob/master/examples/images/rivgraph_overview.PNG)

The figure above demonstrates the core components of RivGraph, but many other features are provided, including:

- Morphologic metrics (lengths, widths, branching angles, braiding indices)
- Algebraic representations of the channel network graph
- Topologic metrics (both topologic and dynamic such as alternative paths, flux sharing, entropies, mutual information, etc.)
- Tools for cleaning and preparing your binary channel network mask
- Island detection, metrics, and filtering
- Mesh generation for characterizing along-river characteristics
- (beta) Tools for centerline migration analysis

All of RivGraph's functionality maintains and respects georeferencing information. If you start with a georeferenced mask (e.g. a GeoTIFF), RivGraph exports your results in the CRS (coordinate reference system) of your mask for convenient mapping, analysis, and fusion with other datasets in a GIS.

You can see some description of RivGraph's functionality via this [AGU poster](https://www.researchgate.net/publication/329845073_Automatic_Extraction_of_Channel_Network_Topology_RivGraph), and the flow directionality logic and validation is described in our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html). Examples demonstrating the basic RivGraph features are available for a [delta channel network](https://github.com/jonschwenk/RivGraph/blob/master/examples/delta_example.py.ipynb) and a [braided river](https://github.com/jonschwenk/RivGraph/blob/master/examples/braided_river_example.ipynb).

Check out the [documentation](https://jonschwenk.github.io/RivGraph/).

As of 11/29/2020, we are working on buttoning down RivGraph, including additional documentation, examples, and packaging. Please check back soon, and feel free to interact by opening an issue or emailing j........k@gmail.com. Now is a good time to add feature requests!

Installing
-----
RivGraph v0.3 is hosted on the anaconda channel [jschwenk](https://anaconda.org/jschwenk/rivgraph). We recommend installing into a fresh conda environment to minimize the risk of dependency clashes. The easiest way to do this is to download the [environment.yml](https://github.com/jonschwenk/RivGraph/blob/master/environment.yml) file, then open Terminal (Mac/Unix) or Anaconda Prompt (Windows) and type:

<pre><code>conda env create --file /path/to/environment.yml  # the environment name will be 'rivgraph', but you can change the environment file to name it anything</code></pre>

You may then want to install Spyder or your preferred IDE. Conda should fetch all the required dependencies and handle versioning.

If you want to install RivGraph into an already-existing environment, you can run <pre><code>conda activate myenv
conda install rivgraph -c jschwenk</code></pre>

**Note**: While packaged for all platforms, we have currently only tested the win32, win64, and linux-64 platforms.

Savvy users have been able to install RivGraph from this Github repo, but usually not without dependency headaches. See the [environment file](https://github.com/jonschwenk/RivGraph/blob/master/environment.yml) for dependencies.

How to use?
-----
We are working on adding more documentation and examples. Your best bet for getting started is to reproduce the [Colville delta example](https://github.com/jonschwenk/RivGraph/blob/master/examples/delta_example.ipynb) or the [Brahmaputra river example](https://github.com/jonschwenk/RivGraph/blob/master/examples/braided_river_example.ipynb). Otherwise, use RivGraph by creating either a delta or a river class, then applying the associated methods. Look at the [classes.py](https://github.com/jonschwenk/RivGraph/blob/master/rivgraph/classes.py) script to get a sense for what methods are available and what they're actually doing.

**Note**: there are many functions under the hood that may be useful to you. Check out the [im_utils script](https://github.com/jonschwenk/RivGraph/blob/master/rivgraph/im_utils.py) (image utilities) in particular for functions to help whip your mask into shape!

Task list
-----
These tasks represent what is needed before we "officially" release RivGraph via publication in the [Journal of Open Source Software](https://joss.theoj.org/).

3/29/2020 - Task list created
- [x] [Conda Packaging](anaconda.org/jschwenk/rivgraph) - Updates 5/25/2020
- [x] [Delta example](https://github.com/jonschwenk/RivGraph/blob/master/examples/delta_example.py.ipynb)
- [x] [Braided river example](https://github.com/jonschwenk/RivGraph/blob/master/examples/braided_river_example.ipynb)
- [x] How to fix flow directions (shown in braided river example, section 7.1)
- [x] Function for removing artificial nodes. Restructured code to not add these automatically, but can be added with a [function](https://github.com/jonschwenk/RivGraph/blob/9bc320239443ea7b1673307f77f4edb86251aaf9/rivgraph/ln_utils.py#L724).
- [x] [Unit testing](https://github.com/jonschwenk/RivGraph/tree/master/tests)
- [x] Function documentation
- [ ] How to prepare masks for inputs
- [ ] Where to get masks
- [ ] How to draw shorelines


Contacting us
-------------

The best way to get in touch is to [open an issue](https://github.com/jonschwenk/rivgraph/issues/new) or comment on any open issue or pull request. Otherwise, send an email to j.........k@gmail.com


Contributing
------------
If you think you're not skilled or experienced enough to contribute, think again! We agree wholeheartedly with the sentiments expressed by this [Imposter syndrome disclaimer](https://github.com/Unidata/MetPy#contributing). We welcome all forms of user contributions including feature requests, bug reports, code, documentation requests, and code. Simply open an issue in the [tracker](https://github.com/jonschwenk/RivGraph/issues). For code development contributions, please contact us via email to be added to our slack channel where we can hash out a plan for your contribution. 

Citing RivGraph
------------

Citations help us justify the effort that goes into building and maintaining this project. If you used RivGraph for your research, please consider citing us.

As of 5/26/2020, please cite our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html) and/or our [AGU Presentation](https://www.researchgate.net/publication/329845073_Automatic_Extraction_of_Channel_Network_Topology_RivGraph). We hope to soon publish RivGraph in the Journal of Open Source Software.

License
-------

This is free software: you can redistribute it and/or modify it under the terms of the **BSD 3-clause License**. A copy of this license is provided in [LICENSE.txt](https://github.com/jonschwenk/RivGraph/blob/master/LICENSE.txt).

RivGraph has been assigned number C19049 by the Feynman Center for Innovation.
