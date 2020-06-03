About
-----
RivGraph is a python package that provides tools for converting a binary mask of a channel network into a graph (i.e. a set of connected links and nodes). One major component of RivGraph is its ability to automatically set flow directions in each link of the network. It also computes morphologic metrics (lengths, widths, branching angles, etc.) and topologic metrics. RivGraph also contains a smattering of other tools and features, including some functions for helping clean and prepare your binary mask. 

You can see some of RivGraph's functionality via this [AGU poster](https://www.researchgate.net/publication/329845073_Automatic_Extraction_of_Channel_Network_Topology_RivGraph), and the flow directionality logic and validation is described in our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html).

As of 5/26/2020, we are working on buttoning down RivGraph, including documentation, examples, and packaging. Please check back soon, and feel free to interact by opening an issue or emailing j........k@gmail.com. Now is a good time to add feature requests!

Installing
-----
RivGraph v0.3 is hosted on the anaconda channel [jschwenk](https://anaconda.org/jschwenk/rivgraph). We recommend installing into a fresh conda environment to minimize the risk of dependency clashes. The easiest way to do this is to download the [environment.yml](https://github.com/jonschwenk/RivGraph/blob/master/environment.yml) file, then open Terminal (Mac/Unix) or Anaconda Prompt (Windows)

<pre><code>cd /path/to/where/environment.yml/is/
conda env create --file environment.yml # the environment name will be 'rivgraph', but you can change the environment file to name it anything</code></pre>

You may then want to install Spyder or your preferred IDE. Conda *should* fetch all the required dependencies and handle versioning.
**Note**: While packaged for all platforms, we have currently only tested the win64 platform.

Savvy users have been able to install RivGraph from this Github repo, but usually not without dependency headaches. See the [requirements file](https://github.com/jonschwenk/RivGraph/blob/master/requirements.txt) for dependencies.

How to use?
-----
We are working on documentation and examples. Your best bet for getting started is to reproduce the [Colville delta example](https://github.com/jonschwenk/RivGraph/blob/master/examples/delta_example.py.ipynb). Otherwise, use RivGraph by creating either a delta or a river class, then applying the associated methods. Look at the [classes.py](https://github.com/jonschwenk/RivGraph/blob/master/rivgraph/classes.py) script to get a sense for what methods are available, and what they're actually doing.

**Note**: there are many functions under the hood that may be useful to you. Check out the [im_utils script](https://github.com/jonschwenk/RivGraph/blob/master/rivgraph/im_utils.py) (image utilities) in particular for functions to help whip your mask into shape!

Task list
-----
These tasks represent what is needed before we "officially" release RivGraph via publication in the [Journal of Open Source Software](https://joss.theoj.org/).

3/29/2020 - Task list created
- [x] [Delta example](https://github.com/jonschwenk/RivGraph/blob/master/examples/delta_example.py.ipynb)
- [ ] Braided river example
- [ ] How to prepare masks for inputs
- [ ] Where to get masks
- [ ] How to draw shorelines
- [ ] How to fix flow directions
- [ ] Function documentation
- [ ] Unit testing - in progress (5/11/2020)
- [ ] Function for removing artificial nodes
- [x] [Conda Packaging](anaconda.org/jschwenk/rivgraph) - Updates 5/25/2020 


Contacting us
-------------

The best way to get in touch is to [open an issue](https://github.com/jonschwenk/rivgraph/issues/new) or comment
  on any open issue or pull request. Otherwise, send an email to j.........k@gmail.com

Citing RivGraph
------------

Citations help us justify the effort that goes into building and maintaining this project. If you
used RivGraph for your research, please consider citing us.

As of 5/26/2020, please cite our [ESurf Dynamics paper]((https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html) and/or our [AGU Presentation](https://www.researchgate.net/publication/329845073_Automatic_Extraction_of_Channel_Network_Topology_RivGraph). We hope to soon publish RivGraph in the Journal of Open Source Software.

License
-------

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in [LICENSE.txt](https://github.com/jonschwenk/RivGraph/blob/master/LICENSE.txt).

RivGraph has been assigned number C19049 by the Feynman Center for Innovation.
