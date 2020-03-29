# RivGraph

RivGraph is a python package that assists with converting a binary mask of a channel network into a graph (set of links and nodes). One major component of RivGraph is its ability to automatically set flow directions in each link of the network. It also computes morphologic metrics (lengths, widths, branching angles, etc.) and topologic metrics.

Use RivGraph by creating either a delta or a river class, then applying the associated methods. As there is no documentation as of now, you will have to play with it and dig through the code a bit to understand how to use it. The rivgraph.py file contains these classes, as well as their methods, and should be a great starting place.

8/6/2019 - RivGraph is being prepared for "official release." The code is up-to-date, but has not been cleaned, commented, or sufficiently documented. There is currently [one example](https://github.com/jonschwenk/RivGraph/blob/master/examples/delta_example.py.ipynb) that should get you started; more are underway. 

3/29/2020 - Task list
- [x] Delta example
- [ ] Braided river example
- [ ] How to prepare masks for inputs
- [ ] Where to get masks
- [ ] How to draw shorelines
- [ ] How to fix flow directions
- [ ] Function documentation
- [ ] Unit testing
- [ ] Function for removing artificial nodes


For examples of what RivGraph does, see this AGU poster: https://www.researchgate.net/publication/329845073_Automatic_Extraction_of_Channel_Network_Topology_RivGraph

For description of the flow directionality algorithms, see this paper: https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html.

If you have questions, contact me at j.....k@gmail.com

RivGraph has been assigned number C19049 by the Feynman Center for Innovation.
