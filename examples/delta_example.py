# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Let's demo RivGraph on the Colville Delta!
# This demo shows some of the core functionality and convenient plotting and exporting features provided by RivGraph for analyzing delta channel networks. The basic steps of RivGraph include:
#
# 1. Instantiate delta class 
# 2. Skeletonize the binary mask 
# 3. Compute the network (links and nodes) 
# 4. Prune the network (requires user-created shoreline and input nodes for deltas) 
# 5. Compute morphologic metrics (lengths, widths) 
# 6. Assign flow directions for each link.
# 7. Compute and plot steady-state flux partitioning.
# 8. Compute some topologic metrics.
#
# Along the way, we'll export some geotiffs and GeoPackages (or shapefiles, if you prefer) for inspection in QGIS. RivGraph requires a **binary mask of the channel network**, preferably georeferenced (i.e., a GeoTiff). For deltas, you will also need to create two vector files: one of the **shoreline**, and one of the **inlet nodes**. See section 4 for guidance on how to create these required geovector files.

# %% [markdown]
# ### 1. Instantiate delta class

# %%
from rivgraph.classes import delta
from rivgraph.deltas.delta_metrics import compute_steady_state_link_fluxes
import matplotlib.pyplot as plt

# Define the path to the georeferenced binary image.
mask_path = "./data/Colville_Delta/Colville_mask.tif"

# Results will be saved with this name
name = 'Colville' 

# Where do you want to store the results? This folder will be created if it doesn't exist.
results_folder = './data/Colville_Delta/Results'

# Boot up the delta class! We set verbose=True to see progress of processing.
colville = delta(name, mask_path, results_folder=results_folder, verbose=True) 

# The mask has been re-binarized and stored as an attribute of colville:
plt.imshow(colville.Imask)

# %% [markdown]
# ### 2. Skeletonize the binary mask

# %%
# Simply use the skeletonize() method.
colville.skeletonize()

# After running, colville has a new attribute: Iskel. Let's take a look.
plt.imshow(colville.Iskel)

# %% [markdown]
# The skeleton is hard to see; perhaps we'd like to look at it closer? One option is to save it as a geotiff and pull it up in a GIS (like QGIS).

# %%
# We use the write_geotiff() method with the "skeleton" option.
colville.to_geotiff('skeleton')

# %% [markdown]
# The georeferenced Colville skeleton has been written to disk, so we can pull it up in QGIS along with the georeferenced mask:
#
# ![colville_qgis_mask_skel_large.PNG](images/colville_qgis_mask_skel_large.png)
#
# Or a bit zoomed-in: 
#
# ![colville_qgis_mask_skel_zoom.PNG](images/colville_qgis_mask_skel_zoom.png)
#

# %% [markdown]
# ### 3. Compute the network (links and nodes)

# %%
# Simply use the compute_network() method.
colville.compute_network()

# %%
# Now we can see that the "links" and "nodes" dictionaries have been added as colville attributes:
links = colville.links
nodes = colville.nodes
print('links: {}'.format(links.keys()))
print('nodes: {}'.format(nodes.keys()))

# %% [markdown]
# The *links* dictionary currently contains four keys: 
# -  <i>idx</i>: a list of all the pixel indices that make up the link (indices created with input mask shape and np.ravel_multi_index)
# - <i>conn</i> : a two-element list containing the node *id*s of the link's endpoints
# - <i>id</i>: each link has a unique *id*; the ordering is irrelevant
# - <i>n_networks</i>: the number of disconnected networks (==1 if the input mask contains a single connected blob)
#     
# The *nodes* dictionary currently contains three keys:
# - <i>idx</i>: the index of the node's position within the original image (i.e. np.ravel_multi_index())
# - <i>conn</i>: an N-element list containing the N link *id*s of the links connected to this node.
# - <i>id</i>: each node has a unique *id*; the ordering is irrelevant
#     
# We can visualze the network in a couple of ways. First, we can plot with matplotlib:

# %%
colville.plot('network')

# %% [markdown]
# Nodes and links are labeled with their ids. Kind of hard to see, so we can zoom in OR we can export the network to geovectors and pull 'em into QGIS:

# %%
colville.to_geovectors('network', ftype='gpkg')  # use GeoPackage by default; GeoJSON requires EPSG:4326 or reproject=True

# Let's see where the network geovector files were written:
print(colville.paths['links'])
print(colville.paths['nodes'])

# %% [markdown]
# And dragging these into QGIS:
# ![colville_network_unpruned.PNG](images/colville_network_unpruned.png)
#
# You can query different links and nodes using the Identify tool. Note that their properties ('conn' and 'id') are appended.

# %% [markdown]
# ### 4. Pruning the network

# %% [markdown]
# You notice in the above image that there are many superfluous links along the shoreline. This is a result of skeletonizing such a massive, connected waterbody (i.e. the ocean in this case). Additionally, the network contains a number of "dangling" links, or those that are connected only at one end. We want to keep the inlet and outlet dangling links, but not the others! RivGraph will automatically prune the network, but it requires (for deltas) two additional pieces of information: the location of the inlet nodes, and a delineation of the shoreline.
# We can create both of these in QGIS:
#
# ![colville_shoreline_inlet_outlet.png](images/colville_shoreline_inlet_outlet.png)
#
# <b>Shoreline</b>: Create a polyline vector layer. The shoreline should be drawn to intersect all the outlet links. It should separate all the unwanted ocean links from the actual links of the delta channel network. If you get errors, you may need to adjust your shoreline a little--try to ensure it does not intersect any nodes!
#
# <b>Inlet nodes</b>: Create a point vector layer. Simply place points at nodes that represent the inlets to the network. The placement does not need to be exact; RivGraph will find the closest node to the one(s) you create. These will be marked as inlet nodes and won't be removed during pruning.
#
# <b>Saving</b>: For convencience, these files should be saved in the Results folder that you initialized the class. Save as <i>results_folder/Colville_shoreline.shp</i> and <i>results_folder/Colville_inlet_nodes.shp</i>. However, this is not mandatory as you can also point to the files during pruning. 
#

# %% [markdown]
# Now that we have identified the shoreline and inlet/outlet nodes, let's prune the network!

# %%
# Note that if the shoreline and inlet nodes shapefiles are in the path_results path, we do not need to specify their locations:
# colville.prune_network()

# However, our files are one directory up, so we need to point to them.
colville.prune_network(path_shoreline='data/Colville_Delta/Colville_shoreline.shp', path_inletnodes='data/Colville_Delta/Colville_inlet_nodes.shp')

# Now that we've pruned, we should re-export the network:
colville.to_geovectors(ftype='gpkg')
# Note that this time we did not specify the export target; by default 'network' will be exported.
# We use GeoPackage here because GeoJSON export now requires EPSG:4326 or reproject=True.


# %% [markdown]
# Let's see how the pruned version compares to the unpruned:
#
# ![colville_shoreline_inlet_outlet_pruned.png](images/colville_shoreline_inlet_outlet_pruned.png)
#
# Wow, we really clipped off a lot of links! We also added some new nodes at the shoreline--notice how each link that intersects the shoreline was truncated, and outlet nodes were placed there (RivGraph remembers which nodes are outlet nodes). You may be concerned that some of the dangling links or subnetworks were pruned--this is by design, and if you want to retain any dangling links, you need to mark their upstream-most nodes as inlet nodes in your shapefile.
#
# Compare with the figure above this one; the set of nodes was also reduced. As links were removed from the network, some nodes were no longer needed as they only connected two links.

# %% [markdown]
# ### 5. Compute morphologic metrics (lengths, widths)
# Now that the network is resolved and pruned, we can compute some link metrics.

# %%
# Compute link widths and lengths
colville.compute_link_width_and_length()

# Lets look at histograms of link widths and lengths:
trash = plt.hist(colville.links['len_adj'], bins=50)
plt.ylabel('count')
plt.xlabel('link length (m)')
plt.title('Histogram of link lengths')

# %% [markdown]
# In the above figure, we see that almost all the links are 1 km or shorter, with three being much longer. This histogram will be different for each delta, and can depend on the resolution of your input binary mask. 
#
# <b>Note</b>: the lengths are reported in <b>meters</b> because that is the unit of the original geotiff CRS. You can check this unit with ```print(colville.unit)```. It is highly unadvisable to use degrees (EPSG:4326 and others) to compute distances.

# %%
print(colville.unit)

# %% [markdown]
# <b>Note</b>: we used the 'len_adj' field rather than the 'len' field. The difference is addressed in a separate Jupyter notebook called XXX. 
#
#
#
# We can do the same for the widths:

# %%
trash = plt.hist(colville.links['wid_adj'], bins=50)
plt.ylabel('count')
plt.xlabel('link width (m)')
plt.title('Histogram of link widths')    

# %% [markdown]
# ### 6. Assign flow directions for each link.
# Now we wish to determine the long-term, steady-state flow direction in each link. The algorithms used here are described in [this paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html).

# %%
colville.assign_flow_directions()

# %% [markdown]
# If RivGraph has any problems assigning link directions, it will let us know. Here, we see no error messages, and a message indicating no cycles were found in the graph. Great! 
#
# We also notice that RivGraph mentiones that a .csv file was created for us to manually set flow directions. If we inspect the flow directions and find some that are incorrect, these can be fixed by entering the link ID and the appropriate upstream node in this .csv, and running ```assign_flow_directions()``` again. See the [braided river example](./braided_river_example.ipynb), section 7 for more details. Note that any links entered into this .csv will be forced to have the upstream node as indicated. RivGraph sets links' directions iteratively, so if you find a problematic area in the link directions (i.e. a number of links whose directions are wrong), you can usually fix it by setting a few key links without needing to flip all of them manually.
#
# Let's look at some plots.
#

# %%
# Plot the links with the directionality marked
colville.plot('directions')

# %% [markdown]
# Links are colored such that upstream is cyan and downstream is purple. Similar to the skeleton, we can export the link directions as a geotiff for inspection in a GIS:

# %%
colville.to_geotiff('directions')

# %% [markdown]
# Pulling this into QGIS and applying a similar color ramp, we see
#
# ![colville_link_directions.PNG](images/colville_link_directions.PNG)
#
# The pixel values along each link have been rescaled from 0 (upstream) to 1 (downstream).
#
# Now that flow directions have been computed, we can also compute junction angles at each node. 
#

# %%
# As of 3/4/2020, this method only computes junction angles at nodes that have exactly three connecting links.
colville.compute_junction_angles(weight=None) # See XXX for a description and meaning of the weight options.

# If we check the the nodes dictionary, we should see that three new fields exist: 'int_ang', 'jtype', and 'width_ratio'.
# 'int_ang' is the junction angle. 'jtype' is either 'b' (bifurcation), 'c' (confluence), or -1 for nodes for which the
# junction angles cannot be computed. 'width_ratio' refers to the ratio between the larger and smaller links.
print(colville.nodes.keys())

# %% [markdown]
# ### 7. Compute and plot steady-state flux partitioning
#
# Once link directions are assigned, we can compute how a unit flux introduced at the inlet(s) partitions through the network under steady-state conditions.
# For the Colville network there are multiple inlets, so we pass ``inlet='equal'`` to partition the unit source flux equally among them.
#
# The resulting link fluxes are stored in ``colville.links['flux_ss']`` and can be exported like any other link attribute.

# %%
colville.links = compute_steady_state_link_fluxes(
    None,
    colville.links,
    colville.nodes,
    weight_name='flux_ss',
    routing='width',
    inlet='equal',
)

# Check that the total outlet flux sums to one.
outlet_flux = 0.0
for conn, flux in zip(colville.links['conn'], colville.links['flux_ss']):
    if conn[1] in colville.nodes['outlets']:
        outlet_flux += flux
print(f"Total outlet flux: {outlet_flux:.6f}")

# %%
# Plot the steady-state flux partitioning.
# We disable the basemap here so the example does not require web tiles or contextily.
fig, ax, links_plot, outlets_plot = colville.plot_fluxes(basemap=False)

# %%
outlets_plot[['node_id', 'outlet_flux']].sort_values('outlet_flux', ascending=False).head()

# %% [markdown]
# Thicker blue lines indicate links carrying more of the steady-state flux, while the outlet markers summarize how much of the unit inlet flux exits at each outlet.
# Because ``flux_ss`` is now attached to ``colville.links``, it will also be included if we export the network geovectors again.

# %% [markdown]
# ### 8. Compute topologic metrics
#
# RivGraph will compute a number of topologic metrics for your delta channel network. These metrics are explained and demonstrated in Tejedor et. al 2015a (doi.org/10.1002/2014WR016577) and 2015b (doi.org/10.1002/2014WR016604). Note that some pre-processing is done to the topology to compute these metrics; it is highly recommended that you understand these preprocessing steps and/or compute the metrics yourself.

# %%
colville.compute_topologic_metrics(inlet='equal') # You may get an overflow warning

# The metrics are stored in an attribute dictionary:
print(colville.topo_metrics.keys())

# %%
# Query different metrics by accessing the dictionary by key.
print(colville.topo_metrics['nonlin_entropy_rate'])

# %%
# Most metrics are computed for each outlet node
print(colville.topo_metrics['top_mutual_info']) # The first column are node IDs, the second are the topological mutual information values.

# %% [markdown]
# If you wish to compute your own metrics or perform topological analyses, you'll probably need an adjacency matrix. RivGraph will provide this with the following method:

# %%
# Unweighted, unnormalized adjacency matrix
adj = colville.adjacency_matrix() 
print(adj)

# %%
# You may also want an adjacency matrix weighted by link width.
adj_w = colville.adjacency_matrix(weight='wid_adj') # Can also weight by 'len_adj' or provide a vector of your own weights.
print(adj_w)

# %%
# And you may want this adjacency matrix normalized.
adj_w_n = colville.adjacency_matrix(weight='wid_adj', normalized=True)
print(adj_w_n) # Each row sums to 1
