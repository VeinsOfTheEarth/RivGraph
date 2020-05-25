# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:50:38 2018

@author: Jon
"""
import os
import gdal
import numpy as np
import networkx as nx
import rivgraph.io_utils as io
import rivgraph.geo_utils as gu
import rivgraph.mask_to_graph as m2g
import rivgraph.ln_utils as lnu

import rivgraph.deltas.delta_utils as du
import rivgraph.deltas.delta_directionality as dd
import rivgraph.deltas.delta_metrics as dm
import rivgraph.rivers.river_directionality as rd
import rivgraph.rivers.river_utils as ru

from scipy.ndimage.morphology import distance_transform_edt

## TODO: TEST this implementation! -- create synthetic georeferencing when non-georeferenced image provided

class rivnetwork:
    
    def __init__(self, name, path_to_mask, path_to_results=None, exit_sides=None, verbose=False):

        self.name = name
        self.verbose = verbose
        
        # Get or create georeferencing info
        self.gdobj = gdal.Open(path_to_mask)
        self.rasterPath = path_to_mask
        self.imshape = (self.gdobj.RasterYSize, self.gdobj.RasterXSize)
        
        if self.gdobj.GetProjection == '':
            print('Input mask is unprojected; assigning a dummy projection.')
            # Creates a dummy projection in EPSG:4326 with UL coordinates (0,0) 
            # and pixel resolution = 1. 
            self.gt = (0, 1, 0, 0, 1, 0)
            self.wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
            self.epsg = 4326
            self.unit = 'degree'
        else:
            self.gt = self.gdobj.GetGeoTransform()
            self.wkt = self.gdobj.GetProjection()
            self.epsg = gu.get_EPSG(self.gdobj)
            self.unit = gu.get_unit(self.epsg)
                
        if self.unit == 'degree':
            self.pixarea = 1
            self.pixlen = 1
        elif self.unit in ['metre', 'meter', 'foot', 'feet']:
            self.pixarea = abs(self.gt[1] * self.gt[5])
            self.pixlen = abs(self.gt[1])
            
        # Prepare paths for saving if path_to_results provided
        if path_to_results is not None:
            self.paths = io.prepare_paths(path_to_results, name, path_to_mask )
            
        # Save exit sides
        if exit_sides is not None:
            self.exit_sides = exit_sides.lower()
            
        # Load mask into memory
        self.Imask = self.gdobj.ReadAsArray()
        

    def compute_network(self):
        
        if hasattr(self, 'Iskel') is False:
            self.skeletonize()
            
        if self.verbose is True:
            print('Resolving links and nodes...', end='')
            
        self.links, self.nodes = m2g.skel_to_graph(self.Iskel)
        
        if self.verbose is True:
                print('done.')

        
    def compute_distance_transform(self):
        
        # Load the distance transform if it already exists
        if 'Idist' in self.paths.keys() and os.path.isfile(self.paths['Idist']) is True:
            self.Idist = gdal.Open(self.paths['Idist']).ReadAsArray()
        else:
            if self.verbose is True:
                print('Computing distance transform...', end='')
                
            self.Idist = distance_transform_edt(self.Imask)
        
            if self.verbose is True:
                print('done.')
                
        
    def compute_link_width_and_length(self):
        
        if hasattr(self, 'links') is False:
            self.compute_network()
        
        if hasattr(self, 'Idist') is False:
            self.compute_distance_transform()
                        
        # Widths and lengths are appended to links dict
        self.links = lnu.link_widths_and_lengths(self.links, self.Idist, pixlen=self.pixlen)
        
    
    def compute_junction_angles(self, weight=None):
        """
        weight refers to how much weight should be given to pixels farther away
        from the junction node.
        """
        
        if 'certain' not in self.links.keys():
            print('Junction angles cannot be computed before link directions are set.')
        else:
            self.nodes = lnu.junction_angles(self.links, self.nodes, self.imshape, self.pixlen, weight=weight)
        
        
    def plot(self, *kwargs):
        """
        Plots the network. Optional kwargs are
        'network' - plots labeled links and nodes
        'directions' - plots links with directionality indicated
        By default, both will be plotted.
        """
        ## TODO: add error handling for wrong plotting commands
        
        plt_directions, plt_network = False, False
        if len(kwargs) == 0:
            plt_directions = True
            plt_network = True
        if 'network' in kwargs:
            plt_network = True
        if 'directions' in kwargs:
            plt_directions = True

        if hasattr(self, 'links') is False:
            print('Network has not been computed yet; cannot plot.')
            return
        
        if plt_directions is True:
            if 'certain' not in self.links.keys():
                print('Must assign link directions before plotting link directions.')
                return

            else:
                lnu.plot_dirlinks(self.links, self.imshape)
                
        if plt_network is True:
            lnu.plot_network(self.links, self.nodes, self.Imask, self.name)       
        
            
    def save_network(self, path=None):
        
        if path==None and hasattr(self, 'paths') is False:      
            print('No path is available to load the network.')
        elif path is None:
            path = self.paths['network_pickle']
            try:
                io.pickle_links_and_nodes(self.links, self.nodes, path)
                print('Links and nodes saved to pickle file: {}.'.format(self.paths['network_pickle']))
            except AttributeError:
                print('Network has not been computed yet. Use the compute_network() method first.')
            
    
    def load_network(self, path=None):
        
        if path==None and hasattr(self, 'paths') is False:
            print('No path is available to load the network.')
            return
            
        if path is None:
            path = self.paths['network_pickle']
            
        if os.path.isfile(path) is False:
                print('No file was found at provided path: {}.'.format(path))
        else:
            self.links, self.nodes = io.unpickle_links_and_nodes(path)       
                 

    def adjacency_matrix(self, weight=None, normalized=False):
        
        # Created (weighted) adjacency matrix networkx object
        G = dm.graphiphy(self.links, self.nodes, weight=weight)
        
        if normalized is True:
            A = dm.normalize_adj_matrix(G)
        else:
            A = nx.to_numpy_array(G)

        return A
    
    
    def to_geovectors(self, *kwargs):
        
        if len(kwargs) == 0:
            try:
                # Save nodes
                io.nodes_to_geofile(self.nodes, self.imshape, self.gt, self.epsg, self.paths['nodes'])
                
                # Save links
                io.links_to_geofile(self.links, self.imshape, self.gt, self.epsg, self.paths['links'])
                print('Links and nodes saved to geofiles: {}, {}.'.format(self.paths['nodes'], self.paths['links']))
                    
            except AttributeError:
                print('Links and nodes could not be saved. Ensure network has been computed.')             
        else:
            for kw in kwargs:
                if kw == 'links':
                    io.links_to_geofile(self.links, self.imshape, self.gt, self.epsg, self.paths['links'])
                elif kw == 'nodes':
                    io.nodes_to_geofile(self.nodes, self.imshape, self.gt, self.epsg, self.paths['nodes'])
                elif kw == 'mesh':
                    io.meshlines_to_shapefile(self.meshlines, self.epsg, self.paths['meshlines'], nameid=self.name)
                    io.meshpolys_to_geovectors(self.meshpolys, self.epsg, self.paths['meshpolys'])
                elif kw == 'centerline':
                    io.centerline_to_geovector(self.centerline, self.epsg, self.paths['centerline'])
                elif kw == 'centerline_smooth':
                    io.centerline_to_geovector(self.centerline_smooth, self.epsg, self.paths['centerline_smooth'])
                    
                    
    def write_geotiff(self, writedata='skeleton'):
        """
        Options for "writedata": 
            'links' - geotiff with link directions denoted
            'skeleton' - skeletonized mask
            'distance' - distance transformed mask
        """
        
        possdata = ['links', 'distance', 'skeleton']
        if writedata not in possdata:
            print('Cannot write {}. Choose from {}.'.format(writedata, possdata))
            return
        
        if writedata == 'links':
            io.write_linkdirs_geotiff(self.links, self.gdobj, self.paths['linkdirs'])
            return
        
        if writedata == 'distance':
            raster = self.Idist
            outpath = self.paths['Idist']
            dtype = gdal.GDT_Float32
            color_table = 0
            nbands = 1
        elif writedata == 'skeleton':
            raster = self.Iskel
            outpath = self.paths['Iskel']
            dtype = gdal.GDT_Byte
            color_table = io.colortable('skel')
            nbands = 1
        
        io.write_geotiff(raster, self.gt, self.wkt, outpath, dtype=dtype, color_table=color_table, nbands=nbands)
        
        print('Geotiff written to {}.'.format(outpath))

            

class delta(rivnetwork):
    
    def __init__(self, name, path_to_mask, path_to_results=None, verbose=False):
        
        rivnetwork.__init__(self, name, path_to_mask, path_to_results, verbose=verbose)
            
    
    def skeletonize(self):
        
        if hasattr(self, 'Imask') is False:
            raise AttributeError('Mask array was not provided or was unreadable.')
            
        # Load the skeleton if it already exists
        if 'Iskel' in self.paths.keys() and os.path.isfile(self.paths['Iskel']) is True:
            self.Iskel = gdal.Open(self.paths['Iskel']).ReadAsArray()

        else:
            if self.verbose is True:
                print('Skeltonizing mask...', end='')
    
            self.Iskel = m2g.skeletonize_mask(self.Imask)
            
            if self.verbose is True:
                print('done.')


    def prune_network(self, path_shoreline=None, path_inletnodes=None):
        
        try:
            if path_shoreline is None:
                path_shoreline = self.paths['shoreline']
        except AttributeError:
            raise AttributeError('Could not find shoreline shapefile which should be at {}.'.format(self.paths['shoreline']))
           
        try:
            if path_inletnodes is None:
                path_inletnodes = self.paths['inlet_nodes']
        except AttributeError:
            raise AttributeError('Could not inlet_nodes shapefile which should be at {}.'.format(self.paths['inlet_nodes']))
   
        self.links, self.nodes = du.prune_delta(self.links, self.nodes, path_shoreline, path_inletnodes, self.gdobj)
            
    
    def assign_flow_directions(self):
        
        if hasattr(self, 'links') is False:
            raise AttributeError('Network has not yet been computed.')
            
        if 'inlets' not in self.nodes.keys():
            raise AttributeError('Cannot assign flow direcitons until prune_network has been run.')
        
        if 'len' not in self.links.keys():
            self.compute_link_width_and_length()

        if hasattr(self, 'Idist') is False:
            self.compute_distance_transform()
            
        self.links, self.nodes = dd.set_link_directions(self.links, self.nodes, self.imshape, path_csv=self.paths['fixlinks_csv'])
                    
            
    def set_flow_dirs_manually(self):
       
        try:
            self.links, self.nodes = dd.set_dirs_manually(self.links, self.nodes, self.paths['fixlinks_csv'])
        except AttributeError:
            print('Cannot set flow directions. Ensure network has been computed and pruned.')

    def compute_topologic_metrics(self):
       
        if hasattr(self, 'links') is False:
            raise AttributeError('Network has not yet been computed.')

        if 'certain' not in self.links.keys():           
            raise AttributeError('Link directionality has not been computed.')

        self.topo_metrics = dm.compute_delta_metrics(self.links, self.nodes)

            
class river(rivnetwork):
    
    def __init__(self, name, path_to_mask, path_to_results=None, exit_sides=None, verbose=False):
        
        if exit_sides is None:
            raise Warning('Must provide exit_sides for river class.') 
        
        rivnetwork.__init__(self, name, path_to_mask, path_to_results, exit_sides, verbose=verbose)
            
        
    def skeletonize(self):
        
        if hasattr(self, 'Imask') is False:
            raise AttributeError('Mask array was not provided or was unreadable.')
            
        # Load the skeleton if it already exists
        if 'Iskel' in self.paths.keys() and os.path.isfile(self.paths['Iskel']) is True:
            self.Iskel = gdal.Open(self.paths['Iskel']).ReadAsArray()

        else:
            if self.verbose is True:
                print('Skeltonizing mask...', end='')
    
            self.Iskel = m2g.skeletonize_river_mask(self.Imask, self.exit_sides)
            
            if self.verbose is True:
                print('done.')

            
    def prune_network(self):
        
        if hasattr(self, 'links') is False:
            raise AttributeError('Could not prune river. Check that network has been computed.')
            
        if hasattr(self, 'Iskel') is False:
            self.skeletonize()
                   
        self.links, self.nodes = ru.prune_river(self.links, self.nodes, self.exit_sides, self.Iskel, self.gdobj)


    def compute_centerline(self):
        
        if self.verbose is True:
            print('Computing centerline...', end='')
            
        centerline_pix, self.max_valley_width_pixels = ru.mask_to_centerline(self.Imask, self.exit_sides)
        self.centerline = gu.xy_to_coords(centerline_pix[:,0], centerline_pix[:,1], self.gt)
        
        if self.verbose is True:
            print('done.')
        
    
    def compute_mesh(self, grid_spacing=None, smoothing=0.1, bufferdist=None):
       
        if hasattr(self, 'centerline') is False:
            self.compute_centerline()
                            
        # Need channel widths for parameterizing mesh generation
        if hasattr(self, 'width_chans') is False:
            self.width_chans, self.width_extent = ru.chan_width(self.centerline, self.Imask, pixarea=self.pixarea)
                
        # If not specified, grid spacing is set based on distribution of link lengths        
        if grid_spacing is None:
            
            # Need link widths to parameterize mesh generation (spacing)
            if 'len' not in self.links.keys():
                self.compute_link_width_and_length()
                
            grid_spacing = np.percentile(self.links['len'],25)
        
        # If bufferdistance not specified, set it to 10% larger than the maximum valley width
        if bufferdist is None:
            bufferdist = self.max_valley_width_pixels * self.pixlen * 1.1

        if self.verbose is True:
            print('Generating mesh...', end='')

        self.meshlines, self.meshpolys, self.centerline_smooth = ru.centerline_mesh(self.centerline, self.width_chans, bufferdist, grid_spacing, smoothing=smoothing)
        
        if self.verbose is True:
            print('done.')

    
    def assign_flow_directions(self):
        
        if 'inlets' not in self.nodes.keys():
            raise AttributeError('Cannot assign flow directions until prune_network has been run.')
        
        if hasattr(self, 'centerline') is False:
            self.compute_centerline()
        
        if hasattr(self, 'meshpolys') is False:
            self.compute_mesh()
            
        if hasattr(self, 'Idist') is False:
            self.compute_distance_transform()
        
        if self.verbose is True:
            print('Setting link directionality...', end='')
            
        self.links, self.nodes = rd.set_directionality(self.links, self.nodes, self.Imask, self.exit_sides, self.gt, self.meshlines, self.meshpolys, self.Idist, self.pixlen, self.paths['fixlinks_csv'])

        if self.verbose is True:
            print('done.')
                
                
                
            
