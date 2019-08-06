# -*- coding: utf-8 -*-
"""
rivgraph.py
====================================
Classes for running rivgraph commands on your channel network.

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
    """
    The rivnetwork class organizes data and methods for channel networks. This is 
    a parent class to the delta and river classes inherit rivnetwork methods and 
    attributes. This class thus represents the common elements of river and delta 
    channel networks.    
    """

    
    def __init__(self, name, path_to_mask, results_folder=None, filetypes=None, exit_sides=None, verbose=False):
        """
        Initializes a channelnetwork class.
        
        
        Parameters
        ----------
        name : str
            The name of the channel network; also defines the folder name for storing results.
        path_to_mask : str
           Points to the channel network mask file path
        results_folder : str
            Specifies a directory where results should be stored
        filetypes: str
            Sets the filetypes for writing results. Choose from 'GeoJSON' or 'shapefile' right now. Fiona's Geopackage driver does not work.
        exit_sides : str
            Only required for river channel netowrks. A two-character string (from N, E, S, or W) that denotes which sides of the image the river intersects (upstream first) -- e.g. 'NS', 'EW', 'NW', etc.
            
        
        Attributes
        ----------
        name : str
            the name of the channel network, usually the river or delta's name
        verbose : str
            [True] or False to specify if processing updates should be printed.
        gdobj : osgeo.gdal.Dataset
            object created by gdal.Open() that provides access to geotiff metadata
        mask_path : str
            filepath to the input binary channel network mask
        imshape : tuple
            dimensions of the image (rows, cols)
        gt : tuple
            gdal-type Geotransform of the input mask geotiff
        wkt : str
            well known text representation of coordinate reference system of input mask geotiff
        epsg: int
            epsg code of the coordinate reference system of input mask geotiff
        unit: str
            units of the coordinate reference system; typically 'degree' or 'meter'
        pixarea: int or float
            area of each pixel, in units of 'unit'
        pixlen: int or float
            length of each pixel, assuming sides are equal-length
        paths: dict
            dictionary of strings for managing where files should be read/written
        exit_sides: str
            two-character string denoting which sides of the image the channel network intersects (N,E,S, and/or W). Upstream side should be given first.
        Imask: numpy.ndarray
            binary mask found at mask_path loaded into a numpy array via gdal.Open().ReadAsArray(), dtype=np.bool
        links: dict
            Stores the links of the network and associated properties
        nodes: dict
            Stores the nodes of the network and associated properties
        Idist: numpy.ndarray
            image of the distance transform of the binary mask, dtype=np.float

        """

        self.name = name
        self.verbose = verbose
        
        # Get or create georeferencing info
        self.mask_path = path_to_mask
        self.gdobj = gdal.Open(self.mask_path)
        self.imshape = (self.gdobj.RasterYSize, self.gdobj.RasterXSize)
        
        if self.gdobj.GetProjection() == '':
            print('Input mask is unprojected; assigning a dummy projection.')
            # Creates a dummy projection in EPSG:4326 with UL coordinates (0,0) 
            # and pixel resolution = 1. 
            self.gt = (0, 1, 0, 0, 0, -1)
            self.wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]' # 4326
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
            
        if filetypes is None:
            self.ftype = 'GeoJSON' # default
        else:
            self.ftype = filetypes
        
        valid_ftypes = ['GeoJSON', 'shapefile']
        if self.ftype not in valid_ftypes:
            raise ValueError('filetyes muse be one of {}.'.format(valid_ftypes))
        
        # Prepare paths for saving 
        if results_folder is not None:
            self.paths = io.prepare_paths(results_folder, name, path_to_mask, self.ftype)
        else:
            self.paths = io.prepare_paths(os.path.dirname(os.path.abspath(path_to_mask)) , name, path_to_mask, filetypes)
                
        # Save exit sides
        if exit_sides is not None:
            self.exit_sides = exit_sides.lower()
            
        # Load mask into memory
        self.Imask = self.gdobj.ReadAsArray()
        

    def compute_network(self):
        """
        Computes the links and nodes of the channel network mask. First skeletonizes
        the mask, then resolves the skeleton's graph.
        """
        
        if hasattr(self, 'Iskel') is False:
            self.skeletonize()
            
        if self.verbose is True:
            print('Resolving links and nodes...', end='')
            
        self.links, self.nodes = m2g.skel_to_graph(self.Iskel)
        
        if self.verbose is True:
                print('done.')

        
    def compute_distance_transform(self):
        """
        Computes the distance transform of the channel network mask.
        """
        
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
        """
        Computes widths and lengths of each link in the links dictionary and
        appends them as dictionary attributes.
        
        """
        
        if hasattr(self, 'links') is False:
            self.compute_network()
        
        if hasattr(self, 'Idist') is False:
            self.compute_distance_transform()
                        
        # Widths and lengths are appended to links dict
        self.links = lnu.link_widths_and_lengths(self.links, self.Idist, pixlen=self.pixlen)
        
    
    def compute_junction_angles(self, weight=None):
        """
        Computes the angle at nodes where only three links are connected. Directions
        must be assigned before angles can be computed. Also defines each 3-link
        node as 'confluence' or 'bifurcation' and appends this designation to 
        the nodes dictionary.
        
        
        Parameters
        ----------
        weight : str
            [None], 'exp', or 'lin' to determine how to weight the contribution
            of pixels as we move away from the junction node.
            
        """
        
        if 'certain' not in self.links.keys():
            print('Junction angles cannot be computed before link directions are set.')
        else:
            self.nodes = lnu.junction_angles(self.links, self.nodes, self.imshape, self.pixlen, weight=weight)
        
        
    def set_flow_dirs_manually(self):
       """
       Sets flow directions of links specified by .csv file.
       """
       
       try:
           self.links, self.nodes = dd.set_dirs_manually(self.links, self.nodes, self.paths['fixlinks_csv'])
       except AttributeError:
           print('Cannot set flow directions. Ensure network has been computed and pruned.')

        
    def plot(self, *kwargs):
        """
        Generates matplotlib plots of the network. 
        
        Parameters
        ----------
        *kwargs : str
            If [None], both of the following plots will be generated:    
            'network': links and nodes are plotted, labeled with their ids
            'directions': links are plotted with their directionality indicated
            
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
        """
        Writes the link and nodes dictionaries to a .pkl file.
        
        Parameters
        ----------
        path : str
            path--including extension--to network .pkl file. If [None], file 
            written to path found in paths['network_pickle']
        """
        
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
        """
        Loads the link and nodes dictionaries from a .pkl file.
        
        Parameters
        ----------
        path : str
            path--including extension--to network .pkl file. If [None], file 
            is loaded from path found in paths['network_pickle']
        """
       
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
        """
        Returns the adjacency matrix for a graph defined by links and nodes
        dictionaries.
        
        Parameters
        ----------
        weight : str, optional
            [None] or the attribute in the links dictionary to use for weighting links. Typically 'wid_adj' or 'len'.
        normalized : bool, optional
            If True, each row in the adjacency matrix will sum to one. [False] by default.
        
        Returns
        -------
        
        A : numpy.ndarray
             an NxN matrix representing the connectivity of the graph, where N is the
             number of nodes in the network. See adjacency matrix for more details.
            
        """
        # Create (weighted) adjacency matrix networkx object
        G = dm.graphiphy(self.links, self.nodes, weight=weight)
        
        if normalized is True:
            A = dm.normalize_adj_matrix(G)
        else:
            A = nx.to_numpy_array(G)

        return A
    
    
    def to_geovectors(self, *kwargs):
        """
        Writes the links and nodes of the network to geovectors (currently only
        shapefiles supported).
        
        Parameters
        ----------
        *kwargs: str
            If no kwargs provided, both links and nodes written to disk. Paths are
            defined by paths['links'] and paths['nodes']. For river classes,
            'mesh', 'centerline', and 'centerline_smooth' may also be written.
        """
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
                    
                    
    def write_geotiff(self, writedata):
        """
        Writes geotiffs to disk.
        
        Parameters
        ----------
        writedata : {'links', 'skeleton', 'distance'}
                Write geotiff with link directions noted, skeletonized mask, or
                distance transformed mask, respectively.
                
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
    """
    A class to manage and organize data and methods for analyzing a delta channel network. 
    This class inherets all the attributes and methods of the rivnetwork class, but also includes delta-specific attributes and methods.
    
    
    Attributes
    ----------
    Iskel : np.ndarray
        image of the skeletonized binary mask
    topo_metrics : dict
        Contains a number of connectivity and network metrics.   

    """   

    def __init__(self, name, path_to_mask, results_folder=None, filetypes=None, verbose=False):
        
        """
        
        Parameters
        ----------
        name : str
            The name of the delta channel network; also defines the folder name for storing results.
        path_to_mask : str
           Points to the channel network mask file path
        results_folder : str, optional
            Specifies a directory where results should be stored
        verbose : str, optional
            RivGraph will output processing progress if 'True'. Default is 'False'.            
        
        """
        
        rivnetwork.__init__(self, name, path_to_mask, results_folder, verbose=verbose)
            
    
    def skeletonize(self):
        """
        Skeletonizes the delta binary mask.
        """

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
        
        """
        Prunes the delta by removing spurs and links beyond the provided shoreline. Paths may be provided to shoreline and inlet nodes shapefiles, otherwise their location is specified by paths dictionary.
        """
        
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
        """
        Computes flow directions for each link in the delta channel network.
        """        
        
        if hasattr(self, 'links') is False:
            raise AttributeError('Network has not yet been computed.')
            
        if 'inlets' not in self.nodes.keys():
            raise AttributeError('Cannot assign flow direcitons until prune_network has been run.')
        
        if 'len' not in self.links.keys():
            self.compute_link_width_and_length()

        if hasattr(self, 'Idist') is False:
            self.compute_distance_transform()
            
        self.links, self.nodes = dd.set_link_directions(self.links, self.nodes, self.imshape, path_csv=self.paths['fixlinks_csv'])
                    
            
    def compute_topologic_metrics(self):
        """
        Computes a suite of connectivity and network metrics for a delta channel network.
        """
       
        if hasattr(self, 'links') is False:
            raise AttributeError('Network has not yet been computed.')

        if 'certain' not in self.links.keys():           
            raise AttributeError('Link directionality has not been computed.')

        self.topo_metrics = dm.compute_delta_metrics(self.links, self.nodes)

            
class river(rivnetwork):
    """
    A class to manage and organize data and methods for analyzing a braided river channel network. 
    This class inherets all the attributes and methods of the rivnewtwork class, but also includes delta-specific attributes and methods.
    
    ...
    
    Attributes
    ----------
    Iskel : np.ndarray
        image of the skeletonized binary mask
    topo_metrics : dict
        Contains a number of connectivity and network metrics.    
    centerline : tuple of two numpy.ndarrays
        Centerline of the holes-filled river channel network mask. First element in tuple are x-coordinates; second are y-coordinates.
    centerline_smooth : XXX
        A smooth version of the centerline
    max_valley_width_pixels : np.int 
        The maximum valley width in pixels, defined by widths along the centerline       
    width_chans : XXX
        Average channel width
    width_extent: XXX
        Average width of the holes-filled channel mask
    meshlines : XXX
        The lines of the mesh that are perpendicular to the local river direction
    meshpolys : XXX
        Polygons comprising the along-channel mesh
    
    Methods
    -------
    skeletonize()
        Skeletonizes the river binary mask; uses a different method than for deltas.
    prune_network()
        Prunes the river channel network by removing spurs.
    compute_centerline()
        Computes the centerline of the holes-filled river channel network mask.
    compute_mesh(grid_spacing=None, smoothing=0.1, bufferdist=None)
        Computes a mesh that follows the channel centerline; grid_spacing sets the length of each grid cell; bufferdist sets the width of each grid cell.
    assign_flow_direcions()
        Computes flow directions for each link in the delta channel network.
    set_flow_dirs_manually()
        Reads a user-created .csv file found at paths['fixlinks_csv'] to set flow directions of specified liks.
    """   

    def __init__(self, name, path_to_mask, results_folder=None, filetypes=None, exit_sides=None, verbose=False):
        
        if exit_sides is None:
            raise Warning('Must provide exit_sides for river class.') 
        
        rivnetwork.__init__(self, name, path_to_mask, results_folder, filetypes, exit_sides, verbose=verbose)
            
        
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
            
        centerline_pix, valley_centerline_widths = ru.mask_to_centerline(self.Imask, self.exit_sides)
        self.max_valley_width_pixels = np.max(valley_centerline_widths)
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

        self.meshlines, self.meshpolys, self.centerline_smooth = ru.valleyline_mesh(self.centerline, self.width_chans, bufferdist, grid_spacing, smoothing=smoothing)
        
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
                
                
                
            
