# -*- coding: utf-8 -*-
"""
RivGraph (classes.py)
=====================
Classes for running rivgraph commands on your channel network or centerline.

"""
import os
import sys
from loguru import logger
try:
    from osgeo import gdal
except ModuleNotFoundError:
    import gdal
import numpy as np
import networkx as nx
from pyproj.crs import CRS
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from scipy import signal
import rivgraph.io_utils as io
import rivgraph.geo_utils as gu
import rivgraph.mask_to_graph as m2g
import rivgraph.ln_utils as lnu
import rivgraph.mask_utils as mu
import rivgraph.deltas.delta_utils as du
import rivgraph.deltas.delta_directionality as dd
import rivgraph.deltas.delta_metrics as dm
import rivgraph.rivers.river_directionality as rd
import rivgraph.rivers.river_utils as ru
import rivgraph.rivers.centerline_utils as cu

class rivnetwork:
    """
    Base rivnetwork class.

    The rivnetwork class organizes data and methods for channel networks. This
    is a parent class to the delta and river classes which inherit rivnetwork
    methods and attributes. This class thus represents the common elements of
    river and delta channel networks.

    """


    def __init__(self, name, path_to_mask, results_folder=None,
                 exit_sides=None, verbose=False):
        """
        Initializes a channelnetwork class.


        Parameters
        ----------
        name : str
            The name of the channel network; also defines the folder name for
            storing results.
        path_to_mask : str
            Points to the channel network mask file path
        results_folder : str, optional
            Specifies a directory where results should be stored
        exit_sides : str, optional
            Only required for river channel netowrks. A two-character string
            (from N, E, S, or W) that denotes which sides of the image the
            river intersects (upstream first) -- e.g. 'NS', 'EW', 'NW', etc.
        verbose : bool, optional
            If True, print run information and warnings to the console, default
            is False.


        Attributes
        ----------
        name : str
            the name of the channel network, usually the river or delta's name
        verbose : bool, optional (False by default)
            True or False to specify if processing updates should be printed.
        d : osgeo.gdal.Dataset
            object created by gdal.Open() that provides access to geotiff
            metadata
        mask_path : str
            filepath to the input binary channel network mask
        imshape : tuple
            dimensions of the image (rows, cols)
        gt : tuple
            gdal-type Geotransform of the input mask geotiff
        wkt : str
            well known text representation of coordinate reference system of
            input mask geotiff
        epsg: int
            epsg code of the coordinate reference system of input mask geotiff
        unit: str
            units of the coordinate reference system; typically 'degree' or
            'meter'
        pixarea: int or float
            area of each pixel, in units of 'unit'
        pixlen: int or float
            length of each pixel, assumes sides are equal-length
        paths: dict
            dictionary of strings for managing where files should be
            read/written
        exit_sides: str
            two-character string denoting which sides of the image the channel
            network intersects (N,E,S, and/or W). Upstream side should be given
            first.
        Imask: numpy.ndarray
            binary mask found at mask_path loaded into a numpy array via
            `gdal.Open().ReadAsArray()`, dtype=np.bool
        links: dict
            Stores the links of the network and associated properties
        nodes: dict
            Stores the nodes of the network and associated properties
        Idist: numpy.ndarray
            image of the distance transform of the binary mask, dtype=np.float

        """
        # Store some class attributes
        self.name = name
        self.verbose = verbose

        # Prepare paths for saving
        if results_folder is not None:
            self.paths = io.prepare_paths(results_folder, name, path_to_mask)
        else:
            self.paths = io.prepare_paths(
                            os.path.dirname(
                                os.path.abspath(path_to_mask)), name,
                            path_to_mask)
        self.paths['input_mask'] = os.path.normpath(path_to_mask)

        # init logger - prints out to stdout if verbose is True
        # ALWAYS writes output to log file (doesn't print if verbose is False)
        self.init_logger()

        # Handle georeferencing
        # GA_Update required for setting dummy projection/geotransform
        self.gdobj = gdal.Open(self.paths['input_mask'], gdal.GA_Update)
        self.imshape = (self.gdobj.RasterYSize, self.gdobj.RasterXSize)

        # Create dummy georeferencing if none is supplied
        if self.gdobj.GetProjection() == '':
            logger.info('Input mask is unprojected; assigning a dummy projection.')
            # Creates a dummy projection in EPSG:4326 with UL coordinates (0,0)
            # and pixel resolution = 1.
            self.wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]' # 4326
            self.gdobj.SetProjection(self.wkt)
            self.gdobj.SetGeoTransform((0, 1, 0, self.imshape[1], 0, -1))
        else:
            self.wkt = self.gdobj.GetProjection()
        self.gt = self.gdobj.GetGeoTransform()

        # Store crs as pyproj CRS object for interacting with geopandas
        self.crs = CRS(self.gdobj.GetProjection())
        self.unit = gu.get_unit(self.crs)

        self.pixarea = abs(self.gt[1] * self.gt[5])
        self.pixlen = abs(self.gt[1])

        # Save exit sides
        if exit_sides is not None:
            self.exit_sides = exit_sides.lower()

        # Load mask into memory
        self.Imask = self.gdobj.ReadAsArray()


    def init_logger(self):
        """Function to initialize the logger."""
        if self.verbose is True:
            logger.configure(
                handlers=[
                    dict(sink=self.paths['log'],
                         format="[{time:YYYY-MM-DD at HH:mm:ss}] | {message}"),
                    dict(sink=sys.stdout,
                         format="{message}")
                ],
                activation=[("", True)],
            )
        else:
            logger.configure(
                handlers=[
                    dict(sink=self.paths['log'],
                         format="[{time:YYYY-MM-DD at HH:mm:ss}] | {message}"),
                ],
                activation=[("", True)],
            )
        logger.info("-"*10 + " New Run " + "-"*10)


    def compute_network(self):
        """
        Computes the links and nodes of the channel network mask.
        First skeletonizes the mask if not already done, then resolves the
        skeleton's graph.

        """
        if hasattr(self, 'Iskel') is False:
            self.skeletonize()

        logger.info('Resolving links and nodes...', end='')

        self.links, self.nodes = m2g.skel_to_graph(self.Iskel)

        logger.info('links and nodes have been resolved.')


    def compute_distance_transform(self):
        """
        Computes the distance transform of the channel network mask.

        """
        # Load the distance transform if it already exists
        if 'Idist' in self.paths.keys() and \
            os.path.isfile(self.paths['Idist']) is True:
            self.Idist = gdal.Open(self.paths['Idist']).ReadAsArray()
        else:
            logger.info('Computing distance transform...', end='')

            self.Idist = distance_transform_edt(self.Imask)

            logger.info('distance transform done.')


    def compute_link_width_and_length(self):
        """
        Computes widths and lengths of each link in the links dictionary and
        appends them as dictionary attributes.

        """
        if hasattr(self, 'links') is False:
            self.compute_network()

        if hasattr(self, 'Idist') is False:
            self.compute_distance_transform()

        logger.info('Computing link widths and lengths...')

        # Widths and lengths are appended to links dict
        self.links = lnu.link_widths_and_lengths(self.links, self.Idist,
                                                 pixlen=self.pixlen)

        logger.info('link widths and lengths computed.')


    def compute_junction_angles(self, weight=None):
        """
        Computes the angle at nodes where only three links are connected.
        Directions must be assigned before angles can be computed. Also defines
        each 3-link node as 'confluence' or 'bifurcation' and appends this
        designation to the nodes dictionary.

        Parameters
        ----------
        weight : str
            [None], 'exp' (exponential), or 'lin' (linear) to determine the
            decay of the weights the contributions of pixels as we move away
            from the junction node.

        """
        if 'certain' not in self.links.keys():
            logger.info('Junction angles cannot be computed before link directions are set.')
        else:
            self.nodes = lnu.junction_angles(self.links, self.nodes,
                                             self.imshape, self.pixlen,
                                             weight=weight)


    def get_islands(self, props=['area', 'maxwidth', 'major_axis_length',
                                 'minor_axis_length', 'surrounding_links'],
                          connectivity=2):
        """
        Finds all the islands in the binary mask and computes their morphological
        properties. Can be used to help "clean" masks of small islands. Must
        run compute_network() first.

        Parameters
        ----------
        props : list, optional
            Properties to compute for each island. Properties can be any of those
            provided by rivgraph.im_utils.regionprops.
            The default is ['area', 'maxwidth', 'major_axis_length', 'minor_axis_length'].
        connectivity : int, optional
            If 1, 4-connectivity will be used to determine connected blobs. If
            2, 8-connectivity will be used. The default is 2.

        Returns
        -------
        islands : geopandas GeoDataFrame
             Contains the polygons of each island with the requested property
             attributes as columns. An additional 'remove' attribute is
             initialized to make thresholding easier.

        """
        do_surr = False
        if 'surrounding_links' in props:
            props.remove('surrounding_links')
            if hasattr(self, 'links') is True:
                do_surr = True
            else:
                logger.info('Cannot compute surrounding island links without first computing the network. Skipping.')

        logger.info('Getting island properties...')

        islands, Iislands = mu.get_island_properties(self.Imask, self.pixlen, self.pixarea, self.crs, self.gt, props, connectivity=connectivity)

        logger.info('got island properties.')

        if do_surr is True:
            if hasattr(self.links, 'wid_adj') is False:
                self.compute_link_width_and_length()

            logger.info('Computing surrounding links for each island...')

            islands = mu.surrounding_link_properties(self.links, self.nodes, self.Imask, islands, Iislands, self.pixlen, self.pixarea)

            logger.info('surrounding links computed.')

        # Add a column to be used for thresholding
        islands['remove'] = [False for i in range(len(islands))]

        return islands, Iislands


    def plot(self, *kwargs, axis=None):
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
            logger.info('No path is available to load the network.')
            return

        if plt_directions is True:
            if 'certain' not in self.links.keys():
                print('Must assign link directions before plotting link directions.')
                return
            else:
                d = lnu.plot_dirlinks(self.links, self.imshape)
                return d

        if plt_network is True:
            f = lnu.plot_network(self.links, self.nodes, self.Imask, self.name, axis=axis)
            return f


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
                logger.info('Links and nodes saved to pickle file: {}.'.format(self.paths['network_pickle']))
            except AttributeError:
                logger.info('Network has not been computed yet. Use the compute_network() method first.')


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
            logger.info('No path is available to load the network.')
            return

        if path is None:
            path = self.paths['network_pickle']

        if os.path.isfile(path) is False:
                logger.info('No file was found at provided path: {}.'.format(path))
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
            an NxN matrix representing the connectivity of the graph, where N
            is the number of nodes in the network. See adjacency matrix for more details.

        """
        # Create (weighted) adjacency matrix networkx object
        G = dm.graphiphy(self.links, self.nodes, weight=weight)

        if normalized is True:
            A = dm.normalize_adj_matrix(G)
        else:
            A = nx.to_numpy_array(G)

        return A


    def to_geovectors(self, export='network', ftype='json'):
        """
        Writes the links and nodes of the network to geovectors.

        Parameters
        ----------
        export : str
            Determines which features to export. Choose from:

            - all (exports all available vector data)

            - network (links and nodes)

            - links

            - nodes

            - centerline (river classes only)

            - mesh (centerline mesh, river classes only)

            - centerline_smooth (river classes only)

        ftype : str
            Sets the output file format. Choose from:

            - json (GeoJSON)

            - shp  (ESRI Shapefile)

        """
        # Get extension for requested output type
        if ftype == 'json':
            ext = 'json'
        elif ftype == 'shp':
            ext = 'shp'
        else:
            raise TypeError('Only json and shp output types are supported.')

        # Prepare list of desired exports
        if export == 'all':
            to_export = ['links', 'nodes', 'mesh', 'centerline', 'centerline_smooth']
        elif export == 'network':
            to_export = ['links', 'nodes']
        else:
            to_export = [export]

        # Ensure that each requested vector dataset has been computed, then export it
        for te in to_export:
            if te == 'links':
                if hasattr(self, 'links') is True:
                    self.paths['links'] = os.path.join(self.paths['basepath'], self.name + '_links.' + ext)
                    io.links_to_geofile(self.links, self.imshape, self.gt, self.crs, self.paths['links'])
                else:
                    logger.info('Links have not been computed and thus cannot be exported.')
            if te == 'nodes':
                if hasattr(self, 'nodes') is True:
                    self.paths['nodes'] = os.path.join(self.paths['basepath'], self.name + '_nodes.' + ext)
                    io.nodes_to_geofile(self.nodes, self.imshape, self.gt, self.crs, self.paths['nodes'])
                else:
                    logger.info('Nodes have not been computed and thus cannot be exported.')
            if te == 'mesh':
                if hasattr(self, 'meshlines') is True and type(self) is river:
                    self.paths['meshlines'] = os.path.join(self.paths['basepath'], self.name + '_meshlines.' + ext)
                    self.paths['meshpolys'] = os.path.join(self.paths['basepath'], self.name + '_meshpolys.' + ext)
                    io.shapely_list_to_geovectors(self.meshlines, self.crs, self.paths['meshlines'])
                    io.shapely_list_to_geovectors(self.meshpolys, self.crs, self.paths['meshpolys'])
                else:
                    logger.info('Mesh has not been computed and thus cannot be exported.')
            if te == 'centerline':
                if hasattr(self, 'centerline') is True and type(self) is river:
                    self.paths['centerline'] = os.path.join(self.paths['basepath'], self.name + '_centerline.' + ext)
                    io.centerline_to_geovector(self.centerline, self.crs, self.paths['centerline'])
                else:
                    logger.info('Centerlines has not been computed and thus cannot be exported.')
            if te == 'centerline_smooth':
                if hasattr(self, 'centerline_smooth') is True and type(self) is river:
                    self.paths['centerline_smooth'] = os.path.join(self.paths['basepath'], self.name + '_centerline_smooth.' + ext)
                    io.centerline_to_geovector(self.centerline_smooth, self.crs, self.paths['centerline_smooth'])
                else:
                    logger.info('Smoothed centerline has not been computed and thus cannot be exported.')


    def to_geotiff(self, export):
        """
        Writes geotiffs to disk.

        Parameters
        ----------
        export : str
            Select a raster to write to geotiff. Choose from:
                'directions' - network burned into a raster with link directions from 0 (upstream) to 1 (downstream))
                'skeleton'  - skeletonized mask
                'distance' - distance-transformed mask

        """
        valid_exports = ['directions', 'distance', 'skeleton']
        if export not in valid_exports:
            logger.info('Cannot write {}. Choose from {}.'.format(export, valid_exports))
            return

        if export == 'directions':
            outpath = self.paths['linkdirs']
            io.write_linkdirs_geotiff(self.links, self.gdobj, outpath)
        else:
            if export == 'distance':
                raster = self.Idist
                outpath = self.paths['Idist']
                dtype = gdal.GDT_Float32
                color_table = None
                options = None
                nbands = 1
            elif export == 'skeleton':
                raster = self.Iskel
                outpath = self.paths['Iskel']
                dtype = gdal.GDT_Byte
                color_table = io.colortable('skel')
                options=['COMPRESS=LZW']
                nbands = 1

            io.write_geotiff(raster, self.gt, self.wkt, outpath, dtype=dtype, options=options, color_table=color_table, nbands=nbands)

        logger.info('Geotiff written to {}.'.format(outpath))


class delta(rivnetwork):
    """
    A class to manage and organize data and methods for analyzing a delta channel network.
    This class inherets all the attributes and methods of the rivnetwork class,
    but also includes delta-specific attributes and methods.


    Attributes
    ----------
    Iskel : np.ndarray
        image of the skeletonized binary mask
    topo_metrics : dict
        Contains a number of connectivity and network metrics.

    """

    def __init__(self, name, path_to_mask, results_folder=None, verbose=False):
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
            logger.info('Skeletonizing mask...')

            self.Iskel = m2g.skeletonize_mask(self.Imask)

            logger.info('done skeletonization.')


    def prune_network(self, path_shoreline=None, path_inletnodes=None,
                      prune_less=False):
        """
        Prunes the delta by removing spurs and links beyond the provided shoreline.
        Paths may be provided to shoreline and inlet nodes shapefiles, otherwise
        their location is specified by paths dictionary.


        Parameters
        ----------
        path_shoreline : str, optional
            Path to shoreline shapefile/geosjon. The default is None but will
            check for the file at `paths['shoreline']`.
        path_inletnodes : str, optional
            Path to inlet nodes shapefile/geojson. The default is None but will
            check for the file at `paths['inlet_nodes']`.
        prune_less : bool, optional
            Boolean to optionally prune the network less. The first spur
            removal can create problems, especially for very small/simple
            networks. Default behavior is encouraged, but in the event a bug
            is encountered, toggling this parameter to True may fix the issue.
            Default is False (more pruning).

        Returns
        -------
        :
            None, but saves pruned links and nodes dictionaries to class object.

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

        self.links, self.nodes = du.prune_delta(self.links, self.nodes, path_shoreline, path_inletnodes, self.gdobj, prune_less)


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

        self.links, self.nodes = dd.set_link_directions(self.links, self.nodes, self.imshape, manual_set_csv=self.paths['fixlinks_csv'])


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
    This class inherets all the attributes and methods of the rivnetwork class, but also includes delta-specific attributes and methods.


    Attributes
    ----------
    Iskel : np.ndarray
        Image of the skeletonized binary mask
    topo_metrics : dict
        Contains a number of connectivity and network metrics.
    centerline : tuple of two numpy.ndarrays
        Centerline of the holes-filled river channel network mask. First element in tuple are x-coordinates; second are y-coordinates.
    centerline_smooth : shapely.geometry.LineString
        A smooth version of centerline
    max_valley_width_pixels : np.int
        The maximum valley width in pixels, defined by widths along the centerline
    width_chans : float
        Average channel width
    width_extent: float
        Average width of the holes-filled channel mask
    meshlines : list of shapely.geometry.LineString
        The lines of the mesh that are perpendicular to the local river direction
    meshpolys : list of shapely.geometry.Polygon
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
        Reads a user-created .csv file found at `paths['fixlinks_csv']` to set flow directions of specified links.

    """

    def __init__(self, name, path_to_mask, results_folder=None,
                 exit_sides=None, verbose=False):

        if exit_sides is None:
            raise Warning('Must provide exit_sides for river class.')

        rivnetwork.__init__(self, name, path_to_mask, results_folder, exit_sides, verbose=verbose)


    def skeletonize(self):
        """
        Skeletonizes the river binary mask.

        """
        if hasattr(self, 'Imask') is False:
            raise AttributeError('Mask array was not provided or was unreadable.')

        # Load the skeleton if it already exists
        if 'Iskel' in self.paths.keys() and os.path.isfile(self.paths['Iskel']) is True:
            self.Iskel = gdal.Open(self.paths['Iskel']).ReadAsArray()

        else:
            logger.info('Skeletonizing mask...')

            self.Iskel = m2g.skeletonize_river_mask(self.Imask, self.exit_sides)

            logger.info('skeletonization is done.')


    def prune_network(self):
        """
        Prunes the computed river network.

        """
        if hasattr(self, 'links') is False:
            raise AttributeError('Could not prune river. Check that network has been computed.')

        if hasattr(self, 'Iskel') is False:
            self.skeletonize()

        self.links, self.nodes = ru.prune_river(self.links, self.nodes, self.exit_sides, self.Iskel, self.gdobj)


    def compute_centerline(self):
        """
        Computes the centerline of the holes-filled river binary image.

        """
        logger.info('Computing centerline...')

        centerline_pix, valley_centerline_widths = ru.mask_to_centerline(self.Imask, self.exit_sides)
        self.max_valley_width_pixels = np.max(valley_centerline_widths)
        self.centerline = gu.xy_to_coords(centerline_pix[:,0], centerline_pix[:,1], self.gt)

        logger.info('centerline computation is done.')


    def compute_mesh(self, grid_spacing=None, smoothing=0.1, buf_halfwidth=None):
        """
        Generates an along-centerline mesh that indicates a valley-direction
        of sorts. The mesh is useful for computing spatial statistics as a function
        of downstream distance. The resulting mesh captures the low-frequency
        characteristic of the river corridor.

        This tool is tricky to fully automate, and the user may need to play
        with the smoothing and bufferdist parameters if errors are thrown or
        the result is not satisfying.

        Parameters
        ----------
        grid_spacing : float
            Defines the distance between perpendicular-to-centerline transects.
            Units are defined by input mask CRS.
        smoothing : float
            Defines the smoothing window of the left- and right-valleylines as a fraction
            of the total centerline length. Range is [0, 1].
        buf_halfwidth : float
            Defines the offset distance of the left- and right-valleylines from
            from the centerline. Units correspond to those of the CRS of the
            input mask.

        """
        # Need a centerline
        if hasattr(self, 'centerline') is False:
            self.compute_centerline()

        # Need average channel width for parameterizing mesh generation
        if hasattr(self, 'avg_chan_width') is False:
            if hasattr(self, 'links') is False:
                self.compute_network()
            if hasattr(self.links, 'wid_adj') is False:
                self.compute_link_width_and_length()

            # self.avg_chan_width = np.mean(self.links['wid_a1dj'])
            self.avg_chan_width = np.sum(self.Imask) * self.pixarea / np.sum(self.links['len_adj'])

        # If not specified, grid spacing is set to one channel width
        if grid_spacing is None:
            grid_spacing = self.avg_chan_width

        # If buffer halfwidth is not specified, it is set to 10% larger than the maximum valley width
        if buf_halfwidth is None:
            # Compute the maximum valley width in pixels
            if hasattr(self, 'max_valley_width_pixels') is False:

                logger.info('Computing maximum valley width...')

                self.max_valley_width_pixels = ru.max_valley_width(self.Imask)

                logger.info('valley width computation is done.')

            # Multiply by pixlen to keep units consistent
            buf_halfwidth = self.max_valley_width_pixels * self.pixlen * 1.1

        logger.info('Generating mesh...')

        self.meshlines, self.meshpolys, self.centerline_smooth = ru.valleyline_mesh(self.centerline, self.avg_chan_width, buf_halfwidth, grid_spacing, smoothing=smoothing)

        logger.info('mesh generation is done.')


    def assign_flow_directions(self):
        """
        Automatically sets flow directions for each link in a braided river
        channel network.

        """
        if 'inlets' not in self.nodes.keys():
            raise AttributeError('Cannot assign flow directions until prune_network() has been run.')

        if hasattr(self, 'centerline') is False:
            self.compute_centerline()

        if hasattr(self, 'meshpolys') is False:
            self.compute_mesh()

        if hasattr(self, 'Idist') is False:
            self.compute_distance_transform()

        logger.info('Setting link directionality...')

        self.links, self.nodes = rd.set_directionality(self.links, self.nodes, self.Imask, self.exit_sides, self.gt, self.meshlines, self.meshpolys, self.Idist, self.pixlen, self.paths['fixlinks_csv'])

        logger.info('link directionality has been set.')


class centerline():

    def __init__(self, x, y, attribs=None, crs=None):
        """
        attribs is a dictionary with attributes; can be single values like
        average channel width or one value per coordinate like local width.
        """
        # Store original coordinates
        self.xo = x
        self.yo = y

        # Store crs info if provided
        self.crs = crs

        # Store attributes
        if attribs:
            for a in attribs.keys():

                try:
                    alen = len(attribs[a])
                except Exception:
                    alen = 1

                if alen == 1 or alen == len(x):
                    setattr(self, a, attribs[a])
                else:
                    logger.info('Attribute {} does not have the proper length and is not being stored.'.format(a))

    def __get_x_and_y(self):

        if hasattr(self, 'xrs'):
            x = self.xrs
            y = self.yrs
            vers = 'resampled'
        elif hasattr(self, 'xs'):
            x = self.xs
            y = self.ys
            vers = 'smooth'
        else:
            x = self.xo
            y = self.yo
            vers = 'original'

        return x, y, vers


    def smooth(self, window=None, n=1, k=3, x=None, y=None):
        """
        Smooths the x and y coordinates of the centerline using a k-th order
        Savitzky-Golay filter.

        window refers to the number of points to use in the moving window;
        must be odd n is the number of times to perform the smoothing.
        """
        if x is None:
            x, y, _ = self.__get_x_and_y()

        if window is None:
            if hasattr(self, 'window_cl'):
                window = self.window_cl
            else:
                logger.info('Must provide a smoothing window.')
                return

        # Ensure window is integer and odd
        window = int(window)
        if window % 2 == 0:
            window = window + 1

        self.xs = signal.savgol_filter(x, window_length=window, polyorder=k,
                                       mode='interp')
        self.ys = signal.savgol_filter(y, window_length=window, polyorder=k,
                                       mode='interp')

        # Could make this recursive but if a non-default x,y are passed in, it would not function as expected
        if n > 1:
            for i in range(1,n-1):
                self.xs = signal.savgol_filter(self.xs, window_length=window,
                                               polyorder=3, mode='interp')
                self.ys = signal.savgol_filter(self.ys, window_length=window,
                                               polyorder=3, mode='interp')

    def resample(self, N, x=None, y=None):
        """
        If no arguments are provided for x and y, will resample the smoothed
        coordinates if available, else will resample the original coordinates.

        N is the number of points that the resulting centerline
        should contain.
        """
        if x is None:
            x, y, _ = self.__get_x_and_y()

        xy, spline = cu.evenly_space_line(x, y, npts=N)
        self.xrs = xy[0]
        self.yrs = xy[1]

    def s(self, x=None, y=None):

        if x is None:
            x, y, _ = self.__get_x_and_y()

        sss, _ = cu.s_ds(x, y)
        return sss

    def ds(self, x=None, y=None):

        if x is None:
            x, y, _ = self.__get_x_and_y()

        _, dss = cu.s_ds(x, y)
        return dss

    def C(self, x=None, y=None):
        """
        Important: curvatures are negativized to match the zs approach
        """
        if x is None:
            x, y, _ = self.__get_x_and_y()

        Cs, _, _ = cu.curvars(x, y, unwrap=True)
        Cs = np.insert(Cs, 0, 0)
        return -Cs

    def Csmooth(self, window=None, x=None, y=None):

        if window is None:
            if hasattr(self, 'window_C'):
                window = self.window_C
            else:
                logger.info('Must provide a smoothing window.')
                return

        Cs = self.C()
        Cs = signal.savgol_filter(Cs, window_length=window, polyorder=3,
                                  mode='interp')
#            Cs = signal.medfilt(Cs,kernel_size=5)
        return Cs

    def infs(self, N, x=None, y=None):
        """
        Finds inflection points.

        N is the number of expected inflection points. It can be estimated
        from N ~= centerline length / 10W, but visual inspection is usually
        best.
        """

        if x is None:
            x, y, _ = self.__get_x_and_y()

        # Use centerline oversmoothing to find inflection points
        self.infs_os, _ = cu.inflection_pts_oversmooth(x, y, n_infs=N)

    def infsC(self, x=None, y=None):

        if not hasattr(self, 'C'):
            self.curvature()

        # Use curvature to find inflection points
        self.infs_C = cu.inflection_points(self.C)


    def intersection_points(self, x2, y2, x1=None, y1=None):

        if x1 is None:
            x1, y1, _ = self.__get_x_and_y()

        ls1 = LineString(zip(x1, y1))
        ls2 = LineString(zip(x2, y2))
        ls_intersections = ls1.intersection(ls2)
        self.ints_all = np.unique(np.sort([np.argmin(np.sqrt((x1-pt.coords.xy[0][0])**2 + (y1-pt.coords.xy[1][0])**2)) for pt in ls_intersections])) # locations of zero migration

        # Map the intersection points so that there is one point for every
        # pair of inflection points in inf_os
        # If there is only one intersection point, use it.
        # If none, use the first inflection point?
        # If multiple, use the one closest to the first inflection point
        if hasattr(self, 'infs_os'):

            s = self.s()

            # Compute the average bend length from the inflection points
            ints = []
            s = self.s()
#            abl = (s[self.infs_os[-1]] - s[self.infs_os[0]])/(len(self.infs_os)-1)

            for i in range(len(self.infs_os)):

                i0 = self.infs_os[i]

                # Find nearest intersection point for first inflection
                if i == 0:
                    intidx = np.argmin(np.abs(s[i0] - s[self.ints_all]))
                    ints.append(self.ints_all[intidx])

                # Else find the nearest interesection point that is downstream of the bend's first inflection point
                else:
                    possible_ints = self.ints_all[self.ints_all > ints[i-1]]
                    dists = np.abs(s[possible_ints] - s[i0])
                    ints.append(possible_ints[np.argmin(dists)])

                if i == len(self.infs_os)-1:
                    break

            self.ints = np.array(ints)

        else:
            logger.info('Could not map intersections to inflection point pairs because infs_os not computed. Run infs() first.')


    def mig_rate_transect_matching(self, x2, y2, dt_years, path_matchers, x1=None, y1=None, mig_spacing=None, window=None, path_mig_vectors=None):
        """
        Compute migration rate using "transect matching". Requires a user to
        provide a geovector file (e.g. shapefile, geopackage, etc.) of that
        contains transects that intersect both centerlines at their common
        points.

        Also computes a smoothed version of the migration rates, and a smoothed
        version with cutoff-affected points set to NaN.
        """

        # If no migration rate smoothing parameter is provided, use the same
        # one used for smoothing curvatures, else window size is 5.
        if window is None:
            if hasattr(self, 'window_C'):
                window = self.window_C
            else:
                window = 5  # must be greater than the polyorder, which is 3 by default

        if x1 is None:
            x1, y1, _ = self.__get_x_and_y()

        # If no spacing is provided, use 1/8 channel width
        if mig_spacing is None:
            mig_spacing = self.W/8

        self.mr_tm, pts_cl1, pts_cl2 = cu.cl_migration_transect_matching(path_matchers, x1, y1, x2, y2, dt_years, mig_spacing)

        # Export migration vectors if path provided
        if path_mig_vectors is not None:
            if self.crs is None:
                logger.info('Cannot export migration vectors until crs is set.')
            else:
                # Migration vectors export
                mvs = []
                for p1, p2 in zip(pts_cl1, pts_cl2):
                    mvs.append(LineString((p1, p2)))
                gdf_mvs = gpd.GeoDataFrame(geometry=mvs, crs=self.crs)
                gdf_mvs.to_file(path_mig_vectors, driver=io.get_driver(path_mig_vectors))


        # Smooth the migration rates
        self.mr_tm_sm = signal.savgol_filter(self.mr_tm, window_length=window,
                                             polyorder=3, mode='interp')

        # Set cutoff-affected and erodibility-affected bends to NaN
        self.mr_tm_nan = self.mr_tm.copy()
        self.mr_tm_sm_nan = self.mr_tm_sm.copy()
        if hasattr(self, 'cut_ids'):
            for c in self.cut_ids:
                self.mr_tm_nan[self.infs_os[c]:self.infs_os[c+1]] = np.NaN
                self.mr_tm_sm_nan[self.infs_os[c]:self.infs_os[c+1]] = np.NaN

        if hasattr(self, 'erode_ids'):
            for e in self.erode_ids:
                self.mr_tm_nan[self.infs_os[e]:self.infs_os[e+1]] = np.NaN
                self.mr_tm_sm_nan[self.infs_os[e]:self.infs_os[e+1]] = np.NaN


    def mig_rate_zs(self, x2, y2, dt_years, x1=None, y1=None, window=None):
        """
        Compute migration rate using Sylvester et al's method of
        dynamic time warping. Also computes a smoothed version of the
        migration rates, and a smoothed version with cutoff-affected
        points set to NaN.
        """
        if x1 is None:
            x1, y1, _ = self.__get_x_and_y()

        # If no migration rate smoothing parameter is provided, use the same
        # one used for smoothing curvatures, else window size is 5.
        if window is None:
            if hasattr(self, 'window_C'):
                window = self.window_C
            else:
                window = 5  # must be greater than the polyorder, which is 3 by default

        import os
        import sys
        script_dir = r"C:\Users\Jon\Desktop\Research\Koyukukon\Normalize migration rates\Code\curvaturepy-master"
        sys.path.append(os.path.abspath(script_dir))
        import cline_analysis as ca

        self.mr_zs, self.mrs_zs, self.p_zs, self.q_zs = ca.get_migr_rate(x1, x2, y1, y2, dt_years, 0)

        # Smooth the migration rates
        self.mr_zs_sm = signal.savgol_filter(self.mr_zs, window_length=window,
                                             polyorder=3, mode='interp')

        # Set cutoff-affected and erodibility-affected bends to NaN
        self.mr_zs_nan = self.mr_zs.copy()
        self.mr_zs_sm_nan = self.mr_zs_sm.copy()
        if hasattr(self, 'cut_ids'):
            for c in self.cut_ids:
                self.mr_zs_nan[self.infs_os[c]:self.infs_os[c+1]] = np.NaN
                self.mr_zs_sm_nan[self.infs_os[c]:self.infs_os[c+1]] = np.NaN

        if hasattr(self, 'erode_ids'):
            for e in self.erode_ids:
                self.mr_zs_nan[self.infs_os[e]:self.infs_os[e+1]] = np.NaN
                self.mr_zs_sm_nan[self.infs_os[e]:self.infs_os[e+1]] = np.NaN


    def plot(self, x=None, y=None):

        if x is None:
            x, y, version = self.__get_x_and_y()
        else:
            version = ''

        fig, ax = plt.subplots()
        legend = []
        ax.plot(x, y, 'k')
        legend.append(version + ' centerline')

        if hasattr(self, 'infs_os'):
            ax.plot(x[self.infs_os], y[self.infs_os], 'rs')
            legend.append('inflection points')

        if hasattr(self, 'ints_all'):
            ax.plot(x[self.ints_all],  y[self.ints_all], 'go')
            legend.append('intersection points')

        if hasattr(self, 'ints'):
            ax.plot(x[self.ints], y[self.ints], 'b^')
            legend.append('intersection points (mapped)')

        plt.legend(legend)
        plt.axis('equal')


    def zs_plot(self, window=None):
        """
        Copied verbatim from https://github.com/zsylvester/curvaturepy/blob/master/Purus_2_migration_rates.ipynb
        Slight modifications for meshing in the centerline class.
        """

        if hasattr(self, 'infs_os') is False:
            logger.info('Must compute inflection points first.')
            return

        if hasattr(self, 'ints') is False:
            logger.info('Must compute intersections first.')
            return

        if hasattr(self, 'mr_zs_nan') is False:
            logger.info('Must compute migration rates first.')
            return
#            elif hasattr(self, 'mr_zs_sm_nan'):
#                migr_rate = self.mr_zs_sm_nan
        else:
            migr_rate = self.mr_zs_nan

        if hasattr(self, 'cut_ids') is False:
            cutoff_inds = []
        else:
            cutoff_inds = self.cut_ids

        if hasattr(self, 'erode_ids') is False:
            erodibility_inds = []
        else:
            erodibility_inds = self.erode_ids

        if window is None:
            if hasattr(self, 'window_C'):
                window = self.window_C
            else:
                logger.info('Must provide a smoothing window.')
                return

        LZC = self.infs_os
        LZM = self.ints
        s = self.s()
        curv = self.Csmooth()
        W = self.W

        fig, ax1 = plt.subplots(figsize=(18,4))
#            plt.tight_layout()

        y1 = 0.7
        y2 = 0.0
        y3 = -0.87
        y4 = -1.25

        for i in range(0,len(LZC)-1,2):
            xcoords = [s[LZC[i]],s[LZC[i+1]],s[LZC[i+1]],s[LZM[i+1]],s[LZM[i+1]],s[LZM[i]],s[LZM[i]],s[LZC[i]]]
            ycoords = [y1,y1,y2,y3,y4,y4,y3,y2]
            ax1.fill(xcoords,ycoords,color=[0.85,0.85,0.85],zorder=0)

        ax1.fill_between(s, 0, curv*W)
        ax2 = ax1.twinx()
        ax2.fill_between(s, 0, migr_rate, facecolor='green')

        ax1.plot([0,max(s)],[0,0],'k--')
        ax2.plot([0,max(s)],[0,0],'k--')

        ax1.set_ylim(y4,y1)
        ax2.set_ylim(-15,40)
        ax1.set_xlim(s[LZC[0]],s[-1])

        for i in erodibility_inds:
            xcoords = [s[LZC[i]],s[LZC[i+1]],s[LZC[i+1]],s[LZM[i+1]],s[LZM[i+1]],s[LZM[i]],s[LZM[i]],s[LZC[i]]]
            ycoords = [y1,y1,y2,y3,y4,y4,y3,y2]
            ax1.fill(xcoords,ycoords,color=[1.0,0.85,0.85],zorder=0)

        for i in cutoff_inds:
            xcoords = [s[LZC[i]],s[LZC[i+1]],s[LZC[i+1]],s[LZM[i+1]],s[LZM[i+1]],s[LZM[i]],s[LZM[i]],s[LZC[i]]]
            ycoords = [y1,y1,y2,y3,y4,y4,y3,y2]
            ax1.fill(xcoords,ycoords,color=[0.85,1.0,0.85],zorder=0)

        for i in range(len(LZC)-1):
            if np.sum(np.isnan(migr_rate[LZM[i]:LZM[i+1]]))>0:
                xcoords = [s[LZC[i]],s[LZC[i+1]],s[LZC[i+1]],s[LZM[i+1]],s[LZM[i+1]],s[LZM[i]],s[LZM[i]],s[LZC[i]]]
                ycoords = [y1,y1,y2,y3,y4,y4,y3,y2]
                ax1.fill(xcoords,ycoords,color='w')

        for i in range(len(LZC)-1):
            if np.sum(np.isnan(migr_rate[LZM[i]:LZM[i+1]]))>0:
                xcoords = [s[LZC[i]],s[LZC[i+1]],s[LZC[i+1]],s[LZM[i+1]],s[LZM[i+1]],s[LZM[i]],s[LZM[i]],s[LZC[i]]]
                ycoords = [35,35,20.7145,0,-15,-15,0,20.7145]
                ax2.fill(xcoords,ycoords,color='w')

        for i in range(0,len(LZC)-1,2):
            ax1.text(s[LZC[i]],0.5,str(i),fontsize=12)
