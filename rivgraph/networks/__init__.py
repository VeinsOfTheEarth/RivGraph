#!/usr/bin/env python
import os.path as osp
import sys
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
import networkx as nx

from rivgraph.classes import river as RGRiver
import rivgraph.ln_utils as rgln
from rivgraph.ordered_set import OrderedSet
import rivgraph.io_utils as io
import rivgraph.directionality as dy


from . import network_directionality as nd


class RiverNetwork(RGRiver):

    def __init__(self, links, nodes, idx_shape, results_folder='.', res=30, name=None,
                 verbose=True):

        # Store some class attributes
        self.name = name or 'River network'
        self.verbose = verbose

        # Prepare paths for saving
        if results_folder is not None:
            self.paths = io.prepare_paths(results_folder, self.name, '.')
        else:
            self.paths = io.prepare_paths(osp.dirname(osp.abspath(".")), self.name, ".")

        # init logger - prints out to stdout if verbose is True
        # ALWAYS writes output to log file (doesn't print if verbose is False)
        self.init_logger()

        # read input
        nodes = pd.read_csv(nodes, index_col=0) if type(nodes) == str else nodes
        links = pd.read_csv(links, index_col=0) if type(links) == str else links
        self.nodes_input = nodes
        self.links_input = links

        self.nodes = {
            'id': OrderedSet(assert_unique(nodes.index, "id of nodes")),
            'idx': OrderedSet(assert_unique(nodes.idx, "idx of nodes")),
            'conn': nodes.conn.tolist(),
            'inlets': nodes.index[nodes.inlet].tolist(),
            'outlets': nodes.index[nodes.outlet].tolist(),
        }
        self.nodes.update({k: nodes[k].values for k in nodes.columns if k not in self.nodes})
        ln = nodes.conn.apply(len)
        ln_min_conn = np.min((ln[links["start_node"]], ln[links["end_node"]]), axis=0)

        self.links = {
            'id': OrderedSet(assert_unique(links.index, "id of links")),
            'idx': links.idx.tolist(),
            'conn': links[["start_node", "end_node"]].values.tolist(),
            'wid_pix': links.wid_pix.apply(lambda s: np.array(s)).values.tolist(),
            'dangle': ln_min_conn == 1,
        }
        self.links.update({k: links[k].values for k in links.columns if k not in self.links})

        self.pixarea = res**2
        self.pixlen = res
        self.imshape = idx_shape
        class Mask:
            shape = idx_shape
        self.Imask = Mask()
        self.gt, self.Idist = None, None
        self.meshlines, self.meshpolys, self.centerline = None, None, None
        self.exit_sides = 'ns'

    def link_widths_and_lengths(self):
        self.links = link_widths_and_lengths(self.links, self.imshape, pixlen=self.pixlen)
        # width combos
        width = [[self.links["wid_adj"][self.links['id'].index(i)] for i in con] for con in self.nodes["conn"]]
        self.nodes["width_combo"] = [np.array(w) / np.array([np.sum(w) - i for i in w]) for w in width]
        return self.links

    def find_parallel_links(self):
        from rivgraph.ln_utils import find_parallel_links
        self.links, self.nodes = find_parallel_links(self.links, self.nodes)

    @property
    def links_direction_info(self):
        cols = tuple(set(self.links.keys()) - set(list(self.links_input.columns)+["n_networks", "parallels", "conn"]))
        dat = {c: self.links[c] for c in cols if len(np.array(self.links[c]).shape) == 1}
        # add cycles
        dat["cycle"] = pd.Series(0, index=dat['id'])
        dat.pop("cycles", [])
        for ic, cyc in enumerate(self.links["cycles"]):
            dat['cycle'][cyc] = ic + 1
        # cycle = -1 for ignored cycle links 
        if "cycles_ignore_links" in dat:
            dat["cycle"][dat.pop("cycles_ignore_links")] = -1
        link_dir_info = pd.DataFrame(dat).set_index("id")
        return link_dir_info

    @property
    def nodes_direction_info(self):
        df = pd.DataFrame({c: self.nodes[c] for c in ["int_ang", "jtype", "width_ratio", "id"]}).set_index("id")
        # add cycles
        df["cycle"] = 0
        for ic, cyc in enumerate(self.nodes["cycles"]):
            df.loc[cyc, "cycle"] = ic + 1
        df['continuity_violated'] = 0
        df.loc[self.nodes["continuity_violated"], "continuity_violated"] = 1
        return df

    @property
    def flipped_links(self):
        inp = self.links_input[["start_node", "end_node"]].values.tolist()
        flpd = [self.links["id"][i] for i, l in enumerate(inp) if self.links["conn"][i] != l]
        return flpd

    def run(self, output=True):
        """Set directions and write output."""
        self.find_parallel_links()
        self.link_widths_and_lengths()
        self.assign_flow_directions()
        self.find_sink_source_nodes()
        if self.nodes["sinks"]:
            self.reset_sink_flow_catchments()
            self.assign_flow_directions()
        self.compute_junction_angles()
        self.find_mainstem()

        # write output
        if output:
            self.links_direction_info.to_csv(osp.join(self.paths['basepath'], 'links_direction_info.csv'))
            self.nodes_direction_info.to_csv(osp.join(self.paths['basepath'], 'nodes_direction_info.csv'))
            with open(osp.join(self.paths['basepath'], 'flipped_links.csv'), 'w') as f:
                f.writelines(['%s\n' % i for i in self.flipped_links])
        return


    def assign_flow_directions(self):
        """
        Automatically sets flow directions for each link in a braided river
        channel network.

        """
        logger.info('Setting link directionality...')

        self.links, self.nodes = nd.set_directionality(
            self.links, self.nodes, self.Imask, self.exit_sides, self.gt,
            self.meshlines, self.meshpolys, self.Idist, self.pixlen, self.paths['fixlinks_csv'])

        logger.info('link directionality has been set.')
        return

    def find_sink_source_nodes(self):
        """Find sink and source nodes from cycles and discontinuities.
        """
        assert ("cycles" in self.links and "cycles" in self.nodes), "Directions not yet set."

        self.nodes["sinks"] = []
        self.nodes["sources"] = []

        # check cycles
        for cycle_nodes in self.nodes["cycles"]:
            G = nx.MultiDiGraph()
            allcon = set([l for n in cycle_nodes
                          for l in self.nodes["conn"][self.nodes["id"].index(n)]])
            G.add_edges_from([tuple(self.links["conn"][self.links["id"].index(l)]) + ({"id": l},)
                            for l in allcon])
            # cycle nodes where outflows are connected
            outflows = [list(G.in_edges(n))[0][0] for n in G
                        if len(G.out_edges(n)) == 0 and len(G.in_edges(n)) == 1]
            # cycle nodes where inflows are connected
            inflows = [list(G.out_edges(n))[0][1] for n in G
                       if len(G.out_edges(n)) == 1 and len(G.in_edges(n)) == 0]
            # check if sink or source
            if (not outflows) and inflows:
                self.nodes["sinks"] += cycle_nodes
            if (not inflows) and outflows:
                self.nodes["sources"] += cycle_nodes

        logger.info(f'Found {len(self.nodes["sinks"])} cycle sink nodes and '
                    f'{len(self.nodes["sources"])} cycle source nodes.')
        # check discontinuities
        G = nx.MultiDiGraph()
        G.add_edges_from(self.links["conn"])
        sink_nodes = [n for n in G if len(G.out_edges(n)) == 0 and len(G.in_edges(n)) > 1]
        self.nodes["sinks"] += sink_nodes
        source_nodes = [n for n in G if len(G.in_edges(n)) == 0 and len(G.out_edges(n)) > 1]
        self.nodes["sources"] += source_nodes
        logger.info(f'Found {len(sink_nodes)} single sink nodes and '
                    f'{len(source_nodes)} single source nodes.')
        return

    def check_width_continuity(self):
        """
        After all directions are assigned, calculate the ratio between total inflow and total outflow width.
        Where this ratio is far from 1 (either very large or very small), directions are likely wrong.
        """
        G = self.nx_graph
        ids = self.nodes["id"]
        self.nodes["in_out_width_ratio"] = np.ones(len(ids)) * np.nan

        for n in tqdm(G, "Checking width in/out ratios"):
            inedg, outedg = G.in_edges(n, data=True), G.out_edges(n, data=True)
            if not (len(inedg) and len(outedg)):
                continue
            inwidth = sum([d["wid_adj"] for _, _, d in inedg])
            outwidth = sum([d["wid_adj"] for _, _, d in outedg])
            nodes["in_out_width_ratio"][ids.index(n)] = inwidth / outwidth

        return

    @property
    def nx_graph(self):
        G = nx.MultiDiGraph()
        ids = self.links["id"]
        G.add_edges_from([tuple(conn) + ({k: v[i] for k, v in self.links.items() if len(v) == len(ids)},)
                          for i, (lid, conn) in enumerate(zip(ids, self.links["conn"]))])
        return G

    def reset_sink_flow_catchments(self):
        """Reset all links upstream of nodes, determine their closest path to an outlet,
        set the directions of that path within their original catchment. The other reset
        directions need to be assigned again.
        """
        sink_nodes = self.nodes["sinks"]
        outlets = self.nodes["outlets"]

        # directed graph with line ids
        G = self.nx_graph

        # get the upstream catchments of all sink nodes
        catchment = G.subgraph([n for cn in sink_nodes for n in nx.bfs_tree(G, cn, reverse=True)])
        catchment_lines_idx = np.array([self.links["id"].index(i) for _,_,i in catchment.edges(data='id')])

        logger.info(f"Resetting directions in {len(catchment_lines_idx)} links upstream of sinks.")
        self.links["certain"][catchment_lines_idx] = 0 
        self.links["certain_alg"][catchment_lines_idx] = 0 
        self.links["certain_order"][catchment_lines_idx] = 0 
        self.links["guess_alg"] = [[] if i in catchment_lines_idx else g for i, g in enumerate(self.links["guess_alg"])]
        self.links["guess"] = [[] if i in catchment_lines_idx else g for i, g in enumerate(self.links["guess"])]
        # reset directions to original otherwise 
        inp = self.links_input[["start_node", "end_node"]].values.tolist()
        self.links["conn"] = [inp[i] if i in catchment_lines_idx else l for i, l in enumerate(self.links["conn"])]

        # path to nearest outlet through undirected graph
        Gund = G.to_undirected()
        # dict of path line ids with upstream nodes
        reset_lines = {}
        for cn in sink_nodes:
            # find shortest path to next outlet
            outlpaths = [nx.shortest_path(Gund, cn, o) for o in outlets]
            shortest = sorted(outlpaths, key=len)[0]
            # get line ids including possible multi links
            # line ids with upstream node and downstream within catchment
            line_path = {lid: f for f, t in zip(shortest[:-1], shortest[1:])
                        for _, _, lid in Gund.subgraph([f, t]).edges(data="id")
                        if t in catchment}
            reset_lines.update(line_path)

        logger.info(f"Setting {len(reset_lines)} lines out of sinks via shortest path to outlet.")
        alg = dy.algmap("sp_links_sink")
        for lid, upn in reset_lines.items():
            lidx = self.links["id"].index(lid)
            self.links, self.nodes = dy.set_link(self.links, self.nodes, lidx, upn, alg=alg)
        return

    def find_mainstem(self):
        """Find network mainstem by following the widest downstream link.
        """
        ms = np.zeros(len(self.links["id"]), dtype=int)

        for inode in tqdm(self.nodes["inlets"], "Finding mainstems"):
            while inode not in self.nodes["outlets"]:
                ils = [self.links["id"].index(i) for i in
                       self.nodes["conn"][self.nodes["id"].index(inode)]]
                # filter links going out of nodes
                ilds = [i for i in ils if inode == self.links["conn"][i][0]]
                if not len(ilds):
                    warnings.warn(f"No ds lines, discontinuity? At node {inode}")
                    break
                ilw = [self.links["wid_adj"][i] for i in ilds]
                il = ilds[np.argmax(ilw)]
                if ms[il] > 0:
                    # already found mainstems here
                    break
                ms[il] = 1
                inode = self.links["conn"][il][1]
                # check if next node is in cycle
                in_cycles = [inode in cy for cy in self.nodes["cycles"]]
                if any(in_cycles):
                    warnings.warn(f"Next node in cylce: {sum(in_cycles)}")
                    break
        self.links["is_mainstem"] = ms
        return ms


def assert_unique(ids_series, name):
    if len(ids_series) != len(ids_series.unique()):
        dups = ids_series[ids_series.duplicated(False)]
        raise RuntimeError(f"{name} is non-unique:\n{dups}")
    return ids_series


def link_widths_and_lengths(links, dims, pixlen=1):
    """
    Compute all link widths and lengths. Adapted from rivgraph.ln_utils

    Computes link widths and lengths for all links in the network. A
    distance transform approach is used where the width of a pixel is its
    distance to the nearest non-max pixel times two.

    There is a slight twist. When a skeleton is computed for a very wide
    channel with a narrow tributary, there is a very straight section of the
    skeleton as it leaves the wide channel to go into the tributary; this
    straight section (so-called "false" pixels) should not be used to compute
    average link width, as it's technically part of the wide channel. The twist
    here accounts for that by elminating the ends of the each link from
    computing widths and lengths, where the distance along each end is equal to
    the half-width of the endmost pixels. Adjusted link widths and lengths are
    also computed that account for this effect.

    The following new attributes are added to the links dictionary:

    - 'len' : the length of the full link

    - 'wid_pix' : the width of each pixel in the link

    - 'wid' : the average width of all pixels of the link

    - 'wid_adj' : the "adjusted" average width of all link pixels excluding
                  "false" pixels

    - 'wid_med' : median of 'adjusted' width values

    - 'len_adj' : the "adjusted" length of the link after excluding
                  "false" pixels

    - 'sinuosity' : simple sinuosity using euclidean distances in the array-
                    space. does not take projection or geoid into account.
                    is length of channel / euclidean distance

    Parameters
    ----------
    links : dict
        Network links and associated properties.
    Idt : np.array
        Distance transform of the original mask.
    pixlen : float or int, optional
        Length (or resolution) of the pixel. If provided, assumes that the
        pixel resolution is the same in the horizontal and vertical directions.
        The default is 1, which corresponds to computing widths and lengths
        in units of pixels.

    Returns
    -------
    links : dict
        Network links with width and length properties appended.

    """
    # Initialize attribute storage
    links['len'] = []
    links['wid'] = []
    links['wid_adj'] = []  # average of all link pixels considered to be part of actual channel
    links['wid_med'] = []  # median of all link px in channel
    links['len_adj'] = []
    links['sinuosity'] = []  # channel sinuosity for adjusted length

    width_mult = 1.1  # fraction of endpoint half-width to trim links before computing link width

    # Compute trimmed/untrimmed link widths and lengths
    for li, widths in tqdm(zip(links['idx'], links['wid_pix']), "Link width and length", total=len(links["idx"])):

        xy = np.unravel_index(li, dims)

        # Compute distances along link
        dists = np.cumsum(np.sqrt(np.diff(xy[0])**2 + np.diff(xy[1])**2))
        dists = np.insert(dists, 0, 0) * pixlen

        # Compute distances along link in opposite direction
        revdists = np.cumsum(np.flipud(np.sqrt(np.diff(xy[0])**2 +
                                               np.diff(xy[1])**2)))
        revdists = np.insert(revdists, 0, 0) * pixlen

        # Find the first and last pixel along the link that is at least a half-width's distance away
        startidx = np.argmin(np.abs(dists - widths[0]/2*width_mult))
        endidx = len(dists) - np.argmin(np.abs(revdists - widths[-1] /
                                               2*width_mult)) - 1

        # Ensure there are enough pixels to trim the ends by the pixel half-width
        if startidx >= endidx:
            links['wid_adj'].append(np.mean(widths))
            links['wid_med'].append(np.median(widths))
            links['len_adj'].append(dists[-1])
            # straight-line distance between first and last pixel of link
            st_dist = np.sqrt((xy[0][0]-xy[0][-1])**2 +
                              (xy[1][0]-xy[1][-1])**2) * pixlen
            # sinuosity =  channel len / straight line length
            links['sinuosity'].append(dists[-1] / st_dist)
        else:
            links['wid_adj'].append(np.mean(widths[startidx:endidx]))
            links['wid_med'].append(np.median(widths[startidx:endidx]))
            links['len_adj'].append(dists[endidx] - dists[startidx])
            # straight-line distance between first and last pixel of link
            st_dist = np.sqrt((xy[0][startidx]-xy[0][endidx])**2 +
                              (xy[1][startidx]-xy[1][endidx])**2) * pixlen
            # sinuosity =  channel len / straight line length
            links['sinuosity'].append((dists[endidx]-dists[startidx])/st_dist)

        # Unadjusted lengths and widths
        links['wid'].append(np.mean(widths))
        links['len'].append(dists[-1])

        # Ensure the minimum width and length equal to the pixel resolution
        links['wid'][-1] = max(pixlen, links['wid'][-1])
        links['len'][-1] = max(pixlen, links['len'][-1])
        links['wid_adj'][-1] = max(pixlen, links['wid_adj'][-1])
        links['len_adj'][-1] = max(pixlen, links['len_adj'][-1])

    return links
