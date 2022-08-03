#!/usr/bin/env python
import os.path as osp
import sys
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from rivgraph.classes import river as RGRiver
import rivgraph.ln_utils as rgln
from rivgraph.ordered_set import OrderedSet
import rivgraph.io_utils as io


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

        self.links = {
            'id': OrderedSet(assert_unique(links.index, "id of links")),
            'idx': links.idx.tolist(),
            'conn': links[["start_node", "end_node"]].values.tolist(),
            'wid_pix': links.wid_pix.apply(lambda s: np.array(s)).values.tolist(),
        }

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
        return self.links

    def find_parallel_links(self):
        from rivgraph.ln_utils import find_parallel_links
        self.links, self.nodes = find_parallel_links(self.links, self.nodes)

    @property
    def links_direction_info(self):
        cols = tuple(set(self.links.keys()) - set(list(self.links_input.columns)+["n_networks", "parallels", "conn"]))
        dat = {c: self.links[c] for c in cols if len(np.array(self.links[c]).shape) == 1}
        dat['wid_pctdiff'] = self.links['wid_pctdiff'].flatten()
        # add cycles
        dat["cycle"] = pd.Series(0, index=dat['id'])
        dat.pop("cycles", [])
        for ic, cyc in enumerate(self.links["cycles"]):
            dat['cycle'][cyc] = ic + 1
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
        self.compute_junction_angles()

        #self.resolve_cycles()
        # write output
        if output:
            self.links_direction_info.to_csv(osp.join(self.paths['basepath'], 'links_direction_info.csv'))
            self.nodes_direction_info.to_csv(osp.join(self.paths['basepath'], 'nodes_direction_info.csv'))
            with open(osp.join(self.paths['basepath'], 'flipped_links.csv'), 'w') as f:
                f.writelines(['%s\n' % i for i in self.flipped_links])
        return
    
    def resolve_cycles(self):
        """Set directions by accumulation in cycle links."""
        for cyl, cyn in zip(self.links["cycles"], self.nodes["cycles"]):
            # make sure widest river is flowing into node with greatest accumulation
            for l in cyl:
                stenn = self.links["conn"][self.links["id"].index(l)]
                stac, enac = self.nodes_input.loc[stenn, "accumulation"]
                if stac > enac:
                    rgln.flip_link(self.links, l)
        return


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
