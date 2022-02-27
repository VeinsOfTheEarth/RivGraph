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
            'id': OrderedSet(nodes.index),
            'idx': OrderedSet(nodes.idx),
            'conn': nodes.conn.tolist(),
            'inlets': nodes.index[nodes.inlet].tolist(),
            'outlets': nodes.index[nodes.outlet].tolist(),
        }

        self.links = {
            'id': OrderedSet(links.index),
            'idx': links.idx.tolist(),
            'conn': links[["start_node", "end_node"]].values.tolist(),
            'wid_pix': links.wid_pix.apply(lambda s: np.array(eval(s))).values.tolist(),
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
        dat["cycles"] = pd.Series(0, index=dat['id'])
        for ic, cyc in enumerate(self.links["cycles"]):
            dat['cycles'][cyc] = ic + 1
        link_dir_info = pd.DataFrame(dat).set_index("id")
        return link_dir_info

    @property
    def nodes_direction_info(self):
        self.compute_junction_angles()
        df = pd.DataFrame({c: self.nodes[c] for c in ["int_ang", "jtype", "width_ratio", "id"]}).set_index("id")
        # add cycles
        df["is_cycle"] = 0
        for ic, cyc in enumerate(self.nodes["cycles"]):
            df.loc[cyc, "is_cycle"] = ic + 1
        return df

    @property
    def flipped_links(self):
        inp = self.links_input[["start_node", "end_node"]].values.tolist()
        flpd = [self.links["id"][i] for i, l in enumerate(inp) if self.links["conn"][i] != l]
        return flpd

    def compute_strahler_order(self, loop_raise=False):

        order = {}
        fromto = np.array(self.links["conn"])
        orderX = fromto[[f in self.nodes["inlets"] for f, t in fromto], :]
        i = 1
        # as long as there is one routed that isnt the largest outlet
        while len(orderX) >= 1 and any([n not in self.nodes["outlets"] for n in orderX[:, 1]]):
            # loop over each node and correct order in order dict
            sys.stdout.write('\rCalculating stream order %s' % i)
            for sb in orderX[:, 0]:
                if sb in order:
                    order[sb] = max([order[sb], i])
                else:
                    order[sb] = i
            # get downstream node ids
            ins = np.unique(orderX[:, 1])
            # get their info
            orderX = fromto[np.array([s in ins for s in fromto[:, 0]])]
            # increase order
            i += 1
            # raise or warn
            if i > len(fromto) + 1:
                msg = "Order is exceeding the number of links, which indicates loops. Remaining nodes: %s" % orderX
                if loop_raise:
                    raise RuntimeError(msg)
                else:
                    warnings.warn(msg)
                    break

        sys.stdout.write('\n')
        self.links["strahler_order"] = np.array([order[i] for i, ii in self.links["conn"]])
        return

    def run(self, output=True):
        """Set directions and write output."""
        self.find_parallel_links()
        self.link_widths_and_lengths()
        self.assign_flow_directions()
        self.resolve_cycles()
        self.compute_strahler_order()
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


def prune_single_flow_network(nodes, links):
    # prune single flow reaches and check the direction by accumulation
    sfd_in = [None]
    flips = []
    order = 1
    while True:
        # inlet nodes and lines with single flow
        sfd_inl = nodes[(nodes.n_lines == 1) & nodes.single_flow & ~nodes.outlet]
        sfd_lns = links.loc[sfd_inl.conn.apply(lambda x: x[0])]
        nxt_nodes = nodes.loc[
            np.where(sfd_lns.end_node == sfd_inl.index, sfd_lns["start_node"], sfd_lns["end_node"])
        ]
        # make sure inlet - outlet links have the right direction (includes outlet - inlet lines)
        flip = (sfd_lns.end_node == sfd_inl.index) & nxt_nodes.single_flow.values
        flips.extend(sfd_lns.index[flip].tolist())
        # receiving node also needs to be a single flow node
        sfd_inl = sfd_inl[(nxt_nodes.single_flow).values]
        sfd_lns = sfd_lns[(nxt_nodes.single_flow).values]
        if len(sfd_inl) == 0:
            break
        # recalculate next nodes
        nxt_nodes = nodes.loc[
            np.where(sfd_lns.end_node == sfd_inl.index, sfd_lns["start_node"], sfd_lns["end_node"])
        ]
        # recalculate node connections
        for n, l in zip(nxt_nodes.index, sfd_lns.index):
            if l in nodes.loc[n, 'conn']:
                nodes.loc[n, 'conn'].remove(l)
        nodes.loc[nxt_nodes.index, 'n_lines'] = nodes.loc[nxt_nodes.index, 'conn'].apply(lambda c: len(c))
        # drop from nodes and lines
        nodes.drop(sfd_inl.index, inplace=True)
        nodes.drop(nxt_nodes.index[nxt_nodes.outlet], inplace=True)
        links.drop(sfd_lns.index, inplace=True)
        print(f'Pruning {len(sfd_lns)} single flow lines of order {order}')
        order += 1

    return nodes, links, flips


def multi_network_rivgraph(links, nodes, idx_shape, results_folder='.', res=30):
    """Run RivgraphNetwork on multiple networks.
    """
    nodes = pd.read_csv(nodes, index_col=0) if type(nodes) == str else nodes
    nodes['conn'] = nodes.conn.apply(eval)
    links = pd.read_csv(links, index_col=0) if type(links) == str else links
    links['idx'] = links.idx.apply(eval)

    # sanity check input
    #  segments with less than 2 pixels?
    links_length = links.idx.apply(lambda s: len(s))
    assert links_length.min() >= 2
    #  no inlet also assigned as outlet
    assert len(set(nodes.index[nodes.inlet]) & set(nodes.index[nodes.outlet])) == 0
    #  all node idx are at start/end of line idx
    assert (nodes.loc[links.start_node.values, "idx"].values == links['idx'].apply(lambda l: l[0])).all()
    assert (nodes.loc[links.end_node.values, "idx"].values == links['idx'].apply(lambda l: l[-1])).all()
    # no loops
    assert (links.start_node == links.end_node).sum() == 0

    nodes, links, flips = prune_single_flow_network(nodes, links)

    nix, lix = nodes.groupby("component").groups, links.groupby("component").groups
    assert len(nix) == len(lix)
    print(f"Found {len(nix)} networks.")
    links_info, nodes_info = [], []
    for n in nix:
        rgn = RiverNetwork(links.loc[lix[n]], nodes.loc[nix[n]], idx_shape,
                           results_folder=results_folder, res=res,
                           name="component_%04i" % n)
        rgn.run(output=False)
        links_info.append(rgn.links_direction_info.copy())
        nodes_info.append(rgn.nodes_direction_info.copy())
        flips.extend(rgn.flipped_links)
    pd.concat(links_info).to_csv(osp.join(results_folder, 'links_direction_info.csv'))
    pd.concat(nodes_info).to_csv(osp.join(results_folder, 'nodes_direction_info.csv'))
    with open(osp.join(results_folder, 'flipped_links.csv'), 'w') as f:
        f.writelines(['%s\n' % i for i in flips])
    return


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
