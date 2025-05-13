# -*- coding: utf-8 -*-
"""
delta_metrics
=============
Created on Mon May 21 09:00:01 2018

@author: Jon
"""
import numpy as np
import networkx as nx
import rivgraph.directionality as dy
import rivgraph.ln_utils as lnu

"""
This script contains algorithms that were ported from Matlab scripts provided
by Alejandro Tejedor to compute topologic and dynamic metrics on deltas. The
provided Matlab script required the bioinformatics toolbox; here we use
networkx to achieve the same result. Ported by Jon Schwenk.
The conversion was tested by computing metrics for the Wax Lake Delta
(provided by AT) and the Yenesei Delta (provided by JS)--perfect agreement
was found for all metrics, for both deltas, using both the original Matlab
scripts and the Python functions provided here.
JS has made some efficiency improvments to the code; otherwise most variable
names and code structure was matched to the original Matlab scripts.

Use at your own risk.
"""


def compute_delta_metrics(links, nodes):
    """Compute delta metrics."""
    # Delta metrics require a single apex node
    # This is not the ideal way to force a single inlet; adding the super-apex
    # is generally a much better approach. It has yet to be tested thoroughly,
    # though.
    links_m, nodes_m = ensure_single_inlet(links, nodes)

    # Ensure we have a directed, acyclic graph; also include widths as weights
    G = graphiphy(links_m, nodes_m, weight='wid_adj')

    if nx.is_directed_acyclic_graph(G) is not True:
        raise RuntimeError('Cannot proceed with metrics as graph is not acyclic.')

    # Compute the intermediate variables required to compute delta metrics
    deltavars = intermediate_vars(G)

    # Compute metrics
    metrics = dict()
    ner, pexc, ner_randomized = delta_nER(deltavars, N=200)
    metrics['nonlin_entropy_rate'] = ner
    metrics['nER_prob_exceedence'] = pexc
    metrics['nER_randomized'] = ner_randomized
    metrics['top_mutual_info'], metrics['top_conditional_entropy'] = top_entropy_based_topo(deltavars)
    metrics['top_link_sharing_idx'] = top_link_sharing_index(deltavars)
    metrics['n_alt_paths'] = top_number_alternative_paths(deltavars)
    metrics['resistance_distance'] = top_resistance_distance(deltavars)
    metrics['top_pairwise_dependence'] = top_s2s_topo_pairwise_dep(deltavars)
    metrics['flux_sharing_idx'] = dyn_flux_sharing_index(deltavars)
    metrics['leakage_idx'] = dyn_leakage_index(deltavars)
    metrics['dyn_pairwise_dependence'] = dyn_pairwise_dep(deltavars)
    metrics['dyn_mutual_info'], metrics['dyn_conditional_entropy'] = dyn_entropy_based_dyn(deltavars)

    return metrics


def ensure_single_inlet(links, nodes):
    """
    Ensure only a single apex node exists. This dumbly just prunes all inlet
    nodes+links except the widest one. Recommended to use the super_apex()
    approach instead if you want to preserve all inlets.

    All the delta metrics here require a single apex node, and that that node
    be connected to at least two downstream links. This function ensures these
    conditions are met; where there are multiple inlets, the widest is chosen.
    This function also ensures that the inlet node is attached to at least two
    links--this is important for computing un-biased delta metrics.
    The links and nodes dicts are copied so they remain unaltered; the altered
    copies are returned.

    """
    # Copy links and nodes so we preserve the originals
    links_edit = dict()
    links_edit.update(links)
    nodes_edit = dict()
    nodes_edit.update(nodes)

    # Find the widest inlet
    in_wids = []
    for i in nodes_edit['inlets']:
        linkid = nodes_edit['conn'][nodes_edit['id'].index(i)][0]
        linkidx = links_edit['id'].index(linkid)
        in_wids.append(links_edit['wid_adj'][linkidx])
    widest_inlet_idx = in_wids.index(max(in_wids))
    inlets_to_remove = nodes_edit['inlets'][:]

    # Remove inlet nodes and links until continuity is no longer broken
    badnodes = dy.check_continuity(links_edit, nodes_edit)
    if len(badnodes) > 0:
        raise RuntimeError('Provided (links, nodes) has source or sink at nodes: {}.'.format(badnodes))

    # Keep the widest inlet - delete all others (and remove their subnetworks)
    main_inlet = inlets_to_remove.pop(widest_inlet_idx)
    for i in inlets_to_remove:
        nodes_edit['inlets'].remove(i)
        badnodes = dy.check_continuity(links_edit, nodes_edit)
        while len(badnodes) > 0:
            badnode = badnodes.pop()
            # Remove the links connected to the bad node:
            # the hanging node will also be removed
            connlinks = nodes_edit['conn'][nodes_edit['id'].index(badnode)]
            for cl in connlinks:
                links_edit, nodes_edit = lnu.delete_link(links_edit,
                                                         nodes_edit, cl)

            badnodes = dy.check_continuity(links_edit, nodes_edit)

    # Ensure there are at least two links emanating from the inlet node
    conn = nodes_edit['conn'][nodes_edit['id'].index(main_inlet)]
    while len(conn) == 1:
        main_inlet_new = links_edit['conn'][links_edit['id'].index(conn[0])][:]
        main_inlet_new.remove(main_inlet)
        links_edit, nodes_edit = lnu.delete_link(links_edit, nodes_edit,
                                                 conn[0])

        # Update new inlet node
        nodes_edit['inlets'].remove(main_inlet)
        main_inlet = main_inlet_new[0]
        nodes_edit['inlets'] = nodes_edit['inlets'] + [main_inlet]
        conn = nodes_edit['conn'][nodes_edit['id'].index(main_inlet)]

    return links_edit, nodes_edit


def add_super_apex(links, nodes, imshape):
    """
    If multiple inlets are present, this creates a "super apex" that is
    directly upstream of all the inlet nodes. The synthetic links created
    have zero length and widths equal to the sum of the widths of the links
    connected to their respective inlet node.
    """

    # Get inlet nodes
    ins = nodes['inlets']

    if len(ins) <= 1:
        return links, nodes

    # Find the location of the super-apex by averaging the inlets' locations
    ins_idx = [nodes['idx'][nodes['id'].index(i)] for i in ins]
    rs, cs = np.unravel_index(ins_idx, imshape)
    apex_r, apex_c = np.mean(rs, dtype=int), np.mean(cs, dtype=int)
    apex_idx = np.ravel_multi_index((apex_r, apex_c), imshape)

    # Get the widths of the super-apex links -- these are just the summed
    # widths of all the links connected to each inlet node
    sa_widths = []
    for i in ins:
        lconn = nodes['conn'][nodes['id'].index(i)]
        sa_widths.append(sum([links['wid_adj'][links['id'].index(lid)] for lid in lconn]))

    # Add links from the super-apex to the inlet nodes
    # Widths are computed above
    # lengths are set to zero for these synthetic links
    for i, wid in zip(ins, sa_widths):
        in_idx = nodes['idx'][nodes['id'].index(i)]
        idcs = [apex_idx, in_idx]
        links, nodes = lnu.add_link(links, nodes, idcs)
        links['wid_adj'].append(wid)
        # we also append to the other attributes to keep fields the same length
        links['wid'].append(wid)
        links['wid_med'].append(wid)
        links['sinuosity'].append(wid)
        links['len'].append(0)
        links['len_adj'].append(0)

    # Add the super apex node field to the nodes dictionary
    # nodes = ln_utils.add_node(nodes, apex_idx, sa_lids)
    nodes['super_apex'] = nodes['id'][-1]

    return links, nodes


def delete_super_apex(links, nodes):
    """
    If you have a super apex, this function deletes it and connecting links.
    """

    # Get super apex node
    if 'super_apex' not in nodes:
        raise ValueError('no super apex detected.')

    # identify super apex
    super_apex = nodes['super_apex'][0]

    # identify connecting links
    super_links = nodes['conn'][nodes['id'].index(super_apex)]

    # delete links first
    for i in super_links:
        links, nodes = lnu.delete_link(links, nodes, i)

    # then delete super apex
    nodes = lnu.delete_node(nodes, super_apex, warn=True)

    return links, nodes


def graphiphy(links, nodes, weight=None, inletweights=None):
    """Converts RivGraph links and nodes into a NetworkX graph object.

    Converts the RivGraph links and nodes dictionaries into a NetworkX graph
    object.

    Parameters
    ----------
    links : dict
        RivGraph links and their attributes
    nodes : dict
        RivGraph nodes and their attributes
    weight : str, optional
        Link attribute to use to weight the NetworkX graph. If not provided or
        None, the graph will be unweighted (links of 1 and 0)
    inletweights : list, optional
        Optional manual weights for the inlets when using the super-apex
        functionality. Overrides the weight set by the inlet link attribute
        in favor of values from the provided list. List must be in the same
        order and have the same length as nodes['inlets'].

    Returns
    -------
    G : networkx.DiGraph
        Returns a NetworkX DiGraph object weighted by the link attribute
        specified in the optional parameter `weight`
    """
    if weight is not None and weight not in links.keys():
        raise RuntimeError('Provided weight key not in links dictionary.')

    if weight is None:
        weights = np.ones((len(links['conn']), 1))
    else:
        weights = np.array(links[weight])

    # Check weights
    if np.sum(weights <= 0) > 0:
        raise Warning('One or more of your weights is =< 0. This could cause problems later.')

    if inletweights is not None:
        if 'super_apex' not in nodes.keys():
            raise RuntimeError('Can only specify weights if super-apex has been added.')
        if len(inletweights) != len(nodes['inlets']):
            raise RuntimeError('graphiphy requires {} weights but {} were provided.'.format(len(nodes['inlets']), len(inletweights)))
        # Set weights of inlet links
        for inw, inl in zip(inletweights, nodes['inlets']):
            lconn = nodes['conn'][nodes['id'].index(inl)][:]
            lconn = [lc for lc in lconn if lc in nodes['conn'][nodes['id'].index(nodes['super_apex'])]]
            lidx = links['id'].index(lconn[0])
            weights[lidx] = inw

    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc, wt, lid in zip(links['conn'], weights, links['id']):
        G.add_edge(lc[0], lc[1], weight=wt, linkid=lid)
    return G


def normalize_adj_matrix(G):
    """
    Normalize adjacency matrix.

    Normalizes a graph's adjacency matrix so the sum of weights of each row
    equals one. G is a networkx Graph with weights assigned.

    """
    # First, get adjacency matrix
    A = nx.to_numpy_array(G)
    # Normalize each node
    for r in range(A.shape[0]):
        rowsum = np.sum(A[r, :])
        if rowsum > 0:
            A[r, :] = A[r, :] / np.sum(A[r, :])

    return A


def intermediate_vars(G):
    """
    Compute interemediate variables and matrices.

    Computes the intermediate variables and matrices required to compute
    delta metrics. This function prevents the re-computation of many matrices
    required in the metric functions.

    """
    deltavars = dict()

    # The weighted adjacency matrix (A) of a Directed Acyclic Graph (DAG) has
    # entries a_{uv} that correspond to the fraction of the flux
    # present at node v that flows through the channel (vu). Flux partitioning
    # is done via channel widths.

    # Compute normalized weighted adjacency matrix
    A = normalize_adj_matrix(G)

    deltavars['A_w'] = A.T

    # Apex and outlet nodes
    deltavars['apex'], deltavars['outlets'] = find_inlet_outlet_nodes(deltavars['A_w'])

    """ Weighted Adj """
    deltavars['F_w'], deltavars['SubN_w'] = delta_subN_F(deltavars['A_w'])

    """ Weighted transitional"""
    deltavars['A_w_trans'] = np.matmul(deltavars['A_w'], np.linalg.pinv(np.diag(np.sum(deltavars['A_w'], axis=0))))
    deltavars['F_w_trans'], deltavars['SubN_w_trans'] = delta_subN_F(deltavars['A_w_trans'])

    """ Unweighted Adj"""
    deltavars['A_uw'] = np.array(deltavars['A_w'].copy(), dtype=bool)
    deltavars['F_uw'], deltavars['SubN_uw'] = delta_subN_F(deltavars['A_uw'])

    """ Unweighted transitional"""
    deltavars['A_uw_trans'] = np.matmul(deltavars['A_uw'], np.linalg.pinv(np.diag(np.sum(deltavars['A_uw'], axis=0))))
    deltavars['F_uw_trans'], deltavars['SubN_uw_trans'] = delta_subN_F(deltavars['A_uw_trans'])

    return deltavars


def find_inlet_outlet_nodes(A):
    """
    Find inlet and outlet nodes.

    Given an input adjacency matrix (A), returns the inlet and outlet nodes.
    The graph should contain a single apex
    (i.e. run ensure_single_inlet first).

    """
    apex = np.where(np.sum(A, axis=1) == 0)[0]
    if apex.size != 1:
        raise RuntimeError('The graph contains more than one apex.')
    outlets = np.where(np.sum(A, axis=0) == 0)[0]

    return apex, outlets


def compute_steady_state_link_fluxes(G, links, nodes, weight_name='flux_ss'):
    """Compute steady state fluxes through the network graph.

    Computes the steady state fluxes through links given a directed, weighted,
    NetworkX graph. The network should have only a single inlet (use either
    ensure_single_inlet() or add_super_apex() to do this). Additionally,
    this method will fail if the network has parallel edges. You should first
    run ln_utils artificial_nodes() function to break parallel edges, then
    re-compute link widths and lengths. Method after Tejedor et al 2015 [1]_.

    .. [1] Tejedor, Alejandro, et al. "Delta channel networks: 1. A
       graph‐theoretic approach for studying connectivity and steady state
       transport on deltaic surfaces."
       Water Resources Research 51.6 (2015): 3998-4018.

    Parameters
    ----------
    G : networkx.DiGraph
        NetworkX DiGraph object from graphiphy()
    links : dict
        RivGraph links dictionary
    nodes : dict
        RivGraph nodes dictionary
    weight_name : str, optional
        Name to give the new attribute in the links dictionary, is optional,
        if not provided will be 'flux_ss' (flux steady-state)

    Returns
    -------
    links : dict
        RivGraph links dictionary with new attribute
    """
    # Normalize the adjacency matrix
    An = normalize_adj_matrix(G)
    # Transposed adjacency required for computing F
    An_t = np.transpose(An)
    # Compute steady-state flux distribution
    fluxes, _ = delta_subN_F(An_t)

    # Fluxes are at-a-node and need to be translated to links
    fluxes = np.expand_dims(fluxes, 1)
    # Expand node-fluxes back to full adjacency matrix
    fw = fluxes * An
    # All nonzero elements in fw represent links where there is flux
    rows, cols = np.where(fw > 0)
    Gnodes = list(G.nodes)
    linkfluxes = np.zeros((len(links['id']), 1))  # Preallocate storage
    for (r, c) in zip(rows, cols):
        u = Gnodes[r]
        v = Gnodes[c]
        link_id = G.edges[u, v]['linkid']
        linkfluxes[links['id'].index(link_id)] = fw[r, c]

    # Store the fluxes in the links dict
    links[weight_name] = np.array(linkfluxes).flatten().tolist()

    return links


def delta_subN_F(A, epsilon=10**-10):
    """
    Compute steady state flux distribution.

    Computes the steady state flux distribution in the delta nodes when the
    system is fed with a constant unity influx from the Apex. Also defines the
    subnetworks apex-to-outlet.
    The SubN is an NxM matrix, where N is number of nodes and M is the number
    of outlets. For each mth outlet, its contributing subnetwork is given by
    the nonzero entries in SubN. The values in SubN are the degree of
    "belongingness" of each node to its subnetwork. If SubN(m,n) = 0, the m'th
    node does not belong to the n'th subnetwork; but if SubN(m,n) = 1, the m'th
    node belongs *only* to the n'th subnetwork. The values in SubN may be
    interpreted as the percentage of tracers that pass through node m that
    eventually make their way to the outlet of subnetwork n.

    """
    ApexID, OutletsID = find_inlet_outlet_nodes(A)

    """ Computing the steady-state flux, F """
    # To avoid boundary conditions and with the purpose of computing F, we
    # create a cycled version of the graph by connecting the outlet nodes
    # to the apex
    AC = A.copy()
    AC[ApexID, OutletsID] = 1

    # F is proportional to the eigenvector corresponding to the zero eigenvalue
    # of L=I-AC
    L = np.identity(AC.shape[0]) - np.matmul(AC,
                                             np.linalg.pinv(
                                                np.diag(np.sum(AC, axis=0))))
    d, v = np.linalg.eig(L)
    # Renormalize eigenvectors so that F at apex equals 1
    I = np.where(np.abs(d) < epsilon)[0]
    F = np.abs(v[:, I] / v[ApexID, I])

    """ Computing subnetworks """
    # R is null space of L(Gr)=Din(Gr-Ar(Gr)) - where Gr is the reverse graph,
    # Din the in-degree matrix, and Ar the adjacency matrix of Gr
    Ar = np.transpose(A)
    Din = np.diag(np.sum(Ar, axis=1))
    L = Din - Ar
    d, v = np.linalg.eig(L)
    # Renormalize eigenvectors to one
    for i in range(v.shape[1]):
        # set values below epsilon to 0
        v[:, i][v[:, i] < epsilon] = 0
        if np.max(v[:, i]) == 0:
            continue
        else:
            v[:, i] = v[:, i] / np.max(v[:, i])

    # Null space basis
    SubN = v[:, np.where(np.abs(d) < epsilon)[0]]
    I = np.where(SubN < epsilon)
    SubN[I[0], I[1]] = 0

    return np.squeeze(F), SubN


def nl_entropy_rate(A):
    """
    Compute nonlocal entropy rate.

    Computes the nonlocal entropy rate (nER) corresponding to the delta
    (inlcuding flux partition) represented by matrix A

    """
    # Compute steady-state flux and subnetwork structure
    F, SubN = delta_subN_F(A)
    F = F/np.sum(F)

    # Entropy per node
    Entropy = []
    for i in range(len(F)):
        I = np.where(SubN[i, :] > 0)[0]
        ent = -np.sum(SubN[i, I]*np.log2(SubN[i, I]))
        if len(I) > 1:
            ent = ent / np.log2(len(I))
        Entropy.append(ent)

    nER = np.sum(F*np.array(Entropy))

    return nER


def delta_nER(deltavars, N=500):
    """
    Compute nonlocal entropy rate.

    Compute the nonlocal entrop rate (nER) corresponding to the delta
    (including flux partition) represented by adjacency matrix A, and compares
    its value with the nER resulting from randomizing the flux partition.

    Returns
    -------
    pExc :
        the probability that the value of nER for a randomization of the fluxes
        on the topology dictated by A exceeds the actual value of nER. If the
        value of pExc is lower than 0.10, we considered that the actual partition
        of fluxes is an extreme value of nER
    nER_Delta :
        the nonlinear entropy rate for the provided adjacency matrix
    nER_randA :
        the nonlinear entropy rates for the N randomized deltas

    """
    A = deltavars['A_w_trans'].copy()
    nER_Delta = nl_entropy_rate(A)

    nER_randA = []
    for i in range(N):
        A_rand = A.copy()
        I = np.where(A_rand > 0)
        rand_weights = np.random.uniform(0, 1, (1, len(I[0])))
        A_rand[I] = rand_weights
        A_rand = np.matmul(A_rand, np.linalg.pinv(np.diag(np.sum(A_rand, axis=0))))
        nER_randA.append(nl_entropy_rate(A_rand))

    pExc = len(np.where(nER_randA > nER_Delta)[0]) / len(nER_randA)

    return nER_Delta, pExc, nER_randA


def top_entropy_based_topo(deltavars, epsilon=10**-10):
    """
    Compute topologic mutual information and conditional entropies.

    Computes the Topologic Mutual Information (TMI) and the Topologic
    Conditional Entropy for each subnetwork.

    """
    outlets = deltavars['outlets']

    # Fluxes at each node F and subnetworks subN
    F = deltavars['F_uw_trans'].copy()
    F = F/np.sum(F)
    SubN = deltavars['SubN_uw_trans'].copy()

    # Fluxes at links
    L_F = np.matmul(deltavars['A_uw_trans'], np.diag(F))

    TMI = np.empty((SubN.shape[1], 2))
    TCE = np.empty((SubN.shape[1], 2))
    for i in range(SubN.shape[1]):

        # Nodes that belong to subnetwork i
        nodes_in = np.where(SubN[:, i] > epsilon)[0]
        # Nodes that don't belong to subnetwork i
        nodes_out = np.where(SubN[:, i] < epsilon)[0]
        outlet_SubN = list(set(outlets).intersection(set(nodes_in)))[0]
        # Fluxes within subnetwork i - remove nodes_out
        subN_F = L_F.copy()
        subN_F[:, nodes_out] = 0
        subN_F[nodes_out, :] = 0

        # Compute fluxes leaving (Fn_out) and entering (Fn_in) each node in
        # the subnetwork, and total flux in the subnetwork (FS)
        Fn_out = np.sum(subN_F, axis=0)
        Fn_in = np.sum(subN_F, axis=1)
        FS = np.sum(subN_F)

        # Normalize all fluxes by FS
        subN_F = subN_F / FS
        Fn_out = Fn_out / FS
        Fn_in = Fn_in / FS

        # Compute TMI and TCE
        TMI_sum = 0
        TCE_sum = 0
        for ni in nodes_in:
            downN = np.where(subN_F[:, ni] > 0)[0]
            if len(downN) != 0:
                for d in downN:
                    TMI_sum = TMI_sum + subN_F[d, ni] * \
                              np.log2(subN_F[d, ni] / (Fn_in[d] * Fn_out[ni]))
                    TCE_sum = TCE_sum - subN_F[d, ni] * \
                              np.log2(subN_F[d, ni] * \
                              subN_F[d, ni] / Fn_in[d] / Fn_out[ni])
        TMI[i, 0] = outlet_SubN
        TMI[i, 1] = TMI_sum
        TCE[i, 0] = outlet_SubN
        TCE[i, 1] = TCE_sum

    return TMI, TCE


def top_link_sharing_index(deltavars, epsilon=10**-10):
    """
    Compute the link sharing index.

    Computes the Link Sharing Index (LSI) which quantifies the overlapping
    (in terms of links) of each subnetwork with other subnetworks in the
    delta.
    """
    outlets = deltavars['outlets']

    # Don't need weights
    A = deltavars['A_uw'].copy()

    # Set of links in the network (r, c)
    r, c = np.where(A==True)
    NL = r.shape[0]
    LinkBelong = np.zeros((NL, 1))

    # SubN indicates which nodes belong to each subnetwork
    SubN = deltavars['SubN_uw'].copy()
    NS = SubN.shape[1]
    SubN_Links = [[] for sl in range(NS)]

    # Evalueate LinkBelong and SubN_Links
    for i in range(NL):
        for k in range(NS):
            if SubN[r[i], k] > 0 and SubN[c[i], k] > 0:
                LinkBelong[i] = LinkBelong[i] + 1
                SubN_Links[k].append(i)

    # LSI is defined for each subnetwork as one minus the average
    # inverse LinkBelong
    LSI = np.empty((NS, 2))
    for k in range(NS):
        I = np.where(SubN[outlets, k] > epsilon)[0]
        LSI[k, 0] = outlets[I]
        LSI[k, 1] = 1 - np.nanmean(1 / LinkBelong[SubN_Links[k]])

    return LSI


def top_number_alternative_paths(deltavars, epsilon=10**-15):
    """
    Compute number of alternative paths.

    Computes the number of alternative paths (Nap) in the combinatorics sense
    from the Apex to each of the shoreline outlets.

    """
    apexid = deltavars['apex']
    outlets = deltavars['outlets']

    # Don't need weights
    A = deltavars['A_uw'].copy()

    # To compute Nap we need to find the null space of L==I*-A', where I* is
    # the Identity matrix with zeros for the diagonal entries that correspond
    # to the outlets.
    D = np.ones((A.shape[0], 1))
    D[outlets] = 0
    L = np.diag(np.squeeze(D)) - A.T
    d, v = np.linalg.eig(L)
    d = np.abs(d)
    null_space_v = np.where(np.logical_and(d < epsilon, d > -epsilon))[0]

    # Renormalize eigenvectors of the null space to have one at the outlet entry
    vN = np.abs(v[:, null_space_v])
    paths = np.empty((null_space_v.shape[0], 2))
    for i in range(null_space_v.shape[0]):
        I = np.where(vN[outlets, i] > epsilon)[0]
        vN[:, i] = vN[:, i] / vN[outlets[I], i]
        paths[i, 0] = outlets[I]
        paths[i, 1] = vN[apexid, i]

    return paths


def top_resistance_distance(deltavars, epsilon=10**-15):
    """
    Compute the topologic resistance distance.

    NOTE! TopoDist was not supplied with this function--can use networkX to
    compute shortest path but need to know what "shortest" means
    This function will not work until TopoDist is resolved.
    Computes the resistance distance (RD) from the Apex to each of the
    shoreline outlets. The value of RD between two nodes is the effective
    resistance between the two nodes when each link in the network is replaced
    by a 1 ohm resistor.
    """
    print("Warning: resistance distances are incorrect. See https://github.com/VeinsOfTheEarth/RivGraph/issues/103.")

    apexid = deltavars['apex']
    outlets = deltavars['outlets']

    # Don't need weights
    As = deltavars['A_uw'].copy()

    # Compute the RD within each subnetwork
    SubN = deltavars['SubN_w'].copy()

    RD = np.empty((SubN.shape[1], 2))
    for i in range(SubN.shape[1]):
        # Nodes that don't belong to subnetwork
        I = np.where(np.abs(SubN[:, i]) < epsilon)[0]
        # Zero columns and rows of nodes that are not present in subnetwork i
        As_i = As.copy()
        As_i[I, :] = 0
        As_i[:, I] = 0
        # Laplacian L and its pseudoinverse
        L = np.diag(np.sum(As_i, axis=0)) - As_i
        invL = np.linalg.pinv(L)

        # Compute RD
        I = np.where(SubN[outlets, i] > epsilon)[0]
        o = outlets[I]
        a = apexid
        RD[i, 0] = o

        # Distance between the apex and the ith outlet
        TopoDist = graphshortestpath(As_i, a[0], o[0])

        # RD is normalized by TopoDist to be able to compare networks of different size
        RD[i, 1] = (invL[a, a] + invL[o, o] - invL[a, o] - \
                   invL[o, a]) / TopoDist

    return RD


def graphshortestpath(A, start, finish):
    """
    Find the shortest path.

    Uses networkx functions to find the shortest path along a graph defined
    by A; path is simply defined as the number of links. Actual length not
    considered. Number of links in the shortest path is returned.

    """
    G = nx.from_numpy_array(A)
    sp = nx.shortest_path_length(G, start, finish)

    return sp


def top_s2s_topo_pairwise_dep(deltavars, epsilon=10**-10):
    """
    Compute subnetwork topologic pairwise dependence.

    This  function computes the Subnetwork to Subnetwork Topologic Pairwise
    Dependence (TPD) which quantifies the overlapping for all pairs of
    subnetworks in terms of links.

    """
    outlets = deltavars['outlets']

    # Don't need weights
    A = deltavars['A_uw'].copy()

    # Set of links
    r, c = np.where(A > 0)
    NL = len(r)

    # SubN indicates which nodes belong to each subnetwork
    SubN = deltavars['SubN_uw'].copy()

    NS = SubN.shape[1]
    SubN_Links = [[] for i in range(NS)]

    # Evaluate SubN_Links
    for i in range(NL):
        for k in range(NS):
            if SubN[r[i], k] > 0 and SubN[c[i], k] > 0:
                SubN_Links[k].append(i)

    # Compute TDP
    TDP = np.empty((len(outlets), len(outlets)))
    for i in range(NS):
        for k in range(NS):
            TDP[k, i] = len(set(SubN_Links[i]).intersection(set(SubN_Links[k]))) / len(SubN_Links[k])

    return TDP


def dyn_flux_sharing_index(deltavars, epsilon=10**-10):
    """
    Compute the flux sharing index.

    Computes the Flux Sharing Index (LSI) which quantifies the overlapping
    (in terms of flux) of each subnetwork with other subnetworks in the
    delta.

    """
    outlets = deltavars['outlets']

    # Set of links in the network (r, c)
    r, c = np.where(deltavars['A_w'] > 0)
    NL = r.shape[0]

    # SubN indicates which nodes belong to each subnetwork
    SubN = deltavars['SubN_w'].copy()

    NS = SubN.shape[1]
    SubN_Links = [[] for sl in range(NS)]

    # Evalueate SubN_Links
    for i in range(NL):
        for k in range(NS):
            if SubN[r[i], k] > 0 and SubN[c[i], k] > 0:
                SubN_Links[k].append(i)

    # FSI is defined for each subnetwork as one minus the average inverse SubN
    FSI = np.empty((NS, 2))
    for k in range(NS):
        I = np.where(SubN[outlets, k] > epsilon)[0]
        if len(I) != 0:
            FSI[k, 0] = outlets[I]
            # Downstream nodes of all the links in the subnetwork
            NodesD = r[SubN_Links[k]]
            FSI[k, 1] = 1 - np.nanmean(SubN[NodesD, k])
        else:
            FSI[k, 0] = np.nan
            FSI[k, 1] = np.nan

    return FSI


def dyn_leakage_index(deltavars, epsilon=10**-10):
    """
    Compute the leakage index.

    Computes the LI which accounts for the fraction of flux in subnetwork i
    leaked to other subnetworks.

    """
    apexid = deltavars['apex']
    outlets = deltavars['outlets']

    A = deltavars['A_w'].copy()

    # Check that the inlet node is at a bifurcation
    a = apexid
    I = np.where(A[:, a] > 0)[0]
    if len(I) < 2:
        print('Warning: the apex of the delta has only one node downstream. It is recommended that there be at least two downstream links from the apex to avoid biases.')

    # Fluxes at each node F and subnetworks subN
    F = deltavars['F_w'].copy()
    SubN = deltavars['SubN_w'].copy()

    # Link fluxes
    L_F = np.matmul(deltavars['A_w_trans'], np.diag(F))

    # Mathematically LI is computed for each subnetwork as the difference on
    # the fluxes at the nodes minus the links normalized by the total flux
    # in the links
    LI = np.empty((SubN.shape[1], 2))
    for i in range(SubN.shape[1]):
        # Nodes that belong to subnetwork i
        nodes_in = np.where(SubN[:, i] > epsilon)[0]
        # Nodes that do not belong to subnetwork i
        nodes_out = np.where(SubN[:, i] < epsilon)
        outlet_subN = set(outlets).intersection(set(nodes_in))
        if len(outlet_subN) != 0:
            outlet_subN = outlet_subN.pop()
            # Fluxes within subnetwork i -- remove nodes_out
            subN_F = L_F.copy()
            subN_F[:, nodes_out] = 0
            subN_F[nodes_out, :] = 0
            # Active links within subnetwork
            links_subN = np.where(subN_F > 0)
            # Sum of the fluxes in all the li nks in the subnetwork
            sum_links = np.sum(subN_F[links_subN[0], links_subN[1]])
            # Sum of the fluxes in all the nodes (except the outlet--since it
            # cannot leak out by definition)
            sum_nodes = np.sum(F[nodes_in]) - F[outlet_subN]

            LI[i,0] = outlet_subN
            LI[i,1] = (sum_nodes - sum_links) / sum_nodes
        else:
            LI[i,:] = np.nan

    return LI


def dyn_pairwise_dep(deltavars, epsilon=10**-10):
    """
    Compute subnetwork dynamic pairwise dependence.

    Computes the subnetwork to subnetwork dynamic pairwise dependence (DPD)
    which quantifies the overlapping for all pairs of subnetworks in terms of
    flux.

    """
    A = deltavars['A_w_trans'].copy()

    # Set of links in the network (r, c)
    r, c = np.where(A > 0)
    NL = r.shape[0]

    # SubN indicates which nodes belong to each subnetwork
    F = deltavars['F_w_trans'].copy()
    SubN = deltavars['SubN_w_trans'].copy()

    NS = SubN.shape[1]
    SubN_Links = [[] for sl in range(NS)]

    # Evalueate SubN_Links
    for i in range(NL):
        for k in range(NS):
            if SubN[r[i], k] > 0 and SubN[c[i], k] > 0:
                SubN_Links[k].append(i)

    # Link fluxes
    L_F = np.matmul(A, np.diag(F))

    # Compute DPD
    DPD = np.empty((NS, NS))
    for i in range(NS):
        for k in range(NS):
            link_intersect = list(set(SubN_Links[i]).intersection(set(SubN_Links[k])))
            links_in_s = SubN_Links[k]
            DPD[k, i] = np.sum(L_F[r[link_intersect],
                               c[link_intersect]]) / np.sum(L_F[r[links_in_s],
                                                            c[links_in_s]])

    return DPD


def dyn_entropy_based_dyn(deltavars, epsilon=10**-10):
    """
    Compute dynamic mutual information and dynamic conditional entropy.

    Computes the Dynamic Mutual Information (DMI) and the Dynamic
    Conditional Entropy for each subnetwork.

    """
    outlets = deltavars['outlets']
    A = deltavars['A_w_trans'].copy()

    # Fluxes at each node F and subnetworks subN
    F = deltavars['F_w_trans'].copy()
    SubN = deltavars['SubN_w_trans'].copy()
    F = F / np.sum(F)
    # Fluxes at links
    L_F = np.matmul(A, np.diag(F))

    DMI = np.empty((SubN.shape[1], 2))
    DCE = np.empty((SubN.shape[1], 2))
    for i in range(SubN.shape[1]):

        # Nodes that belong to subnetwork i
        nodes_in = np.where(SubN[:, i] > epsilon)[0]
        # Nodes that don't belong to subnetwork i
        nodes_out = np.where(SubN[:, i] < epsilon)[0]
        outlet_SubN = list(set(outlets).intersection(set(nodes_in)))[0]
        # Fluxes within subnetwork i - remove nodes_out
        subN_F = L_F.copy()
        subN_F[:, nodes_out] = 0
        subN_F[nodes_out, :] = 0

        # Compute fluxes leaving (Fn_out) and entering (Fn_in) each node in
        # the subnetwork, and total flux in the subnetwork (FS)
        Fn_out = np.sum(subN_F, axis=0)
        Fn_in = np.sum(subN_F, axis=1)
        FS = np.sum(subN_F)

        # Normalize all fluxes by FS
        subN_F = subN_F / FS
        Fn_out = Fn_out / FS
        Fn_in = Fn_in / FS

        # Compute TMI and TCE
        DMI_sum = 0
        DCE_sum = 0
        for ni in nodes_in:
            downN = np.where(subN_F[:, ni] > 0)[0]
            if len(downN) != 0:
                for d in downN:
                    DMI_sum = DMI_sum + subN_F[d, ni] * \
                              np.log2(subN_F[d, ni] / (Fn_in[d] * Fn_out[ni]))
                    DCE_sum = DCE_sum - subN_F[d, ni] * \
                              np.log2(subN_F[d, ni] * subN_F[d, ni] / Fn_in[d] / Fn_out[ni])
        DMI[i, 0] = outlet_SubN
        DMI[i, 1] = DMI_sum
        DCE[i, 0] = outlet_SubN
        DCE[i, 1] = DCE_sum

    return DMI, DCE


def dist_from_apex(nodes, imshape):
    """Calculate normalized distance from apex.

    Does this for nodes. Calculates a normalized distances from apex, ignores
    pixel resolution.

    Parameters
    ----------
    nodes : dict
        RivGraph dictionary of nodes
    imshape : tuple
        Tuple of the shape of the domain (e.g., Imask.shape)

    Returns
    -------
    norm_dist : list
        List of normalized straight line distances between each node and the
        inlet in the same order as the nodes come in the input nodes
        dictionary.
    """
    # id row/coord of the apex (or representative location)
    apex_id = nodes['inlets']
    if len(apex_id) < 1:
        raise ValueError('No inlets')
    elif len(apex_id) > 1:
        # average inlet locations to a single point
        ins_idx = [nodes['idx'][nodes['id'].index(i)] for i in apex_id]
        rs, cs = np.unravel_index(ins_idx, imshape)
        apex_xy = np.mean(rs, dtype=int), np.mean(cs, dtype=int)
    else:
        apex_idx = nodes['idx'][nodes['id'].index(apex_id)]
        apex_xy = np.unravel_index(apex_idx, imshape)

    # calculate distances to all nodes from apex location
    def calc_dist(apex_xy, node_xy):
        """Euclidean distance function."""
        return np.sqrt((apex_xy[0]-node_xy[0])**2 +
                       (apex_xy[1]-node_xy[1])**2)

    # get coordinates of all nodes in xy space
    node_xy = [np.unravel_index(i, imshape) for i in nodes['idx']]
    node_dists = [calc_dist(apex_xy, i) for i in node_xy]
    # normalize and return this normalized distance
    norm_dist = list(np.array(node_dists) / np.max(node_dists))

    return norm_dist


def calc_QR(links, nodes, wt='wid_adj', new_at='graphQR'):
    """Clunky solution (target for optimization) to get QR at bifurcations.

    QR is defined as the larger branch Q / smaller branch Q per
    Edmonds & Slingerland 2008 [2]_. This measure of flux partitioning at a
    bifurcation does not scale beyond bifurcations to trifurcations etc.
    The graph-based flux partitioning scheme also assumes flow is routed
    in a steady-state manner based on the width (or some other attribute)
    of the links in the network. Therefore the actual flux value doesn't
    matter, we can calculate QR as larger width / smaller width from the two
    branches as that will be the same as if we'd calculated the steady-state
    fluxes and taken their ratio.
    The function is written flexibly to allow one to assuming flux weighting
    by an attribute other than the link width if desired.

    .. warning::

      QR values calculated at nodes located at confluences, polyfurcations,
      or any other non-bifurcating location will be incorrect!

    .. [2] Edmonds, D. A., and R. L. Slingerland. "Stability of delta
       distributary networks and their bifurcations."
       Water Resources Research 44.9 (2008).

    Parameters
    ----------
    links : dict
        RivGraph links dictionary
    nodes : dict
        RivGraph nodes dictionary
    wt : str, optional
        String pointing to the link attribute to use when calculating ratios,
        optional, default is 'wid_adj' which is the adjusted link widths
    new_at : str, optional
        Name of the new attribute to add to the nodes dictionary, optional,
        default is 'graphQR' to indicate the graph calculated QR value
    Returns
    -------
    nodes : dict
        RivGraph dictionary with new_at attribute added
    """
    # check links for wt attribute
    if wt not in links.keys():
        raise ValueError('wt attribute not in the links dictionary')

    # set up list of zeros
    nodes[new_at] = np.zeros_like(nodes['id'], dtype=float)

    for i in range(len(nodes['id'])):
        # for bifurcations
        if len(nodes['conn'][i]) == 3:
            # get the 3 connected link ids
            link_ids = nodes['conn'][i]
            # get upstream node for each link, its "start" point
            link_starts = [links['conn'][links['id'].index(link_ids[0])][0],
                           links['conn'][links['id'].index(link_ids[1])][0],
                           links['conn'][links['id'].index(link_ids[2])][0]]
            # figure out which two links are the ones leaving this node
            # and get the width of each
            # (which controls the local flux partitioning anyway)
            if link_starts[0] == link_starts[1]:
                # check if 1st and 2nd match
                wid_1 = links[wt][links['id'].index(link_ids[0])]
                wid_2 = links[wt][links['id'].index(link_ids[1])]
            elif link_starts[0] == link_starts[2]:
                # check if 1st and 3rd match
                wid_1 = links[wt][links['id'].index(link_ids[0])]
                wid_2 = links[wt][links['id'].index(link_ids[2])]
            else:
                # then 2nd and 3rd must match
                wid_1 = links[wt][links['id'].index(link_ids[1])]
                wid_2 = links[wt][links['id'].index(link_ids[2])]

        # for inlets w/ only 2 connecting links
        elif nodes['id'][i] in nodes['inlets'] and len(nodes['conn'][i]) == 2:
            link_ids = nodes['conn'][i]
            wid_1 = links[wt][links['id'].index(link_ids[0])]
            wid_2 = links[wt][links['id'].index(link_ids[1])]

        # catch-all for other scenarios: QR will be -1
        else:
            wid_1 = -1
            wid_2 = 1

        # calculate and assign QR to the node of interest
        wid_big = np.max([wid_1, wid_2])
        wid_small = np.min([wid_1, wid_2])
        nodes[new_at][i] = wid_big / wid_small

    # coerce into list
    nodes[new_at] = list(nodes[new_at])

    # if junction angles are known, make non-bifurcation node QR values NaNs
    if 'jtype' in nodes.keys():
        for i in range(len(nodes['jtype'])):
            if nodes['jtype'][i] != 'b':
                nodes[new_at][i] = np.nan

    return nodes
