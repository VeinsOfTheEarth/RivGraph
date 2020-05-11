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
by Alej Tejedor to compute topologic and dynamic metrics on deltas. The provided
Matlab script required the bioinformatics toolbox; here we use networkx to
achieve the same result. Ported by Jon Schwenk.

The conversion was tested by computing metrics for the Wax Lake Delta
(provided by AT) and the Yenesei Delta (provided by JS)--perfect agreement
was found for all metrics, for both deltas, using both the original Matlab
scripts and the Python functions provided here.

JS has made some efficiency improvments to the code; otherwise most variable
names and code structure was matched to the original Matlab scripts.
"""

def compute_delta_metrics(links, nodes):

    # Delta metrics require a single apex node
    links_m, nodes_m = ensure_single_inlet(links, nodes)

    # Ensure we have a directed, acyclic graph; also include widths as weights
    G = graphiphy(links, nodes, weight='wid_adj')

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
    TMI, TCE = top_entropy_based_topo(deltavars)
    metrics['top_mutual_info'] = TMI
    metrics['top_conditional_entropy'] = TCE
    metrics['top_link_sharing_idx'] = top_link_sharing_index(deltavars)
    metrics['n_alt_paths'] = top_number_alternative_paths(deltavars)
    metrics['resistance_distance'] = top_resistance_distance(deltavars)
    metrics['top_pairwise_dependence'] = top_s2s_topo_pairwise_dep(deltavars)
    metrics['flux_sharing_idx'] = dyn_flux_sharing_index(deltavars)
    metrics['leakage_idx'] = dyn_leakage_index(deltavars)
    metrics['dyn_pairwise_dependence'] = dyn_pairwise_dep(deltavars)
    DMI, DCE = dyn_entropy_based_dyn(deltavars)
    metrics['dyn_mutual_info'] = DMI
    metrics['dyn_conditional_entropy'] = DCE

    return metrics


def graphiphy(links, nodes, weight=None):

    if weight is not None and weight not in links.keys():
        raise RuntimeError('Provided weight key not in nodes dictionary.')

    if weight is None:
        weights = np.ones((len(links['conn']),1))
    else:
        weights = links[weight]

    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], weights):
        G.add_edge(lc[0], lc[1], weight=wt)

    return G


def normalize_adj_matrix(G):
    """
    Normalizes a graph's adjacency matrix so the sum of weights of each row
    equals one. G is a networkx Graph with weights assigned.
    """

    # First, get adjacency matrix
    A = nx.to_numpy_array(G)
    # Normalize each node
    for r in range(A.shape[0]):
        rowsum = np.sum(A[r,:])
        if rowsum > 0:
            A[r,:] = A[r,:] / np.sum(A[r,:])

    return A



def intermediate_vars(G):
    """
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
    deltavars['A_uw'] = np.array(deltavars['A_w'].copy(), dtype=np.bool)
    deltavars['F_uw'], deltavars['SubN_uw'] = delta_subN_F(deltavars['A_uw'])

    """ Unweighted transitional"""
    deltavars['A_uw_trans'] = np.matmul(deltavars['A_uw'], np.linalg.pinv(np.diag(np.sum(deltavars['A_uw'], axis=0))))
    deltavars['F_uw_trans'], deltavars['SubN_uw_trans'] = delta_subN_F(deltavars['A_uw_trans'])

    return deltavars


def ensure_single_inlet(links, nodes):
    """
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
            # Remove the links connected to the bad node - the hanging node will also be removed
            connlinks = nodes_edit['conn'][nodes_edit['id'].index(badnode)]
            for cl in connlinks:
                links_edit, nodes_edit = lnu.delete_link(links_edit, nodes_edit, cl)

            badnodes = dy.check_continuity(links_edit, nodes_edit)

    # Ensure there are at least two links emanating from the inlet node
    conn = nodes_edit['conn'][nodes_edit['id'].index(main_inlet)]
    while len(conn) == 1:
        main_inlet_new = links_edit['conn'][links_edit['id'].index(conn[0])][:]
        main_inlet_new.remove(main_inlet)
        links_edit, nodes_edit = lnu.delete_link(links_edit, nodes_edit, conn[0])

        # Update new inlet node
        nodes_edit['inlets'].remove(main_inlet)
        main_inlet = main_inlet_new[0]
        nodes_edit['inlets'] = nodes_edit['inlets'] + [main_inlet]
        conn = nodes_edit['conn'][nodes_edit['id'].index(main_inlet)]

    return links_edit, nodes_edit


def find_inlet_outlet_nodes(A):
    """
    Given an input adjacency matrix (A), returns the inlet and outlet nodes.
    The graph should contain a single apex (i.e. run ensure_single_inlet first).
    """
    apex = np.where(np.sum(A, axis=1)==0)[0]
    if apex.size != 1:
        raise RuntimeError('The graph contains more than one apex.')
    outlets = np.where(np.sum(A, axis=0)==0)[0]

    return apex, outlets


def delta_subN_F(A, epsilon = 10**-10):
    """
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
    # create a cycled version of the graph by connecting the outlet nodes to the apex
    AC = A.copy()
    AC[ApexID, OutletsID] = 1

    # F is proportional to the eigenvector corresponding to the zero eigenvalue
    # of L=I-AC
    L = np.identity(AC.shape[0]) - np.matmul(AC, np.linalg.pinv(np.diag(np.sum(AC,axis=0))))
    d, v = np.linalg.eig(L)
    # Renormalize eigenvectors so that F at apex equals 1
    I = np.where(np.abs(d)<epsilon)[0]
    F = np.abs(v[:,I] / v[ApexID, I])

    """ Computing subnetworks """
    # R is null space of L(Gr)=Din(Gr-Ar(Gr)) - where Gr is the reverse graph,
    # Din the in-degree matrix, and Ar the adjacency matrix of Gr
    Ar = np.transpose(A)
    Din = np.diag(np.sum(Ar, axis=1))
    L = Din - Ar
    d, v = np.linalg.eig(L)
    # Renormalize eigenvectors to one
    for i in range(v.shape[1]):
        if np.max(v[:,i]) == 0:
            continue
        else:
            v[:,i] = v[:,i] / np.max(v[:,i])

    # Null space basis
    SubN = v[:, np.where(np.abs(d)<epsilon)[0]]
    I = np.where(SubN<epsilon)
    SubN[I[0], I[1]] = 0

    return np.squeeze(F), SubN


def nl_entropy_rate(A):
    """
    Computes the nonlocal entropy rate (nER) corresponding to the delta
    (inlcuding flux partition) represented by matrix A
    """
    # Compute steady-state flux and subnetwork structure
    F, SubN = delta_subN_F(A)
    F = F/np.sum(F)

    # Entropy per node
    Entropy = []
    for i in range(len(F)):
        I = np.where(SubN[i,:]>0)[0]
        ent = -np.sum(SubN[i,I]*np.log2(SubN[i,I]))
        if len(I) > 1:
            ent = ent / np.log2(len(I))
        Entropy.append(ent)

    nER = np.sum(F*np.array(Entropy))

    return nER


def delta_nER(deltavars, N=500):
    """
    Compute the nonlocal entrop rate (nER) corresponding to the delta
    (including flux partition) represented by adjacency matrix A, and compares
    its value with the nER resulting from randomizing the flux partition.

    OUTPUTS:
        pExc: the probability that the value of nER for a randomization of the fluxes
              on the topology dictated by A exceeds the actual value of nER. If the
              value of pExc is lower than 0.10, we considered that the actual partition
              of fluxes is an extreme value of nER
      nER_Delta: the nonlinear entropy rate for the provided adjacency matrix
      nER_randA: the nonlinear entropy rates for the N randomized deltas
    """

    A = deltavars['A_w_trans'].copy()
    nER_Delta = nl_entropy_rate(A)

    nER_randA = []
    for i in range(N):
        A_rand = A.copy()
        I = np.where(A_rand>0)
        rand_weights = np.random.uniform(0,1,(1,len(I[0])))
        A_rand[I] = rand_weights
        A_rand = np.matmul(A_rand, np.linalg.pinv(np.diag(np.sum(A_rand, axis=0))))
        nER_randA.append(nl_entropy_rate(A_rand))

    pExc = len(np.where(nER_randA > nER_Delta)[0]) / len(nER_randA)

    return nER_Delta, pExc, nER_randA


def top_entropy_based_topo(deltavars, epsilon = 10**-10):
    """
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

    TMI = np.empty((SubN.shape[1],2))
    TCE = np.empty((SubN.shape[1],2))
    for i in range(SubN.shape[1]):

        # Nodes that belong to subnetwork i
        nodes_in = np.where(SubN[:,i] > epsilon)[0]
        # Nodes that don't belong to subnetwork i
        nodes_out = np.where(SubN[:,i] < epsilon)[0]
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
            downN = np.where(subN_F[:,ni] > 0)[0]
            if len(downN) != 0:
                for d in downN:
                    TMI_sum = TMI_sum + subN_F[d, ni] * np.log2(subN_F[d, ni] / (Fn_in[d] * Fn_out[ni]))
                    TCE_sum = TCE_sum - subN_F[d, ni] * np.log2(subN_F[d, ni] * subN_F[d, ni] / Fn_in[d] / Fn_out[ni])
        TMI[i,0] = outlet_SubN
        TMI[i,1] = TMI_sum
        TCE[i,0] = outlet_SubN
        TCE[i,1] = TCE_sum

    return TMI, TCE


def top_link_sharing_index(deltavars, epsilon= 10**-10):
    """
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
    LinkBelong = np.zeros((NL,1))

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

    # LSI is defined for each subnetwork as one minus the average inverse LinkBelong
    LSI = np.empty((NS,2))
    for k in range(NS):
        I = np.where(SubN[outlets, k] > epsilon)[0]
        LSI[k,0] = outlets[I]
        LSI[k,1] = 1 - np.nanmean(1 / LinkBelong[SubN_Links[k]])

    return LSI


def top_number_alternative_paths(deltavars, epsilon= 10**-15):
    """
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
    D = np.ones((A.shape[0],1))
    D[outlets] = 0
    L = np.diag(np.squeeze(D)) - A.T
    d, v = np.linalg.eig(L)
    d = np.abs(d)
    null_space_v = np.where(np.logical_and(d < epsilon, d > -epsilon))[0]

    # Renormalize eigenvectors of the null space to have one at the outlet entry
    vN = np.abs(v[:, null_space_v])
    paths = np.empty((null_space_v.shape[0],2))
    for i in range(null_space_v.shape[0]):
        I = np.where(vN[outlets, i] > epsilon)[0]
        vN[:,i] = vN[:,i] / vN[outlets[I], i]
        paths[i,0] = outlets[I]
        paths[i,1] = vN[apexid, i]

    return paths


def top_resistance_distance(deltavars, epsilon=10**-15):
    """
    NOTE! TopoDist was not supplied with this function--can use networkX to compute shortest path but need to know what "shortest" means
    This function will not work until TopoDist is resolved.

    Computes the resistance distance (RD) from the Apex to each of the
    shoreline outlets. The value of RD between two nodes is the effective
    resistance between the two nodes when each link in the network is replaced
    by a 1 ohm resistor.
    """

    apexid = deltavars['apex']
    outlets = deltavars['outlets']

    # Don't need weights
    As = deltavars['A_uw'].copy()

    # Compute the RD within each subnetwork
    SubN = deltavars['SubN_w'].copy()

    RD = np.empty((SubN.shape[1],2))
    for i in range(SubN.shape[1]):
        # Nodes that don't belong to subnetwork
        I = np.where(np.abs(SubN[:,i]) < epsilon)[0]
        # Zero columns and rows of nodes that are not present in subnetwork i
        As_i = As.copy()
        As_i[I,:] = 0
        As_i[:,I] = 0
        # Laplacian L and its pseudoinverse
        L = np.diag(np.sum(As_i, axis=0)) - As_i
        invL = np.linalg.pinv(L)

        # Compute RD
        I = np.where(SubN[outlets, i] > epsilon)[0]
        o = outlets[I]
        a = apexid
        RD[i,0] = o

        # Distance between the apex and the ith outlet
        TopoDist = graphshortestpath(As_i, a[0], o[0])

        # RD is normalized by TopoDist to be able to compare networks of different size
        RD[i,1] = (invL[a,a] + invL[o,o] - invL[a,o] - invL[o,a]) / TopoDist

    return RD


def graphshortestpath(A, start, finish):
    """
    Uses networkx functions to find the shortest path along a graph defined
    by A; path is simply defined as the number of links. Actual length not
    considered. Number of links in the shortest path is returned.
    """
    import networkx as nx

    G = nx.from_numpy_matrix(A)
    sp = nx.shortest_path_length(G, start, finish)

    return sp


def top_s2s_topo_pairwise_dep(deltavars, epsilon=10**-10):
    """
    This  function computes the Subnetwork to Subnetwork Topologic Pairwise
    Dependence (TPD) which quantifies the overlapping for all pairs of subnetworks
    in terms of links.
    """

    outlets = deltavars['outlets']

    # Don't need weights
    A = deltavars['A_uw'].copy()

    # Set of links
    r, c = np.where(A>0)
    NL = len(r)

    # SubN indicates which nodes belong to each subnetwork
    SubN = deltavars['SubN_uw'].copy()

    NS = SubN.shape[1]
    SubN_Links = [[] for i in range(NS)]

    # Evaluate SubN_Links
    for i in range(NL):
        for k in range(NS):
            if SubN[r[i],k] > 0 and SubN[c[i],k] > 0:
                SubN_Links[k].append(i)

    # Compute TDP
    TDP = np.empty((len(outlets), len(outlets)))
    for i in range(NS):
        for k in range(NS):
            TDP[k,i] = len(set(SubN_Links[i]).intersection(set(SubN_Links[k]))) / len(SubN_Links[k])

    return TDP


def dyn_flux_sharing_index(deltavars, epsilon=10**-10):
    """
    Computes the Flux Sharing Index (LSI) which quantifies the overlapping
    (in terms of flux) of each subnetwork with other subnetworks in the
    delta.
    """
    outlets = deltavars['outlets']

    # Set of links in the network (r, c)
    r, c = np.where(deltavars['A_w']>0)
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
    FSI = np.empty((NS,2))
    for k in range(NS):
        I = np.where(SubN[outlets, k] > epsilon)[0]
        if len(I) != 0:
            FSI[k,0] = outlets[I]
            NodesD = r[SubN_Links[k]] # Downstream nodes of all the links in the subnetwork
            FSI[k,1] = 1 - np.nanmean(SubN[NodesD,k])
        else:
            FSI[k,0] = np.nan
            FSI[k,1] = np.nan


    return FSI


def dyn_leakage_index(deltavars, epsilon=10**-10):
    """
    Computes the LI which accounts for the fraction of flux in subnetwork i
    leaked to other subnetworks.
    """
    apexid = deltavars['apex']
    outlets = deltavars['outlets']

    A = deltavars['A_w'].copy()

    # Check that the inlet node is at a bifurcation
    a = apexid
    I = np.where(A[:,a] > 0)[0]
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
    LI = np.empty((SubN.shape[1],2))
    for i in range(SubN.shape[1]):
        # Nodes that belong to subnetwork i
        nodes_in = np.where(SubN[:,i] > epsilon)[0]
        # Nodes that do not belong to subnetwork i
        nodes_out = np.where(SubN[:,i] < epsilon)
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
            # Sum of the fluxes in all the nodes (except the outlet--since it cannot
            # leak out by definition)
            sum_nodes = np.sum(F[nodes_in]) - F[outlet_subN]

            LI[i,0] = outlet_subN
            LI[i,1] = (sum_nodes - sum_links) / sum_nodes
        else:
            LI[i,:] = np.nan

    return LI


def dyn_pairwise_dep(deltavars, epsilon=10**-10):
    """
    Computes the subnetwork to subnetwork dynamic pairwise dependence (DPD)
    which quantifies the overlapping for all pairs of subnetworks in terms of
    flux.
    """
    A = deltavars['A_w_trans'].copy()

    # Set of links in the network (r, c)
    r, c = np.where(A>0)
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
    DPD = np.empty((NS,NS))
    for i in range(NS):
        for k in range(NS):
            link_intersect = list(set(SubN_Links[i]).intersection(set(SubN_Links[k])))
            links_in_s = SubN_Links[k]
            DPD[k,i] = np.sum(L_F[r[link_intersect], c[link_intersect]]) / np.sum(L_F[r[links_in_s], c[links_in_s]])

    return DPD


def dyn_entropy_based_dyn(deltavars, epsilon = 10**-10):
    """
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

    DMI = np.empty((SubN.shape[1],2))
    DCE = np.empty((SubN.shape[1],2))
    for i in range(SubN.shape[1]):

        # Nodes that belong to subnetwork i
        nodes_in = np.where(SubN[:,i] > epsilon)[0]
        # Nodes that don't belong to subnetwork i
        nodes_out = np.where(SubN[:,i] < epsilon)[0]
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
            downN = np.where(subN_F[:,ni] > 0)[0]
            if len(downN) != 0:
                for d in downN:
                    DMI_sum = DMI_sum + subN_F[d, ni] * np.log2(subN_F[d, ni] / (Fn_in[d] * Fn_out[ni]))
                    DCE_sum = DCE_sum - subN_F[d, ni] * np.log2(subN_F[d, ni] * subN_F[d, ni] / Fn_in[d] / Fn_out[ni])
        DMI[i,0] = outlet_SubN
        DMI[i,1] = DMI_sum
        DCE[i,0] = outlet_SubN
        DCE[i,1] = DCE_sum

    return DMI, DCE
