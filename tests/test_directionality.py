"""Light-weight tests for directionality helpers."""
from __future__ import annotations

import networkx as nx

from tests._helpers import require_gdal_bindings


def _directionality_module():
    require_gdal_bindings()
    from rivgraph import directionality as di

    return di


def test_directionlity_trackers():
    di = _directionality_module()
    links = {'id': [0, 1]}
    nodes = {'id': [0, 1]}
    di.add_directionality_trackers(links, nodes, 'delta')
    assert 'certain' in links.keys()
    assert 'certain_order' in links.keys()
    assert 'certain_alg' in links.keys()
    assert 'guess' in links.keys()
    assert 'guess_alg' in links.keys()
    assert 'maxang' not in links.keys()
    di.add_directionality_trackers(links, nodes, 'river')
    assert 'maxang' in links.keys()


def test_algmap_values():
    di = _directionality_module()
    expected = {
        'sourcesinkfix': -2,
        'manual_set': -1,
        'inletoutlet': 0,
        'continuity': 1,
        'parallels': 2,
        'artificials': 2.1,
        'main_chans': 4,
        'bridges': 5,
        'known_fdr': 6,
        'known_fdr_rs': 6.1,
        'syn_dem': 10,
        'syn_dem_med': 10.1,
        'sym_dem_leftover': 10.2,
        'sp_links': 11,
        'sp_nodes': 12,
        'longest_steepest': 13,
        'three_agree': 15,
        'syn_dem_and_sp': 16,
        'cl_dist_guess': 20,
        'cl_ang_guess': 21,
        'cl_dist_set': 22,
        'cl_ang_set': 23,
        'cl_ang_rs': 23.1,
        'cl_dist_and_ang': 24,
        'short_no_bktrck': 25,
        'wid_pctdiff': 26,
    }
    for key, value in expected.items():
        assert di.algmap(key) == value


def test_merge_list_of_lists():
    di = _directionality_module()
    inlist = [[1, 2, 3], [1, 2, 4], [2, 4, 6]]
    merged = di.merge_list_of_lists(inlist)
    assert merged == [[1, 2, 3, 4, 6]]


def test_flip_links_in_G():
    di = _directionality_module()
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G = di.flip_links_in_G(G, 'all')
    [x, y] = G.edges.data()
    assert x == (2, 1, {})
    assert y == (3, 2, {})
