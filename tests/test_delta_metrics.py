from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from rivgraph.deltas import delta_metrics as dm


def _line_network():
    links = {
        'id': [10, 11],
        'conn': [[0, 1], [1, 2]],
        'wid_adj': [1.0, 1.0],
    }
    nodes = {
        'id': [0, 1, 2],
        'conn': [[10], [10, 11], [11]],
        'inlets': [0],
        'outlets': [2],
    }
    return links, nodes


def _diamond_network():
    links = {
        'id': [10, 11, 12, 13],
        'conn': [[0, 1], [0, 2], [1, 3], [2, 3]],
        'wid_adj': [1.0, 1.0, 1.0, 1.0],
    }
    nodes = {
        'id': [0, 1, 2, 3],
        'conn': [[10, 11], [10, 12], [11, 13], [12, 13]],
        'inlets': [0],
        'outlets': [3],
    }
    return links, nodes




def _parallel_link_network():
    links = {
        'id': [10, 11, 12],
        'conn': [[0, 1], [0, 1], [1, 2]],
        'wid_adj': [1.0, 2.0, 3.0],
        'idx': [[0, 1], [0, 5, 1], [1, 2]],
    }
    nodes = {
        'id': [0, 1, 2],
        'conn': [[10, 11], [10, 11, 12], [12]],
        'inlets': [0],
        'outlets': [2],
    }
    return links, nodes

def _two_inlet_network():
    links = {
        'id': [10, 11, 12],
        'conn': [[0, 2], [0, 3], [1, 0]],
        'wid_adj': [5.0, 5.0, 1.0],
        'certain': [1, 1, 1],
        'idx': [[0, 2], [0, 3], [1, 0]],
    }
    nodes = {
        'id': [0, 1, 2, 3],
        'idx': [0, 1, 2, 3],
        'conn': [[10, 11, 12], [12], [10], [11]],
        'inlets': [0, 1],
        'outlets': [2, 3],
    }
    return links, nodes


def test_compute_delta_metrics_warns_once_and_supports_subset(monkeypatch):
    links, nodes = _diamond_network()
    monkeypatch.setattr(dm, '_EXPERIMENTAL_METRICS_WARNING_EMITTED', False)

    with pytest.warns(dm.ExperimentalDeltaMetricWarning, match='experimental convenience metrics'):
        metrics = dm.compute_delta_metrics(
            links,
            nodes,
            metrics=['n_alt_paths', 'resistance_distance'],
            warn_experimental=True,
        )

    with warnings_not_raised(dm.ExperimentalDeltaMetricWarning):
        metrics2, deltavars = dm.compute_delta_metrics(
            links,
            nodes,
            metrics='n_alt_paths',
            warn_experimental=True,
            return_intermediates=True,
        )

    assert set(metrics.keys()) == {'n_alt_paths', 'resistance_distance'}
    assert set(metrics2.keys()) == {'n_alt_paths'}
    assert 'A_w' in deltavars and 'SubN_w' in deltavars


class warnings_not_raised:
    def __init__(self, category):
        self.category = category

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._records = self._ctx.__enter__()
        warnings.simplefilter('always')
        return self._records

    def __exit__(self, exc_type, exc, tb):
        self._ctx.__exit__(exc_type, exc, tb)
        unexpected = [w for w in self._records if issubclass(w.category, self.category)]
        if unexpected:
            raise AssertionError(f'unexpected warnings raised: {unexpected}')
        return False


def test_synthetic_graph_metrics_make_sense():
    line_links, line_nodes = _line_network()
    line_metrics = dm.compute_delta_metrics(
        line_links,
        line_nodes,
        metrics=['n_alt_paths', 'resistance_distance'],
        warn_experimental=False,
    )
    assert np.allclose(line_metrics['n_alt_paths'][:, 1], [1.0])
    assert np.allclose(line_metrics['resistance_distance'][:, 1], [1.0])

    diamond_links, diamond_nodes = _diamond_network()
    diamond_metrics = dm.compute_delta_metrics(
        diamond_links,
        diamond_nodes,
        metrics=['n_alt_paths', 'resistance_distance'],
        warn_experimental=False,
    )
    assert np.allclose(diamond_metrics['n_alt_paths'][:, 1], [2.0])
    assert np.allclose(diamond_metrics['resistance_distance'][:, 1], [0.5])


def test_prune_to_single_inlet_does_not_mutate_inputs():
    links, nodes = _two_inlet_network()
    links_before = deepcopy(links)
    nodes_before = deepcopy(nodes)

    links_after, nodes_after = dm._prune_to_single_inlet(links, nodes)

    assert links == links_before
    assert nodes == nodes_before
    assert links_after != links_before
    assert nodes_after != nodes_before
    assert nodes_after['inlets'] == [0]


def test_multi_inlet_metrics_require_explicit_policy():
    links, nodes = _two_inlet_network()

    with pytest.raises(ValueError, match='Multiple inlet nodes detected'):
        dm.compute_delta_metrics(links, nodes, warn_experimental=False)


def test_multi_inlet_metrics_report_internal_super_apex_metadata():
    links, nodes = _two_inlet_network()
    links_before = deepcopy(links)
    nodes_before = deepcopy(nodes)

    metrics, deltavars = dm.compute_delta_metrics(
        links,
        nodes,
        inlet='equal',
        metrics='n_alt_paths',
        warn_experimental=False,
        return_intermediates=True,
    )

    meta = deltavars['metric_metadata']
    assert set(metrics.keys()) == {'n_alt_paths'}
    assert meta['n_inlets_original'] == 2
    assert meta['multi_inlet_strategy'] == 'virtual_source_super_apex'
    assert meta['used_super_apex'] is True
    assert np.allclose(list(meta['inlet_weights_normalized'].values()), [0.5, 0.5])
    assert links == links_before
    assert nodes == nodes_before
    assert 'super_apex' not in nodes



def test_metric_preflight_catches_inconsistent_node_link_references():
    links, nodes = _line_network()
    nodes['conn'] = [[10], [10], [11]]

    with pytest.raises(ValueError, match='Invalid network for delta metric computation'):
        dm.compute_delta_metrics(links, nodes, warn_experimental=False)



def test_metric_preflight_catches_unknown_inlet_node():
    links, nodes = _line_network()
    nodes['inlets'] = [999]

    with pytest.raises(ValueError, match=r"nodes\['inlets'\] contains unknown node ids"):
        dm.compute_delta_metrics(links, nodes, warn_experimental=False)



def test_steady_state_preflight_requires_positive_width_weights():
    links, nodes = _line_network()
    links['wid_adj'] = [1.0, 0.0]

    with pytest.raises(ValueError, match=r"links\['wid_adj'\] must be strictly positive"):
        dm.compute_steady_state_link_fluxes(
            None,
            links,
            nodes,
            routing='width',
            validate=True,
        )


def test_parallel_links_are_split_internally_for_metrics_with_warning():
    links, nodes = _parallel_link_network()
    links_before = deepcopy(links)
    nodes_before = deepcopy(nodes)

    with pytest.warns(UserWarning, match='Parallel links detected'):
        metrics, deltavars = dm.compute_delta_metrics(
            links,
            nodes,
            metrics='n_alt_paths',
            warn_experimental=False,
            return_intermediates=True,
        )

    meta = deltavars['metric_metadata']
    assert set(metrics.keys()) == {'n_alt_paths'}
    assert meta['used_parallel_link_split'] is True
    assert meta['n_parallel_link_sets'] == 1
    assert meta['n_artificial_parallel_nodes'] == 1
    assert deltavars['A_w'].shape[0] == 4
    assert links == links_before
    assert nodes == nodes_before
    assert 'arts' not in nodes
