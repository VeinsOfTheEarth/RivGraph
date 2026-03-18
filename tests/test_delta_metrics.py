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


def test_ensure_single_inlet_does_not_mutate_inputs():
    links, nodes = _two_inlet_network()
    links_before = deepcopy(links)
    nodes_before = deepcopy(nodes)

    links_after, nodes_after = dm.ensure_single_inlet(links, nodes)

    assert links == links_before
    assert nodes == nodes_before
    assert links_after != links_before
    assert nodes_after != nodes_before
    assert nodes_after['inlets'] == [0]
