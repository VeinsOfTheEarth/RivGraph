"""Small tests for walk.py helpers."""
from __future__ import annotations

import numpy as np

from tests._helpers import require_raster_runtime


def _walk_module():
    require_raster_runtime()
    from rivgraph import walk

    return walk


def test_idcs_no_turnaround_cases():
    walk = _walk_module()
    Iskel = np.zeros((5, 5))
    expected = {
        (0, 6): [7, 11, 12],
        (0, 5): [9, 10, 11],
        (0, 4): [3, 8, 9],
        (3, 4): [0, 5, 10],
        (4, 0): [-5, -4, 1],
        (5, 0): [-6, -5, -4],
        (6, 0): [-1, -6, -5],
    }
    for idcs, out in expected.items():
        poss_walk_idcs = walk.idcs_no_turnaround(list(idcs), Iskel)
        assert np.all(poss_walk_idcs == out)
