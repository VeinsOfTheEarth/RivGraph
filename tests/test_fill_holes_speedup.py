from __future__ import annotations

import numpy as np

from rivgraph import im_utils as iu


def test_fill_holes_fills_only_small_enclosed_holes():
    I = np.ones((8, 8), dtype=bool)
    I[2, 2] = 0
    I[4:6, 4:6] = 0

    filled = iu.fill_holes(I, maxholesize=1)

    assert filled[2, 2]
    assert not filled[4, 4]
    assert not filled[5, 5]


def test_fill_holes_does_not_fill_boundary_connected_voids():
    I = np.ones((7, 7), dtype=bool)
    I[:, 0] = 0
    I[3, 3] = 0

    filled = iu.fill_holes(I, maxholesize=10)

    assert not filled[1, 0]
    assert filled[3, 3]
