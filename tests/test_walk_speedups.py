from __future__ import annotations

import numpy as np

from rivgraph import walk


def _sample_branch_skeleton():
    I = np.zeros((9, 9), dtype=bool)
    I[4, 2:7] = True
    I[2:7, 4] = True
    I[3, 3] = True
    return np.pad(I, 4)


def test_get_neighbors_fast_path_matches_legacy_path():
    Iskel = _sample_branch_skeleton()
    ctx = {"flat": np.ravel(Iskel), "ncols": Iskel.shape[1]}

    for idx in np.flatnonzero(Iskel):
        legacy = walk.get_neighbors(int(idx), Iskel)
        fast = walk.get_neighbors(int(idx), Iskel, walk_ctx=ctx)
        assert set(legacy) == set(fast)


def test_branchpoint_lookup_matches_direct_is_bp():
    Iskel = _sample_branch_skeleton()
    ctx = walk.make_walk_context(Iskel)

    for idx in np.flatnonzero(Iskel):
        direct = walk.is_bp(int(idx), Iskel, use_bp_lookup=False)
        cached = walk.is_bp(int(idx), Iskel, walk_ctx=ctx)
        assert direct == cached
