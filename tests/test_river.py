"""Tests for broader `rivgraph.classes.river` functions."""
import pytest
import sys
import os
import io
import numpy as np
from rivgraph.classes import river
from rivgraph import geo_utils
from rivgraph.rivers import river_utils as ru
from rivgraph.rivers import centerline_utils as cu


def test_river_ne(tmp_path):
    """Test river with exit sides 'ne'."""
    img_path = os.path.normpath(
        'tests/integration/data/Brahma/brahma_mask_clip.tif')
    out_path = os.path.join(tmp_path, 'cropped.tif')
    geo_utils.crop_geotif(img_path, npad=10, outpath=out_path)
    test_ne = river('Brahmclip', out_path,
                    os.path.join(tmp_path, 'brahma'),
                    exit_sides='ne')
    test_ne.compute_network()
    test_ne.compute_mesh()
    test_ne.prune_network()

    # make assertions
    assert len(test_ne.nodes['inlets']) == 1
    assert len(test_ne.nodes['outlets']) == 1
    assert test_ne.exit_sides == 'ne'


def test_river_sw(tmp_path):
    """Test river with exit sides 'sw'."""
    test_sw = river('Brahmclip',
                    os.path.join(tmp_path, 'cropped.tif'),
                    os.path.join(tmp_path, 'brahma'),
                    exit_sides='sw')
    test_sw.compute_network()
    test_sw.compute_mesh()
    test_sw.prune_network()

    # make assertions
    assert len(test_sw.nodes['inlets']) == 1
    assert len(test_sw.nodes['outlets']) == 1
    assert test_sw.exit_sides == 'sw'
