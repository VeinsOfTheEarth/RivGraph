"""Targeted tests for light-weight IO helpers."""
from __future__ import annotations

import os
import numpy as np

from tests._helpers import require_io_utils


import pytest


@pytest.mark.parametrize(
    ("kind", "expected_entries"),
    [
        ("binary", {0: (0, 0, 0, 0), 1: (255, 255, 255, 100)}),
        ("mask", {0: (0, 0, 0, 0), 1: (0, 128, 0, 100)}),
        ("tile", {0: (0, 0, 0, 0), 1: (0, 0, 255, 100)}),
        ("GSW", {0: (0, 0, 0, 0), 1: (0, 0, 0, 0), 2: (176, 224, 230, 100)}),
    ],
)
def test_colortable_entries(kind, expected_entries):
    io_utils = require_io_utils()
    color_table = io_utils.colortable(kind)

    for idx, rgba in expected_entries.items():
        assert np.all(color_table.GetColorEntry(idx) == rgba)


def test_create_manual_dir_csv(tmp_path):
    io_utils = require_io_utils()
    csvpath = tmp_path / "csvtest.csv"
    io_utils.create_manual_dir_csv(str(csvpath))
    assert csvpath.is_file()


def test_prep_paths(tmp_path):
    io_utils = require_io_utils()
    resultsfolder = tmp_path
    name = "new"
    basetiff = tmp_path / "dummy_mask.tif"
    paths = io_utils.prepare_paths(resultsfolder, name, str(basetiff))
    assert isinstance(paths, dict)
    assert paths["basepath"] == os.path.normpath(resultsfolder)
    assert paths["maskpath"] == str(basetiff)
    assert paths["Iskel"] == os.path.join(tmp_path, "new_skel.tif")
    assert paths["Idist"] == os.path.join(tmp_path, "new_dist.tif")
    assert paths["network_pickle"] == os.path.join(tmp_path, "new_network.pkl")
    assert paths["fixlinks_csv"] == os.path.join(tmp_path, "new_fixlinks.csv")
    assert paths["linkdirs"] == os.path.join(tmp_path, "new_link_directions.tif")
    assert paths["metrics"] == os.path.join(tmp_path, "new_metrics.pkl")
    assert paths["shoreline"] == os.path.join(tmp_path, "new_shoreline.shp")
    assert paths["inlet_nodes"] == os.path.join(tmp_path, "new_inlet_nodes.shp")
