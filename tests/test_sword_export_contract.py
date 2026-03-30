"""Contract tests for SWORD-style exports."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS

from rivgraph.export_schema import (
    SWORD_NODE_PLACEHOLDER_COLUMNS,
    SWORD_NODE_SCHEMA_COLUMNS,
    SWORD_REACH_PLACEHOLDER_COLUMNS,
    SWORD_REACH_SCHEMA_COLUMNS,
)
import rivgraph.ln_utils as lnu
from tests._helpers import require_io_utils


@pytest.fixture()
def sword_geo_context():
    dims = (4, 5)
    gt = (500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0)
    crs = CRS.from_epsg(32615)
    return dims, gt, crs


@pytest.fixture()
def sword_network():
    nodes = {
        "id": [10, 11, 12],
        "idx": [1, 13, 18],
        "conn": [[100], [100, 101], [101]],
        "inlets": [10],
        "outlets": [12],
    }
    links = {
        "id": [100, 101],
        "idx": [np.array([0, 1, 2, 7, 12]), np.array([12, 13, 18])],
        "conn": [[10, 11], [11, 12]],
        "wid_pix": [np.array([1, 2, 3, 2, 1]), np.array([2, 2, 2])],
        "wid_adj": [5.0, 6.0],
        "len": [120.0, 60.0],
        "certain": [True, True],
        "flux_ss": [0.7, 0.7],
    }
    return lnu.mark_network_ids_finalized(links, nodes)


def _read_pair(nodes_path: Path, reaches_path: Path):
    return gpd.read_file(nodes_path), gpd.read_file(reaches_path)


def test_build_sword_geodataframes_uses_stable_contract_order(sword_geo_context, sword_network):
    io_utils = require_io_utils()
    dims, gt, crs = sword_geo_context
    links, nodes = sword_network

    sword_nodes, sword_reaches = io_utils.build_sword_geodataframes(
        links, nodes, dims, gt, crs, "meter", metadata={"network": "demo", "custom_tag": "x"}
    )

    expected_node_prefix = list(SWORD_NODE_SCHEMA_COLUMNS) + list(SWORD_NODE_PLACEHOLDER_COLUMNS)
    expected_reach_prefix = list(SWORD_REACH_SCHEMA_COLUMNS) + list(SWORD_REACH_PLACEHOLDER_COLUMNS)

    assert sword_nodes.crs.to_epsg() == 4326
    assert sword_reaches.crs.to_epsg() == 4326
    assert sword_nodes.columns.tolist()[: len(expected_node_prefix)] == expected_node_prefix
    assert sword_reaches.columns.tolist()[: len(expected_reach_prefix)] == expected_reach_prefix
    assert sword_nodes.columns.tolist()[-2:] == ["custom_tag", "geometry"]
    assert sword_reaches.columns.tolist()[-2:] == ["custom_tag", "geometry"]
    assert sword_nodes["network"].astype(str).tolist() == ["demo"] * len(sword_nodes)
    assert sword_reaches["network"].astype(str).tolist() == ["demo"] * len(sword_reaches)


@pytest.mark.parametrize("ftype", ["gpkg", "json"])
def test_sword_export_contract_roundtrip(tmp_path, sword_geo_context, sword_network, ftype):
    io_utils = require_io_utils()
    dims, gt, crs = sword_geo_context
    links, nodes = sword_network

    nodes_path = tmp_path / f"nodes_sword.{ftype}"
    reaches_path = tmp_path / f"reaches_sword.{ftype}"
    io_utils.export_for_sword(
        links,
        nodes,
        dims,
        gt,
        crs,
        {"nodes_sword": str(nodes_path), "reaches_sword": str(reaches_path)},
        "meter",
        metadata={"network": "demo"},
        flux_attr="flux_ss",
    )

    sword_nodes, sword_reaches = _read_pair(nodes_path, reaches_path)

    expected_node_prefix = list(SWORD_NODE_SCHEMA_COLUMNS) + list(SWORD_NODE_PLACEHOLDER_COLUMNS)
    expected_reach_prefix = list(SWORD_REACH_SCHEMA_COLUMNS) + list(SWORD_REACH_PLACEHOLDER_COLUMNS)

    assert sword_nodes.crs.to_epsg() == 4326
    assert sword_reaches.crs.to_epsg() == 4326
    assert sword_nodes.columns.tolist()[: len(expected_node_prefix)] == expected_node_prefix
    assert sword_reaches.columns.tolist()[: len(expected_reach_prefix)] == expected_reach_prefix
    assert sword_nodes.columns.tolist()[-1] == "geometry"
    assert sword_reaches.columns.tolist()[-1] == "geometry"

    assert sword_reaches["reach_id_R"].tolist() == [100, 101]
    assert sword_reaches["fdir_set"].astype(bool).tolist() == [True, True]
    assert sword_reaches["rg_us_nd"].tolist() == [10, 11]
    assert sword_reaches["rg_ds_nd"].tolist() == [11, 12]
    assert sword_reaches["rg_inlet"].astype(bool).tolist() == [True, False]
    assert sword_reaches["rg_outlet"].astype(bool).tolist() == [False, True]
    assert np.allclose(sword_reaches["rg_flux"].astype(float), [0.7, 0.7])
    assert np.isnan(float(sword_reaches["rg_outflx"].iloc[0]))
    assert float(sword_reaches["rg_outflx"].iloc[1]) == pytest.approx(0.7)
    assert sword_reaches["network"].astype(str).tolist() == ["demo"] * len(sword_reaches)

    assert len(sword_nodes) >= len(links["id"])
    assert set(sword_nodes["reach_id_R"].tolist()) == {100, 101}
    assert set(sword_nodes["fdir_set"].astype(bool).tolist()) == {True}
    assert np.allclose(sword_nodes["rg_flux"].astype(float), 0.7)


def test_sword_export_requires_meter_units(sword_geo_context, sword_network):
    io_utils = require_io_utils()
    dims, gt, crs = sword_geo_context
    links, nodes = sword_network

    with pytest.raises(TypeError, match="meters-based CRS"):
        io_utils.build_sword_geodataframes(links, nodes, dims, gt, crs, "degree")
