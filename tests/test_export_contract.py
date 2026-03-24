"""Contract tests for canonical RivGraph vector exports."""
from __future__ import annotations

import copy

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS

from rivgraph.export_schema import (
    EXPORT_SCHEMA_VERSION,
    RG_LINK_SCHEMA_COLUMNS,
    RG_NODE_SCHEMA_COLUMNS,
    get_extension_for_format,
    normalize_geovector_format,
)
import rivgraph.ln_utils as lnu
from tests._helpers import require_io_utils


@pytest.fixture()
def geo_context():
    dims = (4, 5)
    gt = (500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0)
    crs = CRS.from_epsg(32615)
    return dims, gt, crs


@pytest.fixture()
def synthetic_nodes():
    nodes = {
        "id": [10, 11],
        "idx": [1, 13],
        "flux": [1.25, 2.5],
        "state": ["inlet", "outlet"],
        "conn": [[100], [101, 102]],
        "inlets": [10],
        "outlets": [11],
    }
    _, nodes = lnu.mark_network_ids_finalized(None, nodes)
    return nodes


@pytest.fixture()
def synthetic_links():
    links = {
        "id": [100, 101],
        "idx": [np.array([0, 1, 2]), np.array([6, 11, 16])],
        "flux": [3.5, 4.5],
        "certain": [True, False],
        "conn": [[10, 12], [11, 13]],
        "wid_pix": [np.array([1, 2, 3]), np.array([2, 2, 2])],
        "wid_adj": [5.0, 6.0],
    }
    links, _ = lnu.mark_network_ids_finalized(links, None)
    return links


@pytest.fixture()
def node_gpkg(tmp_path, geo_context, synthetic_nodes):
    io_utils = require_io_utils()
    dims, gt, crs = geo_context
    path = tmp_path / "nodes_contract.gpkg"
    io_utils.nodes_to_geofile(synthetic_nodes, dims, gt, crs, str(path))
    return gpd.read_file(path)


@pytest.fixture()
def link_gpkg(tmp_path, geo_context, synthetic_links, synthetic_nodes):
    io_utils = require_io_utils()
    dims, gt, crs = geo_context
    path = tmp_path / "links_contract.gpkg"
    io_utils.links_to_geofile(synthetic_links, dims, gt, crs, str(path), nodes=synthetic_nodes)
    return gpd.read_file(path)


def test_node_export_starts_with_canonical_schema_columns(node_gpkg):
    expected_prefix = list(RG_NODE_SCHEMA_COLUMNS) + ["schema_rg"]
    assert node_gpkg.columns.tolist()[: len(expected_prefix)] == expected_prefix
    assert node_gpkg["schema_rg"].astype(str).tolist() == [EXPORT_SCHEMA_VERSION] * len(node_gpkg)


def test_link_export_starts_with_canonical_schema_columns(link_gpkg):
    expected_prefix = list(RG_LINK_SCHEMA_COLUMNS) + ["schema_rg"]
    assert link_gpkg.columns.tolist()[: len(expected_prefix)] == expected_prefix
    assert link_gpkg["schema_rg"].astype(str).tolist() == [EXPORT_SCHEMA_VERSION] * len(link_gpkg)


def test_extra_attributes_follow_canonical_prefix(node_gpkg, link_gpkg):
    node_prefix_len = len(RG_NODE_SCHEMA_COLUMNS) + 1
    link_prefix_len = len(RG_LINK_SCHEMA_COLUMNS) + 1
    assert node_gpkg.columns.tolist()[node_prefix_len:-1] == ["flux", "state"]
    assert link_gpkg.columns.tolist()[link_prefix_len:-1] == ["flux", "certain", "wid_pix", "wid_adj"]
    assert node_gpkg.columns.tolist()[-1] == "geometry"
    assert link_gpkg.columns.tolist()[-1] == "geometry"


@pytest.mark.parametrize(
    ("token", "ext", "driver"),
    [
        ("gpkg", "gpkg", "GPKG"),
        ("geopackage", "gpkg", "GPKG"),
        ("json", "json", "GeoJSON"),
        ("geojson", "json", "GeoJSON"),
        ("shp", "shp", "ESRI Shapefile"),
        ("shapefile", "shp", "ESRI Shapefile"),
    ],
)
def test_format_normalization_accepts_common_aliases(token, ext, driver):
    fmt = normalize_geovector_format(token)
    assert fmt.ext == ext
    assert fmt.driver == driver
    assert get_extension_for_format(token) == ext


def test_format_normalization_rejects_unknown_formats():
    with pytest.raises(TypeError, match="Only json, shp, and gpkg"):
        normalize_geovector_format("kml")


def test_export_warns_when_ids_are_not_finalized(tmp_path, geo_context):
    io_utils = require_io_utils()
    dims, gt, crs = geo_context
    nodes = {
        "id": [10, 11],
        "idx": [1, 13],
        "conn": [[100], [101]],
        "inlets": [10],
        "outlets": [11],
    }
    path = tmp_path / "nodes_warn.gpkg"
    with pytest.warns(UserWarning, match="provisional IDs"):
        io_utils.nodes_to_geofile(nodes, dims, gt, crs, str(path))
