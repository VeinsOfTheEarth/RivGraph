"""Migration-focused geospatial IO regression tests."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS

import rivgraph.ln_utils as lnu
from tests._helpers import require_io_utils, require_rasters


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


def _read_geofile(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    assert len(gdf) > 0
    assert gdf.crs is not None
    assert gdf.geometry.notna().all()
    assert (~gdf.geometry.is_empty).all()
    return gdf


def test_nodes_to_geofile_roundtrip_preserves_crs_geometry_and_attrs(
    tmp_path,
    geo_context,
    synthetic_nodes,
):
    io_utils = require_io_utils()
    dims, gt, crs = geo_context
    path = tmp_path / "nodes_roundtrip.gpkg"

    io_utils.nodes_to_geofile(synthetic_nodes, dims, gt, crs, str(path))
    gdf = _read_geofile(path)

    assert gdf.crs.to_epsg() == crs.to_epsg()
    assert set(gdf.geometry.geom_type) == {"Point"}
    assert gdf["id_node"].tolist() == synthetic_nodes["id"]
    assert gdf["idx_node"].tolist() == synthetic_nodes["idx"]
    assert np.allclose(gdf["flux"].astype(float), synthetic_nodes["flux"])
    assert gdf["state"].astype(str).tolist() == synthetic_nodes["state"]
    assert gdf["id_links"].astype(str).str.contains("100|101").all()
    assert gdf["n_links"].tolist() == [1, 2]
    assert gdf["is_inlet"].astype(bool).tolist() == [True, False]
    assert gdf["is_outlet"].astype(bool).tolist() == [False, True]
    assert gdf["type_io"].astype(str).tolist() == ["inlet", "outlet"]

    expected_xy = [
        (500045.0, 4099985.0),
        (500105.0, 4099925.0),
    ]
    actual_xy = [(geom.x, geom.y) for geom in gdf.geometry]
    assert np.allclose(actual_xy, expected_xy)


def test_links_to_geofile_roundtrip_preserves_crs_geometry_and_attrs(
    tmp_path,
    geo_context,
    synthetic_links,
    synthetic_nodes,
):
    io_utils = require_io_utils()
    dims, gt, crs = geo_context
    path = tmp_path / "links_roundtrip.gpkg"

    io_utils.links_to_geofile(synthetic_links, dims, gt, crs, str(path), nodes=synthetic_nodes)
    gdf = _read_geofile(path)

    assert gdf.crs.to_epsg() == crs.to_epsg()
    assert set(gdf.geometry.geom_type) == {"LineString"}
    assert gdf["id_link"].tolist() == synthetic_links["id"]
    assert np.allclose(gdf["flux"].astype(float), synthetic_links["flux"])
    assert gdf["certain"].astype(str).tolist() == ["True", "False"]
    assert gdf["id_nodes"].astype(str).str.contains("10|11|12|13").all()
    assert gdf["n_nodes"].tolist() == [2, 2]
    assert gdf["id_us_node"].tolist()[0] == 10
    assert np.isnan(gdf["id_us_node"].tolist()[1])
    assert gdf["id_ds_node"].tolist()[0] == 12
    assert np.isnan(gdf["id_ds_node"].tolist()[1])
    assert gdf["wid_pix"].astype(str).str.contains("1|2|3").all()
    assert gdf["is_inlet"].astype(bool).tolist() == [True, False]
    assert gdf["is_outlet"].astype(bool).tolist() == [False, True]
    assert gdf["type_io"].astype(str).tolist() == ["inlet", "outlet"]

    start_coords = [tuple(geom.coords[0]) for geom in gdf.geometry]
    expected_starts = [
        (500015.0, 4099985.0),
        (500045.0, 4099955.0),
    ]
    assert np.allclose(start_coords, expected_starts)


def test_write_geotiff_roundtrip_preserves_values_and_metadata(tmp_path, geo_context):
    io_utils = require_io_utils()
    rasters = require_rasters()
    import rasterio

    _, gt, crs = geo_context
    raster = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16)
    path = tmp_path / "roundtrip.tif"

    io_utils.write_geotiff(
        raster,
        gt,
        crs.to_wkt(),
        str(path),
        dtype='uint16',
        nodata=999,
    )

    with rasterio.open(path) as ds:
        assert ds is not None
        assert ds.shape == raster.shape
        assert rasters.affine_to_geotransform(ds.transform) == gt
        assert CRS(ds.crs).to_epsg() == crs.to_epsg()
        assert ds.nodata == 999
        assert np.array_equal(ds.read(1), raster)


def test_geojson_export_requires_explicit_reproject_flag(tmp_path, geo_context, synthetic_nodes):
    io_utils = require_io_utils()
    dims, gt, crs = geo_context
    path = tmp_path / "nodes_roundtrip.json"

    with pytest.raises(ValueError, match="pass reproject=True"):
        io_utils.nodes_to_geofile(synthetic_nodes, dims, gt, crs, str(path))


def test_geojson_export_reprojects_when_requested(tmp_path, geo_context, synthetic_nodes):
    io_utils = require_io_utils()
    dims, gt, crs = geo_context
    path = tmp_path / "nodes_roundtrip.json"

    io_utils.nodes_to_geofile(synthetic_nodes, dims, gt, crs, str(path), reproject=True)
    gdf = _read_geofile(path)

    assert gdf.crs.to_epsg() == 4326
    assert gdf["id_node"].tolist() == synthetic_nodes["id"]


@pytest.mark.parametrize(
    ("path", "expected_driver"),
    [
        ("network.json", "GeoJSON"),
        ("network.shp", "ESRI Shapefile"),
        ("network.gpkg", "GPKG"),
    ],
)
def test_get_driver_supports_modern_vector_targets(path, expected_driver):
    io_utils = require_io_utils()
    assert io_utils.get_driver(path) == expected_driver
