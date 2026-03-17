"""Contract tests for the future Rasterio-backed ``rivgraph.rasters`` module.

These tests are intentionally written against the target API for the raster
backend extraction. They are skipped until ``rivgraph.rasters`` exists, so they
can land before the implementation work starts without breaking the suite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
import warnings

from tests._helpers import require_rasters


@pytest.fixture()
def projected_raster_path(tmp_path: Path) -> Path:
    path = tmp_path / "projected.tif"
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    transform = from_origin(500000.0, 4100000.0, 30.0, 30.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:32615",
        transform=transform,
        nodata=999,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture()
def ungeoreferenced_raster_path(tmp_path: Path) -> Path:
    path = tmp_path / "unprojected.tif"
    data = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
        ) as dst:
            dst.write(data, 1)
    return path


def test_open_raster_reads_array_and_metadata(projected_raster_path: Path):
    rasters = require_rasters()

    rst = rasters.open_raster(projected_raster_path)

    assert isinstance(rst.array, np.ndarray)
    assert rst.array.shape == (3, 4)
    assert rst.shape == (3, 4)
    assert rst.height == 3
    assert rst.width == 4
    assert tuple(rst.gt) == (500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0)
    assert rst.transform == from_origin(500000.0, 4100000.0, 30.0, 30.0)
    assert CRS(rst.crs).to_epsg() == 32615
    assert CRS.from_wkt(rst.wkt).to_epsg() == 32615
    assert rst.pixlen == 30.0
    assert rst.pixarea == 900.0


def test_open_raster_assigns_dummy_georef_without_mutating_source(ungeoreferenced_raster_path: Path):
    rasters = require_rasters()

    rst = rasters.open_raster(ungeoreferenced_raster_path, allow_dummy_georef=True)

    assert rst.shape == (2, 3)
    assert tuple(rst.gt) == (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
    assert CRS(rst.crs).to_epsg() == 4326
    assert CRS.from_wkt(rst.wkt).to_epsg() == 4326
    assert rst.pixlen == 1.0
    assert rst.pixarea == 1.0

    with rasterio.open(ungeoreferenced_raster_path) as src:
        assert src.crs is None


def test_coordinate_helpers_are_self_consistent(projected_raster_path: Path):
    rasters = require_rasters()
    rst = rasters.open_raster(projected_raster_path)

    cols = np.array([0, 1, 3])
    rows = np.array([0, 2, 1])
    xs, ys = rasters.xy_to_coords(cols, rows, rst.gt)
    back = rasters.coords_to_xy(xs, ys, rst.gt)

    assert np.array_equal(back[:, 0], cols)
    assert np.array_equal(back[:, 1], rows)

    idx = np.ravel_multi_index((rows, cols), rst.shape)
    idx_xs, idx_ys = rasters.idx_to_coords(idx, rst.shape, rst.gt)
    assert np.allclose(idx_xs, xs)
    assert np.allclose(idx_ys, ys)


def test_write_geotiff_roundtrip_preserves_multiband_data_and_metadata(tmp_path: Path):
    rasters = require_rasters()
    crs = CRS.from_epsg(32615)
    gt = (500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0)
    path = tmp_path / "multiband_roundtrip.tif"
    raster = np.dstack(
        [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint8),
        ]
    )

    rasters.write_geotiff(raster, gt, crs.to_wkt(), path, nodata=255)

    with rasterio.open(path) as src:
        assert src.count == 2
        assert src.shape == raster.shape[:2]
        assert src.transform.to_gdal() == gt
        assert CRS(src.crs).to_epsg() == 32615
        assert src.nodata == 255
        data = np.moveaxis(src.read(), 0, -1)
        assert np.array_equal(data, raster)


def test_crop_geotiff_updates_data_window_and_transform(tmp_path: Path):
    rasters = require_rasters()
    source = tmp_path / "crop_source.tif"
    out = tmp_path / "crop_out.tif"
    data = np.zeros((6, 7), dtype=np.uint8)
    data[2:4, 3:6] = 1
    transform = from_origin(1000.0, 2000.0, 10.0, 10.0)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:32615",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    returned = rasters.crop_geotiff(source, npad=1, outpath=out)
    assert Path(returned) == out

    with rasterio.open(out) as src:
        cropped = src.read(1)
        assert cropped.shape == (4, 5)
        assert np.array_equal(cropped, data[1:5, 2:7])
        assert src.transform.to_gdal() == (1020.0, 10.0, 0.0, 1990.0, 0.0, -10.0)


def test_downsample_binary_geotiff_updates_resolution_and_values(tmp_path: Path):
    rasters = require_rasters()
    source = tmp_path / "downsample_source.tif"
    out = tmp_path / "downsample_out.tif"
    data = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    transform = from_origin(500.0, 1000.0, 20.0, 20.0)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:32615",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    returned = rasters.downsample_binary_geotiff(source, ds_factor=0.5, output_name=out)
    assert Path(returned) == out

    with rasterio.open(out) as src:
        down = src.read(1)
        assert down.shape == (2, 2)
        assert np.array_equal(down, np.array([[1, 0], [0, 1]], dtype=down.dtype))
        assert src.transform.to_gdal() == (500.0, 40.0, 0.0, 1000.0, 0.0, -40.0)
        assert set(np.unique(down)).issubset({0, 1})
