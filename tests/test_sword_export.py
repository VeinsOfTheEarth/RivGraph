"""Unit tests for SWORD export helpers."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
from pyproj import CRS


class _DummyModule(types.ModuleType):
    """Minimal stub for GDAL/OGR/OSR modules during import-only tests."""

    def __getattr__(self, name):  # pragma: no cover - trivial shim
        return 0


class _DummyDataset:
    """Minimal GDAL-like dataset for coordinate transforms in export tests."""

    RasterYSize = 20
    RasterXSize = 20

    def GetGeoTransform(self):
        return (0.0, 10.0, 0.0, 1000.0, 0.0, -10.0)



def _import_io_utils_with_gdal_stubs():
    for name in ("gdal", "ogr", "osr"):
        if name not in sys.modules:
            sys.modules[name] = _DummyModule(name)

    if "osgeo" not in sys.modules:
        osgeo = types.ModuleType("osgeo")
        osgeo.gdal = sys.modules["gdal"]
        osgeo.ogr = sys.modules["ogr"]
        osgeo.osr = sys.modules["osr"]
        sys.modules["osgeo"] = osgeo

    return importlib.import_module("rivgraph.io_utils")


@pytest.fixture()
def simple_directed_network():
    links = {
        "id": [10, 11, 12],
        "conn": [[1, 2], [2, 3], [2, 4]],
        "idx": [
            np.array([21, 22, 23]),
            np.array([23, 24, 25]),
            np.array([23, 43, 63]),
        ],
        "wid_pix": [
            np.array([20.0, 22.0, 24.0]),
            np.array([18.0, 18.0, 18.0]),
            np.array([12.0, 12.0, 12.0]),
        ],
        "wid_adj": [22.0, 18.0, 12.0],
        "len": [30.0, 30.0, 30.0],
        "certain": np.array([1, 1, 1]),
        "flux_ss": [1.0, 0.6, 0.4],
    }
    nodes = {
        "id": [1, 2, 3, 4],
        "conn": [[10], [10, 11, 12], [11], [12]],
        "inlets": [1],
        "outlets": [3, 4],
    }
    return links, nodes


def test_build_sword_geodataframes_exports_direction_and_flux(simple_directed_network):
    io_utils = _import_io_utils_with_gdal_stubs()
    links, nodes = simple_directed_network

    sword_nodes, sword_reaches = io_utils.build_sword_geodataframes(
        links,
        nodes,
        _DummyDataset(),
        CRS.from_epsg(32615),
        unit="meter",
        metadata={"network": "demo"},
        flux_attr="flux_ss",
    )

    assert set(["fdir_set", "rg_flux"]).issubset(sword_nodes.columns)
    assert set(["rch_id_up", "rch_id_dn", "fdir_set", "rg_flux", "rg_outflx", "rg_us_nd", "rg_ds_nd", "rg_inlet", "rg_outlet"]).issubset(sword_reaches.columns)
    assert set(sword_reaches["network"]) == {"demo"}
    assert sword_nodes.crs.to_epsg() == 4326
    assert sword_reaches.crs.to_epsg() == 4326

    reaches_by_id = sword_reaches.set_index("reach_id_R")
    assert bool(reaches_by_id.loc[10, "fdir_set"]) is True
    assert reaches_by_id.loc[10, "n_rch_up"] == 0
    assert reaches_by_id.loc[10, "n_rch_down"] == 2
    assert reaches_by_id.loc[10, "rch_id_dn"] == "11 12"
    assert reaches_by_id.loc[10, "rg_us_nd"] == 1
    assert reaches_by_id.loc[10, "rg_ds_nd"] == 2
    assert bool(reaches_by_id.loc[10, "rg_inlet"]) is True
    assert bool(reaches_by_id.loc[10, "rg_outlet"]) is False
    assert reaches_by_id.loc[10, "rg_flux"] == pytest.approx(1.0)
    assert reaches_by_id.loc[11, "rg_outflx"] == pytest.approx(0.6)
    assert reaches_by_id.loc[12, "rg_outflx"] == pytest.approx(0.4)

    node_flux_by_reach = sword_nodes.groupby("reach_id_R")["rg_flux"].first().to_dict()
    assert node_flux_by_reach == pytest.approx({10: 1.0, 11: 0.6, 12: 0.4})
    assert sword_nodes["fdir_set"].tolist() == [True, True, True]



def test_build_sword_geodataframes_gracefully_handles_missing_directions_and_flux(simple_directed_network):
    io_utils = _import_io_utils_with_gdal_stubs()
    links, nodes = simple_directed_network
    links = dict(links)
    links.pop("certain")
    links.pop("flux_ss")

    sword_nodes, sword_reaches = io_utils.build_sword_geodataframes(
        links,
        nodes,
        _DummyDataset(),
        CRS.from_epsg(32615),
        unit="meter",
    )

    assert sword_reaches["fdir_set"].tolist() == [False, False, False]
    assert sword_reaches["n_rch_up"].tolist() == [0, 0, 0]
    assert sword_reaches["n_rch_down"].tolist() == [0, 0, 0]
    assert sword_reaches["rg_flux"].isna().all()
    assert sword_reaches["rg_outflx"].isna().all()
    assert sword_nodes["rg_flux"].isna().all()



def test_get_driver_supports_gpkg():
    io_utils = _import_io_utils_with_gdal_stubs()
    assert io_utils.get_driver("demo.gpkg") == "GPKG"
