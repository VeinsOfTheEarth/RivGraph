from pathlib import Path

import numpy as np
import pytest
import rasterio

from rivgraph import mask_utils as mu


REALCASE_MASK = Path(__file__).resolve().parent / "data" / "Orinoco_filled.tif"


def _read_mask_metadata(path):
    with rasterio.open(path) as src:
        imask = src.read(1).astype(bool)
        pixlen = float(abs(src.transform.a))
        pixarea = float(abs(src.transform.a * src.transform.e))
        gt = src.transform.to_gdal()
        return imask, pixlen, pixarea, src.crs, gt


def test_orinoco_rasterio_island_polygons_preserve_ids_and_labels():
    imask, pixlen, pixarea, crs, gt = _read_mask_metadata(REALCASE_MASK)

    islands, iislands = mu.get_island_properties(
        imask, pixlen, pixarea, crs, gt, ["area"], connectivity=2
    )

    assert len(islands) == 128
    assert islands.id.tolist() == sorted(islands.id.tolist())
    assert set(np.unique(iislands)) - {0} == set(islands.id.tolist())


def test_orinoco_rasterio_polygons_match_labeled_pixel_areas():
    imask, pixlen, pixarea, crs, gt = _read_mask_metadata(REALCASE_MASK)

    islands, _ = mu.get_island_properties(
        imask, pixlen, pixarea, crs, gt, ["area"], connectivity=2
    )

    rasterio_geom_area = float(islands.geometry.area.sum())
    pixel_area_sum = float(islands.Area.sum())

    assert rasterio_geom_area == pytest.approx(pixel_area_sum, rel=0, abs=1e-3)
