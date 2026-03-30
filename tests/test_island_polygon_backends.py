import numpy as np
import pytest

from rivgraph import mask_utils as mu


GT = (0, 1, 0, 11, 0, -1)


def _square_island_mask():
    imask = np.ones((11, 11), dtype=bool)
    imask[4:7, 4:7] = 0
    return imask


def _diagonal_island_mask():
    imask = np.ones((11, 11), dtype=bool)
    imask[4, 4] = 0
    imask[5, 5] = 0
    imask[6, 6] = 0
    return imask


def _border_and_internal_islands_mask():
    imask = np.ones((11, 11), dtype=bool)
    imask[0:2, 0:2] = 0  # should merge with the exterior and be removed
    imask[6:8, 6:8] = 0
    return imask


@pytest.mark.parametrize(
    "imask, expected_count",
    [(_square_island_mask(), 1), (_diagonal_island_mask(), 1), (_border_and_internal_islands_mask(), 1)],
)
def test_island_polygon_rasterio_preserves_island_ids_and_labels(imask, expected_count):
    islands, iislands = mu.get_island_properties(
        imask, 1, 1, None, GT, ["area"], connectivity=2
    )

    assert len(islands) == expected_count
    assert islands.id.tolist() == sorted(islands.id.tolist())
    assert set(np.unique(iislands)) - {0} == set(islands.id.tolist())


@pytest.mark.parametrize(
    "imask, expected_area",
    [(_square_island_mask(), 9.0), (_diagonal_island_mask(), 3.0)],
)
def test_rasterio_island_polygons_match_labeled_pixel_area(imask, expected_area):
    islands, iislands = mu.get_island_properties(
        imask, 1, 1, None, GT, ["area"], connectivity=2
    )

    assert len(islands) == 1
    assert islands.iloc[0].Area == pytest.approx(expected_area)
    assert islands.geometry.iloc[0].area == pytest.approx(expected_area)

    island_id = islands.id.iloc[0]
    assert np.count_nonzero(iislands == island_id) == expected_area

