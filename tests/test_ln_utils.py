"""Targeted regression tests for ``rivgraph.ln_utils``."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from rivgraph import ln_utils as lnu


def _build_straight_strip(width: int, full_length: int = 31, link_length: int = 11, pad: int = 4):
    """Return a synthetic straight channel and a centerline segment."""
    mask = np.zeros((width + 2 * pad, full_length + 2 * pad), dtype=np.uint8)
    mask[pad : pad + width, pad : pad + full_length] = 1

    # Keep the test link away from the strip ends so endpoint trimming does not
    # affect the width check.
    center_row = pad + width // 2
    start_col = pad + (full_length - link_length) // 2
    rr = np.full(link_length, center_row, dtype=int)
    cc = np.arange(start_col, start_col + link_length, dtype=int)
    links = {"idx": [np.ravel_multi_index((rr, cc), mask.shape)]}

    return mask, links


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6, 7])
def test_link_widths_match_pixel_width_for_straight_strip(width):
    """Odd-width strips should not be biased one pixel too wide."""
    mask, links = _build_straight_strip(width)
    idt = distance_transform_edt(mask)

    out = lnu.link_widths_and_lengths(links, idt, pixlen=1)

    np.testing.assert_allclose(out["wid_pix"][0], np.full(11, float(width)))
    assert out["wid"][0] == pytest.approx(float(width))
    assert out["wid_adj"][0] == pytest.approx(float(width))
    assert out["wid_med"][0] == pytest.approx(float(width))


def test_link_widths_respect_pixlen_scaling():
    mask, links = _build_straight_strip(5)
    idt = distance_transform_edt(mask)

    out = lnu.link_widths_and_lengths(links, idt, pixlen=2.5)

    np.testing.assert_allclose(out["wid_pix"][0], np.full(11, 12.5))
    assert out["wid"][0] == pytest.approx(12.5)
    assert out["wid_adj"][0] == pytest.approx(12.5)
    assert out["wid_med"][0] == pytest.approx(12.5)
