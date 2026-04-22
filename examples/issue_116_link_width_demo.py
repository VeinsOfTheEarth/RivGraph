"""Toy reproducer for GitHub issue #116.

This script builds straight synthetic channels with known pixel widths and
runs ``rivgraph.ln_utils.link_widths_and_lengths`` on a centerline segment.
If the current implementation is unchanged, the reported widths follow the
pattern ``2, 2, 4, 4, 6, 6, ...`` for true widths ``1, 2, 3, 4, 5, 6, ...``.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import distance_transform_edt

# Avoid matplotlib cache warnings when running in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from rivgraph.ln_utils import link_widths_and_lengths


def build_strip(width: int, full_length: int = 31, link_length: int = 11, pad: int = 4):
    """Return a straight strip mask and a centered synthetic link."""
    mask = np.zeros((width + 2 * pad, full_length + 2 * pad), dtype=np.uint8)
    mask[pad : pad + width, pad : pad + full_length] = 1

    # Use a centerline segment well away from the strip ends so endpoint
    # trimming does not affect the width comparison.
    center_row = pad + width // 2
    start_col = pad + (full_length - link_length) // 2
    rr = np.full(link_length, center_row)
    cc = np.arange(start_col, start_col + link_length)
    link_idx = np.ravel_multi_index((rr, cc), mask.shape)

    return mask, {"idx": [link_idx]}, rr, cc


def measure_strip(width: int) -> dict[str, object]:
    """Measure a strip with the current RivGraph width code."""
    mask, links, rr, cc = build_strip(width)
    idt = distance_transform_edt(mask)
    out = link_widths_and_lengths(links, idt, pixlen=1)

    true_width = int(mask[:, cc[len(cc) // 2]].sum())
    return {
        "true_width": true_width,
        "wid": float(out["wid"][0]),
        "wid_adj": float(out["wid_adj"][0]),
        "wid_pix_unique": sorted(set(float(x) for x in out["wid_pix"][0])),
        "dt_unique": sorted(set(float(idt[r, c]) for r, c in zip(rr, cc))),
    }


def main() -> None:
    print("true_width  wid  wid_adj  error  wid_pix_unique  dt_unique")
    for width in range(1, 8):
        result = measure_strip(width)
        error = result["wid"] - result["true_width"]
        print(
            f"{result['true_width']:>10}"
            f"{result['wid']:>5.1f}"
            f"{result['wid_adj']:>9.1f}"
            f"{error:>7.1f}"
            f"  {result['wid_pix_unique']!s:<14}"
            f"  {result['dt_unique']}"
        )


if __name__ == "__main__":
    main()
