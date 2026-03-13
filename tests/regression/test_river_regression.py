"""End-to-end regression test for a real river workflow."""
from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import require_rivgraph_classes


def test_river_brahma_clipped_end_to_end(river_case_brahma_clipped, tmp_path):
    _, river, _ = require_rivgraph_classes()

    case = river_case_brahma_clipped
    export_formats = case.check("export_formats", ["shp"])

    r = river(
        case.name,
        str(case.input_path("mask")),
        results_folder=str(tmp_path),
        exit_sides=str(case.param("exit_sides", "ns")),
        verbose=False,
    )
    r.paths["fixlinks_csv"] = str(case.input_path("fixlinks"))

    r.compute_network()
    assert hasattr(r, "Iskel")
    assert np.count_nonzero(r.Iskel) > 0
    assert len(r.links["id"]) > 0
    assert len(r.nodes["id"]) > 0

    r.prune_network()
    assert "inlets" in r.nodes
    assert "outlets" in r.nodes
    assert len(r.nodes["inlets"]) >= 1
    assert len(r.nodes["outlets"]) >= 1

    r.assign_flow_directions()
    assert "certain" in r.links
    assert len(r.links["certain"]) == len(r.links["id"])

    assert hasattr(r, "meshpolys")
    assert hasattr(r, "meshlines")
    assert len(r.meshpolys) > 0
    assert len(r.meshlines) > 0

    supported_formats = [fmt for fmt in export_formats if fmt in {"json", "shp"}] or ["shp"]
    for fmt in supported_formats:
        r.to_geovectors(export="network", ftype=fmt)
        assert r.paths["links"].endswith(f".{fmt}")
        assert r.paths["nodes"].endswith(f".{fmt}")
