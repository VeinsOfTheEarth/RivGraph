"""End-to-end regression tests for a real delta workflow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers import require_rivgraph_classes


SUPPORTED_GENERIC_EXPORTS = {"json", "shp"}


def _build_delta(case, tmp_path: Path, *, use_fixlinks: bool):
    delta, _, _ = require_rivgraph_classes()

    d = delta(case.name, str(case.input_path("mask")), results_folder=str(tmp_path), verbose=False)
    if use_fixlinks:
        fixlinks = case.input_path("fixlinks")
        if fixlinks is None or not fixlinks.exists():
            pytest.skip("This regression case does not provide fixlinks.csv yet.")
        d.paths["fixlinks_csv"] = str(fixlinks)

    d.compute_network()

    assert hasattr(d, "Iskel")
    assert np.count_nonzero(d.Iskel) > 0
    assert len(d.links["id"]) > 0
    assert len(d.nodes["id"]) > 0

    d.prune_network(
        path_shoreline=str(case.input_path("shoreline")),
        path_inletnodes=str(case.input_path("inlet_nodes")),
    )
    assert "inlets" in d.nodes
    assert "outlets" in d.nodes
    assert len(d.nodes["inlets"]) >= 1
    assert len(d.nodes["outlets"]) >= 1
    assert len(d.links["id"]) >= 1

    d.assign_flow_directions()
    assert "certain" in d.links
    assert len(d.links["certain"]) == len(d.links["id"])
    assert np.all(np.isin(np.asarray(d.links["certain"]), [0, 1, True, False]))

    return d


def _read_sword_outputs(d):
    import geopandas as gpd

    reaches = gpd.read_file(d.paths["reaches_sword"])
    nodes = gpd.read_file(d.paths["nodes_sword"])
    assert len(reaches) > 0
    assert len(nodes) > 0
    assert "fdir_set" in reaches.columns
    return reaches, nodes


def test_delta_mossy_end_to_end_without_fixlinks(delta_case_mossy, tmp_path):
    case = delta_case_mossy
    export_formats = case.check("export_formats", ["shp"])

    d = _build_delta(case, tmp_path, use_fixlinks=False)

    # The automatic-direction case is valuable even if the resulting directed
    # graph is cyclic. It should still export direction-aware SWORD outputs.
    d.to_geovectors(export="sword", ftype="shp")
    reaches, nodes = _read_sword_outputs(d)
    assert len(reaches) > 0
    assert len(nodes) > 0

    supported_formats = [fmt for fmt in export_formats if fmt in SUPPORTED_GENERIC_EXPORTS] or ["shp"]
    for fmt in supported_formats:
        d.to_geovectors(export="network", ftype=fmt)
        assert d.paths["links"].endswith(f".{fmt}")
        assert d.paths["nodes"].endswith(f".{fmt}")


def test_delta_mossy_end_to_end_with_fixlinks_and_flux(delta_case_mossy, tmp_path):
    delta, _, _ = require_rivgraph_classes()
    from rivgraph.deltas import delta_metrics

    case = delta_case_mossy
    d = _build_delta(case, tmp_path, use_fixlinks=True)

    d.compute_link_width_and_length()
    try:
        d.links = delta_metrics.compute_steady_state_link_fluxes(
            None,
            d.links,
            d.nodes,
            weight_name="flux_ss",
        )
    except ValueError as exc:
        if "not acyclic" in str(exc):
            pytest.fail(
                "Mossy still contains a directed cycle after applying fixlinks.csv, "
                "so steady-state fluxes could not be computed."
            )
        raise

    assert "flux_ss" in d.links
    fluxes = np.asarray(d.links["flux_ss"], dtype=float)
    assert np.all(np.isfinite(fluxes))
    assert np.all(fluxes >= 0)
    assert np.max(fluxes) > 0

    # Preferred behavior is that SWORD export packages the fluxes directly.
    # Accept a transitional exported name while the export schema settles.
    d.to_geovectors(export="sword", ftype="shp")
    reaches, _ = _read_sword_outputs(d)

    flux_cols = [col for col in ("rg_flux", "flux_ss") if col in reaches.columns]
    assert flux_cols, (
        "Expected the SWORD reaches export to include a flux field after "
        "steady-state flux computation."
    )
