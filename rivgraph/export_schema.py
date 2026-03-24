"""Canonical vector export schemas and format helpers for RivGraph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "RG_NODE_SCHEMA_COLUMNS",
    "RG_LINK_SCHEMA_COLUMNS",
    "SWORD_NODE_SCHEMA_COLUMNS",
    "SWORD_NODE_PLACEHOLDER_COLUMNS",
    "SWORD_REACH_SCHEMA_COLUMNS",
    "SWORD_REACH_PLACEHOLDER_COLUMNS",
    "RG_NODE_RESERVED_INPUT_KEYS",
    "RG_LINK_RESERVED_INPUT_KEYS",
    "normalize_geovector_format",
    "get_driver_for_path",
    "get_extension_for_format",
    "ordered_export_columns",
]


EXPORT_SCHEMA_VERSION = "rg-v1"


@dataclass(frozen=True)
class VectorFormat:
    """Normalized description of a supported vector export format."""

    name: str
    ext: str
    driver: str
    preserves_native_crs: bool
    is_lossy: bool


_GEOVECTOR_FORMATS: dict[str, VectorFormat] = {
    "json": VectorFormat(
        name="GeoJSON",
        ext="json",
        driver="GeoJSON",
        preserves_native_crs=False,
        is_lossy=False,
    ),
    "geojson": VectorFormat(
        name="GeoJSON",
        ext="json",
        driver="GeoJSON",
        preserves_native_crs=False,
        is_lossy=False,
    ),
    "shp": VectorFormat(
        name="ESRI Shapefile",
        ext="shp",
        driver="ESRI Shapefile",
        preserves_native_crs=True,
        is_lossy=True,
    ),
    "shapefile": VectorFormat(
        name="ESRI Shapefile",
        ext="shp",
        driver="ESRI Shapefile",
        preserves_native_crs=True,
        is_lossy=True,
    ),
    "gpkg": VectorFormat(
        name="GeoPackage",
        ext="gpkg",
        driver="GPKG",
        preserves_native_crs=True,
        is_lossy=False,
    ),
    "geopackage": VectorFormat(
        name="GeoPackage",
        ext="gpkg",
        driver="GPKG",
        preserves_native_crs=True,
        is_lossy=False,
    ),
}


RG_NODE_SCHEMA_COLUMNS: tuple[str, ...] = (
    "id_node",
    "idx_node",
    "id_links",
    "n_links",
    "is_inlet",
    "is_outlet",
    "type_io",
)


RG_LINK_SCHEMA_COLUMNS: tuple[str, ...] = (
    "id_link",
    "idx_link",
    "id_nodes",
    "n_nodes",
    "id_us_node",
    "id_ds_node",
    "is_inlet",
    "is_outlet",
    "type_io",
)


SWORD_NODE_SCHEMA_COLUMNS: tuple[str, ...] = (
    "x",
    "y",
    "node_id_rg",
    "node_len",
    "reach_id_R",
    "width",
    "width_var",
    "max_width",
    "sinuosity",
    "fdir_set",
    "rg_flux",
)

SWORD_NODE_PLACEHOLDER_COLUMNS: tuple[str, ...] = (
    "node_id",
    "reach_id",
    "wse",
    "wse_var",
    "facc",
    "n_chan_max",
    "n_chan_mod",
    "obstr_type",
    "grod_id",
    "hfalls_id",
    "dist_out",
    "lakeflag",
    "manual_add",
    "meand_len",
    "type",
    "river_name",
    "edit_flag",
    "trib_flag",
    "path_freq",
    "path_order",
    "path_segs",
    "main_side",
    "strm_order",
    "end_reach",
    "network",
)

SWORD_REACH_SCHEMA_COLUMNS: tuple[str, ...] = (
    "x",
    "y",
    "reach_id_R",
    "reach_len",
    "n_nodes",
    "width",
    "width_var",
    "max_width",
    "rch_id_up",
    "rch_id_dn",
    "n_rch_up",
    "n_rch_down",
    "fdir_set",
    "conn_reach",
    "rg_us_nd",
    "rg_ds_nd",
    "rg_inlet",
    "rg_outlet",
    "rg_flux",
    "rg_outflx",
)

SWORD_REACH_PLACEHOLDER_COLUMNS: tuple[str, ...] = (
    "wse",
    "wse_var",
    "facc",
    "n_chan_max",
    "n_chan_mod",
    "obstr_type",
    "grod_id",
    "hfalls_id",
    "dist_out",
    "lakeflag",
    "swot_orbit",
    "swot_obs",
    "type",
    "river_name",
    "edit_flag",
    "trib_flag",
    "path_freq",
    "path_order",
    "path_segs",
    "main_side",
    "strm_order",
    "end_reach",
    "network",
)


RG_NODE_RESERVED_INPUT_KEYS: frozenset[str] = frozenset({"id", "idx", "conn", "inlets", "outlets"})
RG_LINK_RESERVED_INPUT_KEYS: frozenset[str] = frozenset({"id", "idx", "conn", "n_networks"})


def normalize_geovector_format(ftype: str | None) -> VectorFormat:
    """Normalize a user-facing vector format string to a supported format."""
    if ftype is None:
        raise TypeError("A geovector format is required.")

    key = str(ftype).strip().lower()
    try:
        return _GEOVECTOR_FORMATS[key]
    except KeyError as exc:
        raise TypeError(
            "Only json, shp, and gpkg output types are supported. "
            f"Got: {ftype!r}."
        ) from exc


def get_driver_for_path(path_file: str) -> str:
    """Return the canonical OGR/Fiona driver for *path_file*."""
    ext = path_file.rsplit(".", 1)[-1].lower()
    return normalize_geovector_format(ext).driver


def get_extension_for_format(ftype: str) -> str:
    """Return the canonical extension for a user-facing format token."""
    return normalize_geovector_format(ftype).ext


def ordered_export_columns(canonical_columns: Iterable[str], extra_columns: Iterable[str]) -> list[str]:
    """Return stable export column order with geometry always last."""
    cols = list(canonical_columns)
    for col in extra_columns:
        if col not in cols and col != "geometry":
            cols.append(col)
    cols.append("geometry")
    return cols
