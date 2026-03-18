"""Canonical vector export schemas and format helpers for RivGraph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "RG_NODE_SCHEMA_COLUMNS",
    "RG_LINK_SCHEMA_COLUMNS",
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
