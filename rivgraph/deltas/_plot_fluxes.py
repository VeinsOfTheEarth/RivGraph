from __future__ import annotations

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import contextily as cx
from pyproj import CRS
try:
    from osgeo import gdal
except ModuleNotFoundError:
    import gdal

from rivgraph import ln_utils as lnu
from rivgraph import geo_utils as gu


def _scale(values, vmin, vmax):
    vals = [float(v) for v in values]
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return [0.5 * (vmin + vmax)] * len(vals)
    return [vmin + (v - lo) * (vmax - vmin) / (hi - lo) for v in vals]


def _resolve_basemap_source(basemap):
    """Resolve a basemap request to a contextily provider or None."""
    if basemap in (None, False):
        return None
    if basemap is True:
        return cx.providers.OpenStreetMap.Mapnik
    if not isinstance(basemap, str):
        return basemap

    key = basemap.strip()
    lowered = key.lower()
    aliases = {
        "osm": "OpenStreetMap.Mapnik",
        "openstreetmap": "OpenStreetMap.Mapnik",
        "openstreetmap.mapnik": "OpenStreetMap.Mapnik",
        "positron": "CartoDB.Positron",
        "cartodb.positron": "CartoDB.Positron",
        "voyager": "CartoDB.Voyager",
        "cartodb.voyager": "CartoDB.Voyager",
    }
    key = aliases.get(lowered, key)

    provider = cx.providers
    try:
        for part in key.split('.'):
            provider = provider[part] if isinstance(provider, dict) else getattr(provider, part)
    except Exception as exc:
        raise ValueError(
            f"Could not resolve basemap provider '{basemap}'. "
            "Pass False/None to disable, True for OpenStreetMap, or a valid "
            "contextily provider path like 'CartoDB.Positron'."
        ) from exc
    return provider


def _add_north_arrow(ax, xy=(0.94, 0.92), size=0.08):
    """Add a simple north arrow in axes-fraction coordinates."""
    x, y = xy
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - size),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="black",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
        bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.7), ec="none"),
        zorder=10,
    )


def _mask_extent(gt, ncols, nrows):
    x0 = gt[0]
    x1 = gt[0] + gt[1] * ncols
    y0 = gt[3]
    y1 = gt[3] + gt[5] * nrows
    return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))


def _build_mask_rgba(mask, alpha=0.18, rgb=(80, 80, 80)):
    alpha_uint8 = int(np.clip(alpha, 0.0, 1.0) * 255)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = np.where(np.asarray(mask) > 0, alpha_uint8, 0).astype(np.uint8)
    return rgba


def _warp_rgba(rgba, gdobj, target_crs):
    target_crs = CRS.from_user_input(target_crs)
    src_wkt = gdobj.GetProjection()
    src_crs = CRS.from_wkt(src_wkt) if src_wkt else None
    if src_crs is not None and src_crs == target_crs:
        gt = gdobj.GetGeoTransform()
        extent = _mask_extent(gt, gdobj.RasterXSize, gdobj.RasterYSize)
        return rgba, extent

    mem_driver = gdal.GetDriverByName("MEM")
    src_ds = mem_driver.Create("", gdobj.RasterXSize, gdobj.RasterYSize, 4, gdal.GDT_Byte)
    src_ds.SetGeoTransform(gdobj.GetGeoTransform())
    src_ds.SetProjection(src_wkt)
    for band_idx in range(4):
        src_ds.GetRasterBand(band_idx + 1).WriteArray(rgba[..., band_idx])

    warped = gdal.Warp(
        "",
        src_ds,
        format="MEM",
        dstSRS=target_crs.to_wkt(),
        resampleAlg=gdal.GRA_NearestNeighbour,
    )
    arr = np.stack([warped.GetRasterBand(i + 1).ReadAsArray() for i in range(4)], axis=-1)
    extent = _mask_extent(warped.GetGeoTransform(), warped.RasterXSize, warped.RasterYSize)
    return arr, extent


def _add_mask_underlay(ax, gdobj, plot_crs, *, alpha=0.18, rgb=(80, 80, 80)):
    mask = gdobj.ReadAsArray()
    rgba = _build_mask_rgba(mask, alpha=alpha, rgb=rgb)
    rgba_plot, extent = _warp_rgba(rgba, gdobj, plot_crs)
    ax.imshow(rgba_plot, extent=extent, origin="upper", interpolation="nearest", zorder=1)


def outlet_flux_gdf(links, nodes, gdobj, flux_attr="flux_ss"):
    flux_by_outlet = {nid: 0.0 for nid in nodes["outlets"]}
    node_ids = list(nodes["id"])
    flux_values = links[flux_attr]
    for conn, flux in zip(links["conn"], flux_values):
        ds = conn[1]
        if ds in flux_by_outlet:
            flux_by_outlet[ds] += float(flux)

    outlet_idxs = [nodes["idx"][node_ids.index(nid)] for nid in nodes["outlets"]]
    xs, ys = gu.idx_to_coords(outlet_idxs, gdobj)
    gdf = gpd.GeoDataFrame(
        {
            "node_id": list(nodes["outlets"]),
            "outlet_flux": [flux_by_outlet[nid] for nid in nodes["outlets"]],
        },
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=links_to_gdf(links, gdobj, attrs=[flux_attr]).crs,
    )
    return gdf


def links_to_gdf(links, gdobj, attrs=None):
    gdf = lnu.links_to_gpd(links, gdobj).copy()
    attrs = [] if attrs is None else list(attrs)
    for key in ["id", *attrs]:
        if key in links and len(links[key]) == len(links["id"]):
            gdf[key] = links[key]
    return gdf


def plot_flux_map(
    links,
    nodes,
    gdobj,
    *,
    line_attr="flux_ss",
    basemap="OpenStreetMap.Mapnik",
    show_mask=False,
    mask_alpha=0.18,
    north_arrow=True,
    north_arrow_xy=(0.94, 0.92),
    outlet_style="size",
    outlet_marker_size=110.0,
    outlet_cmap="viridis",
):
    """Plot steady-state flux partitioning.

    Parameters
    ----------
    line_attr : str, optional
        Link attribute to use for linewidth scaling. Defaults to ``flux_ss``.
    basemap : bool, str, or contextily provider, optional
        Basemap control. ``True`` or the default string uses OpenStreetMap.
        Pass ``False``/``None`` to disable, or provide another provider such as
        ``"CartoDB.Positron"``.
    show_mask : bool, optional
        If ``True``, overlay the user's input mask as a semi-transparent underlay.
    mask_alpha : float, optional
        Opacity for non-zero mask pixels when ``show_mask=True``.
    north_arrow : bool, optional
        If ``True``, add a north arrow to the map.
    outlet_style : {"size", "color"}, optional
        ``"size"`` scales outlet marker area by flux. ``"color"`` uses equal-size
        markers colored by outlet flux.
    outlet_marker_size : float, optional
        Marker size to use when ``outlet_style='color'``.
    outlet_cmap : str, optional
        Matplotlib colormap used when ``outlet_style='color'``.
    """
    if line_attr not in links:
        raise KeyError(f"'{line_attr}' was not found in links.")
    if outlet_style not in {"size", "color"}:
        raise ValueError("outlet_style must be either 'size' or 'color'.")

    links_gdf = links_to_gdf(links, gdobj, attrs=[line_attr, "wid_adj"])
    outlets_gdf = outlet_flux_gdf(links, nodes, gdobj, flux_attr=line_attr)

    plot_crs = "EPSG:3857"
    links_plot = links_gdf.to_crs(plot_crs)
    outlets_plot = outlets_gdf.to_crs(plot_crs)

    line_widths = _scale(links_plot[line_attr], 0.4, 6.0)
    marker_sizes = _scale(outlets_plot["outlet_flux"], 20, 300)

    fig, ax = plt.subplots(figsize=(10, 10))

    total_bounds = links_plot.total_bounds
    if len(outlets_plot) > 0:
        ob = outlets_plot.total_bounds
        total_bounds = np.array([
            min(total_bounds[0], ob[0]),
            min(total_bounds[1], ob[1]),
            max(total_bounds[2], ob[2]),
            max(total_bounds[3], ob[3]),
        ])
    pad_x = max((total_bounds[2] - total_bounds[0]) * 0.03, 1.0)
    pad_y = max((total_bounds[3] - total_bounds[1]) * 0.03, 1.0)
    ax.set_xlim(total_bounds[0] - pad_x, total_bounds[2] + pad_x)
    ax.set_ylim(total_bounds[1] - pad_y, total_bounds[3] + pad_y)

    basemap_source = _resolve_basemap_source(basemap)
    if basemap_source is not None:
        cx.add_basemap(ax, source=basemap_source, crs=plot_crs, zorder=0)

    if show_mask:
        _add_mask_underlay(ax, gdobj, plot_crs, alpha=mask_alpha)

    links_plot.plot(ax=ax, linewidth=line_widths, color="tab:blue", alpha=0.85, zorder=2)

    if outlet_style == "size":
        outlets_plot.plot(
            ax=ax,
            markersize=marker_sizes,
            color="crimson",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
    else:
        outlet_artist = outlets_plot.plot(
            ax=ax,
            markersize=outlet_marker_size,
            column="outlet_flux",
            cmap=outlet_cmap,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            legend=False,
        )
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(outlet_cmap))
        sm.set_array(outlets_plot["outlet_flux"].to_numpy())
        sm.set_clim(vmin=float(outlets_plot["outlet_flux"].min()), vmax=float(outlets_plot["outlet_flux"].max()))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("Outlet flux")

    if north_arrow:
        _add_north_arrow(ax, xy=north_arrow_xy)

    ax.set_axis_off()
    ax.set_title("Steady-state flux partitioning")
    plt.tight_layout()
    return fig, ax, links_plot, outlets_plot
