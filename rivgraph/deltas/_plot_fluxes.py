from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx

from rivgraph import ln_utils as lnu
from rivgraph import geo_utils as gu


def _scale(values, vmin, vmax):
    vals = [float(v) for v in values]
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return [0.5 * (vmin + vmax)] * len(vals)
    return [vmin + (v - lo) * (vmax - vmin) / (hi - lo) for v in vals]


def outlet_flux_gdf(links, nodes, imshape, gt, wkt):
    flux_by_outlet = {nid: 0.0 for nid in nodes["outlets"]}
    node_ids = list(nodes["id"])
    for conn, flux in zip(links["conn"], links["flux_ss"]):
        ds = conn[1]
        if ds in flux_by_outlet:
            flux_by_outlet[ds] += float(flux)

    outlet_idxs = [nodes["idx"][node_ids.index(nid)] for nid in nodes["outlets"]]
    xs, ys = gu.idx_to_coords(outlet_idxs, imshape, gt)
    gdf = gpd.GeoDataFrame(
        {
            "node_id": list(nodes["outlets"]),
            "outlet_flux": [flux_by_outlet[nid] for nid in nodes["outlets"]],
        },
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=links_to_gdf(links, imshape, gt, wkt).crs,
    )
    return gdf


def links_to_gdf(links, imshape, gt, wkt):
    gdf = lnu.links_to_gpd(links, imshape, gt, wkt).copy()
    for key in ("id", "flux_ss", "wid_adj"):
        if key in links and len(links[key]) == len(links["id"]):
            gdf[key] = links[key]
    return gdf


def plot_flux_map(links, nodes, imshape, gt, wkt, *, line_attr="flux_ss", basemap=True):
    links_gdf = links_to_gdf(links, imshape, gt, wkt)
    outlets_gdf = outlet_flux_gdf(links, nodes, imshape, gt, wkt)

    plot_crs = "EPSG:3857"
    links_plot = links_gdf.to_crs(plot_crs)
    outlets_plot = outlets_gdf.to_crs(plot_crs)

    line_widths = _scale(links_plot[line_attr], 0.4, 6.0)
    marker_sizes = _scale(outlets_plot["outlet_flux"], 20, 300)

    fig, ax = plt.subplots(figsize=(10, 10))
    links_plot.plot(ax=ax, linewidth=line_widths, color="tab:blue", alpha=0.85)
    outlets_plot.plot(
        ax=ax,
        markersize=marker_sizes,
        color="crimson",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    if basemap:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except Exception as exc:
            warnings.warn(
                f"Basemap could not be added; continuing without basemap. Original error: {exc}",
                UserWarning,
                stacklevel=2,
            )

    ax.set_axis_off()
    ax.set_title("Steady-state flux partitioning")
    plt.tight_layout()
    return fig, ax, links_plot, outlets_plot
