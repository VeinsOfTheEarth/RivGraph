"""Raster helpers built on Rasterio.

This module centralizes raster opening, metadata access, coordinate
conversions, and GeoTIFF writing. It exposes a small, RivGraph-specific API
expressed in terms of affine transforms, geotransforms, and arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

import rivgraph.im_utils as im


__all__ = [
    "Raster",
    "ColorTable",
    "affine_to_geotransform",
    "geotransform_to_affine",
    "affine_to_gt",
    "gt_to_affine",
    "open_raster",
    "xy_to_coords",
    "coords_to_xy",
    "idx_to_coords",
    "write_geotiff",
    "crop_geotiff",
    "downsample_binary_geotiff",
]


_DTYPE_CODE_MAP = {
    1: 'uint8',
    2: 'uint16',
    3: 'int16',
    4: 'uint32',
    5: 'int32',
    6: 'float32',
    7: 'float64',
    8: 'complex64',
    9: 'complex64',
    10: 'complex64',
    11: 'complex128',
}


class ColorTable:
    """Simple RGBA color table for single-band raster exports."""

    def __init__(self):
        self._entries: dict[int, tuple[int, int, int, int]] = {}

    def SetColorEntry(self, index: int, rgba):
        self._entries[int(index)] = tuple(int(v) for v in rgba)

    def GetColorEntry(self, index: int):
        return self._entries.get(int(index))

    def as_mapping(self):
        return dict(self._entries)


@dataclass
class Raster:
    """In-memory raster plus the metadata RivGraph needs downstream."""

    array: np.ndarray
    transform: Affine
    crs: CRS | None = None
    nodata: float | int | None = None
    path: str | None = None
    source_georeferenced: bool = True

    @property
    def shape(self):
        return tuple(np.asarray(self.array).shape[-2:])

    @property
    def height(self):
        return int(self.shape[0])

    @property
    def width(self):
        return int(self.shape[1])

    @property
    def count(self):
        arr = np.asarray(self.array)
        return 1 if arr.ndim == 2 else int(arr.shape[0])

    @property
    def geotransform(self):
        return affine_to_geotransform(self.transform)

    @property
    def gt(self):
        return self.geotransform

    @property
    def crs_wkt(self):
        return self.crs.to_wkt() if self.crs is not None else ''

    @property
    def wkt(self):
        return self.crs_wkt

    @property
    def pixel_length(self):
        return abs(self.geotransform[1])

    @property
    def pixlen(self):
        return self.pixel_length

    @property
    def pixel_area(self):
        return abs(self.geotransform[1] * self.geotransform[5])

    @property
    def pixarea(self):
        return self.pixel_area

    @property
    def dtype(self):
        return np.asarray(self.array).dtype


def affine_to_geotransform(transform: Affine) -> tuple[float, float, float, float, float, float]:
    """Convert an affine transform to RivGraph's 6-tuple geotransform."""
    return (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)


# Short alias retained for internal brevity.
affine_to_gt = affine_to_geotransform


def geotransform_to_affine(geotransform) -> Affine:
    """Convert a RivGraph 6-tuple geotransform to an affine transform."""
    return Affine(geotransform[1], geotransform[2], geotransform[0], geotransform[4], geotransform[5], geotransform[3])


# Short alias retained for internal brevity.
gt_to_affine = geotransform_to_affine


def _default_transform(shape):
    """Return RivGraph's default geotransform for ungeoreferenced rasters."""
    return from_origin(0.0, float(shape[1]), 1.0, 1.0)


def _normalize_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, np.dtype):
        return dtype.name
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return np.dtype(dtype).name
    if isinstance(dtype, int):
        return _DTYPE_CODE_MAP.get(dtype, 'uint16')
    try:
        return np.dtype(dtype).name
    except Exception:
        return str(dtype)


def open_raster(path, assign_default_georef=True, allow_dummy_georef=None):
    """Open a raster with Rasterio and return RivGraph raster metadata.

    Parameters
    ----------
    path : str or path-like
        Raster path to open.
    assign_default_georef : bool, optional
        When True, rasters without georeferencing are assigned RivGraph's
        default 1x1, origin-at-(0, nrows) georeferencing in memory.
    allow_dummy_georef : bool, optional
        Deprecated compatibility alias for ``assign_default_georef``.
    """
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data = src.read()
            nodata = src.nodata
            if data.shape[0] == 1:
                data = data[0]
            crs = src.crs
            transform = src.transform
            source_georeferenced = (
                src.crs is not None and src.transform != Affine.identity()
            )

    if allow_dummy_georef is not None:
        assign_default_georef = allow_dummy_georef

    if not source_georeferenced:
        if not assign_default_georef:
            raise ValueError(f'Raster {path} has no georeferencing.')
        transform = _default_transform(data.shape[-2:] if np.asarray(data).ndim == 3 else data.shape)
        crs = CRS.from_epsg(4326)

    return Raster(
        np.asarray(data),
        transform if isinstance(transform, Affine) else geotransform_to_affine(transform),
        crs=CRS.from_user_input(crs) if crs is not None else None,
        nodata=nodata,
        path=str(path),
        source_georeferenced=source_georeferenced,
    )


def xy_to_coords(xs, ys, gt):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    cx = gt[0] + (xs + 0.5) * gt[1]
    cy = gt[3] + (ys + 0.5) * gt[5]
    return cx, cy


def coords_to_xy(xs, ys, gt):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    cols = np.floor((xs - gt[0]) / gt[1]).astype(int)
    rows = np.floor((ys - gt[3]) / gt[5]).astype(int)
    return np.column_stack((cols, rows))


def idx_to_coords(idx, shape, gt):
    rows, cols = np.unravel_index(idx, shape)
    return xy_to_coords(cols, rows, gt)


def _parse_creation_options(options):
    profile = {}
    if not options:
        return profile
    for opt in options:
        if '=' not in opt:
            continue
        key, value = opt.split('=', 1)
        key = key.strip().lower()
        value = value.strip()
        if key == 'compress':
            profile['compress'] = value.lower()
        elif key == 'tiled':
            profile['tiled'] = value.upper() in {'YES', 'TRUE', '1'}
        elif key == 'blockxsize':
            profile['blockxsize'] = int(float(value))
        elif key == 'blockysize':
            profile['blockysize'] = int(float(value))
    return profile


def write_geotiff(raster, gt, wkt, path_export, dtype='uint16', options=None,
                  nbands=1, nodata=None, color_table=None):
    arr = np.asarray(raster)
    if arr.ndim == 2:
        arr_to_write = arr[np.newaxis, :, :]
        count = 1
    elif arr.ndim == 3:
        if arr.shape[-1] == nbands:
            arr_to_write = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] == nbands:
            arr_to_write = arr
        else:
            arr_to_write = np.moveaxis(arr, -1, 0)
        count = arr_to_write.shape[0]
    else:
        raise ValueError('Raster must be 2-D or 3-D.')

    dtype_name = _normalize_dtype(dtype) or np.asarray(arr_to_write).dtype.name
    transform = geotransform_to_affine(gt)
    crs = CRS.from_wkt(wkt) if wkt else None

    profile = {
        'driver': 'GTiff',
        'height': int(arr_to_write.shape[1]),
        'width': int(arr_to_write.shape[2]),
        'count': int(count),
        'dtype': dtype_name,
        'transform': transform,
    }
    if crs is not None:
        profile['crs'] = crs
    if nodata is not None:
        profile['nodata'] = nodata
    profile.update(_parse_creation_options(options))

    with rasterio.open(path_export, 'w', **profile) as dst:
        dst.write(arr_to_write.astype(dtype_name, copy=False))
        if color_table is not None:
            dst.write_colormap(1, color_table.as_mapping())


def crop_geotiff(tif, cropto='first_nonzero', npad=0, outpath=None):
    tif = Path(tif)
    output_file = tif.with_name(f'{tif.stem}_cropped.tif') if outpath is None else Path(outpath)

    src = open_raster(tif)
    tiffull = src.array
    if cropto != 'first_nonzero':
        raise NotImplementedError('Only cropto="first_nonzero" is supported.')

    idcs = np.where(tiffull > 0)
    t = np.min(idcs[0])
    b = np.max(idcs[0]) + 1
    l = np.min(idcs[1])
    r = np.max(idcs[1]) + 1
    tifcropped = tiffull[t:b, l:r]
    if npad != 0:
        tifcropped = np.pad(tifcropped, npad, mode='constant', constant_values=False)

    gt = src.gt
    ulx = gt[0] + (l - npad) * gt[1]
    uly = gt[3] + (t - npad) * gt[5]
    crop_gt = (ulx, gt[1], gt[2], uly, gt[4], gt[5])
    dtype_name = _normalize_dtype(src.array.dtype)
    options = ['BLOCKXSIZE=128', 'BLOCKYSIZE=128', 'TILED=YES']
    if np.issubdtype(np.dtype(dtype_name), np.integer):
        options.append('COMPRESS=LZW')

    write_geotiff(tifcropped, crop_gt, src.wkt, output_file, dtype=dtype_name, options=options)
    return str(output_file)


def downsample_binary_geotiff(input_file, ds_factor, output_name, thresh=None):
    if ds_factor >= 1.0:
        raise ValueError('ds_factor must be < 1.')

    src = open_raster(input_file)
    gt = src.gt
    img = src.array.astype(np.int32)
    img_x, img_y = np.shape(img)
    modfactor = 1 / ds_factor
    if (img_x % modfactor > 0) or (img_y % modfactor > 0):
        img_x += img_x % modfactor
        img_y += img_y % modfactor
    old_x, old_y = np.shape(img)
    npad = int(np.max([(img_x - old_x), (img_y - old_y)]))

    newimg = np.pad(img, npad, mode='constant')
    newgt = (gt[0] - npad * gt[1], gt[1], gt[2], gt[3] - npad * gt[5], gt[4], gt[5])

    rs_x = int(img_x * ds_factor)
    rs_y = int(img_y * ds_factor)
    if thresh is None:
        img_rs = im.downsample_binary_image(newimg, (rs_x, rs_y))
    else:
        img_rs = im.downsample_binary_image(newimg, (rs_x, rs_y), thresh)

    dest_gt = (newgt[0], (newgt[1] * img_x) / rs_x, newgt[2], newgt[3], newgt[4], (newgt[5] * img_y) / rs_y)
    write_geotiff(img_rs, dest_gt, src.wkt, output_name, dtype='uint8')
    return str(output_name)
