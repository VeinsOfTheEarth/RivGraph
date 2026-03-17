
"""Raster helpers and a small GDAL-compatible compatibility layer.

This module centralizes raster handling on top of Rasterio while exposing a
light-weight dataset interface for the rest of RivGraph during the backend
transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import warnings

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

import rivgraph.im_utils as im


_LEGACY_GDAL_DTYPES = {
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


@dataclass
class RasterBand:
    """Small stand-in for GDAL's raster band object."""

    dataset: 'RasterDataset'
    band_index: int = 1

    @property
    def DataType(self):
        return np.dtype(self.dataset.dtype).name

    def GetNoDataValue(self):
        return self.dataset.nodata


class ColorTable:
    """Simple color table with the subset of GDAL-like methods RivGraph uses."""

    def __init__(self):
        self._entries: dict[int, tuple[int, int, int, int]] = {}

    def SetColorEntry(self, index: int, rgba):
        self._entries[int(index)] = tuple(int(v) for v in rgba)

    def GetColorEntry(self, index: int):
        return self._entries.get(int(index))

    def as_mapping(self):
        return dict(self._entries)


class RasterDataset:
    """In-memory raster dataset metadata used as a GDAL replacement."""

    def __init__(self, array, transform, crs=None, nodata=None, path=None):
        arr = np.asarray(array)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        self.array = arr
        self.transform = transform if isinstance(transform, Affine) else Affine.from_gdal(*transform)
        self.crs = CRS.from_user_input(crs) if crs else None
        self.nodata = nodata
        self.path = str(path) if path is not None else None
        self._update_cached_metadata()

    def _update_cached_metadata(self):
        if self.array.ndim == 2:
            self.height, self.width = self.array.shape
            self.count = 1
        elif self.array.ndim == 3:
            self.count = self.array.shape[0]
            self.height, self.width = self.array.shape[1:]
        else:
            raise ValueError('Raster array must be 2-D or 3-D.')
        self.shape = (self.height, self.width)
        self.gt = self.transform.to_gdal()
        self.wkt = self.crs.to_wkt() if self.crs is not None else ''
        self.pixlen = abs(self.gt[1])
        self.pixarea = abs(self.gt[1] * self.gt[5])
        self.dtype = self.array.dtype
        self.RasterYSize = self.height
        self.RasterXSize = self.width

    def GetGeoTransform(self):
        return self.gt

    def SetGeoTransform(self, gt):
        self.transform = Affine.from_gdal(*gt)
        self._update_cached_metadata()

    def GetProjection(self):
        return self.wkt

    def SetProjection(self, wkt):
        self.crs = CRS.from_wkt(wkt) if wkt else None
        self._update_cached_metadata()

    def ReadAsArray(self, xoff=0, yoff=0, xsize=None, ysize=None):
        if xsize is None:
            xsize = self.width - xoff
        if ysize is None:
            ysize = self.height - yoff
        row_slice = slice(int(yoff), int(yoff) + int(ysize))
        col_slice = slice(int(xoff), int(xoff) + int(xsize))
        if self.array.ndim == 2:
            return np.asarray(self.array[row_slice, col_slice])
        return np.asarray(self.array[:, row_slice, col_slice])

    def GetRasterBand(self, band_index):
        return RasterBand(self, band_index)


def _dummy_transform(shape):
    return from_origin(0.0, float(shape[1]), 1.0, 1.0)


def _normalize_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, np.dtype):
        return dtype.name
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return np.dtype(dtype).name
    if isinstance(dtype, int):
        return _LEGACY_GDAL_DTYPES.get(dtype, 'uint16')
    try:
        return np.dtype(dtype).name
    except Exception:
        return str(dtype)


def open_raster(path, allow_dummy_georef=True):
    """Open a raster with Rasterio and return RivGraph's compatibility wrapper."""
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

    if crs is None or transform is None:
        if not allow_dummy_georef:
            raise ValueError(f'Raster {path} has no georeferencing.')
        transform = _dummy_transform(data.shape[-2:] if np.asarray(data).ndim == 3 else data.shape)
        crs = CRS.from_epsg(4326)

    return RasterDataset(data, transform, crs=crs, nodata=nodata, path=path)


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
    transform = Affine.from_gdal(*gt)
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
    if outpath is None:
        output_file = tif.with_name(f'{tif.stem}_cropped.tif')
    else:
        output_file = Path(outpath)

    src = open_raster(tif)
    tiffull = src.ReadAsArray()
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

    gt = src.GetGeoTransform()
    ulx = gt[0] + (l - npad) * gt[1]
    uly = gt[3] + (t - npad) * gt[5]
    crop_gt = (ulx, gt[1], gt[2], uly, gt[4], gt[5])
    dtype_name = _normalize_dtype(src.GetRasterBand(1).DataType)
    options = ['BLOCKXSIZE=128', 'BLOCKYSIZE=128', 'TILED=YES']
    if np.issubdtype(np.dtype(dtype_name), np.integer):
        options.append('COMPRESS=LZW')

    write_geotiff(tifcropped, crop_gt, src.GetProjection(), output_file,
                 dtype=dtype_name, options=options)
    return str(output_file)


def downsample_binary_geotiff(input_file, ds_factor, output_name, thresh=None):
    if ds_factor >= 1.0:
        raise ValueError('ds_factor must be < 1.')

    og = open_raster(input_file)
    gm = og.GetGeoTransform()
    img = og.ReadAsArray().astype(np.int32)
    img_x, img_y = np.shape(img)
    modfactor = 1 / ds_factor
    if (img_x % modfactor > 0) or (img_y % modfactor > 0):
        img_x += img_x % modfactor
        img_y += img_y % modfactor
    old_x, old_y = np.shape(img)
    npad = int(np.max([(img_x - old_x), (img_y - old_y)]))

    newimg = np.pad(img, npad, mode='constant')
    newgm = (gm[0] - npad * gm[1], gm[1], gm[2],
             gm[3] - npad * gm[5], gm[4], gm[5])

    rs_x = int(img_x * ds_factor)
    rs_y = int(img_y * ds_factor)
    if thresh is None:
        img_rs = im.downsample_binary_image(newimg, (rs_x, rs_y))
    else:
        img_rs = im.downsample_binary_image(newimg, (rs_x, rs_y), thresh)

    dest_gm = (newgm[0], (newgm[1] * img_x) / rs_x, newgm[2],
               newgm[3], newgm[4], (newgm[5] * img_y) / rs_y)
    write_geotiff(img_rs, dest_gm, og.GetProjection(), output_name, dtype='uint8')
    return str(output_name)
