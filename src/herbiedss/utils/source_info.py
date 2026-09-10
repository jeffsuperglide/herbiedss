from dataclasses import dataclass

import numpy as np
from osgeo import gdal, osr


@dataclass(frozen=True)
class SourceGridInfo:
    is_geographic: bool
    is_global: bool
    uses_360_longitudes: bool
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    xres: float
    yres: float
    projection_name: str


def _source_grid_info(ds: gdal.Dataset) -> SourceGridInfo:
    """
    Inspect GDAL's source dataset extent in its native coordinate system.

    For geographic grids:
      - x is longitude in degrees
      - y is latitude in degrees

    For projected grids:
      - x/y are native projected coordinates, normally meters.
    """
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        raise ValueError(
            "Source GRIB has no affine geotransform. "
            "This may be a curvilinear/geolocation-array grid that needs "
            "a different warp workflow."
        )

    if not np.isclose(gt[2], 0.0) or not np.isclose(gt[4], 0.0):
        raise ValueError(
            "Source GRIB has rotated pixels. This workflow assumes a north-up "
            "affine grid; inspect it for GCPs/geolocation arrays."
        )

    x0, xres, _, y0, _, yres = gt
    x1 = x0 + ds.RasterXSize * xres
    y1 = y0 + ds.RasterYSize * yres

    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjectionRef())

    is_geographic = bool(src_srs.IsGeographic())
    width = xmax - xmin
    height = ymax - ymin

    # Tolerances avoid classifying a nearly-global GFS grid incorrectly
    # due to half-pixel coordinate conventions.
    is_global = is_geographic and width >= 359.0 and height >= 179.0

    # A source whose x extent is approximately 0..360 is a wrapped global
    # geographic source. Test this BEFORE using it with a -180..180 AOI.
    uses_360_longitudes = (
        is_geographic and xmin >= -0.5 and xmax > 180.0 and xmax <= 360.5
    )

    return SourceGridInfo(
        is_geographic=is_geographic,
        is_global=is_global,
        uses_360_longitudes=uses_360_longitudes,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        xres=xres,
        yres=yres,
        projection_name=src_srs.GetName() or "unknown",
    )
