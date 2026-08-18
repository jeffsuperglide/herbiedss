"""
herbiedss.utils.reproject

Reprojection helpers for taking NOAA model grids (HRRR, GFS, GEFS, etc.), as
returned by Herbie's `.xarray()` accessor, and warping them onto the USACE
Standard Hydrologic Grid (SHG) -- optionally clipped to a watershed boundary
supplied by the user as a vector file (shapefile, GeoJSON, GeoPackage, etc.).

Design notes
------------
- SHG is EPSG:5070-equivalent (NAD83 / Conus Albers), i.e. Albers Equal-Area
  Conic with standard parallels 29.5N/45.5N, central meridian 96W, latitude
  of origin 23N, on the NAD83 datum, with a native 2000 m cell size used by
  HEC-DSS.
- HRAP is a polar-stereographic projection (sphere R=6371200 m, true at
  60N, centered on 105W) with a native 4762.5 m cell size.
- Herbie's `da.herbie.crs` returns a *Cartopy* CRS describing the model's
  native grid (e.g. HRRR's Lambert Conformal). We convert that to a pyproj
  CRS, write it onto the DataArray/Dataset with rioxarray, then reproject.
- Reprojection (`.rio.reproject`) and clipping (`.rio.clip`) are separate
  steps: reproject first (resample onto the new CRS/cell size), then clip
  to the watershed polygon. `reproject_match` is NOT used here because the
  user is supplying a *vector* boundary file, not a reference raster --
  reproject_match requires a raster template (CRS + transform + shape) and
  cannot consume a shapefile directly.

Rectilinear vs. curvilinear grids
----------------------------------
rioxarray requires a 1D 'x' and 'y' dimension coordinate with real numeric
values (in the source CRS's own units) to build its affine transform for
`.rio.reproject()`. Herbie/cfgrib output comes in two shapes:

1. Rectilinear (e.g. GFS/GEFS regular lat-lon grids): 'latitude'/'longitude'
   are already 1D dimension coordinates. A straight rename to x/y is safe.
2. Curvilinear (e.g. HRRR's native Lambert Conformal grid): 'latitude'/
   'longitude' are 2D auxiliary coordinates that vary with both grid
   indices. Renaming 2D lat/lon straight to x/y is WRONG -- it either
   raises `rioxarray.exceptions.DimensionMissingCoordinateError: x missing
   coordinates` (no 1D values found) or, worse, silently produces incorrect
   georeferencing if some other coordinate happens to occupy the 'x' dim.
   For this case we look for existing 1D projected coordinates under
   common alternate names (e.g. 'xgrid_0'/'ygrid_0', 'west_east'/
   'south_north') and rename those instead of touching lat/lon at all. If
   no such 1D projected coordinate exists, we reconstruct one by
   round-tripping a single row/column of the 2D lat/lon through the
   source CRS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pyproj
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling

GridSystem = Literal["shg", "hrap"]

SHG_PROJ4 = (
    "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
    "+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
)
HRAP_PROJ4 = (
    "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 "
    "+x_0=0 +y_0=0 +R=6371200 +units=m +no_defs"
)

SHG_CELL_SIZE_M = 2000.0
HRAP_CELL_SIZE_M = 4762.5

SHG_CRS = pyproj.CRS.from_proj4(SHG_PROJ4)
HRAP_CRS = pyproj.CRS.from_proj4(HRAP_PROJ4)

_GRID_REGISTRY: dict[GridSystem, tuple[pyproj.CRS, float]] = {
    "shg": (SHG_CRS, SHG_CELL_SIZE_M),
    "hrap": (HRAP_CRS, HRAP_CELL_SIZE_M),
}

# Alternate 1D projected-dim names seen in various GRIB/GRIB2 decodes of
# Lambert Conformal / polar-stereographic model grids (HRRR, WRF-derived
# products, etc.) that should be treated as x/y once found.
_ALT_X_DIMS = ("xgrid_0", "west_east", "west_east_stag")
_ALT_Y_DIMS = ("ygrid_0", "south_north", "south_north_stag")


def get_target_crs_and_cellsize(grid: GridSystem) -> tuple[pyproj.CRS, float]:
    """Look up the target CRS and native cell size for 'shg' or 'hrap'."""
    try:
        return _GRID_REGISTRY[grid]
    except KeyError as exc:
        raise ValueError(
            f"Unknown grid system '{grid}'. Expected one of {list(_GRID_REGISTRY)}."
        ) from exc


def _source_crs_from_herbie(da: xr.DataArray) -> pyproj.CRS:
    """
    Pull the model's native CRS off the Herbie accessor (a Cartopy CRS) and
    convert it to a pyproj CRS that rioxarray/rasterio can consume. Falls
    back to plain WGS84 if the accessor is unavailable (e.g. the dataset was
    already regridded to a lat/lon grid upstream). This is the key step for
    handling HRRR (Lambert Conformal), GFS/GEFS (regular lat-lon), and other
    NOAA model grids generically -- Herbie already knows each model's native
    projection, so we don't hardcode per-model logic here.
    """
    try:
        cartopy_crs = da.herbie.crs
        return pyproj.CRS.from_user_input(cartopy_crs)
    except Exception:  # noqa: BLE001
        return pyproj.CRS.from_epsg(4326)


def _is_curvilinear(da: xr.DataArray) -> bool:
    """True if latitude/longitude are 2D auxiliary coordinates."""
    lon = da.coords.get("longitude")
    return lon is not None and lon.ndim == 2


def _rename_alt_projected_dims(da: xr.DataArray) -> xr.DataArray:
    """Rename known alternate 1D projected-dim names to x/y, if present."""
    dim_rename: dict[str, str] = {}
    for dim in da.dims:
        if dim in _ALT_X_DIMS:
            dim_rename[dim] = "x"
        elif dim in _ALT_Y_DIMS:
            dim_rename[dim] = "y"
    return da.rename(dim_rename) if dim_rename else da


def _reconstruct_projected_xy(da: xr.DataArray, src_crs: pyproj.CRS) -> xr.DataArray:
    """
    Rebuild 1D projected x/y dimension coordinates for a curvilinear grid
    (2D latitude/longitude, e.g. HRRR) by round-tripping a single row and a
    single column of the 2D lat/lon through the source CRS. This recovers
    the regular meter-spaced axis that the native Lambert Conformal (or
    other projected) grid actually sits on, which cfgrib/Herbie may not
    have exposed directly as a 1D coordinate.

    Assumes the grid dims are ('y', 'x') order for the 2D lat/lon fields,
    which matches standard cfgrib/eccodes output.
    """
    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError(
            "Cannot reconstruct projected x/y: DataArray does not have dims "
            f"named 'x' and 'y' to assign coordinates onto. Found dims: {da.dims}."
        )

    lon2d = da.coords["longitude"].values
    lat2d = da.coords["latitude"].values

    transformer = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

    x1d, _ = transformer.transform(lon2d[0, :], lat2d[0, :])
    _, y1d = transformer.transform(lon2d[:, 0], lat2d[:, 0])

    da = da.assign_coords(x=("x", np.asarray(x1d)), y=("y", np.asarray(y1d)))
    return da


def _ensure_xy_dims(da: xr.DataArray, src_crs: pyproj.CRS) -> xr.DataArray:
    """
    Ensure `da` has valid 1D 'x'/'y' dimension coordinates with real numeric
    values, handling both rectilinear (1D lat/lon) and curvilinear (2D
    lat/lon) source grids. Raises a clear error rather than letting
    rioxarray fail downstream with an opaque
    `DimensionMissingCoordinateError` if reconstruction isn't possible.
    """
    if _is_curvilinear(da):
        da = _rename_alt_projected_dims(da)
        if "x" in da.dims and "y" in da.dims and "x" in da.coords and "y" in da.coords:
            return da
        # No usable 1D projected coordinate was already present -- rebuild
        # one from the 2D lat/lon via the source CRS. This requires the
        # underlying dims to at least be named/renameable to x/y first.
        if "x" not in da.dims or "y" not in da.dims:
            raise ValueError(
                "Curvilinear grid detected (2D latitude/longitude) and no "
                "1D projected x/y dimension names were found (checked "
                f"{_ALT_X_DIMS + _ALT_Y_DIMS}). Cannot safely reconstruct "
                f"coordinates. Actual dims: {da.dims}. Inspect da.dims / "
                "da.coords manually and extend _ALT_X_DIMS/_ALT_Y_DIMS if "
                "this model uses a different naming convention."
            )
        return _reconstruct_projected_xy(da, src_crs)

    # Rectilinear grid: latitude/longitude are 1D dimension coordinates.
    rename_map = {}
    if "longitude" in da.dims and "x" not in da.dims:
        rename_map["longitude"] = "x"
    if "latitude" in da.dims and "y" not in da.dims:
        rename_map["latitude"] = "y"
    if rename_map:
        da = da.rename(rename_map)

    if "x" not in da.coords or "y" not in da.coords:
        raise ValueError(
            "Rectilinear grid, but 'x'/'y' coordinate values are still "
            f"missing after rename. Actual dims: {da.dims}, "
            f"coords: {list(da.coords)}."
        )
    return da


def reproject_to_grid(
    da: xr.DataArray,
    grid: GridSystem,
    *,
    resampling: Resampling = Resampling.bilinear,
    nodata: float = np.nan,
) -> xr.DataArray:
    """
    Reproject a single-variable Herbie DataArray (HRRR, GFS, GEFS, etc.) onto
    SHG (Albers) or HRAP (polar stereographic), snapped to that grid's native
    cell size (2000 m for SHG, 4762.5 m for HRAP).

    Parameters
    ----------
    da : xr.DataArray
        A single 2D (or squeezable-to-2D) field pulled out of Herbie's
        `.xarray()` result, e.g. `ds["t2m"]`.
    grid : {"shg", "hrap"}
        Target hydrologic grid system.
    resampling : rasterio.enums.Resampling
        Resampling algorithm. Use `Resampling.bilinear` or `.cubic` for
        continuous fields (temperature, precipitation depth) and
        `Resampling.nearest` for categorical fields.
    nodata : float
        Fill value for cells outside the source data's coverage.

    Returns
    -------
    xr.DataArray
        The field resampled onto the target grid's CRS at its native cell
        size, with `.rio.crs` set accordingly.
    """
    target_crs, cellsize = get_target_crs_and_cellsize(grid)

    src_crs = _source_crs_from_herbie(da)
    da = _ensure_xy_dims(da, src_crs)
    da = da.rio.write_crs(src_crs, inplace=False)

    da_proj = da.rio.reproject(
        target_crs,
        resolution=cellsize,
        resampling=resampling,
        nodata=nodata,
    )
    return da_proj


def clip_to_boundary(
    da: xr.DataArray,
    boundary_path: str | Path,
    *,
    drop: bool = True,
    all_touched: bool = True,
) -> xr.DataArray:
    """
    Clip an already-reprojected DataArray to a watershed boundary supplied
    as a vector file (shapefile, GeoJSON, GeoPackage -- anything geopandas
    can read). The boundary is reprojected to match `da`'s CRS before
    clipping, so the input file can be in any CRS (including WGS84).

    Parameters
    ----------
    da : xr.DataArray
        Field already reprojected onto the target grid (SHG/HRAP), with
        `.rio.crs` set.
    boundary_path : str or Path
        Path to the watershed boundary vector file.
    drop : bool
        If True, trim the output array to the boundary's bounding box
        (smaller output). If False, keep the original extent and just mask
        cells outside the polygon with nodata.
    all_touched : bool
        If True, include any cell touched by the boundary polygon, not just
        cells whose center falls inside it -- generally what you want for
        watershed-edge cells in a coarse hydrologic grid.

    Returns
    -------
    xr.DataArray
        The clipped/masked field.
    """
    import geopandas as gpd

    if da.rio.crs is None:
        raise ValueError(
            "DataArray has no CRS set. Call reproject_to_grid() before clipping."
        )

    boundary = gpd.read_file(str(boundary_path))
    boundary = boundary.to_crs(da.rio.crs)

    da_clipped = da.rio.clip(
        boundary.geometry.values,
        boundary.crs,
        drop=drop,
        all_touched=all_touched,
    )
    return da_clipped


def reproject_and_clip(
    da: xr.DataArray,
    grid: GridSystem,
    boundary_path: str | Path | None = None,
    *,
    resampling: Resampling = Resampling.bilinear,
    nodata: float = np.nan,
    drop: bool = True,
    all_touched: bool = True,
) -> xr.DataArray:
    """
    Convenience wrapper: reproject a Herbie field onto SHG or HRAP, then
    (optionally) clip it to a user-supplied watershed boundary file.

    If `boundary_path` is None, only the reprojection step runs.
    """
    da_proj = reproject_to_grid(da, grid, resampling=resampling, nodata=nodata)
    if boundary_path is not None:
        da_proj = clip_to_boundary(
            da_proj, boundary_path, drop=drop, all_touched=all_touched
        )
    return da_proj
