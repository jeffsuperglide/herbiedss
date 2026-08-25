from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from herbiedss.utils.reproject import (
    SHG_LOWER_LEFT,
    GridSystem,
    clip_to_boundary,
    get_target_crs_and_cellsize,
    reproject_to_grid,
)


def _pd_to_datetime(value: Any) -> datetime:
    """Coerce a numpy/pandas datetime64 scalar into a stdlib datetime (UTC-naive)."""
    import pandas as pd

    return pd.Timestamp(value).to_pydatetime()


def _pd_to_timedelta(value: Any) -> timedelta:
    """Coerce a numpy/pandas timedelta64 scalar into a stdlib timedelta."""
    import pandas as pd

    return pd.Timedelta(value).to_pytimedelta()


def _dss_time(dt: datetime, *, is_end: bool = False) -> str:
    """
    Format a datetime per HEC-DSS grid D/E-part convention: DDMMMYYYY:HHMM (UTC),
    with midnight as 0000 for a start time and 2400 for an end time.

    Parameters
    ----------
    dt : datetime
        UTC timestamp to format.
    is_end : bool, optional
        If True, format `dt` as the end of an interval (D-part convention
        represents a midnight end time as "2400" on the previous calendar
        day rather than "0000" on the current day). Defaults to False.

    Returns
    -------
    str
        Timestamp formatted as "DDMMMYYYY:HHMM" (month abbreviation
        upper-cased), e.g. "25DEC2005:1200" or "24DEC2005:2400" for a
        midnight end time.
    """

    if is_end and dt.hour == 0 and dt.minute == 0:
        prev_day = dt - timedelta(days=1)
        date_part = prev_day.strftime("%d%b%Y").upper()
        return f"{date_part}:2400"
    date_part = dt.strftime("%d%b%Y").upper()
    time_part = dt.strftime("%H%M")
    return f"{date_part}:{time_part}"


def _build_dss_pathname(
    a_part: str, b_part: str, c_part: str, f_part: str, meta: dict
) -> str:
    """
    Build a DSS grid pathname /A/B/C/D/E/F/ where D and E come from the grid's own
    start/end time (UTC), formatted DDMMMYYYY:HHMM. E is left blank for instantaneous
    grids (no GRIB_stepRange / no end_time).

    Parameters
    ----------
    a_part : str
        A-part: grid reference system (e.g. "SHG", "HRAP", or a model name).
    b_part : str
        B-part: region, watershed, or location name.
    c_part : str
        C-part: data parameter (e.g. "PRECIP", "TMP2M").
    f_part : str
        F-part: version or other descriptive label.
    meta : dict
        Metadata dictionary produced by `_extract_grid_and_metadata`, used
        to source the D-part (`meta["start_time"]`) and E-part
        (`meta["end_time"]`) of the pathname.

    Returns
    -------
    str
        Upper-cased DSS pathname of the form "/A/B/C/D/E/F/". The D-part is
        the grid's start time and the E-part is its end time, both
        formatted via `_dss_time`; either is left as an empty string if
        the corresponding metadata value is missing (e.g. E is blank for
        instantaneous grids).
    """
    start_time = meta.get("start_time")
    end_time = meta.get("end_time")

    d_part = _dss_time(start_time) if start_time else ""
    e_part = _dss_time(end_time, is_end=True) if end_time else ""

    return f"/{a_part}/{b_part}/{c_part}/{d_part}/{e_part}/{f_part}/".upper()

def _extract_grid_and_metadata(
    ds: Any,
    var_name: str | None,
    *,
    grid_system: GridSystem | None = None,
    boundary_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Pull a single 2D numpy grid and its georeferencing + timing metadata out of
    whatever Herbie's .xarray() call returns (a Dataset with one or more variables,
    or a single DataArray).

    If `grid_system` is given ("shg" or "hrap"), the field is first reprojected
    from its native model grid (HRRR Lambert Conformal, GFS/GEFS lat-lon, etc.)
    onto that hydrologic grid, using the CRS Herbie exposes via `da.herbie.crs`.
    If `boundary_path` is also given, the reprojected field is clipped to that
    watershed boundary vector file (shapefile/GeoJSON/GeoPackage) after
    reprojection.

    Parameters
    ----------
    ds : Any
        Object returned by `Herbie.xarray()`: either an `xarray.Dataset`
        containing one or more variables, or a single `xarray.DataArray`.
    var_name : str or None
        Name of the data variable to extract if `ds` is a `Dataset` with
        multiple variables. If `None`, the first variable in the dataset
        is used. Ignored if `ds` is already a `DataArray`.
    grid_system : GridSystem or None, optional
        Target hydrologic grid ("shg" or "hrap") to reproject the field
        onto before extracting the final numpy array. If `None`, the
        field is kept in its native model projection.
    boundary_path : Path or None, optional
        Path to a watershed boundary vector file used to clip the
        reprojected field. Only applied when `grid_system` is also
        provided, since clipping happens after reprojection.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        A 2-tuple of:

        - A 2D numpy array of the field's values (squeezed from any
          extra singleton dimensions).
        - A metadata dictionary including (where available): `variable`,
          `units`, `long_name`, `dims`, `valid_time`, `start_time`,
          `end_time`, `center`, native `lat_min`/`lat_max`/`lon_min`/
          `lon_max`, `source_crs`, and, if `grid_system` was supplied,
          `grid_system`, `target_crs`, `cell_size`, `lower_left_x`,
          `lower_left_y`, and final `shape`.

    Raises
    ------
    ValueError
        If `ds` is a `Dataset` with no data variables, if `var_name` is
        given but not present in the dataset, or if the extracted field
        cannot be squeezed down to exactly 2 dimensions.
    """
    import xarray as xr

    if isinstance(ds, xr.Dataset):
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError("xarray() returned a Dataset with no data variables.")
        name = var_name or data_vars[0]
        if name not in ds.data_vars:
            raise ValueError(f"Variable '{name}' not found. Available: {data_vars}")
        da = ds[name]
    else:
        da = ds  # already a DataArray

    meta: dict = {
        "variable": getattr(da, "name", var_name) or "unknown",
        "units": da.attrs.get("units"),
        "long_name": da.attrs.get("long_name") or da.attrs.get("GRIB_name"),
        "dims": da.dims,
        "step_type": da.attrs.get("GRIB_stepType"),
        "max_data_value": float(np.nanmax(da.values)),
        "min_data_value": float(np.nanmin(da.values)),
        "mean_data_value": float(np.nanmean(da.values)),
        "missing_value":da.attrs.get("GRIB_missingValue")
    }

    # --- Timing metadata must be captured from the *original* Herbie output,
    # before any reprojection touches dims/coords. ---
    ref_time = da.coords.get("time")
    step = da.coords.get("step")
    valid_time = da.coords.get("valid_time")

    if valid_time is not None:
        meta["valid_time"] = _pd_to_datetime(valid_time.values)
    elif ref_time is not None and step is not None:
        meta["valid_time"] = _pd_to_datetime(ref_time.values) + _pd_to_timedelta(
            step.values
        )
    else:
        meta["valid_time"] = None

    step_range = da.attrs.get("GRIB_stepRange")  # e.g. "5-6" for an accumulated field
    if step_range and "-" in str(step_range) and ref_time is not None:
        start_h, end_h = (float(x) for x in str(step_range).split("-"))
        base = _pd_to_datetime(ref_time.values)
        meta["start_time"] = base + timedelta(hours=start_h)
        meta["end_time"] = base + timedelta(hours=end_h)
    elif meta["valid_time"] is not None:
        meta["start_time"] = meta["valid_time"]
        meta["end_time"] = None  # instantaneous grid -> DSS E-part left blank

    try:
        meta["center"] = da.herbie.center
    except Exception:  # noqa: BLE001
        meta["center"] = None

    # --- Native lat/lon bbox, captured before reprojection touches coords. ---
    lat = da.coords.get("latitude")
    lon = da.coords.get("longitude")
    if lat is not None and lon is not None:
        meta["lat_min"] = float(np.nanmin(lat.values))
        meta["lat_max"] = float(np.nanmax(lat.values))
        meta["lon_min"] = float(np.nanmin(lon.values))
        meta["lon_max"] = float(np.nanmax(lon.values))

    try:
        meta["source_crs"] = str(da.herbie.crs)
    except Exception:  # noqa: BLE001
        meta["source_crs"] = None

    # --- Reproject onto SHG/HRAP (and optionally clip to a watershed) before
    # squeezing to a plain numpy array. This works for HRRR (Lambert
    # Conformal), GFS/GEFS (regular lat-lon), or any other model Herbie
    # supports, since the source CRS comes from `da.herbie.crs` generically
    # rather than being hardcoded per model. ---
    if grid_system is not None:
        target_crs, cellsize, grid_type = get_target_crs_and_cellsize(grid_system)
        da = reproject_to_grid(da, grid_system)
        if boundary_path is not None:
            da = clip_to_boundary(da, boundary_path)

        meta["grid_system"] = grid_system.upper()
        meta["target_crs"] = target_crs.to_wkt("WKT1_ESRI")
        meta["cell_size"] = cellsize
        meta["grid_type"] = grid_type

        # DSS grid records key off the lower-left cell index in the target
        # grid's own coordinate system, not lat/lon -- capture x/y bounds
        # post-reprojection.
        x_coord = da.coords.get("x")
        y_coord = da.coords.get("y")
        if grid_system.lower() == "shg":
            meta["lower_left_y"] = SHG_LOWER_LEFT[0]
            meta["lower_left_x"] = SHG_LOWER_LEFT[-1]
        elif x_coord is not None and y_coord is not None:
            meta["lower_left_y"] = float(np.nanmin(y_coord.values))
            meta["lower_left_x"] = float(np.nanmin(x_coord.values))
    else:
        meta["grid_system"] = None
        meta["target_crs"] = None
        meta["cell_size"] = None

    arr = da.values
    arr_flipped = np.flipud(arr)
    # lat and long sizes
    meta["sizes"] = arr_flipped.shape

    grid = np.asarray(np.flipud(arr))
    # grid = np.asarray(arr)
    if grid.ndim > 2:
        grid = grid.squeeze()
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D grid after squeeze, got shape {grid.shape}.")

    meta["shape"] = grid.shape
    return grid, meta


# DSS Grid Types
class DssGridType(IntEnum):
    t400 = 400
    t401 = 401
    t410 = 410
    t411 = 411
    t420 = 420
    t421 = 421
    t430 = 430
    t431 = 431


# DSS Grid Data Types
class DssDataType(Enum):
    """
    HEC-DSS data type: pairs the DSS string code
    with an application-defined numeric code.
    """

    PER_AVER = "PER-AVER"
    PER_CUM = "PER-CUM"
    INST_VAL = "INST-VAL"
    INST_CUM = "INST-CUM"
    FREQ = "FREQ"
    INVALID = "INVALID"

    @property
    def code(self) -> int:
        return {
            DssDataType.PER_AVER: 0,
            DssDataType.PER_CUM: 1,
            DssDataType.INST_VAL: 2,
            DssDataType.INST_CUM: 3,
            DssDataType.FREQ: 4,
            DssDataType.INVALID: 5,
        }[self]
