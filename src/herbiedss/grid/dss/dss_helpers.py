from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import typer

from herbiedss.grid.dss.dssprops import DSS_UNDEFINED_VALUE


class BBox(NamedTuple):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def _parse_path_or_bbox(value: str | None) -> Path | BBox | None:
    """
    Accepts either:
      - a file path; returns Path
      - four integers (comma or space separated) returns; tuple[int, int, int, int]
      - fall through to None
    """
    if value is None:
        return

    if value:
        # treat input string as a path
        path = Path(value)
        if path.exists() or path.suffix.lower() in {".shp", ".geojson", ".json"}:
            return path
        
        # Try to parse as four integers first
        try:
            parts = value.replace(",", " ").split()
        except ValueError:
            raise typer.BadParameter(
                "must be a vector-file path or four separated integers (comma or space): "
                "xmin,ymin,xmax,ymax or 'xmin ymin xmax ymax'"
            )
        
        if len(parts) != 4:
            raise typer.BadParameter(
                "must be a vector-file path or four separated integers (comma or space): "
                "xmin,ymin,xmax,ymax or 'xmin ymin xmax ymax'"
            )


        bbox = BBox(*(int(x) for x in parts))
        if bbox.xmin >= bbox.xmax or bbox.ymin >= bbox.ymin:
            raise typer.BadParameter(
                "Bounding box must satisfy xmin < xmax and ymin < ymax"
            )
        return bbox


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
    Format a datetime for the D- or E-part of an HEC-DSS gridded-data pathname.

    HEC-DSS grid pathname time parts use the format ``DDMMMYYYY:HHMM``. This
    helper applies the DSS interval-end convention for midnight timestamps:
    when a timestamp is the end of an interval and occurs exactly at 00:00,
    it is represented as ``2400`` on the preceding calendar date rather than
    as ``0000`` on the current date.

    For example, an interval ending at midnight UTC on 25 December 2005 is
    formatted as ``24DEC2005:2400`` when `is_end=True`. The same datetime is
    formatted as ``25DEC2005:0000`` when `is_end=False`.

    Parameters
    ----------
    dt : datetime
        Datetime to format. The caller should supply a UTC datetime; this
        function does not convert time zones, inspect `tzinfo`, round seconds,
        or otherwise alter the supplied timestamp except for the special
        midnight interval-end representation.
    is_end : bool, optional
        Whether `dt` represents the end of a time interval.

        If `True` and `dt` falls exactly at midnight (hour and minute are both
        zero), the function subtracts one calendar day and formats the time as
        ``2400``. For all other times, the timestamp is formatted normally.

        If `False`, midnight is formatted as ``0000`` on the datetime's own
        calendar date. Defaults to `False`.

    Returns
    -------
    str
        A DSS-compatible time string in ``DDMMMYYYY:HHMM`` format. The month
        abbreviation is converted to uppercase by the caller if a fully
        uppercase DSS pathname is required.

        Examples include:

        - ``"25DEC2005:1200"`` for noon on 25 December 2005.
        - ``"25DEC2005:0000"`` for a start time at midnight on 25 December
          2005.
        - ``"24DEC2005:2400"`` for an interval ending at midnight on
          25 December 2005.

    Notes
    -----
    Seconds and microseconds are not included in the DSS time representation.
    Consequently, callers should ensure that `dt` has already been normalized
    to the intended minute before calling this function. A timestamp at
    ``00:00:30`` is treated as midnight by the current hour/minute test and
    will be represented as ``2400`` when `is_end=True`.
    """
    if is_end and dt.hour == 0 and dt.minute == 0:
        prev_day = dt - timedelta(days=1)
        date_part = prev_day.strftime("%d%b%Y")
        return f"{date_part}:2400"
    date_part = dt.strftime("%d%b%Y")
    time_part = dt.strftime("%H%M")

    return f"{date_part}:{time_part}"


def _build_dss_pathname(
    a_part: str,
    b_part: str,
    c_part: str,
    f_part: str,
    meta: dict,
    duration: int | None,
) -> str:
    """
    Build an HEC-DSS gridded-data pathname from DSS pathname parts and GRIB
    band metadata.

    The returned pathname follows the standard DSS form:

        /A/B/C/D/E/F/

    The A-, B-, C-, and F-parts are supplied by the caller. The D- and E-parts
    are derived from GDAL GRIB metadata:

    - For instantaneous fields, identified here by product definition template
      number (`GRIB_PDS_PDTN`) other than 8, the D-part is the GRIB valid time
      and the E-part is blank.
    - For interval/accumulation fields using template 8, the D-part represents
      the beginning of the processing interval and the E-part represents the
      interval end/valid time.
    - When template 8 reports zero forecast seconds, the GRIB reference time is
      used as the interval start; otherwise, the interval start is calculated
      by subtracting the interval duration from the valid time.
    - The explicit `duration` argument, when provided, takes precedence over
      `GRIB_FORECAST_SECONDS` when calculating the interval length.

    All times are interpreted as Unix timestamps in UTC and formatted by
    `_dss_time`. The resulting pathname is returned exactly as assembled; this
    function does not normalize the case of individual pathname parts.

    Parameters
    ----------
    a_part : str
        DSS A-part, usually the grid reference system or model/grid identifier,
        such as ``"SHG"``, ``"HRAP"``, or ``"HRRR"``.
    b_part : str
        DSS B-part identifying the spatial area, watershed, basin, region, or
        other location label.
    c_part : str
        DSS C-part identifying the parameter or variable, such as
        ``"PRECIP"``, ``"TMP2M"``, or ``"WIND"``.
    f_part : str
        DSS F-part containing a version, source, model-product label, forecast
        identifier, or other descriptive qualifier.
    meta : dict
        GDAL GRIB band metadata. The following keys are required and must be
        convertible to integers:

        - ``"GRIB_REF_TIME"``: Model/reference time as a Unix timestamp.
        - ``"GRIB_VALID_TIME"``: Valid/end time as a Unix timestamp.
        - ``"GRIB_FORECAST_SECONDS"``: Forecast lead time or processing
          interval length in seconds.
        - ``"GRIB_PDS_PDTN"``: GRIB product definition template number.

        Template number 8 is treated as an interval-based product; all other
        template numbers are treated as instantaneous products.
    duration : int | None
        Optional processing-interval duration in hours. For template 8
        products, a non-``None`` value overrides the interval inferred from
        ``GRIB_FORECAST_SECONDS``. Ignored for non-template-8 products.

    Returns
    -------
    str
        A DSS pathname in the form ``"/A/B/C/D/E/F/"``.

        For instantaneous products, the returned path has a blank E-part:

        ``/A/B/C/VALID_TIME//F/``

        For template 8 interval products, the path includes both interval
        bounds:

        ``/A/B/C/START_TIME/END_TIME/F/``

    Notes
    -----
    This function assumes GRIB metadata timestamps are UTC Unix epoch seconds.
    It recognizes `GRIB_PDS_PDTN == 8` as an interval/accumulation product.
    Other product definition templates are written as instantaneous grids,
    regardless of whether their underlying meteorological quantity is commonly
    interpreted over a period.
    """
    grib_ref_time = int(meta["GRIB_REF_TIME"])
    grib_valid_time = int(meta["GRIB_VALID_TIME"])
    grib_forecast_seconds = int(meta["GRIB_FORECAST_SECONDS"])
    grib_pds_pdtn = int(meta["GRIB_PDS_PDTN"])

    start_time = datetime.fromtimestamp(grib_ref_time, tz=UTC)
    end_time = datetime.fromtimestamp(grib_valid_time, tz=UTC)

    # assuming what comes in is grib_pds_pdtn == 0
    d_part = _dss_time(end_time, is_end=True)
    e_part = ""

    if grib_pds_pdtn == 8:
        grib_forecast_seconds_interval = (
            timedelta(hours=duration)
            if duration
            else timedelta(seconds=grib_forecast_seconds)
        )
        start_time_process = end_time - grib_forecast_seconds_interval
        d_part = (
            _dss_time(start_time)
            if grib_forecast_seconds == 0
            else _dss_time(start_time_process)
        )
        e_part = _dss_time(end_time, is_end=True)

    return f"/{a_part}/{b_part}/{c_part}/{d_part}/{e_part}/{f_part}/"


def _extract_grid_metadata(
    ds: Any,
    var_name: str | None,
) -> dict:
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

    Returns
    -------
    dict
        A metadata dictionary including (where available): `variable`,
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
        return {}

    meta: dict = {
        "variable": getattr(da, "name", var_name) or "unknown",
        "units": da.attrs.get("units"),
        "long_name": da.attrs.get("long_name") or da.attrs.get("GRIB_name"),
        "step_type": da.attrs.get("GRIB_stepType"),
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

    try:
        meta["source_crs"] = str(da.herbie.crs)
    except Exception:  # noqa: BLE001
        meta["source_crs"] = None

    return meta


def _gdal_warp_options(
    boundary: Path | BBox | None,
    epsg: int | None,
) -> dict:
    if boundary is None:
        return {}

    if isinstance(boundary, Path):
        if not boundary.exists():
            raise FileNotFoundError(f"Boundary file not found: {boundary}")
        kwargs = {
            "cutlineDSName": boundary.as_posix(),
            "cropToCutline": True,
            "warpOptions": ["CUTLINE_ALL_TOUCHED=TRUE"],
        }
        if epsg:
            kwargs = {"cutlineSRS": f"EPSG:{epsg}", **kwargs}
        return kwargs
    if isinstance(boundary, tuple):
        if len(boundary) != 4:
            raise ValueError("Bounding box must be a 4-tuple (minx, miny, maxx, maxy)")
        if epsg is None:
            raise typer.BadParameter(
                "When --boundary is a bounding box you must also supply --bbox-epsg",
                param_hint="--bbox-epsg",
            )
        return {
            "outputBounds": boundary,
            "outputBoundsSRS": f"EPSG:{epsg}",
        }
    # Should never reach here if the type hints are respected
    raise TypeError(f"Unsupported boundary type: {type(boundary)}")


def _dss_undefined_cells(data: np.ndarray, nodata) -> np.ndarray:
    if nodata is not None:
        data[data == nodata] = DSS_UNDEFINED_VALUE
    data[np.isnan(data)] = DSS_UNDEFINED_VALUE

    return data
