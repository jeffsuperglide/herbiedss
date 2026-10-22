import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import typer

from herbiedss.grid.dss.dssprops import DSS_UNDEFINED_VALUE


class BBox(NamedTuple):
    """Axis-aligned bounding box as ``(xmin, ymin, xmax, ymax)`` floats.

    Coordinates are in the CRS given separately (see ``--bbox-epsg``); the
    tuple itself carries no CRS information.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float


def _parse_path_or_bbox(value: str | None) -> Path | BBox | None:
    """
    Parse a CLI option that is either a vector-file path or a bounding box.

    The value is interpreted in this order:

    1. If it names an existing path, or its suffix (case-insensitive) is one of
       ``.shp``, ``.geojson`` or ``.json``, it is returned as a `Path`. The
       file's existence is not required for those suffixes.
    2. Otherwise it is split on commas and/or whitespace and must yield exactly
       four numbers ``xmin, ymin, xmax, ymax``, returned as a `BBox` of floats.

    Parameters
    ----------
    value : str or None
        Raw option value, e.g. ``"basin.shp"``, ``"-100,30,-90,40"`` or
        ``"-100 30 -90 40"``.

    Returns
    -------
    Path or BBox or None
        `None` if ``value`` is `None` or an empty string; a `Path` for
        file-like input; a `BBox` for coordinate input.

    Raises
    ------
    typer.BadParameter
        If the value does not split into exactly four parts, or if the box
        violates ``xmin < xmax`` and ``ymin < ymax``.

    Notes
    -----
    Values are converted with `float`, so non-integer coordinates are accepted.
    A part that cannot be converted to a number raises a plain `ValueError`
    from `float`, not `typer.BadParameter`.
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

        bbox = BBox(*(float(x) for x in parts))
        if bbox.xmin >= bbox.xmax or bbox.ymin >= bbox.ymax:
            raise typer.BadParameter(
                "Bounding box must satisfy xmin < xmax and ymin < ymax"
            )
        return bbox


def _parse_preferred_units(value: str) -> dict[str, str]:
    """
    Parse a JSON string mapping source units to preferred units.

    Parameters
    ----------
    value : str
        JSON object whose keys and values are all strings,
        e.g. ``'{"K": "C"}'``.

    Returns
    -------
    dict[str, str]
        The parsed mapping.

    Raises
    ------
    typer.BadParameter
        If the string is not valid JSON, or is not a JSON object with only
        string keys and string values.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise typer.BadParameter(
            'Expected a JSON object of string pairs, e.g. \'{"K": "C"}\''
        )

    return parsed


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
    formatted as ``24Dec2005:2400`` when `is_end=True`. The same datetime is
    formatted as ``25Dec2005:0000`` when `is_end=False`.


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
        A time string in ``DDMMMYYYY:HHMM`` format. The month abbreviation is
        produced by ``strftime("%b")``, so it is mixed case (``Dec``) and
        locale-dependent; it is NOT upper-cased here. Callers that need a
        fully uppercase DSS pathname must upper-case the result.


        Examples include:


        - ``"25Dec2005:1200"`` for noon on 25 December 2005.
        - ``"25Dec2005:0000"`` for a start time at midnight on 25 December
          2005.
        - ``"24Dec2005:2400"`` for an interval ending at midnight on
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
) -> str:
    """
    Build an HEC-DSS gridded-data pathname from caller-supplied pathname parts
    and the timing metadata produced by `_extract_grid_metadata`.


    The returned pathname follows the standard DSS form:


        /A/B/C/D/E/F/


    The A-, B-, C-, and F-parts are supplied by the caller. The D- and E-parts
    are derived from ``meta["start_time"]`` and ``meta["end_time"]``:


    - The D-part is ``start_time`` formatted with `_dss_time` (``is_end=False``).
    - The E-part is ``end_time`` formatted with `_dss_time` (``is_end=True``),
      so an end time at exactly 00:00 is written as ``2400`` on the previous
      day.


    The parts are inserted exactly as given; this function does not normalize
    the case of any part (including the month abbreviation from `_dss_time`).


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
        Metadata dictionary as returned by `_extract_grid_metadata`. Required
        keys:


        - ``"start_time"``: `datetime` for the D-part.
        - ``"end_time"``: `datetime` for the E-part.


    Returns
    -------
    str
        A DSS pathname in the form ``"/A/B/C/D/E/F/"``.


    Raises
    ------
    KeyError
        If ``start_time`` or ``end_time`` is missing from `meta`.
    AttributeError
        If ``end_time`` is `None`. `_extract_grid_metadata` sets it to `None`
        for instantaneous grids, which this function does not currently
        handle (no blank E-part is produced).


    Notes
    -----
    The current implementation assumes interval-style metadata, as noted by
    the inline comment in the code (GRIB product definition template 0 is
    assumed). It does not read ``GRIB_PDS_PDTN``, ``GRIB_REF_TIME``,
    ``GRIB_VALID_TIME`` or ``GRIB_FORECAST_SECONDS``, and it has no
    ``duration`` override. Times are assumed to be UTC.
    """
    start_time = meta["start_time"]
    end_time = meta["end_time"]

    # assuming what comes in is grib_pds_pdtn == 0
    d_part = _dss_time(start_time)
    e_part = _dss_time(end_time, is_end=True)

    return f"/{a_part}/{b_part}/{c_part}/{d_part}/{e_part}/{f_part}/"


def _extract_grid_metadata(
    ds: Any,
    var_name: str | None,
) -> dict:
    """
    Select one variable from the object returned by Herbie's ``.xarray()`` and
    collect its descriptive and timing metadata.


    Only the metadata is gathered here. The function does not reproject, clip,
    squeeze or otherwise modify the data, and it does not return the array.


    Parameters
    ----------
    ds : Any
        Object returned by `Herbie.xarray()`. Only an `xarray.Dataset` is
        processed; any other type (including a bare `xarray.DataArray`)
        causes an empty dict to be returned.
    var_name : str or None
        Name of the data variable to use. If `None`, the first data variable
        in the dataset is used.


    Returns
    -------
    dict
        Empty if `ds` is not a `Dataset`. Otherwise contains:


        - ``variable``: variable name, or ``"unknown"``.
        - ``units``: ``units`` attribute, or `None`.
        - ``long_name``: ``long_name`` attribute, falling back to
          ``GRIB_name``.
        - ``step_type``: ``GRIB_stepType`` attribute, or `None`.
        - ``valid_time``: `datetime` from the ``valid_time`` coordinate, else
          ``time + step``, else `None`.
        - ``start_time`` / ``end_time``: if ``GRIB_stepRange`` is of the form
          ``"a-b"`` (hours) and a ``time`` coordinate exists, these are the
          reference time plus ``a`` and ``b`` hours. Otherwise, if
          ``valid_time`` is known, ``start_time`` equals ``valid_time`` and
          ``end_time`` is `None` (instantaneous grid). If neither applies,
          both keys are absent.
        - ``center``: ``da.herbie.center``, or `None` if unavailable.
        - ``source_crs``: ``str(da.herbie.crs)``, or `None` if unavailable.


    Raises
    ------
    ValueError
        If `ds` is a `Dataset` with no data variables, or if `var_name` is
        given but is not one of its data variables.


    Notes
    -----
    Timing is read from the original Herbie output, and all datetimes are
    UTC-naive. Native bounds (``lat_min`` etc.), grid-system information
    and final array shape are not produced by this function.
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
    """
    Build the keyword arguments for ``gdal.WarpOptions`` that crop or clip a
    raster to a boundary.


    Parameters
    ----------
    boundary : Path or BBox or None
        - `None`: no spatial restriction.
        - `Path`: vector file (shapefile, GeoJSON, etc.) used as a cutline.
        - `BBox` (any 4-tuple): ``(xmin, ymin, xmax, ymax)`` output bounds.
    epsg : int or None
        EPSG code of the boundary coordinates. Required for a bounding box.
        For a vector file it is optional and, if given, is used as the cutline
        SRS (otherwise GDAL uses the SRS stored in the file).


    Returns
    -------
    dict
        - `{}` if `boundary` is `None`.
        - For a `Path`: ``cutlineDSName``, ``cropToCutline=True`` and
          ``warpOptions=["CUTLINE_ALL_TOUCHED=TRUE"]`` (cells touched by the
          polygon are kept), plus ``cutlineSRS`` when `epsg` is set.
        - For a bounding box: ``outputBounds`` and ``outputBoundsSRS``
          (``"EPSG:<epsg>"``).


    Raises
    ------
    FileNotFoundError
        If `boundary` is a `Path` that does not exist.
    ValueError
        If `boundary` is a tuple that does not have four elements.
    typer.BadParameter
        If `boundary` is a bounding box and `epsg` is `None` (hint:
        ``--bbox-epsg``).
    TypeError
        If `boundary` is of any other type.
    """
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
            raise ValueError("Bounding box must be a 4-tuple (xmin, ymin, xmax, ymax)")
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
    """
    Replace nodata and NaN cells with ``DSS_UNDEFINED_VALUE``, in place.


    Parameters
    ----------
    data : numpy.ndarray
        Array to modify. It is mutated directly, not copied, and must be of a
        dtype that can hold ``DSS_UNDEFINED_VALUE``.
    nodata : float or None
        Sentinel nodata value, matched by exact equality. Ignored if `None`.
        A NaN sentinel never matches by equality; NaN cells are handled by the
        separate NaN check.


    Returns
    -------
    numpy.ndarray
        The same array object passed in, after replacement.


    See Also
    --------
    _to_dss_undefined : Non-mutating variant that also handles infinities and
        uses a tolerance when matching nodata values.
    """
    if nodata is not None:
        data[data == nodata] = DSS_UNDEFINED_VALUE
    data[np.isnan(data)] = DSS_UNDEFINED_VALUE

    return data


def _to_dss_undefined(
    data: np.ndarray,
    *nodata_values: float | None,
) -> np.ndarray:
    """
    Return a float32 copy of `data` with invalid cells set to
    ``DSS_UNDEFINED_VALUE``.


    A cell is treated as invalid if it is non-finite (NaN, +inf or -inf) or
    matches any of the supplied nodata values. Finite nodata values are
    matched with `numpy.isclose` (default tolerances); a NaN nodata value
    matches NaN cells.


    Parameters
    ----------
    data : numpy.ndarray
        Input array. It is not modified; it is converted to float32 and copied.
    *nodata_values : float or None
        Zero or more additional nodata sentinels. `None` entries are skipped.


    Returns
    -------
    numpy.ndarray
        New float32 array of the same shape as `data`.
    """
    result = np.asarray(data, dtype=np.float32).copy()

    invalid = ~np.isfinite(result)

    for value in nodata_values:
        if value is None:
            continue
        if np.isnan(value):
            invalid |= np.isnan(result)
        else:
            invalid |= np.isclose(result, value)

    result[invalid] = DSS_UNDEFINED_VALUE
    return result
