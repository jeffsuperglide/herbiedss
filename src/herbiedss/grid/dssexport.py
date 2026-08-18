"""
dssexport.py

Typer command that downloads NWP model output via Herbie, loads it as an
xarray object, optionally reprojects/clips it onto a hydrologic grid (SHG
or HRAP), and writes the resulting 2D grid into a HEC-DSS file as a
gridded-data record with a pathname derived from the grid's own valid
time.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from hecdss import HecDss
from hecdss.gridded_data import GriddedData
from herbie.core import Herbie
from rich.console import Console

from herbiedss.options import (
    DateOption,
    FxxOption,
    ModelOption,
    OverwriteOption,
    ProductOption,
    SaveDirOption,
    SepOption,
    SubsetOption,
    VerboseOption,
)
from herbiedss.utils.reproject import (
    GridSystem,
    clip_to_boundary,
    get_target_crs_and_cellsize,
    reproject_to_grid,
)
from herbiedss.utils.validate import parse_date_values, parse_option_values

app = typer.Typer()

console = Console()
error_console = Console(stderr=True, style="bold red")

DEFAULT_DSS = "herbiedss.dss"


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
        target_crs, cellsize = get_target_crs_and_cellsize(grid_system)
        da = reproject_to_grid(da, grid_system)
        if boundary_path is not None:
            da = clip_to_boundary(da, boundary_path)

        meta["grid_system"] = grid_system.upper()
        meta["target_crs"] = target_crs.to_wkt()
        meta["cell_size"] = cellsize

        # DSS grid records key off the lower-left cell index in the target
        # grid's own coordinate system, not lat/lon -- capture x/y bounds
        # post-reprojection.
        x_coord = da.coords.get("x")
        y_coord = da.coords.get("y")
        if x_coord is not None and y_coord is not None:
            meta["lower_left_x"] = float(np.nanmin(x_coord.values))
            meta["lower_left_y"] = float(np.nanmin(y_coord.values))
    else:
        meta["grid_system"] = None
        meta["target_crs"] = None
        meta["cell_size"] = None

    grid = np.asarray(da.values)
    if grid.ndim > 2:
        grid = grid.squeeze()
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D grid after squeeze, got shape {grid.shape}.")

    meta["shape"] = grid.shape
    return grid, meta


# @app.command()
def dssexport(
    date: DateOption,
    model: ModelOption = "hrrr",
    product: ProductOption = "sfc",
    fxx: FxxOption = "0",
    sep: SepOption = ",",
    save_dir: SaveDirOption = None,
    subset: SubsetOption = None,
    verbose: VerboseOption = False,
    overwrite: OverwriteOption = False,
    remove_grib: Annotated[
        bool,
        typer.Option(
            "--remove-grib",
            help="Delete the local GRIB2 file after loading it into xarray.",
        ),
    ] = False,
    dssfile: Annotated[
        str,
        typer.Option(
            "--dssfile",
            help="Output HEC-DSS file path.",
        ),
    ] = DEFAULT_DSS,
    apart: Annotated[
        str,
        typer.Option(
            "--apart",
            help="A-part: grid reference system (e.g. SHG, HRAP, model name).",
        ),
    ] = "",
    bpart: Annotated[
        str,
        typer.Option(
            "--bpart",
            help="B-part: region / watershed / location name.",
        ),
    ] = "",
    cpart: Annotated[
        str,
        typer.Option(
            "--cpart",
            help="C-part: data parameter (e.g. PRECIP, TMP2M).",
        ),
    ] = "",
    fpart: Annotated[
        str,
        typer.Option(
            "--fpart",
            help="F-part: version / descriptive label.",
        ),
    ] = "",
    variable: Annotated[
        str | None,
        typer.Option(
            "--variable",
            help="Explicit xarray variable name if --subset matches more than one.",
        ),
    ] = None,
    grid_system: Annotated[
        GridSystem | None,
        typer.Option(
            "--grid-system",
            help=(
                "Reproject the model grid (HRRR/GFS/GEFS/etc.) onto this hydrologic "
                "grid before writing to DSS. Choices: 'shg' (Albers, 2000 m native "
                "cell size) or 'hrap' (polar stereographic, 4762.5 m native cell "
                "size). If omitted, the grid is written in its native projection."
            ),
        ),
    ] = "shg",
    boundary_file: Annotated[
        Path | None,
        typer.Option(
            "--boundary-file",
            help=(
                "Path to a watershed boundary vector file (shapefile, GeoJSON, "
                "GeoPackage, etc.) to clip the reprojected grid to. Requires "
                "--grid-system to also be set, since clipping happens after "
                "reprojection. The file's CRS can be anything (e.g. WGS84) -- "
                "it is reprojected to match the target grid automatically."
            ),
        ),
    ] = None,
) -> None:
    """
    Load each date/fxx combination with Herbie's xarray(), optionally reproject
    the field onto SHG or HRAP (clipping to a watershed boundary if supplied),
    extract a 2D numpy grid and its metadata, build a DSS pathname whose D/E
    parts come from the grid's own start/end time, and write the record into a
    HEC-DSS file.

    For every combination of model run date and forecast lead time, this
    command builds a `Herbie` object, opens the (optionally subset) GRIB2
    output as an xarray Dataset/DataArray via `Herbie.xarray()`, extracts a
    single 2D field with `_extract_grid_and_metadata` (reprojecting onto a
    hydrologic grid and clipping to a watershed boundary if requested),
    derives a DSS pathname from the user-supplied A/B/C/F parts and the
    grid's own start/end time, and writes the resulting `GriddedData`
    record into the target HEC-DSS file. Failures for an individual
    date/fxx combination (xarray/extraction/reprojection errors, or
    `dss.put()` failures) are logged to stderr and that combination is
    skipped rather than aborting the whole run.

    Parameters
    ----------
    date : DateOption
        One or more model initialization dates/times, as a
        separator-delimited string (see `sep`). Parsed via
        `parse_date_values` into a list of date strings Herbie can
        interpret.
    model : ModelOption, optional
        Name of the NWP model to download (e.g. "hrrr", "gfs", "rap").
        Case-insensitive. Defaults to "hrrr".
    product : ProductOption, optional
        Output variable product/file type for the model (e.g. "sfc",
        "prs"). Model-dependent. Defaults to "sfc".
    fxx : FxxOption, optional
        One or more forecast lead times in hours, as a
        separator-delimited string (see `sep`). Parsed via
        `parse_option_values` into a list of integers. Defaults to "0".
    sep : SepOption, optional
        Delimiter used to split multiple values passed to `date` and
        `fxx` into lists. Defaults to ",".
    save_dir : SaveDirOption, optional
        Local directory used both for Herbie's downloaded GRIB2 files and,
        if `dssfile` is left at its default name, as the directory in
        which the output DSS file is created. If `None`, Herbie's default
        data directory is used for downloads and the DSS file is created
        in the current working directory.
    subset : SubsetOption, optional
        Regular-expression search string passed to `Herbie.xarray(search=...)`
        to restrict which GRIB messages/variables are loaded. If `None`,
        the entire file's contents are loaded.
    verbose : VerboseOption, optional
        If `True`, print additional diagnostic output while Herbie locates
        and downloads each file. Defaults to `False`.
    overwrite : OverwriteOption, optional
        If `True`, re-download and overwrite the local GRIB2 file even if
        it already exists. Defaults to `False`.
    remove_grib : bool, optional
        If `True`, delete the local GRIB2 file after it has been loaded
        into xarray (only if Herbie itself downloaded it during this run).
        Passed through to `Herbie.xarray(remove_grib=...)`. Defaults to
        `False`.
    dssfile : str, optional
        Path to the output HEC-DSS file. If left at the default
        `"herbiedss.dss"` and `save_dir` is provided, the file is created
        inside `save_dir`. Defaults to `"herbiedss.dss"`.
    apart : str, optional
        A-part of the DSS pathname: grid reference system (e.g. "SHG",
        "HRAP") or model name. Defaults to `""`.
    bpart : str, optional
        B-part of the DSS pathname: region, watershed, or location name.
        Defaults to `""`.
    cpart : str, optional
        C-part of the DSS pathname: data parameter (e.g. "PRECIP",
        "TMP2M"). Defaults to `""`.
    fpart : str, optional
        F-part of the DSS pathname: version or descriptive label.
        Defaults to `""`.
    variable : str or None, optional
        Explicit xarray variable name to extract when `subset` matches
        more than one variable in the returned Dataset. If `None`, the
        first variable is used. Defaults to `None`.
    grid_system : GridSystem or None, optional
        Hydrologic grid to reproject the model field onto before writing
        to DSS: `"shg"` (Albers equal-area, 2000 m native cell size) or
        `"hrap"` (polar stereographic, 4762.5 m native cell size). If
        `None`, the grid is written in its native model projection.
        Defaults to `"shg"`.
    boundary_file : Path or None, optional
        Path to a watershed boundary vector file (shapefile, GeoJSON,
        GeoPackage, etc.) used to clip the reprojected grid. Requires
        `grid_system` to also be set, since clipping is applied after
        reprojection; the boundary file's own CRS may differ from the
        target grid's, as it is reprojected automatically to match.
        Defaults to `None`.

    Returns
    -------
    None
        This command does not return a value. Progress and results for
        each date/fxx combination (grid shape, variable, units, grid
        system, DSS pathname, and write confirmation) are printed to
        stdout via `console.print`; errors are printed to stderr via
        `error_console.print`.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if `boundary_file` is supplied without
        `grid_system` also being set, since clipping cannot be performed
        without a target grid to reproject onto first.
    """
    if boundary_file is not None and grid_system is None:
        error_console.print(
            "--boundary-file requires --grid-system (shg|hrap) to also be set."
        )
        raise typer.Exit(code=1)

    resolved_date: list[str] = parse_date_values(date, sep)
    resolved_fxx: list[int] = parse_option_values(fxx, sep)

    if dssfile == DEFAULT_DSS and save_dir:
        dssfile = (Path(save_dir) / dssfile).as_posix()

    kwargs = {
        "model": model,
        "product": product,
        "verbose": verbose,
        "overwrite": overwrite,
    }
    if save_dir is not None:
        kwargs["save_dir"] = str(save_dir)

    with HecDss(str(dssfile)) as dss:
        for dt in resolved_date:
            for hr in resolved_fxx:
                kwargs["date"] = dt
                kwargs["fxx"] = hr

                H = Herbie(**kwargs)
                try:
                    xr_kwargs = {
                        "search": subset,
                        "remove_grib": remove_grib,
                    }
                    print(xr_kwargs)
                    ds = H.xarray(**xr_kwargs)
                    grid, meta = _extract_grid_and_metadata(
                        ds,
                        var_name=variable,
                        grid_system=grid_system,
                        boundary_path=boundary_file,
                    )
                except Exception as exc:  # noqa: BLE001
                    error_console.print(
                        f"{date} F{hr:03d}: xarray()/extraction/reprojection failed: {exc}"
                    )
                    continue

                dss_path = _build_dss_pathname(apart, bpart, cpart, fpart, meta)

                console.print(
                    f"[cyan]{date} F{hr:03d}[/cyan] grid shape={meta['shape']} "
                    f"variable={meta['variable']} units={meta['units']} "
                    f"grid_system={meta.get('grid_system')} "
                    f"path={dss_path}"
                )

                record = GriddedData.create(
                    data=grid.astype(np.float64),
                    path=dss_path,
                )
                # record.data = grid.astype(np.float64)
                # record.id = dss_path

                # attach whatever additional metadata fields the
                # installed hecdss record actually exposes. Confirm the real
                # attribute names for your hecdss version.
                for attr_name, value in {
                    # "dataUnits": meta["units"],
                    # "data_type": "PER-CUM" if meta.get("end_time") else "INST-VAL",
                    # "grid_reference_system": meta.get("grid_system"),
                    "cell_size": meta.get("cell_size"),
                    # "lowerLeftCellX": meta.get("lower_left_x", meta.get("lon_min")),
                    # "lowerLeftCellY": meta.get("lower_left_y", meta.get("lat_min")),
                    "srsDefinition": meta.get("target_crs"),
                }.items():
                    if value is not None and hasattr(record, attr_name):
                        setattr(record, attr_name, value)

                try:
                    dss.put(record)
                except Exception as exc:  # noqa: BLE001
                    error_console.print(f"{date} F{hr:03d}: dss.put() failed: {exc}")
                    continue

                console.print(
                    f"[green]{date} F{hr:03d} written to DSS:[/green] {dss_path}"
                )
