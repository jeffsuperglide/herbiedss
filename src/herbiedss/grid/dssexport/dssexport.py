"""
dssexport.py

Typer command that downloads NWP model output via Herbie, loads it as an
xarray object, optionally reprojects/clips it onto a hydrologic grid (SHG
or HRAP), and writes the resulting 2D grid into a HEC-DSS file as a
gridded-data record with a pathname derived from the grid's own valid
time.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from hecdss import HecDss
from hecdss.gridded_data import GriddedData
from herbie.core import Herbie
from rich.console import Console

from herbiedss.grid.dssexport.helpers import (
    DssDataType,
    _build_dss_pathname,
    _extract_grid_and_metadata,
)
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
    SHG_CELL_SIZE_M,
    GridSystem,
)
from herbiedss.utils.untis import Units
from herbiedss.utils.validate import parse_date_values, parse_option_values

app = typer.Typer()

console = Console()
error_console = Console(stderr=True, style="bold red")

DEFAULT_DSS = "herbiedss.dss"


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
    dss_data_type: Annotated[
        DssDataType,
        typer.Option(
            "--dss-data-type",
            help="A Dss data type.",
        ),
    ] = DssDataType.PER_CUM,
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
        GridSystem,
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
    cellsize: Annotated[
        int, typer.Option("--cellsize", help="Cell size for the DSS grid")
    ] = 2000,
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

                # check DSS parts and create the path
                _apart = grid_system.upper() if len(apart) <= 0 else apart
                _bpart = (
                    boundary_file.name.replace(boundary_file.suffix, "")
                    if boundary_file is not None and len(bpart) <= 0
                    else bpart
                )
                _cpart = meta["long_name"] if len(cpart) <= 0 else cpart
                if "precipitation" in _cpart.lower():
                    _cpart = "PRECIP"
                _fpart = f"{model}-{product}-{hr:03d}".upper()

                dss_path = _build_dss_pathname(_apart, _bpart, _cpart, _fpart, meta)

                # convert the units
                meta["units"] = Units().to_preferred(meta["units"])

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
                # datatype = dss_dtype
                for attr_name, value in {
                    "type": dss_grid_type.value,
                    "dataUnits": meta["units"],
                    "dataType": dss_data_type.code,
                    # "grid_reference_system": meta.get("grid_system"),
                    "cellSize": meta.get("cell_size", SHG_CELL_SIZE_M),
                    "lowerLeftCellX": meta.get("lower_left_x", meta.get("lon_min")),
                    "lowerLeftCellY": meta.get("lower_left_y", meta.get("lat_min")),
                    "numberOfCellsY": meta.get("sizes", 0)[0],
                    "numberOfCellsX": meta.get("sizes", 0)[-1],
                    "maxDataValue": meta.get("max_data_value", 0.0),
                    "minDataValue": meta.get("min_data_value", 0.0),
                    "meanDataValue": meta.get("mean_data_value", 0.0),
                    "nullValue": meta.get("missing_value", 3.4028234663852886e38),
                    "srsDefinition": meta.get("target_crs"),
                }.items():
                    if value is not None and hasattr(record, attr_name):
                        setattr(record, attr_name, value)

                try:
                    status = dss.put(record)
                    error_console.print(f"DSS put() status is {status}.")
                except Exception as exc:  # noqa: BLE001
                    error_console.print(f"{date} F{hr:03d}: dss.put() failed: {exc}")
                    continue

                console.print(
                    f"[green]{date} F{hr:03d} written to DSS:[/green] {dss_path}"
                )
