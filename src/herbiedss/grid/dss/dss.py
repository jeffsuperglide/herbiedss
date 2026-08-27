""" """

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from hecdss import HecDss
from hecdss.gridded_data import GriddedData
from herbie.core import Herbie
from osgeo import gdal, gdalconst, osr
from rich.console import Console

from herbiedss.grid.dss.dss_helpers import (
    _build_dss_pathname,
    _extract_grid_metadata,
    _gdal_warp_options,
)
from herbiedss.grid.dss.dssprops import (
    DSS_UNDEFINED_VALUE,
    DssDataType,
    DssGridType,
    GridCellsize,
    GridSystem,
    SpatialReferenceDefinition,
    TimeZone,
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
from herbiedss.utils.untis import Units
from herbiedss.utils.validate import parse_date_values, parse_option_values

app = typer.Typer()

console = Console()
error_console = Console(stderr=True, style="bold red")

DEFAULT_DSS = "herbiedss.dss"


def dss(
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
    dss_data_type: Annotated[
        DssDataType | None,
        typer.Option(
            "--dss-data-type",
            help="A Dss data type.",
        ),
    ] = None,
    cellsize: Annotated[
        GridCellsize,
        typer.Option("--cellsize", help="Cell size for the DSS grid"),
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
    output_bounds: Annotated[
        tuple[int, int, int, int] | None,
        typer.Option(
            "--output-bounds",
            help="Tuple of integers defining a bounding box (e.g. (minX, minY, maxX, maxY)).",
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
    output_bounds: tuple or None, optional
        tuple of integers (minX, minY, maxX, maxY)

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
                    ds = H.xarray(
                        search=subset,
                        remove_grib=remove_grib,
                    )
                    meta = _extract_grid_metadata(
                        ds,
                        var_name=variable,
                    )

                except Exception as exc:  # noqa: BLE001
                    error_console.print(
                        f"{date} F{hr:03d}: xarray()/extraction/ failed: {exc}"
                    )
                    continue

                # check DSS parts and create the path
                _apart = grid_system.upper() if len(apart) <= 0 else apart
                _bpart = (
                    boundary_file.name.replace(boundary_file.suffix, "")
                    if boundary_file is not None and len(bpart) <= 0
                    else bpart
                )
                _cpart = meta.get("long_name", "") if len(cpart) <= 0 else cpart
                if "precipitation" in _cpart.lower():
                    _cpart = "PRECIP"
                _fpart = f"{model}-{product}-{hr:03d}" if len(fpart) <= 0 else fpart

                # build the dsspath
                dss_path = _build_dss_pathname(_apart, _bpart, _cpart, _fpart, meta)

                # convert the units
                if meta.get("units", None) is not None:
                    meta["units"] = Units().to_preferred(meta["units"])

                if meta["units"] == "accum":
                    dss_data_type = DssDataType.PER_CUM

                if dss_data_type is None:
                    dss_data_type = DssDataType.PER_CUM

                # console.print(
                #     f"[cyan]{date} F{hr:03d}[/cyan] grid shape={meta['shape']} "
                #     f"variable={meta['variable']} units={meta['units']} "
                #     f"grid_system={meta.get('grid_system')} "
                #     f"path={dss_path}"
                # )

                # create DSS record
                src = H.download(search=subset)
                dst_osr_srs = osr.SpatialReference()
                dst_srs = "EPSG:5070"
                epsg_code = dst_srs.split(":")[-1]
                dst_osr_srs.ImportFromEPSG(int(epsg_code))
                warp_kwargs = _gdal_warp_options(boundary_file, output_bounds)
                warp_kwargs = {
                    **warp_kwargs,
                    "format": "MEM",
                    "xRes": cellsize,
                    "yRes": cellsize,
                    "dstSRS": dst_osr_srs.ExportToWkt(),
                    "targetAlignedPixels": True,
                    "resampleAlg": gdalconst.GRA_Bilinear,
                    "copyMetadata": False,
                }
                with gdal.Open(src) as ds:
                    warp_ds = gdal.Warp(
                        "",  # empty string => no filename, return a Dataset
                        ds,
                        **warp_kwargs,
                    )

                    band = warp_ds.GetRasterBand(1)
                    nodata = band.GetNoDataValue()
                    data = band.ReadAsArray().astype(np.float32, copy=False)
                    # have to flip the data for DSS
                    data_flip = np.flipud(data)
                    # replace nodata value with NaN
                    if nodata is not None:
                        data_flip[data_flip == nodata] = np.nan

                    xsize = warp_ds.RasterXSize
                    ysize = warp_ds.RasterYSize

                    adfGeoTransform = warp_ds.GetGeoTransform()

                    llx = int(adfGeoTransform[0] / adfGeoTransform[1])
                    lly = int(
                        (adfGeoTransform[5] * ysize + adfGeoTransform[3])
                        / adfGeoTransform[1]
                    )

                dss_grid_type = DssGridType.from_grid_system(grid_system.upper())

                srs_def_default = SpatialReferenceDefinition.from_grid_system("SHG")
                srs_def_from_grid = SpatialReferenceDefinition.from_grid_system(
                    grid_system.upper()
                )
                srs_def = srs_def_default.value if len(srs_def_from_grid.value)==0 else srs_def_from_grid.value

                time_zone = TimeZone.UTC
                with HecDss(dssfile) as dss:
                    # create_options = {
                    #     "path": dss_path,
                    #     "type": dss_grid_type.value,
                    #     "dataType": dss_data_type.code,
                    #     "lowerLeftCellX": llx,
                    #     "lowerLeftCellY": lly,
                    #     "numberOfCellsX": xsize,
                    #     "numberOfCellsY": ysize,
                    #     "srsName": dss_grid_type.name,
                    #     "srsDefinitionType": 1,
                    #     "srsDefinition": srs_def,
                    #     "dataUnits": "mm",
                    #     "dataSource": "INTERNAL",
                    #     "timeZoneID": time_zone.name,
                    #     "timeZoneRawOffset": time_zone.value,
                    #     "isInterval": 1,
                    #     "isTimeStamped": 1,
                    #     "cellSize": cellsize,
                    #     "xCoordOfGridCellZero": 0.0,
                    #     "yCoordOfGridCellZero": 0.0,
                    #     "nullValue": DSS_UNDEFINED_VALUE,
                    #     "data": data_flip,
                    # }
                    record = GriddedData.create(
                        path=dss_path,
                        type=dss_grid_type.value,
                        dataType=dss_data_type.code,
                        lowerLeftCellX=llx,
                        lowerLeftCellY=lly,
                        numberOfCellsX=xsize,
                        numberOfCellsY=ysize,
                        srsName=dss_grid_type.name,
                        srsDefinitionType=1,
                        srsDefinition=srs_def,
                        dataUnits="mm",
                        dataSource="INTERNAL",
                        timeZoneID=time_zone.name,
                        timeZoneRawOffset=time_zone.value,
                        isInterval=1,
                        isTimeStamped=1,
                        cellSize=cellsize,
                        xCoordOfGridCellZero=0.0,
                        yCoordOfGridCellZero=0.0,
                        nullValue=DSS_UNDEFINED_VALUE,
                        data=data,
                    )

                try:
                    status = dss.put(record)
                    if status != 0:
                        error_console.print(f"DSS put() failed.  Status is {status}.")
                    else:
                        console.print(
                            f"[green]{date} F{hr:03d} written to DSS:[/green] {dss_path}"
                        )
                except Exception as exc:  # noqa: BLE001
                    error_console.print(f"{date} F{hr:03d}: dss.put() failed: {exc}")
                    continue
