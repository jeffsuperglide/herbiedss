"""
CLI command for downloading numerical weather prediction GRIB2 data with
Herbie, reprojecting each raster band onto a hydrologic grid, and writing the
resulting gridded records to an HEC-DSS file.

The `dss` command supports one or more model initialization times and forecast
lead times. For each requested model run and forecast hour, it:

1. Downloads the requested GRIB2 product through Herbie.
2. Opens the GRIB2 file with GDAL.
3. Reprojects each GRIB raster band to the requested target spatial reference
   system and cell size.
4. Optionally clips the output to a watershed boundary or explicit output
   bounds.
5. Converts GRIB units to the package's preferred DSS-compatible units.
6. Builds a DSS pathname using user-supplied pathname parts together with GRIB
   metadata such as reference time, valid time, parameter, and duration.
7. Flips the raster vertically and replaces missing values with the DSS
   undefined-value convention.
8. Creates and writes `hecdss.gridded_data.GriddedData` records to an
   HEC-DSS file.

The default output grid system is SHG, using EPSG:5070 and a 2,000-meter cell
size. HRAP or another supported `GridSystem` may be selected through the CLI.
When a boundary file is supplied, the boundary is reprojected automatically to
the target grid coordinate reference system before clipping.

Typical usage
-------------
Write HRRR surface-product fields for a single initialization time and forecast
hour to the default DSS file:

    herbiedss dss 2024-01-01T00:00 --model hrrr --product sfc --fxx 0

Write selected precipitation-related fields, clip them to a watershed boundary,
and save them to a specific DSS file:

    herbiedss dss 2024-01-01T00:00 \
        --model hrrr \
        --product sfc \
        --fxx 1,2,3 \
        --subset "APCP" \
        --boundary watershed.geojson \
        --dssfile output.dss \
        --bpart "MY_WATERSHED"

Notes
-----
- DSS pathname A-, B-, C-, and F-parts may be specified with command-line
  options. When omitted, sensible defaults are derived from the selected grid
  system, boundary filename, GRIB element metadata, model, product, and
  forecast hour.
- Each raster band in the downloaded GRIB2 file becomes a separate DSS
  gridded-data record.
- A failure while writing one date/forecast-hour combination is reported to
  stderr and does not stop processing of later combinations.
- `--boundary` requires a target `--grid-system`, because clipping occurs
  after GDAL reprojection.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from hecdss import HecDss
from hecdss.gridded_data import GriddedData
from herbie.core import Herbie
from rich.console import Console

from herbiedss.grid.dss.dss_helpers import (
    _build_dss_pathname,
    _dss_undefined_cells,
    _gdal_warp_options,
    _parse_path_or_bbox,
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
from herbiedss.utils.grib_element import GribElementResolver
from herbiedss.utils.units import Units
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
    ] = "GRID",
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
    boundary: Annotated[
        str | None,
        typer.Option(
            "--boundary",
            help=(
                "Path to a watershed boundary vector file (shapefile, GeoJSON, "
                "GeoPackage, etc.) to clip the reprojected grid to. Requires "
                "--grid-system to also be set, since clipping happens after "
                "reprojection. The file's CRS can be anything (e.g. WGS84) -- "
                "it is reprojected to match the target grid automatically."
            ),
            parser=_parse_path_or_bbox,
        ),
    ] = None,
    boundary_epsg: Annotated[
        int | None,
        typer.Option(
            "--bbox-epsg",
            help="EPSG code of the bbox (only used when --boundary is a bbox).",
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
    boundary : str or None, optional
        Path to a watershed boundary vector file (shapefile, GeoJSON,
        GeoPackage, etc.) used to clip the reprojected grid. GDAL will try
        to read the spatial reference from the boundary (vector layer) when
        it is available.  Explicitly define the boundary spatial reference
        using the boundary_epsg option.  A boundary defined as a bounding box (bbox)
        requires `"boundary_epsg"` definition.  Bounding box entry is a string formated
        as xmin,ymin,xmax,ymax or "xmin ymin xmax ymax". Defaults to `None`.
    boundary_epsg: integer or None, optional
        EPSG code of the boundary.  Defaults to destination/output CRS if `None`.

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
        Raised with exit code 1 if `boundary` is supplied without
        `grid_system` also being set, since clipping cannot be performed
        without a target grid to reproject onto first.
    """

    # try to import gdal
    try:
        from osgeo import gdal, gdalconst, osr

        osr.UseExceptions()

    except ImportError as exc:
        raise ImportError(
            "\n"
            "GDAL is required to use the `dss` command, but its Python bindings "
            "could not be imported.\n\n"
            "Windows installation instructions:\n"
            "  1. Download a GDAL wheel matching your Python version and system "
            "architecture from:\n"
            "     https://github.com/cgohlke/geospatial-wheels/releases\n"
            "  2. Install the downloaded wheel, for example:\n"
            "     python -m pip install GDAL-<version>-cp<python>-cp<python>-win_amd64.whl\n"
            "  3. Reinstall or rerun herbiedss.\n\n"
            "Choose a wheel whose `cpXXX` tag matches your Python version. For "
            "example, `cp312` is Python 3.12 and `cp313` is Python 3.13. "
            "Use `win_amd64` for most 64-bit Intel/AMD Windows installations.\n\n"
            "The GDAL wheels may require the Microsoft Visual C++ Redistributable "
            "for Visual Studio 2022:\n"
            "  https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist\n"
        ) from exc

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

                # check DSS parts and create the path
                _apart = grid_system.upper() if len(apart) <= 0 else apart.title()
                _bpart = (
                    boundary.name.replace(boundary.suffix, "").title()
                    if isinstance(boundary, Path)
                    else bpart
                )
                _fpart = (
                    f"{model}-{product}-{hr:03d}".upper() if len(fpart) <= 0 else fpart
                )

                if dss_data_type is None:
                    dss_data_type = DssDataType.PER_CUM

                src = H.download(search=subset)

                # Destination dataset spatial reference system.
                # dst_osr_srs = osr.SpatialReference()
                # dst_srs = "EPSG:5070"
                # epsg_code = dst_srs.split(":")[-1]
                # dst_osr_srs.ImportFromEPSG(int(epsg_code))

                with HecDss(dssfile) as dss:
                    #  DSS options
                    dss_grid_type = DssGridType.from_grid_system(grid_system.upper())

                    srs_def_default = SpatialReferenceDefinition.from_grid_system("SHG")
                    srs_def_from_grid = SpatialReferenceDefinition.from_grid_system(
                        grid_system.upper()
                    )
                    srs_def = (
                        srs_def_default.value
                        if len(srs_def_from_grid.value) == 0
                        else srs_def_from_grid.value
                    )

                    time_zone = TimeZone.UTC
                    create_options = {
                        # "path": dss_path,
                        "type": dss_grid_type.value,
                        "dataType": dss_data_type.code,
                        # "lowerLeftCellX": llx,
                        # "lowerLeftCellY": lly,
                        # "numberOfCellsX": xsize,
                        # "numberOfCellsY": ysize,
                        "srsName": dss_grid_type.name,
                        "srsDefinitionType": 1,
                        "srsDefinition": srs_def,
                        # "dataUnits": "mm",
                        "dataSource": "INTERNAL",
                        "timeZoneID": time_zone.name,
                        "timeZoneRawOffset": time_zone.value,
                        "isInterval": 1,
                        "isTimeStamped": 1,
                        "cellSize": cellsize,
                        "xCoordOfGridCellZero": 0.0,
                        "yCoordOfGridCellZero": 0.0,
                        "nullValue": DSS_UNDEFINED_VALUE,
                        # "data": data_flip,
                    }

                    # warp options
                    warp_kwargs = _gdal_warp_options(boundary, boundary_epsg) # type: ignore
                    warp_kwargs = {
                        "format": "MEM",
                        "xRes": cellsize,
                        "yRes": cellsize,
                        "dstSRS": srs_def,
                        "targetAlignedPixels": True,
                        "resampleAlg": gdalconst.GRA_Bilinear,
                        "copyMetadata": False,
                        # "creationOptions": ["COMPRESS=DEFLATE", "TILED=YES"],
                        **warp_kwargs,
                    }

                    # read datasets and write to DSS
                    with gdal.Open(src) as ds:
                        for band_num in range(1, ds.RasterCount + 1):
                            # get the band, its metadata, dataset, and build a dss path
                            band = ds.GetRasterBand(band_num)

                            # getting some meta data
                            # GRIB_REF_TIME is the HRRR cycle or model-run time.
                            band_meta = band.GetMetadata()

                            # build dss path.
                            # cpart comes from user input or GRIB_COMMENT
                            grib_element: str = band_meta.get(
                                "GRIB_ELEMENT", ""
                            ).strip()
                            element_resolver = GribElementResolver()
                            element_info = element_resolver.resolve(grib_element)
                            _cpart = (
                                element_info.dss_base_name.title()
                                if len(cpart) <= 0
                                else cpart
                            )

                            # build the dsspath
                            dss_path = _build_dss_pathname(
                                _apart,
                                _bpart,
                                _cpart,
                                _fpart,
                                band_meta,
                                element_info.duration_value,
                            )

                            # convert the units
                            grib_unit: str = band_meta.get("GRIB_UNIT", "").strip()
                            grib_unit = grib_unit.strip().strip("[]").lower()
                            units = Units().to_preferred(grib_unit)
                            band_ds = band.GetDataset()

                            # warp
                            warp_ds = gdal.Warp(
                                "",  # empty string => no filename, return a Dataset
                                band_ds,
                                **warp_kwargs,
                            )

                            xsize = warp_ds.RasterXSize
                            ysize = warp_ds.RasterYSize

                            adfGeoTransform = warp_ds.GetGeoTransform()

                            llx = int(adfGeoTransform[0] / adfGeoTransform[1])
                            lly = int(
                                (adfGeoTransform[5] * ysize + adfGeoTransform[3])
                                / adfGeoTransform[1]
                            )

                            # read the data as an array from the warp.
                            # have to flip the data for DSS
                            # replace nodata value with NaN
                            data = warp_ds.ReadAsArray().astype(np.float32, copy=False)
                            data_flip = np.flipud(data)
                            nodata = band.GetNoDataValue()
                            data = _dss_undefined_cells(data_flip, nodata)

                            create_options = {
                                "path": dss_path,
                                # "type": dss_grid_type.value,
                                # "dataType": dss_data_type.code,
                                "lowerLeftCellX": llx,
                                "lowerLeftCellY": lly,
                                "numberOfCellsX": xsize,
                                "numberOfCellsY": ysize,
                                # "srsName": dss_grid_type.name,
                                # "srsDefinitionType": 1,
                                # "srsDefinition": srs_def,
                                # "dataUnits": "mm",
                                # "dataSource": "INTERNAL",
                                # "timeZoneID": time_zone.name,
                                # "timeZoneRawOffset": time_zone.value,
                                # "isInterval": 1,
                                # "isTimeStamped": 1,
                                # "cellSize": cellsize,
                                # "xCoordOfGridCellZero": 0.0,
                                # "yCoordOfGridCellZero": 0.0,
                                # "nullValue": DSS_UNDEFINED_VALUE,
                                "dataUnits": units,
                                "data": data,
                                **create_options,
                            }

                            record = GriddedData.create(**create_options)

                            try:
                                status = dss.put(record)
                                if status != 0:
                                    error_console.print(
                                        f"DSS put() failed.  Status is {status}."
                                    )
                                else:
                                    console.print(
                                        f"[green]{date} F{hr:03d} written to DSS:[/green] {dss_path}"
                                    )
                            except Exception as exc:  # noqa: BLE001
                                error_console.print(
                                    f"{date} F{hr:03d}: dss.put() failed: {exc}"
                                )
                                continue
