"""
CLI command for downloading numerical weather prediction GRIB2 data with
Herbie, reprojecting the first raster band onto a hydrologic grid, and writing
the resulting gridded record to an HEC-DSS file.

The `dss` command supports one or more model initialization times and forecast
lead times. For each requested model run and forecast hour, it:

1. Downloads the requested GRIB2 product through Herbie (optionally limited to
   messages matching a `--subset` search string).
2. Reads GRIB message metadata (units, variable, times) with ecCodes and
   resolves the DSS unit, DSS data type, and C-part name from it.
3. Opens the GRIB2 file with GDAL and extracts band 1 as a single-band VRT.
4. Validates the source grid (spatial reference present, no un-rewrapped global
   0..360 longitude axis).
5. Reprojects the band to the target spatial reference system (SHG or HRAP)
   and cell size using bilinear resampling, with target-aligned pixels.
6. Optionally clips the output to a watershed boundary or an explicit bounding
   box.
7. Replaces missing values with the DSS undefined-value convention and flips
   the raster vertically, since DSS grids are stored south-to-north.
8. Verifies that the output grid is unrotated, has the expected cell size, and
   is aligned to the DSS cell grid, then computes the lower-left cell indices.
9. Builds a DSS pathname from user-supplied pathname parts together with GRIB
   metadata such as reference time, valid time, parameter, and duration.
10. Creates and writes a `hecdss.gridded_data.GriddedData` record to an
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
  options. When omitted, defaults are derived as follows: A-part from the
  grid system (e.g. "SHG"), B-part from the boundary filename (title-cased)
  or `--bpart` (default "GRID"), C-part from the resolved GRIB element
  (`GribElementResolver`), and F-part from "<MODEL>-<PRODUCT>" in upper case.
  D- and E-parts come from the GRIB reference/valid times and duration.
- Only the first raster band (band 1) of the downloaded GRIB2 file is
  processed and written as a DSS gridded-data record. Use `--subset` to
  select the message of interest.
- Output units are determined from the GRIB metadata and converted to the
  package's preferred DSS-compatible units (override with `--preferred-units`).
- Errors from `dss.put()` for a given date/forecast-hour combination are
  reported to stderr and processing continues with the next combination.
  Other failures (download, GDAL open/warp, grid validation) raise exceptions
  and are not caught by the command.
- `--boundary` clipping occurs after GDAL reprojection, so the output grid
  system (`--grid-system`, default "shg") determines the clipping CRS.
- GDAL's Python bindings are required; an ImportError with installation
  instructions is raised if they are unavailable.
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
    _gdal_warp_options,
    _parse_path_or_bbox,
    _parse_preferred_units,
    _to_dss_undefined,
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
from herbiedss.utils import eccoding
from herbiedss.utils.grib_element import GribElementResolver
from herbiedss.utils.source_info import _source_grid_info
from herbiedss.utils.units import Units
from herbiedss.utils.validate import parse_date_values, parse_option_values

app = typer.Typer()

console = Console()
error_console = Console(stderr=True, style="bold red")

DEFAULT_DSS = "herbiedss.dss"


def dss(
    date: DateOption,
    model: ModelOption = None,
    product: ProductOption = None,
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
    # variable: Annotated[
    #     str | None,
    #     typer.Option(
    #         "--variable",
    #         help="Explicit xarray variable name if --subset matches more than one.",
    #     ),
    # ] = None,
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
    # dss_data_type: Annotated[
    #     DssDataType | None,
    #     typer.Option(
    #         "--dss-data-type",
    #         help="A Dss data type.",
    #     ),
    # ] = None,
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
    bbox_epsg: Annotated[
        int | None,
        typer.Option(
            "--bbox-epsg",
            help="EPSG code of the bbox (only used when --boundary is a bbox).",
        ),
    ] = None,
    preferred_units: Annotated[
        dict[str, str] | None,
        typer.Option(
            "--preferred-units",
            help="Dictionary of preferred units to convert from/to.",
            parser=_parse_preferred_units,
        ),
    ] = None,
) -> None:
    """
    Download GRIB2 data with Herbie, reproject band 1 onto a hydrologic grid
    with GDAL, and write it to an HEC-DSS file as a gridded record.

    For every combination of model run date and forecast lead time, this
    command builds a `Herbie` object and downloads the (optionally
    subset) GRIB2 file with `Herbie.download()`. It reads message metadata
    with ecCodes (units, variable, reference/valid times) and resolves the
    DSS unit, data type, and C-part via `Units` and `GribElementResolver`.
    It then opens the file with GDAL, builds a single-band VRT of band 1,
    and warps it to the target SHG or HRAP spatial reference at the
    requested cell size. If `boundary` is supplied, the warp is clipped
    to it. Missing cells are set to the DSS undefined value, the array is
    flipped vertically, the output grid is validated (unrotated, expected
    cell size, aligned to the DSS cell grid), and the lower-left cell
    indices are computed. A DSS pathname is built from the A/B/C/F parts
    and the GRIB metadata, and the resulting `GriddedData` record is
    written with `HecDss.put()`.

    Only band 1 of each downloaded file is processed. A non-zero status or
    exception from `dss.put()` is reported to stderr and the loop moves on
    to the next combination. Other errors (download, GDAL open/warp,
    source-grid or alignment validation) raise and stop the run.

    Parameters
    ----------
    date : DateOption
        One or more model initialization dates/times, as a
        separator-delimited string (see `sep`). Parsed via
        `parse_date_values` into a list of date strings Herbie can
        interpret.
    model : ModelOption, optional
        Name of the NWP model to download (e.g. "hrrr", "gfs", "rap").
        Case-insensitive. If `None`, Herbie's default is used. Also used
        in the default F-part and the resolved DSS metadata.
    product : ProductOption, optional
        Output variable product/file type for the model (e.g. "sfc",
        "prs"). Model-dependent. If `None`, Herbie's default is used.
        Also used in the default F-part.
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
        Regular-expression search string passed to
        `Herbie.download(search=...)` to restrict which GRIB messages are
        downloaded. If `None`, the entire file is downloaded.
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
        "HRAP") or model name. If empty, the upper-case `grid_system`
        name is used; otherwise the value is title-cased. Defaults to `""`.
    bpart : str, optional
        B-part of the DSS pathname: region, watershed, or location name.
        Ignored when `boundary` is a file path, in which case the
        title-cased boundary file name (without extension) is used.
        Defaults to `"GRID"`.
    cpart : str, optional
        C-part of the DSS pathname: data parameter (e.g. "PRECIP",
        "TMP2M"). If empty, the DSS base name from `GribElementResolver`
        for the GRIB variable is used. Defaults to `""`.
    fpart : str, optional
        F-part of the DSS pathname: version or descriptive label. If
        empty, "<MODEL>-<PRODUCT>" in upper case is used. Defaults to `""`.
    grid_system : GridSystem, optional
        Hydrologic grid to reproject the model field onto before writing
        to DSS: `"shg"` (Albers equal-area, 2000 m native cell size) or
        `"hrap"` (polar stereographic, 4762.5 m native cell size). This
        also selects the DSS grid type and spatial reference definition
        (falling back to SHG if no WKT is defined for the system).
        Defaults to `"shg"`.
    cellsize : GridCellsize, optional
        Output cell size in meters, used for the warp resolution, the
        alignment checks, and the DSS record. Defaults to 2000.
    boundary : str or Path or None, optional
        Boundary used to clip the reprojected grid, parsed by
        `_parse_path_or_bbox`. Either a path to a vector file (shapefile,
        GeoJSON, GeoPackage, etc.) or a bounding box formatted as
        "xmin,ymin,xmax,ymax" or "xmin ymin xmax ymax". GDAL reads the
        spatial reference from a vector file when available. A bounding
        box requires `bbox_epsg`. Defaults to `None` (no clipping).
    bbox_epsg : int or None, optional
        EPSG code of the boundary bounding box. Defaults to the
        destination/output CRS if `None`.
    preferred_units : dict[str, str] or None, optional
        Mapping of units to preferred units, parsed by
        `_parse_preferred_units` and passed to `Units`. Used to convert
        the GRIB units to the DSS data units. Defaults to `None`, which
        uses the package defaults.

    Returns
    -------
    None
        Progress for each date/fxx combination (GDAL version, source grid
        info, band shape, lower-left cell, valid-cell count, min/max, and
        write confirmation with the DSS pathname) is printed to stdout via
        `console.print`. `dss.put()` errors are printed to stderr via
        `error_console.print`.

    Raises
    ------
    ImportError
        If the GDAL Python bindings cannot be imported.
    RuntimeError
        If band 1 cannot be opened, the source grid has no spatial
        reference, a global 0..360 longitude grid is not rewrapped by GDAL,
        the warp fails or returns an unexpected band count or shape, the
        warped grid is rotated, or the output resolution or origin does not
        match the DSS cell grid.
    """

    # try to import gdal
    try:
        from osgeo import gdal, gdalconst, osr

        osr.UseExceptions()
        gdal.SetConfigOption("GRIB_ADJUST_LONGITUDE_RANGE", "YES")

        console.print(f"GDAL version: {gdal.VersionInfo('--version')}")

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
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

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
                _fpart = f"{model}-{product}".upper() if len(fpart) <= 0 else fpart

                #  DSS options
                dss_grid_type = DssGridType.from_grid_system(grid_system.upper())

                # Getting the WKT for internal grid systems
                dst_srs_default = SpatialReferenceDefinition.from_grid_system("SHG")
                dst_srs_from_grid = SpatialReferenceDefinition.from_grid_system(
                    grid_system.upper()
                )

                dst_srs_wkt = (
                    dst_srs_default.value
                    if len(dst_srs_from_grid.value) == 0
                    else dst_srs_from_grid.value
                )

                # data source spatial reference
                dst_srs = osr.SpatialReference()
                dst_srs.ImportFromWkt(dst_srs_wkt)
                # dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

                time_zone = TimeZone.UTC

                # read datasets and write to DSS
                src = H.download(search=subset) if subset else H.download()

                # use ec codes to get metadata
                eccodes_metadata = eccoding._read_eccodes_metadata(src, message_index=0)
                u = Units(preferred=preferred_units)
                raw_units = eccodes_metadata.get("units")
                units = u.to_dss_unit(str(raw_units))

                resolved_eccodes_metadata = eccoding._resolve_dss_metadata(
                    eccodes_metadata,
                    model=model,  # type: ignore
                    product=product,  # type: ignore
                    fxx=hr,
                    dss_units=units.strip(),
                )

                # build dss path.
                # cpart comes from user input or GRIB_COMMENT
                grib_element: str = resolved_eccodes_metadata.get(
                    "variable", ""
                ).strip()

                resolver = GribElementResolver()

                resolver_info = resolver.resolve(grib_element)

                _cpart = resolver_info.dss_base_name if len(cpart) <= 0 else cpart

                # build the dsspath
                dss_path = _build_dss_pathname(
                    _apart,
                    _bpart,
                    _cpart,
                    _fpart,
                    resolved_eccodes_metadata,  # replacing band_meta
                    # element_info.duration_value,
                )

                with gdal.Open(src) as ds:
                    band_num = 1
                    band = ds.GetRasterBand(band_num)
                    if band is None:
                        raise RuntimeError(
                            f"Could not open GRIB band {band_num}; skipping."
                        )
                    # band_meta = band.GetMetadata()
                    src_nodata = band.GetNoDataValue()

                    # get the band, its metadata, dataset, and build a dss path
                    band_vrt = gdal.Translate(
                        "",
                        ds,
                        format="VRT",
                        bandList=[band_num],
                    )
                    source_info = _source_grid_info(band_vrt)

                    console.print(
                        "[cyan]Source grid:[/cyan] "
                        f"{source_info.projection_name}; "
                        f"extent=({source_info.xmin:.6f}, {source_info.ymin:.6f}, "
                        f"{source_info.xmax:.6f}, {source_info.ymax:.6f}); "
                        f"resolution=({source_info.xres:.8f}, {source_info.yres:.8f}); "
                        f"geographic={source_info.is_geographic}; "
                        f"global={source_info.is_global}; "
                        f"lon_0_360={source_info.uses_360_longitudes}"
                    )

                    if source_info.is_global and source_info.uses_360_longitudes:
                        version_num = int(gdal.VersionInfo("VERSION_NUM"))

                        if version_num < 3040000:
                            raise RuntimeError(
                                "This is a global GRIB grid with a 0..360 longitude axis. "
                                "GDAL < 3.4 does not transparently rewrap it. Upgrade to GDAL "
                                "3.4+ or add a longitude-normalization preprocessing step."
                            )

                        raise RuntimeError(
                            "GDAL still exposed a global 0..360 longitude grid despite "
                            "GRIB_ADJUST_LONGITUDE_RANGE=YES. Inspect the source extent and "
                            "apply explicit longitude normalization before the target warp."
                        )

                    if band_vrt is None:
                        raise RuntimeError(
                            f"Could not build VRT for GRIB band {band_num}."
                        )

                    src_srs = band_vrt.GetSpatialRef()
                    if src_srs is None:
                        raise RuntimeError(
                            f"GRIB band {band_num} has no source spatial reference. "
                            "Cannot safely reproject it."
                        )

                    clip_kwargs = _gdal_warp_options(boundary, bbox_epsg)  # type: ignore
                    clip_warp_options = clip_kwargs.pop("warpOptions", [])

                    single_band_warp_kwargs = {
                        "format": "MEM",
                        "xRes": float(cellsize),
                        "yRes": float(cellsize),
                        "dstSRS": dst_srs.ExportToWkt(),
                        "targetAlignedPixels": True,
                        "resampleAlg": gdalconst.GRA_Bilinear,
                        "copyMetadata": False,
                        "multithread": True,
                        "warpOptions": [
                            "INIT_DEST=NO_DATA",
                            "UNIFIED_SRC_NODATA=YES",
                            *clip_warp_options,
                        ],
                        **clip_kwargs,
                    }

                    WARP_NODATA = np.float32(-9999.0)
                    if src_nodata is not None:
                        single_band_warp_kwargs["srcNodata"] = src_nodata

                    single_band_warp_kwargs["dstNodata"] = float(WARP_NODATA)

                    # warp
                    warp_ds = gdal.Warp(
                        "",  # empty string => no filename, return a Dataset
                        band_vrt,
                        # srcSRS=src_srs.ExportToWkt(),
                        **single_band_warp_kwargs,
                    )

                    if warp_ds is None:
                        raise RuntimeError(
                            f"GDAL warp failed for GRIB band {band_num}."
                        )

                    if warp_ds.RasterCount != 1:
                        raise RuntimeError(
                            f"Expected one output raster band for source band {band_num}; "
                            f"got {warp_ds.RasterCount}."
                        )

                    # read the data as an array from the warp.
                    # have to flip the data for DSS
                    # replace nodata value with NaN
                    dst_nodata = warp_ds.GetRasterBand(band_num).GetNoDataValue()

                    data = warp_ds.GetRasterBand(1).ReadAsArray()
                    if data is None or data.ndim != 2:
                        raise RuntimeError(
                            f"Expected a 2-D result for GRIB band {band_num}; "
                            f"got {None if data is None else data.shape}."
                        )

                    data = data.astype(np.float32, copy=False)
                    # data = _dss_undefined_cells(data, dst_nodata)
                    data = _to_dss_undefined(data, src_nodata, dst_nodata)
                    data = np.flipud(data)

                    if data.ndim != 2:
                        raise RuntimeError(
                            f"Expected a 2-D raster after warp; got array shape {data.shape}."
                        )

                    gt = warp_ds.GetGeoTransform()
                    x0, xres, rx, y0, ry, yres = gt

                    if not np.isclose(rx, 0.0) or not np.isclose(ry, 0.0):
                        raise RuntimeError(
                            "Warp output has rotation terms; HEC-DSS gridded records require "
                            "an unrotated grid."
                        )

                    if not np.isclose(xres, float(cellsize)):
                        raise RuntimeError(
                            f"Output x resolution is {xres}, expected {float(cellsize)}."
                        )

                    if not np.isclose(yres, -float(cellsize)):
                        raise RuntimeError(
                            f"Output y resolution is {yres}, expected {-float(cellsize)}."
                        )

                    xsize = warp_ds.RasterXSize
                    ysize = warp_ds.RasterYSize

                    xmin = x0
                    ymin = y0 + ysize * yres

                    llx_float = xmin / float(cellsize)
                    lly_float = ymin / float(cellsize)

                    if not np.isclose(llx_float, round(llx_float), atol=1e-7):
                        raise RuntimeError(
                            f"Output X origin {xmin} is not aligned to a {cellsize}-m DSS grid."
                        )

                    if not np.isclose(lly_float, round(lly_float), atol=1e-7):
                        raise RuntimeError(
                            f"Output Y origin {ymin} is not aligned to a {cellsize}-m DSS grid."
                        )

                    llx = round(llx_float)
                    lly = round(lly_float)

                    create_options = {
                        "path": dss_path,
                        "type": dss_grid_type.value,
                        "dataType": resolver_info.dss_data_type.code,
                        "lowerLeftCellX": llx,
                        "lowerLeftCellY": lly,
                        "numberOfCellsX": xsize,
                        "numberOfCellsY": ysize,
                        "srsName": dss_grid_type.name,
                        "srsDefinitionType": 1,
                        "srsDefinition": dst_srs_wkt,
                        "dataUnits": units,
                        "dataSource": "INTERNAL",
                        "timeZoneID": time_zone.name,
                        "timeZoneRawOffset": time_zone.value,
                        "isInterval": 1,
                        "isTimeStamped": 1,
                        "cellSize": cellsize,
                        "xCoordOfGridCellZero": 0.0,
                        "yCoordOfGridCellZero": 0.0,
                        "nullValue": DSS_UNDEFINED_VALUE,
                        "data": data,
                    }

                    valid = data[data != DSS_UNDEFINED_VALUE]

                    console.print(
                        f"band={band_num}; shape={data.shape}; "
                        f"ll_cell=({llx}, {lly}); "
                        f"cellsize={cellsize}; "
                        f"valid_cells={valid.size}; "
                        f"min={np.nanmin(valid) if valid.size else 'NA'}; "
                        f"max={np.nanmax(valid) if valid.size else 'NA'}"
                    )
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
