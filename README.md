# herbiedss

A command-line tool for downloading NOAA numerical-weather-prediction output through [Herbie](https://github.com/blaylockbk/Herbie), reprojecting and optionally clipping GRIB2 grids with **GDAL**, and writing the resulting rasters to HEC-DSS for use in HEC-HMS, HEC-RAS, and other USACE tools.

`herbiedss` supports HRRR, GFS, GEFS, and other Herbie-supported models. It can write source data onto the USACE Standard Hydrologic Grid (SHG), the NWS Hydrologic Rainfall Analysis Project (HRAP) grid, or another supported DSS grid definition.

## Features

- Downloads GRIB2 model output through Herbie for one or more initialization times and forecast lead hours.
- Processes every raster band in a downloaded GRIB2 product, or limits processing with a regex `--subset` search.
- Uses GDAL to open GRIB2 files, read per-band GRIB metadata, and warp each band to the requested target grid.
- Reprojects to SHG by default: EPSG:5070 / CONUS Albers Equal Area at a 2,000 m cell size.
- Supports HRAP and other package-defined DSS grid systems, including their DSS grid-type and spatial-reference definitions.
- Clips output after reprojection using either a watershed boundary vector file or explicit output bounds.
- Reads watershed boundaries from formats supported by GDAL/OGR, including shapefiles, GeoJSON, and GeoPackage files.
- Reprojects boundary geometries automatically as part of the GDAL warp operation.
- Converts GRIB units to the package's preferred DSS-compatible units.
- Builds DSS pathname D/E parts from GRIB time metadata and the parameter duration.
- Flips warped grids to DSS row orientation and replaces missing values with the DSS undefined-value convention.
- Writes `hecdss.gridded_data.GriddedData` records to an HEC-DSS file.

## Installation

```bash
pip install herbiedss
```

Python 3.10+ is required. Core dependencies include `herbie-data`, `hecdss`, `numpy`, `typer`, and `rich`.

The `dss` command also requires the GDAL Python bindings (`osgeo`). GDAL installation is platform- and Python-version-specific. Verify that the bindings are available before running an export:

```bash
python -c "from osgeo import gdal, osr; print(gdal.VersionInfo())"
```

### Windows GDAL note

If `herbiedss dss` reports that GDAL cannot be imported:

1. Download a GDAL wheel matching your Python version and system architecture from [cgohlke/geospatial-wheels](https://github.com/cgohlke/geospatial-wheels/releases).
2. Install it, for example:

   ```bash
   python -m pip install GDAL-<version>-cp<python>-cp<python>-win_amd64.whl
   ```

3. Rerun the command.

The wheel's `cpXXX` tag must match your Python version—for example, `cp312` for Python 3.12. Most 64-bit Intel/AMD Windows systems use `win_amd64`. GDAL wheels may also require the [Microsoft Visual C++ Redistributable for Visual Studio 2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Usage

The primary command is `herbiedss dss`. The forecast initialization time is required and must be passed with the `--date` option.

```bash
herbiedss dss --date DATE [OPTIONS]
```

`--date` accepts one or more model initialization dates/times. Use `--sep` to choose the separator when passing multiple dates or forecast hours.

### Basic SHG export

By default, the command warps output to SHG (EPSG:5070) at 2,000 m resolution and writes to `herbiedss.dss`.

```bash
herbiedss dss \
  --date 2026-08-18T00:00 \
  --model hrrr \
  --product sfc \
  --fxx 6 \
  --subset "TMP:2 m above ground" \
  --dssfile output.dss \
  --apart SHG \
  --bpart CONUS \
  --cpart TMP2M
```

### Clip to a watershed boundary

Supply a vector boundary with `--boundary-file`. GDAL uses the cutline during the warp, so the vector file may use a different CRS than the destination grid.

```bash
herbiedss dss \
  --date 2026-08-18T00:00 \
  --model hrrr \
  --product sfc \
  --fxx 1,2,3 \
  --subset "APCP" \
  --grid-system shg \
  --boundary-file watershed.geojson \
  --dssfile output.dss \
  --apart SHG \
  --bpart TRINITY \
  --cpart PRECIP
```

When a boundary file is supplied, its filename (without extension) is used as the default DSS B-part. Provide `--bpart` only when you want to override that behavior.

### Clip to explicit bounds

Use `--output-bounds` to set a target-grid bounding box instead of a vector cutline. Bounds are supplied as integer coordinates in the destination grid coordinate system:

```bash
herbiedss dss \
  --date 2026-08-18T00:00 \
  --model hrrr \
  --fxx 6 \
  --subset "APCP" \
  --grid-system shg \
  --output-bounds "(200000,1200000,800000,1800000)" \
  --dssfile output.dss
```

### Multiple model runs and lead times

```bash
herbiedss dss \
  --date 2026-08-18T00:00,2026-08-18T12:00 \
  --model hrrr \
  --product sfc \
  --fxx 1,2,3,6 \
  --subset "APCP" \
  --dssfile precip.dss
```

The command processes every `--date`/`--fxx` combination. A failure for one combination is reported and does not prevent later combinations from being attempted.

## CLI options

| Option | Description |
| --- | --- |
| `--date` | **Required.** One or more forecast initialization dates/times. Supply multiple values using `--sep`. |
| `--model` | Herbie model name, such as `hrrr`, `gfs`, `gefs`, or `rap`. Default: `hrrr`. |
| `--product` | Model product or subset, such as `sfc` or `prs`. Default: `sfc`. |
| `--fxx` | One or more forecast lead hours. Default: `0`. |
| `--sep` | Separator for multiple `--date` and `--fxx` values. Default: `,`. |
| `--subset` | Regex search string passed to Herbie's download method to limit GRIB messages. |
| `--variable` | Reserved explicit xarray-variable selection option. The GDAL export path writes GRIB raster bands. |
| `--grid-system` | Target DSS grid system. Default: `shg`. |
| `--cellsize` | Target grid cell size in meters. Default: `2000`. |
| `--boundary-file` | Watershed boundary vector file used as a GDAL cutline. Requires a target grid system. |
| `--output-bounds` | Destination-grid bounding box: `(minX, minY, maxX, maxY)`. |
| `--dssfile` | Output HEC-DSS filename or path. Default: `herbiedss.dss`. |
| `--apart` | DSS pathname A-part. Defaults to the uppercase grid system. |
| `--bpart` | DSS pathname B-part. Default: `GRID`; with a boundary file, the boundary filename is used. |
| `--cpart` | DSS pathname C-part. If omitted, it is derived from the GRIB element metadata. |
| `--fpart` | DSS pathname F-part. Defaults to `<MODEL>-<PRODUCT>-<FXX>`. |
| `--dss-data-type` | HEC-DSS data type, such as `PER-CUM` or `INST-VAL`. Default: `PER-CUM`. |
| `--save-dir` | Directory used for downloaded GRIB2 files; also receives the default DSS file when `--dssfile` is not specified. |
| `--remove-grib` | Delete a GRIB2 file after it is loaded, when Herbie downloaded it during this run. |
| `--overwrite` | Re-download and overwrite a local GRIB2 file. |
| `--verbose` | Enable verbose Herbie logging. |

## GDAL processing workflow

For each requested initialization time and forecast lead hour, `herbiedss dss`:

1. Creates a `Herbie` object and downloads the selected GRIB2 product, optionally using `--subset` to restrict the messages downloaded.
2. Opens the GRIB2 file with `osgeo.gdal`.
3. Iterates through each GDAL raster band in the file.
4. Reads GRIB metadata from the band, including element, unit, reference time, valid time, and duration information.
5. Uses `gdal.Warp()` to reproject the band, with aligned target pixels and bilinear resampling.
6. Applies the requested target resolution, output bounds, or vector cutline during that same warp operation.
7. Reads the warped raster array, flips it vertically for DSS convention, and replaces source NoData cells with the DSS undefined value.
8. Derives the DSS pathname and creates a `GriddedData` record with the grid dimensions, lower-left cell indexes, spatial-reference definition, time-zone metadata, units, and grid values.
9. Writes the record to the output DSS file.

## Grid systems and spatial references

### SHG

SHG is the default target grid. The GDAL warp destination is **EPSG:5070** (NAD83 / Conus Albers), using target-aligned pixels and a default 2,000 m cell size. The HEC-DSS record is identified as grid type `SHG` / `ALBERS` and stores the following SHG WKT spatial-reference definition:

```wkt
PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",
  GEOGCS["GCS_North_American_1983",
    DATUM["D_North_American_1983",
      SPHEROID["GRS_1980",6378137.0,298.257222101]],
    PRIMEM["Greenwich",0.0],
    UNIT["Degree",0.0174532925199433]],
  PROJECTION["Albers"],
  PARAMETER["False_Easting",0.0],
  PARAMETER["False_Northing",0.0],
  PARAMETER["Central_Meridian",-96.0],
  PARAMETER["Standard_Parallel_1",29.5],
  PARAMETER["Standard_Parallel_2",45.5],
  PARAMETER["Latitude_Of_Origin",23.0],
  UNIT["Meter",1.0]]
```

### HRAP

The package defines HRAP as a polar-stereographic grid for HEC-DSS metadata. Its spatial-reference definition is:

```wkt
PROJCS["Stereographic_CONUS_HRAP",
  GEOGCS["GCS_Sphere_LFM",
    DATUM["D_Sphere_LFM",
      SPHEROID["Shpere_LFM",6371200.0,0.0]],
    PRIMEM["Greenwich",0.0],
    UNIT["Degree",0.0174532925199433]],
  PROJECTION["Stereographic_North_Pole"],
  PARAMETER["False_Easting",1909762.5],
  PARAMETER["False_Northing",7624762.5],
  PARAMETER["Central_Meridian",-105.0],
  PARAMETER["Standard_Parallel_1",60.0],
  UNIT["Meter",1.0]]
```

When using a grid system other than the default SHG, verify the actual GDAL destination CRS and cell size produced by your installed command version before operational use. The current warp setup explicitly creates an EPSG:5070 destination CRS and applies the configured `--cellsize` to the GDAL output.

## DSS pathname and timing

A DSS pathname has the form:

```text
/A-PART/B-PART/C-PART/D-PART/E-PART/F-PART/
```

- A, B, C, and F parts are supplied by options or derived from the selected grid system, boundary filename, GRIB element, model/product, and forecast hour.
- D and E parts are built from the GRIB band's time metadata and its resolved duration.
- If `--cpart` is omitted, `herbiedss` derives a DSS-compatible parameter name from `GRIB_ELEMENT`.
- GRIB units are normalized through `herbiedss.utils.units.Units` before writing the record.

For accumulated precipitation and other interval fields, check the GRIB metadata and generated DSS pathname to ensure the selected message represents the intended accumulation window.

## Important notes

- **Date required.** Every invocation must include `--date`; `herbiedss dss` has no positional date argument.
- **Test subsets first.** GRIB files can contain multiple messages with similar metadata. Use Herbie's inventory tools to inspect available messages and make `--subset` as specific as necessary.
- **Band-oriented output.** The command writes each GDAL raster band as a separate DSS gridded record. If a product contains several variables or levels, either narrow `--subset` or expect multiple output records.
- **Bilinear resampling.** The GDAL warp uses bilinear resampling. This may be suitable for continuous fields such as temperature but may not preserve accumulation or categorical-field semantics exactly. Validate resampling choices for operational precipitation workflows.
- **Cell size and grid definitions.** The current CLI `GridCellsize` type exposes a 2,000 m value, matching the default SHG workflow. HRAP's conventional spacing differs; validate both dimensions and georeferencing in the written DSS file when using HRAP.
- **CONUS-focused definitions.** SHG and HRAP definitions are intended for CONUS hydrologic workflows. Treat results outside their intended domains with caution.
- **DSS metadata compatibility.** `hecdss` APIs and attribute names can vary by installed version. Inspect output in your target HEC application and confirm projection, cell size, origin, extent, units, data type, and pathname timing before production use.

## License

MIT
