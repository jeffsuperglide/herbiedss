# herbiedss

A command-line tool that fetches NOAA weather model output (HRRR, GFS, GEFS, and other models supported by [Herbie](https://github.com/blaylockbk/Herbie)), optionally reprojects it onto the USACE **Standard Hydrologic Grid (SHG)** or the NWS **Hydrologic Rainfall Analysis Project (HRAP)** grid, clips it to a watershed boundary, and writes the result into a **HEC-DSS** file for use in HEC-HMS, HEC-RAS, or other USACE modeling tools.

## Features

- Downloads GRIB2 model output via Herbie for any date, forecast hour, model, and product combination.
- Filters GRIB messages with regex search strings against Herbie's inventory, so you can target specific fields like hourly (not cumulative) accumulated precipitation.
- Reprojects the native model grid (e.g. HRRR's Lambert Conformal, GFS/GEFS's regular lat-lon) onto SHG (Albers Equal-Area, 2000 m native cell size) or HRAP (polar stereographic, 4762.5 m native cell size).
- Handles both rectilinear (1D lat/lon) and curvilinear (2D lat/lon) source grids automatically.
- Clips the reprojected grid to a user-supplied watershed boundary file (shapefile, GeoJSON, GeoPackage, or anything GeoPandas can read), regardless of that file's own CRS.
- Writes each grid to HEC-DSS with a pathname whose D/E parts are derived from the grid's own start/end time, so instantaneous and accumulated fields are timestamped correctly.

## Installation

```bash
pip install herbiedss
```

Requires Python 3.10+. Core dependencies include `herbie-data`, `hecdss`, `rioxarray`, `pyproj`, `geopandas`, `typer`, and `rich`.

## Usage

### Basic export (native model projection)

```bash
herbiedss dssexport \\
    --date 2026-08-18 \\
    --model hrrr \\
    --fxx 6 \\
    --search ":TMP:2 m above ground:" \\
    --dssfile output.dss \\
    --apart HRRR --bpart CONUS --cpart TMP2M
```

### Reproject to SHG and clip to a watershed

```bash
herbiedss dssexport \\
    --date 2026-08-18 \\
    --model hrrr \\
    --fxx 6 \\
    --search ":APCP:.*:(?:0-1|[1-9]\\d*-\\d+) hour" \\
    --grid-system shg \\
    --boundary-file watershed.shp \\
    --dssfile output.dss \\
    --apart SHG --bpart TRINITY --cpart PRECIP
```

### Reproject to HRAP without clipping

```bash
herbiedss dssexport \\
    --date 2026-08-18 \\
    --model gfs \\
    --fxx 12 \\
    --grid-system hrap \\
    --dssfile output.dss \\
    --apart HRAP --bpart CONUS --cpart PRECIP
```

## CLI Options

| Option                                     | Description                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `--date`                                   | Forecast initialization date(s). Accepts a separator-delimited list via `--sep`.                         |
| `--model`                                  | Herbie model name (`hrrr`, `gfs`, `gefs`, etc.). Default: `hrrr`.                                        |
| `--product`                                | Model product/subset (e.g. `sfc`). Default: `sfc`.                                                       |
| `--fxx`                                    | Forecast lead hour(s). Accepts a separator-delimited list.                                               |
| `--search`                                 | Regex pattern to filter Herbie's GRIB inventory (passed to `.xarray(search=...)`).                       |
| `--variable`                               | Explicit xarray variable name, if `--search` matches more than one.                                      |
| `--grid-system`                            | Reproject onto `shg` or `hrap` before writing to DSS. Omit to keep the model's native projection.        |
| `--boundary-file`                          | Path to a watershed boundary vector file. Requires `--grid-system`.                                      |
| `--dssfile`                                | Output HEC-DSS file path. Default: `herbiedss.dss`.                                                      |
| `--apart`, `--bpart`, `--cpart`, `--fpart` | DSS pathname A/B/C/F parts. D and E parts are derived automatically from the grid's own timing metadata. |
| `--save-dir`                               | Local directory for downloaded GRIB2 files.                                                              |
| `--remove-grib`                            | Delete the local GRIB2 file after loading it into xarray.                                                |
| `--overwrite`                              | Re-download and overwrite existing local GRIB2 files.                                                    |
| `--verbose`                                | Enable verbose Herbie logging.                                                                           |

## How Reprojection Works

Herbie's `.xarray()` accessor exposes each model's native coordinate reference system via `da.herbie.crs` (a Cartopy CRS). The `herbiedss.utils.reproject` module converts that to a `pyproj` CRS, attaches it to the DataArray with `rioxarray`, and calls `.rio.reproject()` onto one of two hardcoded target CRS definitions, since neither has a standard EPSG code:

- **SHG**: `+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +datum=NAD83 +units=m`
- **HRAP**: `+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371200 +units=m`

Model grids come in two shapes, handled automatically:

- **Rectilinear** grids (GFS, GEFS): latitude/longitude are already 1D dimension coordinates, so renaming to `x`/`y` is direct.
- **Curvilinear** grids (HRRR's native Lambert Conformal): latitude/longitude are 2D auxiliary coordinates. The module looks for an existing 1D projected coordinate under alternate names, or reconstructs one by round-tripping a row/column of the 2D lat/lon through the source CRS.

Clipping to a watershed boundary happens *after* reprojection, using `.rio.clip()` with the boundary file's geometry reprojected to match the target grid's CRS — this is a separate step from reprojection and works with any vector format GeoPandas supports, not just rasters.

## Known Limitations

- `hecdss.GriddedData` attribute names for grid reference system, cell size, and lower-left origin vary by installed `hecdss` version. The export step attaches these defensively via `hasattr()` checks — verify with `dir(GriddedData())` on your installed version if georeferencing isn't showing up correctly in the output DSS file.
- Accumulated fields (e.g. `APCP`) can have multiple GRIB messages with overlapping-looking inventory strings (`0-2 hour` vs. `1-2 hour`). Always test your `--search` regex with `H.inventory(search)` before running a full export, since an imprecise pattern can silently match the wrong accumulation window.
- HRAP and SHG are both undefined outside the conterminous United States; reprojecting global or non-CONUS domains onto either grid will produce questionable results.

## License

MIT
