"""
download.py

Typer command that downloads GRIB2 model output (optionally subset to
specific fields) using the `Herbie` package, iterating over every
date/forecast-hour combination requested on the command line.

Raises
------
typer.Exit
    Raised with exit code 1 if `Herbie.download` fails for any requested
    date/forecast-hour combination (e.g. the file is not found at any
    known archive source, or a network/IO error occurs).
"""

import typer
from herbie.core import Herbie
from rich.console import Console

from herbiedss.utils.validate import parse_date_values, parse_option_values

from ..options import (
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

console = Console()
error_console = Console(stderr=True, style="bold red")


# @app.command()
def download(
    # ctx: typer.Context,
    date: DateOption,
    model: ModelOption = "hrrr",
    product: ProductOption = "sfc",
    fxx: FxxOption = "0",
    sep: SepOption = ",",
    save_dir: SaveDirOption = None,
    subset: SubsetOption = None,
    verbose: VerboseOption = False,
    overwrite: OverwriteOption = False,
) -> None:
    """Download the GRIB2 file (optionally subset) for the given date/model/fxx.

    For every combination of model run date and forecast lead time (fxx),
    this command builds a `Herbie` object and downloads the corresponding
    GRIB2 file from whichever archive source (NOMADS, AWS, Google Cloud,
    Azure, ECMWF, Pando, etc.) has it available. If a `subset` search
    string is supplied, only the matching GRIB messages (fields) are
    downloaded instead of the full file, which saves bandwidth and disk
    space.

    Parameters
    ----------
    date : DateOption
        One or more model initialization dates/times to download, as a
        separator-delimited string (see `sep`). Parsed via
        `parse_date_values` into a list of date strings that `Herbie`
        can interpret (e.g. "2024-01-01 12:00").
    model : ModelOption, optional
        Name of the NWP model to download, as defined in Herbie's model
        template folder (e.g. "hrrr", "hrrrak", "rap", "gfs", "ecmwf").
        Case-insensitive. Defaults to "hrrr".
    product : ProductOption, optional
        Output variable product/file type for the model (e.g. "sfc" for
        surface, "prs" for pressure levels, "nat", "subh"). Case-sensitive
        and model-dependent. Defaults to "sfc".
    fxx : FxxOption, optional
        One or more forecast lead times in hours, as a separator-delimited
        string (see `sep`). Parsed via `parse_option_values` into a list
        of integers. Available lead times depend on the model and model
        version. Defaults to "0" (the analysis/initialization hour).
    sep : SepOption, optional
        Delimiter used to split multiple values passed to `date` and
        `fxx` into lists. Defaults to ",".
    save_dir : SaveDirOption, optional
        Local directory in which to save downloaded files. If `None`,
        Herbie's default data directory (from its configuration) is
        used instead.
    subset : SubsetOption, optional
        Regular-expression search string used to filter GRIB messages by
        variable/level (e.g. ":TMP:2 m" or ":500 mb"), matched against the
        file's index (.idx) entries. If provided, only the matching
        messages are downloaded via `Herbie.download(search=subset)`;
        if `None`, the full GRIB2 file is downloaded.
    verbose : VerboseOption, optional
        If `True`, print additional diagnostic output while locating and
        downloading each file. Defaults to `False`.
    overwrite : OverwriteOption, optional
        If `True`, re-download and overwrite the file even if it already
        exists locally. If `False`, an existing local copy is reused.
        Defaults to `False`.

    Returns
    -------
    None
        This command does not return a value. For each successfully
        downloaded file, the resolved local path is printed to stdout
        via `console.print`.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if `Herbie.download` raises any exception
        for a given date/fxx combination. The triggering exception's
        message is printed to stderr before exiting.
    """

    resolved_date: list[str] = parse_date_values(date, sep)
    resolved_fxx: list[int] = parse_option_values(fxx, sep)

    for dt in resolved_date:
        for hr in resolved_fxx:
            kwargs = {
                "date": dt,
                "model": model,
                "product": product,
                "fxx": hr,
                "verbose": verbose,
                "overwrite": overwrite,
            }

            if save_dir is not None:
                kwargs["save_dir"] = str(save_dir)

            H = Herbie(**kwargs)

            try:
                path = H.download(search=subset) if subset else H.download()
            except Exception as exc:  # noqa: BLE001
                error_console.print(f"Download failed: {exc}")
                raise typer.Exit(code=1)

            console.print(f"[green]Saved:[/green] {path}")
