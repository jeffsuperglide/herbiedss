"""
inventory.py

Typer command that lists the GRIB2 messages (fields) available for a
given date/model/forecast-hour combination by reading Herbie's parsed
index (.idx) file, optionally filtered by a search pattern, and prints
the result as a formatted table for each date/fxx pair requested.
"""

import typer
from herbie.core import Herbie
from rich.console import Console
from rich.table import Table

from herbiedss.utils.validate import parse_date_values, parse_option_values

from ..options import (
    DateOption,
    FxxOption,
    ModelOption,
    OverwriteOption,
    ProductOption,
    SepOption,
    SubsetOption,
    VerboseOption,
)

console = Console()
error_console = Console(stderr=True, style="bold red")


# @app.command()
def inventory(
    # ctx: typer.Context,
    date: DateOption,
    model: ModelOption = "hrrr",
    product: ProductOption = "sfc",
    fxx: FxxOption = "0",
    sep: SepOption = ",",
    subset: SubsetOption = None,
    verbose: VerboseOption = False,
    overwrite: OverwriteOption = False,
) -> None:
    """Show the GRIB2 inventory (available fields) for the given date/model/fxx.

    For every combination of model run date and forecast lead time, this
    command builds a `Herbie` object and calls `Herbie.inventory()` to
    read and parse the file's index (.idx) into a pandas DataFrame,
    without downloading the full GRIB2 file. Each row of the resulting
    DataFrame describes one GRIB message: its message number, byte
    range, reference/valid time, variable, level, and forecast time. The
    DataFrame is rendered as a Rich table and printed to the console for
    each date/fxx pair.

    If `subset` is supplied, it is passed to `Herbie.inventory(search=...)`
    as a regular expression that filters messages by matching against
    their variable, level, and forecast-time fields (Herbie's
    `search_this` column), so only the matching rows are shown.

    Parameters
    ----------
    date : DateOption
        One or more model initialization dates/times to inspect, as a
        separator-delimited string (see `sep`). Parsed via
        `parse_date_values` into a list of date strings that `Herbie`
        can interpret.
    model : ModelOption, optional
        Name of the NWP model whose index file should be read (e.g.
        "hrrr", "hrrrak", "rap", "gfs", "ecmwf"). Case-insensitive.
        Defaults to "hrrr".
    product : ProductOption, optional
        Output variable product/file type for the model (e.g. "sfc" for
        surface, "prs" for pressure levels). Model-dependent. Defaults
        to "sfc".
    fxx : FxxOption, optional
        One or more forecast lead times in hours, as a
        separator-delimited string (see `sep`). Parsed via
        `parse_option_values` into a list of integers. Defaults to "0".
    sep : SepOption, optional
        Delimiter used to split multiple values passed to `date` and
        `fxx` into lists. Defaults to ",".
    subset : SubsetOption, optional
        Regular-expression search string used to filter the inventory to
        matching GRIB messages (e.g. ":TMP:2 m" or ":500 mb"). If
        `None`, the full inventory is shown.
    verbose : VerboseOption, optional
        If `True`, print additional diagnostic output while Herbie
        locates and reads each index file. Defaults to `False`.
    overwrite : OverwriteOption, optional
        If `True`, forces Herbie to ignore any locally cached copy of the
        file/index when resolving the source. Defaults to `False`.

    Returns
    -------
    None
        This command does not return a value. For each date/fxx
        combination, a titled Rich table of matching inventory rows is
        printed to stdout via `console.print`, or a warning message if
        no entries match.

    Raises
    ------
    typer.Exit
        Raised with exit code 1 if `Herbie.inventory` raises any
        exception while locating or parsing the index file for a given
        date/fxx combination; the triggering exception's message is
        printed to stderr first. Also raised with exit code 0 (a normal,
        non-error exit) as soon as a date/fxx combination's inventory is
        empty or `None`, which stops processing any remaining
        combinations in the loop.
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

            H = Herbie(**kwargs)

            try:
                df = H.inventory(search=subset) if subset else H.inventory()
            except Exception as exc:  # noqa: BLE001
                error_console.print(f"Failed to fetch inventory: {exc}")
                raise typer.Exit(code=1)

            if df is None or df.empty:
                console.print("[yellow]No matching inventory entries found.[/yellow]")
                raise typer.Exit(code=0)

            table = Table(title=f"Inventory: {model.upper()} {date} F{hr:03d}")
            for col in df.columns:
                table.add_column(str(col))
            for _, row in df.iterrows():
                table.add_row(*(str(v) for v in row))
            console.print(table)
