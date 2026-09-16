"""
tellme.py

Typer command that uses the Herbie tell_me_everything() method providing
information about the Herbie object.

"""

from herbie.core import Herbie
from rich.console import Console
from rich.table import Table

from herbiedss.options import DateOption, ModelOption

console = Console()
error_console = Console(stderr=True, style="bold red")

from pathlib import Path

from rich.console import Console
from rich.pretty import Pretty
from rich.text import Text


def render_value(value):
    if isinstance(value, (dict, list, tuple, set)):
        return Pretty(value, expand_all=True)

    if isinstance(value, Path):
        return Text(str(value), style="yellow")

    if value is None or value == "":
        return Text("—", style="dim italic")

    if isinstance(value, bool):
        return Text(str(value), style="green" if value else "red")

    return str(value)


def tell_me_everything(
    date: DateOption,
    model: ModelOption = None,
):
    kwargs = {
        "date": date,
        "model": model,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    H = Herbie(**kwargs)

    table = Table(
        title="[bold cyan]HRRR / Herbie Details[/bold cyan]",
        header_style="bold magenta",
        border_style="cyan",
    )
    table.add_column("Attribute", style="bold green", no_wrap=True)
    table.add_column("Value", overflow="fold")

    names = [
        "DESCRIPTION",
        "DETAILS",
        "EXPECT_IDX_FILE",
        "IDX_STYLE",
        "LOCALFILE",
        "PRODUCTS",
        "SOURCES",
        "config",
        "fxx",
        "get_localFileName",
        "get_remoteFileName",
        "grib",
        "grib_source",
        "idx_source",
        "model",
        "overwrite",
        "product",
        "product_description",
        "search_help",
    ]

    for name in names:
        value = getattr(H, name, None)
        table.add_row(name, render_value(value))

    console.print(table)
