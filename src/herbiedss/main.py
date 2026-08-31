"""
herbiedss.py - A simple Typer CLI wrapping Herbie's inventory(), download() and xarray() methods.

Usage:
    python -m herbiedss.main inventory --date 2024-01-01T00:00 --model hrrr --fxx 0 --subset ":TMP:2 m"
    python -m herbiedss.main download  --date 2024-01-01T00:00 --model hrrr --fxx 0 --subset ":TMP:2 m"
    python -m herbiedss.main --help

CLI:
    herbiedss inventory --date 2024-01-01T00:00 --model hrrr --fxx 0 --subset ":TMP:2 m"
    herbiedss download  --date 2024-01-01T00:00 --model hrrr --fxx 0 --subset ":TMP:2 m"
    herbiedss --help
"""

from __future__ import annotations

import typer
from rich.console import Console

from .grid.download import download as download_command
from .grid.dss.dss import dss as dss_command
from .grid.inventory import inventory as inventory_command

app = typer.Typer(
    name="herbiedss",
    help="A small CLI around Herbie's inventory() and download() methods.",
    no_args_is_help=True,
)

app.command(name="download")(download_command)
app.command(name="inventory")(inventory_command)
app.command(name="dss")(dss_command)


console = Console()
error_console = Console(stderr=True, style="bold red")


@app.callback()
def main() -> None:
    """
    Global options shared by every subcommand.
    """


if __name__ == "__main__":
    app()
