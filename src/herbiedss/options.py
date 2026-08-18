from __future__ import annotations

from typing import Annotated, TypeAlias

import typer

DateOption: TypeAlias = Annotated[  # noqa: UP040
    str,
    typer.Option(
        "--date",
        "-d",
        help="Model initialization date. Formats: YYYYMMDDHH, YYYY-MM-DD, "
        "YYYY-MM-DDTHH:MM, 'YYYY-MM-DD HH:MM'.",
    ),
]
"""
CLI option for the model initialization date.

Parameters
----------
str
    Model initialization date, provided via ``--date`` / ``-d``.
    Accepted formats are:

    - ``YYYYMMDDHH``
    - ``YYYY-MM-DD``
    - ``YYYY-MM-DDTHH:MM``
    - ``'YYYY-MM-DD HH:MM'``
"""

ModelOption: TypeAlias = Annotated[  # noqa: UP040
    str,
    typer.Option("--model", "-m", help="NWP model name."),
]
"""
CLI option for the NWP model name.

Parameters
----------
str
    Name of the numerical weather prediction (NWP) model, provided via
    ``--model`` / ``-m``.
"""

ProductOption: TypeAlias = Annotated[  # noqa: UP040
    str | None,
    typer.Option("--product", help="Model-specific product, e.g. sfc, prs, 0p25."),
]
"""
CLI option for the model-specific product.

Parameters
----------
str or None
    Model-specific product identifier, provided via ``--product``
    (e.g. ``sfc``, ``prs``, ``0p25``). Defaults to ``None`` if not
    specified.
"""

FxxOption: TypeAlias = Annotated[  # noqa: UP040
    str,
    typer.Option("--fxx", "-f", help="Forecast hour."),
]
"""
CLI option for the forecast hour.

Parameters
----------
str
    Forecast hour (lead time), provided via ``--fxx`` / ``-f``.
"""

SepOption: TypeAlias = Annotated[  # noqa: UP040
    str, typer.Option("--sep", help="Seperator for the FXX option.")
]
"""
CLI option for the forecast hour separator.

Parameters
----------
str
    Separator string used when parsing or formatting the ``--fxx``
    option, provided via ``--sep``.
"""

VerboseOption: TypeAlias = Annotated[  # noqa: UP040
    bool, typer.Option("--verbose", help="Enable Herbie's verbose output.")
]
"""
CLI flag to enable verbose output.

Parameters
----------
bool
    If ``True``, enables Herbie's verbose output, provided via
    ``--verbose``.
"""

OverwriteOption: TypeAlias = Annotated[  # noqa: UP040
    bool, typer.Option("--overwrite", help="Enable Herbie's overwrite.")
]
"""
CLI flag to enable overwriting existing files.

Parameters
----------
bool
    If ``True``, enables Herbie's overwrite behavior, provided via
    ``--overwrite``.
"""

SearchOption: TypeAlias = Annotated[  # noqa: UP040
    str | None,
    typer.Option(
        None,
        "--search",
        help="Regex/search string to filter variables, e.g. ':TMP:2 m'.",
    ),
]
"""
CLI option for filtering variables via a search string.

Parameters
----------
str or None
    Regex or search string used to filter variables, provided via
    ``--search`` (e.g. ``':TMP:2 m'``). Defaults to ``None`` if not
    specified.
"""

SubsetOption: TypeAlias = Annotated[  # noqa: UP040
    str | None,
    typer.Option(
        "--subset",
        help="Regex/search string to filter variables, e.g. ':TMP:2 m'.",
    ),
]
"""
CLI option for subsetting variables via a search string.

Parameters
----------
str or None
    Regex or search string used to subset variables, provided via
    ``--subset`` (e.g. ``':TMP:2 m'``). Defaults to ``None`` if not
    specified.
"""

SaveDirOption: TypeAlias = Annotated[  # noqa: UP040
    str | None,
    typer.Option("--save-dir", help="Root directory to save the downloaded file(s)."),
]
"""
CLI option for the save directory.

Parameters
----------
str or None
    Root directory in which downloaded file(s) will be saved,
    provided via ``--save-dir``. Defaults to ``None`` if not
    specified.
"""
