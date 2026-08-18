from datetime import datetime

import typer


def validate_date(value: str) -> str:
    """Validate that a date string matches one of Herbie's accepted formats.

    Attempts to parse ``value`` against a fixed list of accepted date/time
    formats (compact ``YYYYMMDDHH``, ISO date, and ISO date-time with either
    a ``T`` or space separator). The formats are tried in order and the
    first one that parses successfully short-circuits the loop. The
    original string is returned unchanged on success; no reformatting is
    performed.

    Parameters
    ----------
    value : str
        The date string to validate, e.g. ``"2024-01-01"``,
        ``"2024-01-01T00:00"``, ``"2024-01-01 00:00"``, or
        ``"2024010100"``.

    Returns
    -------
    str
        The same ``value`` that was passed in, if it matches one of the
        accepted formats.

    Raises
    ------
    typer.BadParameter
        If ``value`` does not match any of the accepted formats. The
        exception message lists the accepted formats so Typer can surface
        a helpful error to the CLI user.
    """
    fmts = ["%Y%m%d%H", "%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"]
    for fmt in fmts:
        try:
            if datetime.strptime(value, fmt).astimezone():
                return value
        except ValueError:
            continue
    raise typer.BadParameter(
        f"Could not parse date '{value}'. Try YYYY-MM-DD, YYYY-MM-DDTHH:MM, "
        f"'YYYY-MM-DD HH:MM', or YYYYMMDDHH."
    )


def parse_date_values(value: str, sep: str = ",") -> list[str]:
    """Parse a delimited string of dates into a list of validated date strings.

    Splits ``value`` on ``sep``, strips whitespace from each resulting
    token, discards empty tokens, and validates each remaining token with
    :func:`validate_date`. Intended for use as a Typer callback so a CLI
    option can accept multiple dates in a single comma-separated argument.

    Parameters
    ----------
    value : str
        A delimited string containing one or more date tokens, e.g.
        ``"2024-01-01,2024-01-02"``.
    sep : str, optional
        The delimiter used to split ``value`` into individual date tokens,
        by default ``","``.

    Returns
    -------
    list[str]
        The validated date strings, in the order they appeared in
        ``value``. Empty tokens are omitted from the result.

    Raises
    ------
    typer.BadParameter
        Propagated from :func:`validate_date` if any non-empty token fails
        to match one of the accepted date formats.
    """
    result: list[str] = []
    for token in value.split(sep):
        token = token.strip()
        if not token:
            continue

        result.append(validate_date(token))
    return result


def parse_option_values(value: str, sep: str = ",") -> list[int]:
    """Parse a delimited string of integers and ranges into a flat list of ints.

    Splits ``value`` on ``sep`` and, for each non-empty token, either
    appends a single integer or expands an inclusive numeric range. A
    token is treated as a range if it contains a ``-`` anywhere after its
    first character (this excludes a leading sign character from being
    mistaken for a range separator, so ``"-5"`` is a single negative
    value while ``"1-5"`` and ``"-3--1"`` are ranges). Ranges are split on
    the last ``-`` in the token via ``rpartition``, and the first
    character of the original token is re-attached to the start value to
    preserve a leading sign. Both ascending and descending ranges are
    supported, and the ``end`` value is always included in the output.

    Parameters
    ----------
    value : str
        A delimited string containing integers and/or hyphenated ranges,
        e.g. ``"1,3,5-8"`` or ``"-3,-1-1"``.
    sep : str, optional
        The delimiter used to split ``value`` into individual tokens, by
        default ``","``.

    Returns
    -------
    list[int]
        The expanded list of integers, with ranges unrolled inclusively
        and single-value tokens converted directly to ``int``.

    Raises
    ------
    ValueError
        If a token (or the head/tail of a range token) cannot be
        converted to ``int``. This is not caught here and will propagate
        to the caller as a raw ``ValueError`` rather than a
        ``typer.BadParameter``.
    """
    result: list[int] = []
    for token in value.split(sep):
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:
            head, _, tail = token[1:].rpartition("-")
            head = token[0] + head
            start, end = int(head), int(tail)
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(token))
    return result
