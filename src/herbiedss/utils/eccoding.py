"""Helpers for extracting and resolving GRIB message timing metadata for DSS output.

This module reads metadata from individual GRIB messages through ecCodes. It
then converts that metadata into the start, end, and valid timestamps and the
unit label used when writing grid records to HEC-DSS. Grid values are never
modified here.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import eccodes as ec

from herbiedss.utils.models import FORECAST_INTERVAL_POLICIES


def _read_eccodes_metadata(
    grib_path: str | Path,
    *,
    message_index: int = 0,
) -> dict[str, Any]:
    """
    Read metadata for one zero-based GRIB message.

    The file is scanned sequentially until the message at ``message_index`` is
    reached. Forecast steps are decoded in seconds (``stepUnits`` is set to
    ``"s"``), so the result does not depend on the source step unit.

    Parameters
    ----------
    grib_path : str or pathlib.Path
        Path to the GRIB file.
    message_index : int, default 0
        Zero-based index of the message to read. Must be nonnegative.

    Returns
    -------
    dict of str to Any
        Message metadata with the following keys:

        variable : str
            ecCodes ``shortName``.
        units : str
            Source units.
        long_name : str
            Descriptive field name.
        step_type : str
            ecCodes ``stepType`` (for example ``"instant"``, ``"accum"``,
            or ``"avg"``).
        reference_time : datetime.datetime
            Model run time (UTC).
        valid_time : datetime.datetime
            Reference time plus the end step (UTC).
        start_time : datetime.datetime
            Start of the interval. For instantaneous fields this is the
            valid time.
        end_time : datetime.datetime or None
            End of the interval, or None for instantaneous fields.
        product_definition_template : int
            GRIB2 product definition template number.
        message_index : int
            The index that was requested.

    Raises
    ------
    ValueError
        If ``message_index`` is negative.
    IndexError
        If the file has no message at ``message_index``.
    """
    if message_index < 0:
        raise ValueError("message_index must be nonnegative")

    with Path(grib_path).open("rb") as stream:
        index = 0

        while True:
            gid = ec.codes_grib_new_from_file(stream)
            if gid is None:
                break

            try:
                if index != message_index:
                    index += 1
                    continue

                # Request decoded forecast steps in seconds, rather than
                # assuming the source step unit is hours.
                ec.codes_set(gid, "stepUnits", "s")

                ref_time = datetime.strptime(
                    f"{int(ec.codes_get(gid, 'dataDate')):08d}"  # type: ignore
                    f"{int(ec.codes_get(gid, 'dataTime')):04d}",  # type: ignore
                    "%Y%m%d%H%M",
                ).replace(tzinfo=UTC)

                step_type = str(ec.codes_get(gid, "stepType"))
                start_seconds = float(ec.codes_get(gid, "startStep", float))  # type: ignore
                end_seconds = float(ec.codes_get(gid, "endStep", float))  # type: ignore

                start = ref_time + timedelta(seconds=start_seconds)
                end = ref_time + timedelta(seconds=end_seconds)

                return {
                    "variable": str(ec.codes_get(gid, "shortName")),
                    "units": str(ec.codes_get(gid, "units")),
                    "long_name": str(ec.codes_get(gid, "name")),
                    "step_type": step_type,
                    "reference_time": ref_time,
                    "valid_time": end,
                    "start_time": start if step_type != "instant" else end,
                    "end_time": end if step_type != "instant" else None,
                    "product_definition_template": int(
                        ec.codes_get(gid, "productDefinitionTemplateNumber")  # type: ignore
                    ),
                    "message_index": message_index,
                }

            finally:
                ec.codes_release(gid)

    raise IndexError(f"GRIB message {message_index} not found in {grib_path}")


def _forecast_interval(
    model: str,
    product: str,
    fxx: int,
) -> timedelta:
    """
    Return the configured output interval for a forecast hour.

    The interval is looked up in ``FORECAST_INTERVAL_POLICIES`` by the
    case-insensitive, whitespace-stripped ``(model, product)`` pair. The first
    rule that matches ``fxx`` is used.

    Parameters
    ----------
    model : str
        Model name (for example ``"hrrr"``).
    product : str
        Product name.
    fxx : int
        Forecast hour.

    Returns
    -------
    datetime.timedelta
        Output interval for the forecast hour.

    Raises
    ------
    ValueError
        If no policy exists for the model/product pair, or no rule matches
        ``fxx``.
    """
    key = (model.strip().lower(), product.strip().lower())

    rules = FORECAST_INTERVAL_POLICIES.get(key)
    if rules is None:
        raise ValueError(f"No interval policy configured for model/product {key!r}")

    for rule in rules:
        if rule.matches(fxx):
            return timedelta(hours=rule.interval_hours)

    raise ValueError(
        f"No period interval configured for " f"{model}/{product} at f{fxx:03d}"
    )


def _resolve_dss_metadata(
    meta: dict[str, Any],
    *,
    model: str,
    product: str,
    fxx: int,
    dss_units: str,
) -> dict[str, Any]:
    """
    Resolve DSS timestamps and unit label without modifying grid values.

    Package conventions:

    - Instantaneous fields use the model/product/fxx interval policy.
    - Interval fields use decoded GRIB bounds.
    - ``dss_units`` is the output unit name, e.g. "mm" rather than "mm/h".

    Original temporal metadata and units are retained for traceability.

    Parameters
    ----------
    meta : dict of str to Any
        Metadata from ``_read_eccodes_metadata``.
    model : str
        Model name, used for instantaneous interval lookup.
    product : str
        Product name, used for instantaneous interval lookup.
    fxx : int
        Forecast hour. ``valid_time`` must equal ``reference_time`` plus
        ``fxx`` hours when both are present.
    dss_units : str
        Unit label to write to DSS.

    Returns
    -------
    dict of str to Any
        A copy of ``meta`` with these keys added or replaced:

        source_units, source_step_type, source_start_time, source_end_time
            The original values, kept for traceability.
        start_time, end_time : datetime.datetime
            The resolved interval bounds.
        valid_time : datetime.datetime
            Set to ``end_time``.
        interval : datetime.timedelta
            ``end_time - start_time``.
        interval_source : str
            ``"model_product_forecast_policy"`` or ``"grib_metadata"``.
        units : str
            Set to ``dss_units``.

    Raises
    ------
    ValueError
        If ``valid_time`` does not match ``fxx``, a required timestamp or
        interval bound is missing, an interval field's ``end_time`` differs
        from ``valid_time``, or the resulting interval is not positive.
    """
    step_type = meta.get("step_type")
    valid_time = meta.get("valid_time")
    reference_time = meta.get("reference_time")

    if reference_time is not None and valid_time is not None:
        expected_valid = reference_time + timedelta(hours=fxx)
        if valid_time != expected_valid:
            raise ValueError(
                f"Selected field valid_time={valid_time} does not "
                f"match f{fxx:03d}; expected {expected_valid}"
            )

    if step_type == "instant":
        # In the existing metadata structure, start_time for an
        # instantaneous field can also contain its valid timestamp.
        end = valid_time
        if end is None:
            end = meta.get("start_time")
        if end is None:
            raise ValueError("Instantaneous field has no valid timestamp")

        interval = _forecast_interval(model, product, fxx)
        start = end - interval
        interval_source = "model_product_forecast_policy"

    else:
        # Never infer a replacement interval for an interval product.
        start = meta.get("start_time")
        end = meta.get("end_time")

        if end is None:
            end = valid_time

        if start is None or end is None:
            raise ValueError(
                f"Field with step_type={step_type!r} is missing "
                "its processing interval"
            )

        if valid_time is not None and end != valid_time:
            raise ValueError(
                f"Interval end_time={end} differs from " f"valid_time={valid_time}"
            )

        interval = end - start
        interval_source = "grib_metadata"

    if interval <= timedelta(0):
        raise ValueError(f"Processing interval must be positive: {start} -> {end}")

    result = dict(meta)
    result.update(
        source_units=meta.get("units"),
        source_step_type=step_type,
        source_start_time=meta.get("start_time"),
        source_end_time=meta.get("end_time"),
        start_time=start,
        end_time=end,
        valid_time=end,
        interval=interval,
        interval_source=interval_source,
        units=dss_units,
    )

    return result
