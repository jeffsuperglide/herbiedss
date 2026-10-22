"""
Meteorological unit handling with Pint, specialized for NOAA GRIB / Herbie data.

Provides a :class:`Units` helper that wraps a :class:`pint.UnitRegistry` with a
hydrology context (kg/m² ↔ mm of water), common meteorological aliases, and
mappings to preferred / DSS unit labels.

Examples
--------
>>> from utils.units import Units
>>> u = Units()
>>> q = u.convert(5.0, "kg/m^2", "mm")
>>> print(q.magnitude, q.units)
5.0 mm
"""

from __future__ import annotations

import logging

import pint
from pint import Quantity, UnitRegistry

logger = logging.getLogger(__name__)


class Units:
    """
    Central unit converter for meteorological GRIB data.

    Parameters
    ----------
    water_density : float, default 1000.0
        Water density in kg/m³ used for the classic kg/m² <-> mm conversion.
    preferred : dict of str to str, optional
        Mapping of source unit to preferred meteorological unit. Entries are
        merged over the built-in defaults (precipitation, pressure, and
        temperature), and user entries take precedence.
    extra_definitions : list of str, optional
        Additional Pint unit definitions to register (e.g. custom aliases).
    dss_unit_overrides : dict of str to str, optional
        Mapping of (preferred) unit to the label used for HEC-DSS output.
        Merged over the defaults, and user entries take precedence.

    Attributes
    ----------
    ureg : pint.UnitRegistry
        Registry holding the units, aliases, and the ``hydro`` context.
    water_density : pint.Quantity
        Water density as a quantity in kg/m³.
    preferred : dict of pint.Unit to str
        Normalized source-unit to preferred-unit mapping.
    dss_unit_overrides : dict of pint.Unit to str
        Normalized preferred-unit to DSS label mapping.

    Notes
    -----
    The ``hydro`` context treats 1 kg/m² of water as 1 mm of depth (and the
    equivalent for rates, e.g. kg/m²/s <-> m/s) using ``water_density``. It is
    activated automatically in :meth:`convert` and :meth:`convert_unit`.
    """

    def __init__(
        self,
        water_density: float = 1000.0,
        preferred: dict[str, str] | None = None,
        extra_definitions: list[str] | None = None,
        dss_unit_overrides: dict[str, str] | None = None,
    ):
        self.ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        self.ureg.formatter.default_format = "~D"

        # ----- water density & hydro context -----
        self.water_density = water_density * self.ureg.kg / self.ureg.m**3

        hydro = pint.Context("hydro")
        hydro.add_transformation(
            "[mass] / [length]**2",
            "[length]",
            lambda ureg, x: x / self.water_density,  # type: ignore
        )
        hydro.add_transformation(
            "[length]",
            "[mass] / [length]**2",
            lambda ureg, x: x * self.water_density,  # type: ignore
        )
        hydro.add_transformation(
            "[mass] / [length]**2 / [time]",
            "[length] / [time]",
            lambda ureg, x: x / self.water_density,  # type: ignore
        )
        hydro.add_transformation(
            "[length] / [time]",
            "[mass] / [length]**2 / [time]",
            lambda ureg, x: x * self.water_density,  # type: ignore
        )
        self.ureg.add_context(hydro)
        self._context = "hydro"

        # ----- common meteorological aliases -----
        self.ureg.define("gpm = meter")  # geopotential meter
        self.ureg.define("percent = 0.01 * dimensionless = %")
        self.ureg.define("degC = celsius")
        self.ureg.define("degF = fahrenheit")
        self.ureg.define("dBZ = dimensionless")  # radar reflectivity

        if extra_definitions:
            for defn in extra_definitions:
                self.ureg.define(defn)

        # ----- preferred unit mapping -----
        defined_preferred = {
            "kg/m^2": "mm",
            "kg/m**2": "mm",
            "kg/m**2/s": "mm/h",
            "kg/m^2/s": "mm/h",
            "kg/(m^2 s)": "mm/h",
            "kg m**-2 s**-1": "mm/h",
            "Pa": "hPa",
            "K": "degC",  # uncomment if you always want °C
        }

        self.preferred: dict[pint.Unit, str] = {}

        for mapping in (defined_preferred, preferred or {}):
            for src, dst in mapping.items():
                self.preferred[self._parse(src)] = dst

        default_dss_overrides = {
            "mm/h": "mm",
        }

        # Normalize keys so equivalent spellings match.
        # Process user overrides last so they win.
        self.dss_unit_overrides: dict[pint.Unit, str] = {}

        for mapping in (default_dss_overrides, dss_unit_overrides or {}):
            for src, label in mapping.items():
                self.dss_unit_overrides[self._parse(src)] = label

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _parse(self, unit: str) -> pint.Unit:
        """
        Parse a unit string, tolerating common GRIB/CF variants.

        Parameters
        ----------
        unit : str
            Unit string, e.g. ``"kg m**-2"``, ``"m s-1"``, or ``"W/m**2"``.

        Returns
        -------
        pint.Unit
            Unit parsed by the instance's registry.

        Raises
        ------
        pint.errors.UndefinedUnitError
            If the normalized string contains an unknown unit.
        ValueError
            If the string cannot be parsed.
        """
        unit = unit.strip()
        replacements = {
            "kg/m**2": "kg/m^2",
            "kg m**-2": "kg/m^2",
            "kg/m2": "kg/m^2",
            "kg.m-2": "kg/m^2",
            "kg m**-2 s**-1": "kg/m^2/s",
            "kg/m**2/s": "kg/m^2/s",
            "m s-1": "m/s",
            "m/s**2": "m/s^2",
            "W/m**2": "W/m^2",
            "W m-2": "W/m^2",
            "J/m**2": "J/m^2",
        }
        for old, new in replacements.items():
            unit = unit.replace(old, new)
        return self.ureg.parse_units(unit)

    def convert(
        self,
        value: float | Quantity,
        from_unit: str | None = None,
        to_unit: str = "mm",
    ) -> Quantity:
        """
        Convert a value (or existing Quantity) to a target unit.

        Parameters
        ----------
        value : float or pint.Quantity
            Numeric value (scalar or array) or an existing quantity.
        from_unit : str, optional
            Unit of ``value``. Required when ``value`` is not a Quantity.
        to_unit : str, default "mm"
            Target unit string.

        Returns
        -------
        pint.Quantity
            Converted quantity; use ``.magnitude`` and ``.units``.

        Raises
        ------
        ValueError
            If ``value`` is not a Quantity and ``from_unit`` is None.
        pint.errors.DimensionalityError
            If the units are incompatible, even with the hydro context.

        Examples
        --------
        >>> Units().convert(5.0, "kg/m^2", "mm").magnitude
        5.0
        """
        if isinstance(value, Quantity):
            q = value
        else:
            if from_unit is None:
                raise ValueError("from_unit required when value is not a Quantity")
            q = value * self._parse(from_unit)

        with self.ureg.context(self._context):
            return q.to(self._parse(to_unit))  # type: ignore

    def convert_unit(self, from_unit: str, to_unit: str) -> str:
        """
        Validate a unit conversion and return the clean target unit string.

        Useful when only the unit name is needed, not a numeric conversion.

        Parameters
        ----------
        from_unit : str
            Source unit string.
        to_unit : str
            Target unit string.

        Returns
        -------
        str
            Target unit formatted in short form (``~D``), e.g. ``"mm"``.

        Raises
        ------
        pint.errors.DimensionalityError
            If the two units are incompatible.
        """
        src = self._parse(from_unit)
        dst = self._parse(to_unit)
        with self.ureg.context(self._context):
            self.ureg.Quantity(1.0 * src).to(dst)  # raises if incompatible
        return f"{dst:~D}"

    def to_preferred(self, unit: str) -> str:
        """
        Map a unit string to its preferred meteorological unit.

        Parameters
        ----------
        unit : str
            Source unit string, e.g. ``"kg m**-2 s**-1"`` or ``"Pa"``.

        Returns
        -------
        str
            Preferred unit string (e.g. ``"mm/h"``). If ``unit`` cannot be
            parsed, it is returned unchanged. If no preference is defined, the
            normalized form of ``unit`` is returned.

        Raises
        ------
        pint.errors.DimensionalityError
            If a configured preferred unit is incompatible with ``unit``.
            Invalid preferred conversions are intentionally not hidden.
        """
        try:
            src = self._parse(unit)
        except (pint.errors.PintError, ValueError):
            logger.debug(
                "Could not parse unit %r; returning as-is",
                unit,
                exc_info=True,
            )
            return unit

        dst = self.preferred.get(src)

        if dst is None:
            return f"{src:~D}"

        # Let invalid preferred conversions raise instead of hiding them.
        return self.convert_unit(unit, dst)

    def to_dss_unit(self, unit: str, *, apply_override: bool = True) -> str:
        """
        Return the unit label to use when writing to HEC-DSS.

        Parameters
        ----------
        unit : str
            Source unit string.
        apply_override : bool, default True
            If True, apply ``dss_unit_overrides`` (e.g. ``"mm/h"`` -> ``"mm"``)
            to the preferred unit. If False, return the preferred unit as is.

        Returns
        -------
        str
            DSS unit label, or the preferred unit if no override applies or
            the preferred unit cannot be parsed.
        """
        preferred = self.to_preferred(unit)

        if not apply_override:
            return preferred

        try:
            parsed = self._parse(preferred)
        except (pint.errors.PintError, ValueError):
            return preferred

        return self.dss_unit_overrides.get(parsed, preferred)

    # ------------------------------------------------------------------
    # xarray / Herbie convenience
    # ------------------------------------------------------------------

    def convert_dataarray(
        self,
        da,  # xarray.DataArray
        to_unit: str,
        unit_attr: str = "units",
    ):
        """
        Convert an xarray.DataArray that carries a units attribute.

        Parameters
        ----------
        da : xarray.DataArray
            Array whose ``attrs[unit_attr]`` holds the source unit string.
        to_unit : str
            Target unit string.
        unit_attr : str, default "units"
            Name of the attribute holding the unit.

        Returns
        -------
        xarray.DataArray
            Copy of ``da`` with converted values and the unit attribute
            updated to the target unit.

        Raises
        ------
        ValueError
            If ``da`` has no ``unit_attr`` attribute.
        pint.errors.DimensionalityError
            If the units are incompatible.
        """
        if unit_attr not in da.attrs:
            raise ValueError(f"DataArray has no '{unit_attr}' attribute")

        from_unit = da.attrs[unit_attr]
        q = self.convert(da.values, from_unit, to_unit)

        out = da.copy(data=q.magnitude)
        out.attrs[unit_attr] = f"{q.units:~D}"
        return out


# Optional module-level singleton for convenience
default_units = Units()
