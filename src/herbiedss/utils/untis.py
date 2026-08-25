"""
Meteorological unit handling with Pint, specialized for NOAA GRIB / Herbie data.

Usage
-----
from utils.units import Units

u = Units()                          # default hydro context + common aliases
q = u.convert(5.0, "kg/m^2", "mm")   # → Quantity(5.0, 'mm')
print(q.magnitude, q.units)          # 5.0   mm
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
    water_density : float
        Density used for the classic kg/m² ↔ mm conversion (default 1000 kg/m³).
    preferred : dict, optional
        Mapping of source unit → preferred meteorological unit.
        Defaults cover the most common precipitation / pressure cases.
    extra_definitions : list[str], optional
        Additional unit definitions to register (e.g. custom aliases).
    """

    def __init__(
        self,
        water_density: float = 1000.0,
        preferred: dict[str, str] | None = None,
        extra_definitions: list[str] | None = None,
    ):
        self.ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        self.ureg.formatter.default_format = "~P"

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
        self.preferred = preferred or {
            "kg/m^2": "mm",
            "kg/m²": "mm",
            "kg m-2": "mm",
            "kg/m^2/s": "mm/h",
            "kg m-2 s-1": "mm/h",
            "kg/(m^2 s)": "mm/h",
            "Pa": "hPa",
            "K": "degC",  # uncomment if you always want °C
        }

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _parse(self, unit: str) -> pint.Unit:
        """Parse a unit string, tolerating common GRIB/CF variants."""
        unit = unit.strip()
        replacements = {
            "kg/m**2": "kg/m^2",
            "kg m-2": "kg/m^2",
            "kg/m2": "kg/m^2",
            "kg.m-2": "kg/m^2",
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

        Returns a pure pint.Quantity (use .magnitude and .units).
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
        Convert one unit *string* to another and return the clean target string.
        Useful when you only need the unit name, not a numeric conversion.
        """
        src = self._parse(from_unit)
        dst = self._parse(to_unit)
        with self.ureg.context(self._context):
            self.ureg.Quantity(1.0 * src).to(dst)  # raises if incompatible
        return f"{dst:~P}"

    def to_preferred(self, unit: str) -> str:
        """
        Map a GRIB/CF unit onto the preferred meteorological form.
        Returns the preferred unit string (or the original if no mapping exists).
        """
        try:
            canonical = f"{self._parse(unit):~P}"
        except Exception:
            logger.debug(
                "Could not parse unit %r; returning as-is", unit, exc_info=True
            )
            return unit

        for src, dst in self.preferred.items():
            try:
                if self._parse(src) == self._parse(canonical):
                    return self.convert_unit(canonical, dst)
            except Exception:
                logger.debug(
                    "Preferred-unit match failed for src=%r canonical=%r → %r",
                    src,
                    canonical,
                    dst,
                    exc_info=True,
                )
                continue
        return canonical

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
        Returns a new DataArray with converted values and updated units.
        """
        if unit_attr not in da.attrs:
            raise ValueError(f"DataArray has no '{unit_attr}' attribute")

        from_unit = da.attrs[unit_attr]
        q = self.convert(da.values, from_unit, to_unit)

        out = da.copy(data=q.magnitude)
        out.attrs[unit_attr] = f"{q.units:~P}"
        return out


# Optional module-level singleton for convenience
default_units = Units()
