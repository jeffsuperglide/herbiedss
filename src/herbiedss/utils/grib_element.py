import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from herbiedss.grid.dss.dssprops import DssDataType, TemporalType


@dataclass(frozen=True)
class GribElementDefinition:
    """
    Curated definition of a canonical GRIB element.

    Parameters
    ----------
    parameter_name : str
        Human-readable parameter name (e.g. ``"Temperature"``).
    temporal_type : TemporalType, default TemporalType.UNKNOWN
        How values vary in time (average, accumulation, ...).
    aliases : tuple of str, default ()
        Alternate identifiers (e.g. ECMWF short names such as ``"tp"``)
        that resolve to this element. Matching is case- and
        punctuation-insensitive.
    """

    parameter_name: str
    temporal_type: TemporalType = TemporalType.UNKNOWN
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class GribElementInfo:
    """
    Result of resolving a raw GRIB element string.

    Attributes
    ----------
    raw_element : str
        Input exactly as supplied.
    normalized_element : str
        Upper-cased input with non-alphanumeric characters removed.
    base_element : str
        Canonical element key (duration suffix removed), or the
        normalized input if unresolved, or ``"UNKNOWN"`` if empty.
    parameter_name : str
        Human-readable parameter name.
    duration_value : int or None
        Duration parsed from a suffix (``APCP06`` -> 6), else None.
    duration_unit : str or None
        Unit of the duration; currently always ``"hour"`` when set.
    name_source : str
        Origin of the name: ``"GRIB_ELEMENT_DICT"``,
        ``"GRIB_ELEMENT_RAW"`` (fallback) or ``"UNKNOWN"`` (empty input).
    temporal_type : TemporalType
        Temporal type, from the caller override or the dictionary.
    """

    raw_element: str
    normalized_element: str
    base_element: str
    parameter_name: str
    duration_value: int | None = None
    duration_unit: str | None = None
    name_source: str = "GRIB_ELEMENT_DICT"
    temporal_type: TemporalType = TemporalType.UNKNOWN

    @property
    def dss_data_type(self) -> DssDataType:
        """DssDataType : DSS data type implied by ``temporal_type``."""
        return self.temporal_type.dss_data_type

    @property
    def duration_label(self) -> str | None:
        """str or None : Label such as ``"6 hours"``; None if no duration."""
        if self.duration_value is None:
            return None

        plural = "" if self.duration_value == 1 else "s"
        return f"{self.duration_value} {self.duration_unit}{plural}"

    @property
    def display_name(self) -> str:
        """str : Parameter name, prefixed by the duration label if present."""
        if self.duration_label:
            return f"{self.duration_label} {self.parameter_name}"
        return self.parameter_name

    @property
    def dss_base_name(self) -> str:
        """
        str : Upper-case, hyphen-separated name for use in a DSS pathname.

        Any name containing "precipitation" or "precip" as a whole word
        collapses to ``"Precip"``.
        """
        value = self.parameter_name.upper()

        if re.search(r"\b(?:PRECIPITATION|PRECIP)\b", value):
            return "Precip"

        return re.sub(r"[^A-Z0-9]+", "-", value).strip("-")


class GribElementResolver:
    """
    Resolve GDAL ``GRIB_ELEMENT`` values using a curated dictionary.

    Duration-suffixed variants are supported::

        APCP02 -> APCP, 2 hours
        TMAX06 -> TMAX, 6 hours
        TMIN24 -> TMIN, 24 hours

    A trailing 1-3 digit suffix is treated as a duration only when the
    remaining prefix is a known element. Exact names and aliases always
    take precedence over suffix parsing.

    Parameters
    ----------
    element_names : Mapping of str to GribElementDefinition, optional
        Custom dictionary. Defaults to ``ELEMENT_NAMES``.

    Attributes
    ----------
    element_names : dict of str to GribElementDefinition
        Dictionary keyed by normalized canonical element.

    Raises
    ------
    ValueError
        On first resolve, if an identifier or alias is empty or
        maps to two different definitions.
    """

    ELEMENT_NAMES: Mapping[str, GribElementDefinition] = {
        # Temperature
        "TMP": GribElementDefinition(
            "Temperature",
            TemporalType.AVERAGE,
            aliases=("t",),
        ),
        "TMAX": GribElementDefinition(
            "Maximum Temperature",
            TemporalType.UNKNOWN,
        ),
        "TMIN": GribElementDefinition(
            "Minimum temperature",
            TemporalType.UNKNOWN,
        ),
        "DPT": GribElementDefinition(
            "Dew point temperature",
            TemporalType.UNKNOWN,
        ),
        "DEPR": GribElementDefinition(
            "Dew point depression",
            TemporalType.UNKNOWN,
        ),
        "POT": GribElementDefinition(
            "Potential temperature",
            TemporalType.UNKNOWN,
        ),
        "EPOT": GribElementDefinition(
            "Equivalent potential temperature",
            TemporalType.UNKNOWN,
        ),
        "LAPR": GribElementDefinition(
            "Lapse rate",
            TemporalType.UNKNOWN,
        ),
        # Pressure, height, and vertical motion
        "PRES": GribElementDefinition(
            "Pressure",
            TemporalType.UNKNOWN,
        ),
        "PRMSL": GribElementDefinition(
            "Mean sea level pressure",
            TemporalType.UNKNOWN,
        ),
        "MSLET": GribElementDefinition(
            "Mean sea level pressure",
            TemporalType.UNKNOWN,
        ),
        "GP": GribElementDefinition(
            "Geopotential",
            TemporalType.UNKNOWN,
        ),
        "HGT": GribElementDefinition(
            "Geopotential height",
            TemporalType.UNKNOWN,
        ),
        "DIST": GribElementDefinition(
            "Geometric height",
            TemporalType.UNKNOWN,
        ),
        "VVEL": GribElementDefinition(
            "Vertical velocity",
            TemporalType.UNKNOWN,
        ),
        "DZDT": GribElementDefinition(
            "Vertical velocity",
            TemporalType.UNKNOWN,
        ),
        # Wind
        "UGRD": GribElementDefinition(
            "U-component of wind",
            TemporalType.UNKNOWN,
        ),
        "VGRD": GribElementDefinition(
            "V-component of wind",
            TemporalType.UNKNOWN,
        ),
        "WIND": GribElementDefinition(
            "Wind speed",
            TemporalType.UNKNOWN,
        ),
        "WDIR": GribElementDefinition(
            "Wind direction",
            TemporalType.UNKNOWN,
        ),
        "GUST": GribElementDefinition(
            "Wind gust",
            TemporalType.UNKNOWN,
        ),
        # Moisture and precipitation
        "SPFH": GribElementDefinition(
            "Specific humidity",
            TemporalType.UNKNOWN,
        ),
        "RH": GribElementDefinition(
            "Relative humidity",
            TemporalType.UNKNOWN,
        ),
        "MIXR": GribElementDefinition(
            "Humidity mixing ratio",
            TemporalType.UNKNOWN,
        ),
        "PWAT": GribElementDefinition(
            "Precipitable water",
            TemporalType.UNKNOWN,
        ),
        "PRATE": GribElementDefinition(
            "Precip",
            TemporalType.ACCUMULATION,
        ),
        "APCP": GribElementDefinition(
            "Precip",
            TemporalType.ACCUMULATION,
            aliases=("tp",),
        ),
        "ACPCP": GribElementDefinition(
            "Precip-Convective",
            TemporalType.UNKNOWN,
        ),
        "NCPCP": GribElementDefinition(
            "Precip-Non-convective",
            TemporalType.UNKNOWN,
        ),
        "CPRAT": GribElementDefinition(
            "Precip-Convective rate",
            TemporalType.UNKNOWN,
        ),
        "CPOFP": GribElementDefinition(
            "Precip-Probability of frozen",
            TemporalType.UNKNOWN,
        ),
        # Clouds and hydrometeors
        "TCDC": GribElementDefinition(
            "Total cloud cover",
            TemporalType.UNKNOWN,
        ),
        "LCDC": GribElementDefinition(
            "Low cloud cover",
            TemporalType.UNKNOWN,
        ),
        "MCDC": GribElementDefinition(
            "Medium cloud cover",
            TemporalType.UNKNOWN,
        ),
        "HCDC": GribElementDefinition(
            "High cloud cover",
            TemporalType.UNKNOWN,
        ),
        "CDCON": GribElementDefinition(
            "Convective cloud cover",
            TemporalType.UNKNOWN,
        ),
        "CWAT": GribElementDefinition(
            "Cloud water",
            TemporalType.UNKNOWN,
        ),
        "SNOD": GribElementDefinition(
            "Snow depth",
            TemporalType.UNKNOWN,
        ),
        "WEASD": GribElementDefinition(
            "Water equivalent of accumulated snow depth",
            TemporalType.UNKNOWN,
        ),
        "SNOWC": GribElementDefinition(
            "Snow cover",
            TemporalType.UNKNOWN,
        ),
        "SNOW": GribElementDefinition(
            "Snowfall",
            TemporalType.UNKNOWN,
        ),
        "ICEC": GribElementDefinition(
            "Ice cover",
            TemporalType.UNKNOWN,
        ),
        "ICETK": GribElementDefinition(
            "Ice thickness",
            TemporalType.UNKNOWN,
        ),
        # Radiation and flux
        "DSWRF": GribElementDefinition(
            "Downward shortwave radiation flux",
            TemporalType.UNKNOWN,
        ),
        "USWRF": GribElementDefinition(
            "Upward shortwave radiation flux",
            TemporalType.UNKNOWN,
        ),
        "DLWRF": GribElementDefinition(
            "Downward longwave radiation flux",
            TemporalType.UNKNOWN,
        ),
        "ULWRF": GribElementDefinition(
            "Upward longwave radiation flux",
            TemporalType.UNKNOWN,
        ),
        "NSWRS": GribElementDefinition(
            "Net shortwave radiation flux",
            TemporalType.UNKNOWN,
        ),
        "NLWRS": GribElementDefinition(
            "Net longwave radiation flux",
            TemporalType.UNKNOWN,
        ),
        "GFLUX": GribElementDefinition(
            "Ground heat flux",
            TemporalType.UNKNOWN,
        ),
        "LHTFL": GribElementDefinition(
            "Latent heat flux",
            TemporalType.UNKNOWN,
        ),
        "SHTFL": GribElementDefinition(
            "Sensible heat flux",
            TemporalType.UNKNOWN,
        ),
        # Land surface and hydrology
        "SOILW": GribElementDefinition(
            "Volumetric soil moisture",
            TemporalType.UNKNOWN,
        ),
        "SOILT": GribElementDefinition(
            "Soil temperature",
            TemporalType.UNKNOWN,
        ),
        "LAND": GribElementDefinition(
            "Land-sea mask",
            TemporalType.UNKNOWN,
        ),
        "VEG": GribElementDefinition(
            "Vegetation fraction",
            TemporalType.UNKNOWN,
        ),
        "VEGT": GribElementDefinition(
            "Vegetation type",
            TemporalType.UNKNOWN,
        ),
        "ALBDO": GribElementDefinition(
            "Albedo",
            TemporalType.UNKNOWN,
        ),
        "RUNOF": GribElementDefinition(
            "Runoff",
            TemporalType.UNKNOWN,
        ),
        "BGRUN": GribElementDefinition(
            "Baseflow-groundwater runoff",
            TemporalType.UNKNOWN,
        ),
        "EVP": GribElementDefinition(
            "Evaporation",
            TemporalType.UNKNOWN,
        ),
        "PEVPR": GribElementDefinition(
            "Potential evaporation rate",
            TemporalType.UNKNOWN,
        ),
        # Ocean and wave fields
        "WTMP": GribElementDefinition(
            "Water temperature",
            TemporalType.UNKNOWN,
        ),
        "SST": GribElementDefinition(
            "Sea surface temperature",
            TemporalType.UNKNOWN,
        ),
        "SALTY": GribElementDefinition(
            "Salinity",
            TemporalType.UNKNOWN,
        ),
        "WVDIR": GribElementDefinition(
            "Wave direction",
            TemporalType.UNKNOWN,
        ),
        "WVPER": GribElementDefinition(
            "Wave period",
            TemporalType.UNKNOWN,
        ),
        "WVHGT": GribElementDefinition(
            "Wave height",
            TemporalType.UNKNOWN,
        ),
        "HTSGW": GribElementDefinition(
            "Significant wave height",
            TemporalType.UNKNOWN,
        ),
        # Stability and severe-weather diagnostics
        "CAPE": GribElementDefinition(
            "Convective available potential energy",
            TemporalType.UNKNOWN,
        ),
        "CIN": GribElementDefinition(
            "Convective inhibition",
            TemporalType.UNKNOWN,
        ),
        "LI": GribElementDefinition(
            "Lifted index",
            TemporalType.UNKNOWN,
        ),
        "4LFTX": GribElementDefinition(
            "Best lifted index",
            TemporalType.UNKNOWN,
        ),
        "HLCY": GribElementDefinition(
            "Storm relative helicity",
            TemporalType.UNKNOWN,
        ),
        "VWSH": GribElementDefinition(
            "Vertical wind shear",
            TemporalType.UNKNOWN,
        ),
        "VIS": GribElementDefinition(
            "Visibility",
            TemporalType.UNKNOWN,
        ),
        "REFC": GribElementDefinition(
            "Composite reflectivity",
            TemporalType.UNKNOWN,
        ),
        "REFD": GribElementDefinition(
            "Derived radar reflectivity",
            TemporalType.UNKNOWN,
        ),
        # Categorical parameters
        "CRAIN": GribElementDefinition(
            "Categorical rain",
            TemporalType.UNKNOWN,
        ),
        "CSNOW": GribElementDefinition(
            "Categorical snow",
            TemporalType.UNKNOWN,
        ),
        "CICEP": GribElementDefinition(
            "Categorical ice pellets",
            TemporalType.UNKNOWN,
        ),
        "CFRZR": GribElementDefinition(
            "Categorical freezing rain",
            TemporalType.UNKNOWN,
        ),
        "PTYPE": GribElementDefinition(
            "Precip type",
            TemporalType.UNKNOWN,
        ),
    }

    def __init__(
        self,
        element_names: Mapping[str, GribElementDefinition] | None = None,
    ):
        names = self.ELEMENT_NAMES if element_names is None else element_names

        self.element_names: dict[str, GribElementDefinition] = {
            self._normalize(element): definition
            for element, definition in names.items()
        }

    @cached_property
    def _element_lookup(self) -> dict[str, tuple[str, GribElementDefinition]]:
        """
        dict : Map of normalized identifier -> (canonical, definition).

        Built once per instance; includes canonical names and aliases.
        """
        lookup: dict[str, tuple[str, GribElementDefinition]] = {}

        for element, definition in self.element_names.items():
            canonical = self._normalize(element)

            if not canonical:
                raise ValueError("Canonical GRIB element cannot be empty")

            for identifier in (element, *definition.aliases):
                normalized = self._normalize(identifier)

                if not normalized:
                    raise ValueError(f"Empty alias for GRIB element {element!r}")

                existing = lookup.get(normalized)

                if existing is not None:
                    existing_canonical, existing_definition = existing

                    if (
                        existing_canonical != canonical
                        or existing_definition != definition
                    ):
                        raise ValueError(
                            f"Conflicting GRIB identifier {identifier!r}: "
                            f"{existing_canonical!r} and {canonical!r}"
                        )

                lookup[normalized] = (canonical, definition)

        return lookup

    def resolve(
        self,
        grib_element: str,
        *,
        temporal_type: TemporalType | None = None,
    ) -> GribElementInfo:
        """
        Resolve a raw GRIB element string.

        Parameters
        ----------
        grib_element : str
            Raw element, e.g. ``"APCP06"`` or ``"tp"``.
        temporal_type : TemporalType, optional
            Overrides the dictionary's temporal type (e.g. one derived
            from the ECMWF ``stepType``).

        Returns
        -------
        GribElementInfo
            Resolved info. Unknown elements get ``name_source="GRIB_ELEMENT_RAW"``;
            empty input gets ``base_element="UNKNOWN"``.
        """
        normalized = self._normalize(grib_element)

        if not normalized:
            return GribElementInfo(
                raw_element=str(grib_element),
                normalized_element="",
                base_element="UNKNOWN",
                parameter_name="Unknown parameter",
                name_source="UNKNOWN",
                temporal_type=(
                    TemporalType.UNKNOWN if temporal_type is None else temporal_type
                ),
            )

        duration_value = None
        match = self._element_lookup.get(normalized)

        if match is None:
            candidate, candidate_duration = self._split_duration_suffix(normalized)

            if candidate is not None:
                match = self._element_lookup.get(self._normalize(candidate))

                if match is not None:
                    duration_value = candidate_duration

        if match is None:
            return GribElementInfo(
                raw_element=str(grib_element),
                normalized_element=normalized,
                base_element=normalized,
                parameter_name=self._fallback_name(normalized),
                name_source="GRIB_ELEMENT_RAW",
                temporal_type=(
                    TemporalType.UNKNOWN if temporal_type is None else temporal_type
                ),
            )

        canonical, definition = match

        return GribElementInfo(
            raw_element=str(grib_element),
            normalized_element=normalized,
            base_element=canonical,
            parameter_name=definition.parameter_name,
            duration_value=duration_value,
            duration_unit=("hour" if duration_value is not None else None),
            temporal_type=(
                definition.temporal_type if temporal_type is None else temporal_type
            ),
        )

    def _split_duration_suffix(
        self,
        normalized_element: str,
    ) -> tuple[str | None, int | None]:
        """
        Split a trailing positive 1-3 digit duration from an element.

        Parameters
        ----------
        normalized_element : str
            Normalized element, e.g. ``"APCP06"``.

        Returns
        -------
        base : str or None
            Prefix found in ``element_names``, else None.
        duration : int or None
            Parsed duration, else None.
        """
        for width in range(min(3, len(normalized_element) - 1), 0, -1):
            suffix = normalized_element[-width:]
            base = normalized_element[:-width]

            if suffix.isdigit() and int(suffix) > 0 and base in self.element_names:
                return base, int(suffix)

        return None, None

    @staticmethod
    def _normalize(value: str) -> str:
        """
        Upper-case and strip all non-alphanumeric characters.

        Parameters
        ----------
        value : str
            Raw identifier.

        Returns
        -------
        str
            Normalized identifier, or ``""`` if falsy.
        """
        if not value:
            return ""

        return re.sub(r"[^A-Z0-9]+", "", str(value).upper())

    @staticmethod
    def _fallback_name(element: str) -> str:
        """
        str : Fallback display name for an unknown element.

        Note that ``_normalize`` strips underscores, so the replace is
        effectively a no-op for normalized input.
        """
        return element.replace("_", " ").strip() or "Unknown parameter"
