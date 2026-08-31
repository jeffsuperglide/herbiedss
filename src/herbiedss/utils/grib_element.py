import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GribElementInfo:
    raw_element: str
    normalized_element: str
    base_element: str
    parameter_name: str
    duration_value: int | None = None
    duration_unit: str | None = None
    name_source: str = "GRIB_ELEMENT_DICT"

    @property
    def duration_label(self) -> str | None:
        if self.duration_value is None:
            return None

        plural = "" if self.duration_value == 1 else "s"
        return f"{self.duration_value} {self.duration_unit}{plural}"

    @property
    def display_name(self) -> str:
        if self.duration_label:
            return f"{self.duration_label} {self.parameter_name}"
        return self.parameter_name

    @property
    def dss_base_name(self) -> str:
        value = self.parameter_name.upper()

        if re.search(r"\bPRECIPITATION\b", value):
            return "Precipitation"

        if re.search(r"\bPRECIP\b", value):
            return "Precip"

        return re.sub(r"[^A-Z0-9]+", "-", self.parameter_name).strip("-")


class GribElementResolver:
    """
    Resolve GDAL GRIB_ELEMENT values using only a curated element dictionary.

    Supports duration-suffixed variants such as:
        APCP02 -> APCP, 2 hours
        APCP06 -> APCP, 6 hours
        APCP24 -> APCP, 24 hours
        TMAX06 -> TMAX, 6 hours
        TMIN24 -> TMIN, 24 hours

    The suffix is treated as a duration qualifier only when the remaining
    prefix is present in the element dictionary.
    """

    ELEMENT_NAMES: Mapping[str, str] = {
        # Temperature and thermodynamics
        "TMP": "Temperature",
        "TMAX": "Maximum temperature",
        "TMIN": "Minimum temperature",
        "DPT": "Dew point temperature",
        "DEPR": "Dew point depression",
        "POT": "Potential temperature",
        "EPOT": "Equivalent potential temperature",
        "LAPR": "Lapse rate",
        # Pressure, height, and vertical motion
        "PRES": "Pressure",
        "PRMSL": "Mean sea level pressure",
        "MSLET": "Mean sea level pressure",
        "GP": "Geopotential",
        "HGT": "Geopotential height",
        "DIST": "Geometric height",
        "VVEL": "Vertical velocity",
        "DZDT": "Vertical velocity",
        # Wind
        "UGRD": "U-component of wind",
        "VGRD": "V-component of wind",
        "WIND": "Wind speed",
        "WDIR": "Wind direction",
        "GUST": "Wind gust",
        # Moisture and precipitation
        "SPFH": "Specific humidity",
        "RH": "Relative humidity",
        "MIXR": "Humidity mixing ratio",
        "PWAT": "Precipitable water",
        "PRATE": "Precipitation rate",
        "APCP": "Total precipitation",
        "ACPCP": "Convective precipitation",
        "NCPCP": "Non-convective precipitation",
        "CPRAT": "Convective precipitation rate",
        "CPOFP": "Probability of frozen precipitation",
        # Clouds and hydrometeors
        "TCDC": "Total cloud cover",
        "LCDC": "Low cloud cover",
        "MCDC": "Medium cloud cover",
        "HCDC": "High cloud cover",
        "CDCON": "Convective cloud cover",
        "CWAT": "Cloud water",
        "SNOD": "Snow depth",
        "WEASD": "Water equivalent of accumulated snow depth",
        "SNOWC": "Snow cover",
        "SNOW": "Snowfall",
        "ICEC": "Ice cover",
        "ICETK": "Ice thickness",
        # Radiation and flux
        "DSWRF": "Downward shortwave radiation flux",
        "USWRF": "Upward shortwave radiation flux",
        "DLWRF": "Downward longwave radiation flux",
        "ULWRF": "Upward longwave radiation flux",
        "NSWRS": "Net shortwave radiation flux",
        "NLWRS": "Net longwave radiation flux",
        "GFLUX": "Ground heat flux",
        "LHTFL": "Latent heat flux",
        "SHTFL": "Sensible heat flux",
        # Land surface and hydrology
        "SOILW": "Volumetric soil moisture",
        "SOILT": "Soil temperature",
        "LAND": "Land-sea mask",
        "VEG": "Vegetation fraction",
        "VEGT": "Vegetation type",
        "ALBDO": "Albedo",
        "RUNOF": "Runoff",
        "BGRUN": "Baseflow-groundwater runoff",
        "EVP": "Evaporation",
        "PEVPR": "Potential evaporation rate",
        # Ocean and wave fields
        "WTMP": "Water temperature",
        "SST": "Sea surface temperature",
        "SALTY": "Salinity",
        "WVDIR": "Wave direction",
        "WVPER": "Wave period",
        "WVHGT": "Wave height",
        "HTSGW": "Significant wave height",
        # Stability and severe-weather diagnostics
        "CAPE": "Convective available potential energy",
        "CIN": "Convective inhibition",
        "LI": "Lifted index",
        "4LFTX": "Best lifted index",
        "HLCY": "Storm relative helicity",
        "VWSH": "Vertical wind shear",
        "VIS": "Visibility",
        "REFC": "Composite reflectivity",
        "REFD": "Derived radar reflectivity",
        # Categorical parameters
        "CRAIN": "Categorical rain",
        "CSNOW": "Categorical snow",
        "CICEP": "Categorical ice pellets",
        "CFRZR": "Categorical freezing rain",
        "PTYPE": "Precipitation type",
    }

    def __init__(self, element_names: Mapping[str, str] | None = None):
        names = element_names or self.ELEMENT_NAMES
        self.element_names = {
            self._normalize(value): name for value, name in names.items()
        }

    def resolve(self, grib_element: str) -> GribElementInfo:
        normalized = self._normalize(grib_element)

        if not normalized:
            return GribElementInfo(
                raw_element="",
                normalized_element="",
                base_element="UNKNOWN",
                parameter_name="Unknown parameter",
                name_source="UNKNOWN",
            )

        # Exact dictionary match always wins. This prevents accidentally
        # treating valid codes containing digits, such as 4LFTX, as duration codes.
        if normalized in self.element_names:
            return GribElementInfo(
                raw_element=str(grib_element),
                normalized_element=normalized,
                base_element=normalized,
                parameter_name=self.element_names[normalized],
            )

        base_element, duration_value = self._split_duration_suffix(normalized)

        if base_element and base_element in self.element_names:
            return GribElementInfo(
                raw_element=str(grib_element),
                normalized_element=normalized,
                base_element=base_element,
                parameter_name=self.element_names[base_element],
                duration_value=duration_value,
                duration_unit="hour",
            )

        return GribElementInfo(
            raw_element=str(grib_element),
            normalized_element=normalized,
            base_element=normalized,
            parameter_name=self._fallback_name(normalized),
            name_source="GRIB_ELEMENT_RAW",
        )

    def _split_duration_suffix(
        self,
        normalized_element: str,
    ) -> tuple[str | None, int | None]:
        """
        Interpret a terminal 1- to 3-digit suffix as hours only if its prefix
        resolves to a known GRIB element.

        APCP02 -> ("APCP", 2)
        APCP006 -> ("APCP", 6)
        TMAX24 -> ("TMAX", 24)
        APCP  -> (None, None)
        4LFTX -> (None, None), because it matches exactly before this method.
        """
        match = re.fullmatch(r"([A-Z][A-Z0-9]*?)(\d{1,3})", normalized_element)

        if not match:
            return None, None

        base_element = match.group(1)
        duration_value = int(match.group(2))

        if duration_value <= 0:
            return None, None

        return base_element, duration_value

    @staticmethod
    def _normalize(value: str) -> str:
        if not value:
            return ""

        return re.sub(r"[^A-Z0-9]+", "", str(value).upper())

    @staticmethod
    def _fallback_name(element: str) -> str:
        return element.replace("_", " ").strip() or "Unknown parameter"
