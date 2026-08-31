"""Module 'dssutil' for handling DSS grids

Enumerations, constants and functions

"""

from enum import Enum
from functools import lru_cache
from typing import Literal

GridSystem = Literal["shg", "hrap"]

FLOAT_MAX = 3.40282347e38
FLOAT_UNDEFINED = -FLOAT_MAX

DSS_UNDEFINED = 3.4028234663852886e38
DSS_UNDEFINED_VALUE = -DSS_UNDEFINED

"""str: HRAP WKT"""
HRAP_SRC_DEFINITION: str = 'PROJCS["Stereographic_CONUS_HRAP",\
GEOGCS["GCS_Sphere_LFM",DATUM["D_Sphere_LFM",\
SPHEROID["Shpere_LFM",6371200.0,0.0]],PRIMEM["Greenwich",0.0],\
UNIT["Degree",0.0174532925199433]],\
PROJECTION["Stereographic_North_Pole"],\
PARAMETER["False_Easting",1909762.5],PARAMETER["False_Northing",7624762.5],\
PARAMETER["Central_Meridian",-105.0],PARAMETER["Standard_Parallel_1",60.0],\
UNIT["Meter",1.0]]'

"""str: SHG WKT"""
SHG_SRC_DEFINITION: str = (
    'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",\
GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",\
SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],\
UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],\
PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],\
PARAMETER["Central_Meridian",-96.0],PARAMETER["Standard_Parallel_1",29.5],\
PARAMETER["Standard_Parallel_2",45.5],PARAMETER["Latitude_Of_Origin",23.0],\
UNIT["Meter",1.0]]'
)

"""str: UTM WKT"""
# Parameters: utmZone, utmHemisphere, central_meridian, false_northing
# if (_utmHemisphere.equals("S")) falseNorthing = 10000000;
#  centralMeridian = -183 + _utmZone * 6;
UTM_SRC_DEFINITION: str = 'PROJCS["UTM_ZONE_%s%s_WGS84",\
GEOGCS["WGS_84",DATUM["WGS_1984",SPHEROID["WGS84",6378137,298.257223563]],\
PRIMEM["Greenwich",0],UNIT["degree",0.01745329251994328]],\
UNIT["Meter",1.0],PROJECTION["Transverse_Mercator"],\
PARAMETER["latitude_of_origin",0],\
PARAMETER["central_meridian",%s],\
PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],\
PARAMETER["false_northing",%s],\
AXIS["Easting",EAST],AXIS["Northing",NORTH]]'


# ProjectionDatum
class ProjectionDatum(Enum):
    UNDEFINED_PROJECTION_DATUM = 0
    NAD_27 = 1
    NAD_83 = 2


# CompressionMethod
class CompressionMethod(Enum):
    UNDEFINED_COMPRESSION_METHOD = 0
    NO_COMPRESSION = 1
    ZLIB_COMPRESSION = 26


compression_method = {i.name: i.value for i in CompressionMethod}


# StorageDataType
class StorageDataType(Enum):
    GRID_FLOAT = 0
    GRID_INT = 1
    GRID_DOUBLE = 2
    GRID_LONG = 3


# DataType
class DssDataType(Enum):
    """
    HEC-DSS data type: pairs the DSS string code
    with an application-defined numeric code.
    """

    PER_AVER = "PER-AVER"
    PER_CUM = "PER-CUM"
    INST_VAL = "INST-VAL"
    INST_CUM = "INST-CUM"
    FREQ = "FREQ"
    INVALID = "INVALID"

    @property
    def code(self) -> int:
        return {
            DssDataType.PER_AVER: 0,
            DssDataType.PER_CUM: 1,
            DssDataType.INST_VAL: 2,
            DssDataType.INST_CUM: 3,
            DssDataType.FREQ: 4,
            DssDataType.INVALID: 5,
        }[self]


# GridStructVersion
class GridStructVersion(Enum):
    VERSION_100 = -100


# DssGridType
class DssGridType(Enum):
    UNDEFINED_GRID_TYPE = 400
    HRAP = 410
    SHG = ALBERS = 420
    UTM6N = SPECIFIED_GRID_TYPE = 430

    @classmethod
    @lru_cache(maxsize=1)
    def _name_map(cls):
        return {
            "UNDEFINED_GRID_TYPE": cls.UNDEFINED_GRID_TYPE,
            "HRAP": cls.HRAP,
            "SHG": cls.SHG,
            "ALBERS": cls.ALBERS,
            "UTM6N": cls.UTM6N,
            "SPECIFIED_GRID_TYPE": cls.SPECIFIED_GRID_TYPE,
        }

    @classmethod
    def from_grid_system(cls, grid_system: str):
        return cls._name_map()[grid_system]


# DssGridTypeName
class DssGridTypeName(Enum):
    HRAP = "HRAP"
    SHG = "ALBERS"
    UTM = "UTM%s%s"


# SpatialRefereceDefinition
class SpatialReferenceDefinition(Enum):
    UNDEFINED_GRID_TYPE = SPECIFIED_GRID_TYPE = ""
    HRAP = HRAP_SRC_DEFINITION
    SHG = ALBERS = SHG_SRC_DEFINITION
    UTM6N = UTM_SRC_DEFINITION % ("6", "N", "-147", "0")

    @classmethod
    @lru_cache(maxsize=1)
    def _name_map(cls) -> dict[str, "SpatialReferenceDefinition"]:
        return {
            "UNDEFINED_GRID_TYPE": cls.UNDEFINED_GRID_TYPE,
            "SPECIFIED_GRID_TYPE": cls.SPECIFIED_GRID_TYPE,
            "HRAP": cls.HRAP,
            "SHG": cls.SHG,
            "ALBERS": cls.ALBERS,
            "UTM6N": cls.UTM6N,
        }

    @classmethod
    def from_grid_system(cls, grid_system: str):
        return cls._name_map()[grid_system]


# Grid cellsize
GridCellsize = Literal[2000]


# TimeZones
class TimeZone(Enum):
    GMT = UTC = 0
    AST = 4
    EST = 5
    CST = 6
    MST = 7
    PST = 8
    AKST = 9
    HST = 10
