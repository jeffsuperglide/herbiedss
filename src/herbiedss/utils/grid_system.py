from dataclasses import dataclass

from osgeo import osr

from herbiedss.grid.dss.dssprops import (
    DssGridType,
    GridSystem,
    SpatialReferenceDefinition,
)


@dataclass(frozen=True)
class ResolvedGridSystem:
    """Fully resolved target spatial reference for a DSS output grid."""

    user_value: str | int
    epsg: int | None
    dss_name: str
    display_name: str
    wkt: str
    is_internal: bool


def _srs_from_epsg(epsg: int) -> osr.SpatialReference:
    """Create an OSR spatial reference from an EPSG code or raise clearly."""
    if epsg <= 0:
        raise ValueError(f"EPSG code must be positive; received {epsg}.")

    srs = osr.SpatialReference()
    status = srs.ImportFromEPSG(int(epsg))

    if status != 0:
        raise ValueError(
            f"Could not resolve EPSG:{epsg}. Verify that it exists in the "
            "PROJ/EPSG database bundled with your GDAL installation."
        )

    # Avoid authority axis-order surprises if you later transform points
    # manually. gdal.Warp generally handles normal raster CRS use correctly,
    # but setting traditional GIS order keeps your behavior explicit.
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return srs


def _safe_dss_srs_name(epsg: int) -> str:
    """
    A stable machine-oriented identifier.

    This is deliberately not the CRS display name, because display names can
    contain spaces, punctuation, parentheses, slashes, and wording that may
    change with EPSG/PROJ database updates.
    """
    return f"EPSG_{epsg}"


def _resolve_grid_system(
    grid_system: GridSystem | str | int,
) -> ResolvedGridSystem:
    """
    Resolve a named built-in grid system such as SHG/HRAP, or an EPSG code.

    GridSystem is assumed to be your existing enum-like type. Adapt the
    `SpatialReferenceDefinition` and `DssGridType` accesses to match your
    current implementations.
    """
    # Explicit EPSG integer input.
    if isinstance(grid_system, int) and not isinstance(grid_system, bool):
        srs = _srs_from_epsg(grid_system)
        display_name = srs.GetName() or f"EPSG:{grid_system}"

        return ResolvedGridSystem(
            user_value=grid_system,
            epsg=grid_system,
            dss_name=_safe_dss_srs_name(grid_system),
            display_name=display_name,
            wkt=srs.ExportToWkt(),
            is_internal=False,
        )

    # Optional support for CLI text such as "epsg:5070" or "5070".
    text = str(grid_system).strip()
    upper = text.upper()

    if upper.startswith("EPSG:"):
        epsg_text = text.split(":", 1)[1].strip()

        if not epsg_text.isdigit():
            raise ValueError(
                f"Invalid EPSG value {grid_system!r}. Use an integer such as "
                "5070 or a string such as 'EPSG:5070'."
            )

        return _resolve_grid_system(int(epsg_text))

    if text.isdigit():
        return _resolve_grid_system(int(text))

    # Existing named grid-system behavior.
    name = upper
    try:
        grid_type = DssGridType.from_grid_system(name)
        srs_definition = SpatialReferenceDefinition.from_grid_system(name)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Unknown grid system {grid_system!r}. Use one of your supported "
            "named systems or an EPSG code such as 5070."
        ) from exc

    wkt = srs_definition.value
    if not wkt:
        raise ValueError(
            f"Grid system {name!r} does not provide a spatial-reference WKT."
        )

    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # `AutoIdentifyEPSG()` may find a code for standard WKT, but it can return
    # no authority for custom/internal WKT. Treat it only as optional metadata.
    srs.AutoIdentifyEPSG()
    epsg_text = srs.GetAuthorityCode(None)
    epsg = int(epsg_text) if epsg_text and epsg_text.isdigit() else None

    return ResolvedGridSystem(
        user_value=text,
        epsg=epsg,
        dss_name=grid_type.name,
        display_name=srs.GetName() or name,
        wkt=srs.ExportToWkt(),
        is_internal=True,
    )
