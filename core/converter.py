"""
core/converter.py
Extended conversion module — refactored from json_to_shp.py.

Key improvements over the original script:
  - Configurable HTTP headers (per-call, not hardcoded)
  - Typed ConvertResult dataclass with rich metadata for UI display
  - LatLonDetectionError carries available_columns list for UI dropdowns
  - _check_column_collisions() resolves 10-char truncation collisions
    deterministically (no silent data loss) and returns a warnings list
  - resolve_redirect() returns the redirect URL for provenance display
  - load_from_bytes() for Streamlit UploadedFile content
  - batch_convert() for multi-source batch runs (zip-of-zips or merged)
  - Backward-compatible CLI usage unchanged
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from .exceptions import (
    ConverterError,
    LatLonDetectionError,
    UnreachableURLError,
    UnrecognizedShapeError,
)

# ---------------------------------------------------------------------------
# Field-name lookup tables
# ---------------------------------------------------------------------------

REDIRECT_KEYS: list[str] = [
    "gjDownloadURL",
    "geojson_url",
    "geojsonUrl",
    "downloadUrl",
    "download_url",
    "simplifiedGeometryGeoJSON",
    "staticDownloadLink",
    "geojsonLink",
    "downloadLink",
    "gjUrl",
    "geoJsonDownloadURL",
    "geoJsonUrl",
    "geoBoundariesURL",
]

LAT_KEYS: list[str] = [
    "lat", "latitude", "Latitude", "LAT", "LATITUDE", "Lat", "y", "Y",
    "reclat", "RECLAT",                        # NASA / meteorite datasets
    "decimalLatitude", "decimal_latitude",     # biodiversity / GBIF
    "ylat", "lat_dd", "latitude_dd",
    "_lat",                                    # flattened Socrata geolocation
]
LON_KEYS: list[str] = [
    "lon", "lng", "longitude", "Longitude", "LON", "LONGITUDE", "Lon",
    "long", "LONG", "x", "X",
    "reclong", "RECLONG",                      # NASA / meteorite datasets
    "decimalLongitude", "decimal_longitude",   # biodiversity / GBIF
    "xlon", "lon_dd", "longitude_dd",
    "_lon",                                    # flattened Socrata geolocation
]

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; json2shp/1.0)",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConvertResult:
    """Rich result object returned by convert(); consumed by the UI."""

    zip_path: str
    """Absolute path to the output .zip file."""

    input_type: str
    """Original classification: GEOJSON_DIRECT | METADATA_REDIRECT | TABULAR_POINTS."""

    feature_count: int
    """Number of rows/features in the output GeoDataFrame."""

    geometry_type: str
    """Slash-joined geometry type string, e.g. 'Polygon' or 'Point/MultiPoint'."""

    columns: list[str]
    """Final (truncated) column names in the shapefile (excluding 'geometry')."""

    truncated_columns: dict[str, str]
    """Mapping original column name -> truncated shapefile name (only changed names)."""

    column_warnings: list[str]
    """Human-readable messages about any 10-char collision renames."""

    crs: str
    """CRS string of the output GeoDataFrame, e.g. 'EPSG:4326'."""

    source_name: str
    """Display name / base filename used for this conversion."""

    redirect_url: Optional[str] = None
    """If input_type == METADATA_REDIRECT, the URL that was followed."""

    export_format: str = "shp"
    """Format of the exported file (shp or gdb)."""


@dataclass
class BatchResult:
    """Aggregate result of a batch_convert() call."""

    results: list[dict]
    """Each element is either {'result': ConvertResult, 'name': str}
       or {'error': str, 'name': str, 'exception': Exception,
           'needs_latlon': bool, 'available_columns': list}."""

    batch_zip_path: str
    """Path to the final combined .zip file."""

    total_success: int
    total_error: int
    total_needs_latlon: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_source(source: str, headers: Optional[dict] = None) -> dict | list:
    """Load JSON from a local file path or a URL.

    Args:
        source:  Absolute/relative filesystem path OR http(s):// URL.
        headers: Optional extra HTTP headers merged on top of DEFAULT_HEADERS.

    Raises:
        UnreachableURLError: Network or HTTP-level failure.
        ConverterError:      File not found or invalid JSON.
    """
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}

    if source.startswith("http://") or source.startswith("https://"):
        try:
            resp = requests.get(source, timeout=30, headers=merged_headers)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise UnreachableURLError(
                f"Cannot connect to URL: {source}",
                url=source,
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise UnreachableURLError(
                f"Request timed out after 30 s: {source}",
                url=source,
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise UnreachableURLError(
                f"HTTP {resp.status_code} {resp.reason} from {source}",
                url=source,
                status_code=resp.status_code,
            ) from exc

        try:
            return resp.json()
        except ValueError as exc:
            raise ConverterError(
                f"Response from {source} is not valid JSON: {exc}"
            ) from exc

    # Local file
    if not os.path.exists(source):
        raise ConverterError(f"File not found: {source!r}")
    try:
        with open(source, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConverterError(f"Invalid JSON in {source!r}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConverterError(f"Encoding error reading {source!r}: {exc}") from exc


def load_from_bytes(raw: bytes, name: str = "<upload>") -> dict | list:
    """Parse JSON from raw bytes (e.g. a Streamlit UploadedFile).

    Raises:
        ConverterError: Invalid JSON or encoding problem.
    """
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        try:
            return json.loads(raw.decode("latin-1"))
        except Exception as exc:
            raise ConverterError(f"Cannot decode {name}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConverterError(f"Invalid JSON in {name!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


# ESRI geometry type → Shapely constructor map (populated lazily in helpers)
_ESRI_GEOM_TYPES = {
    "esriGeometryPoint",
    "esriGeometryMultipoint",
    "esriGeometryPolyline",
    "esriGeometryPolygon",
}


def classify(data: dict | list) -> str:
    """Return the input-type string for *data*.

    Returns one of:
      'GEOJSON_DIRECT'      — GeoJSON Feature / FeatureCollection / list of Features
      'METADATA_REDIRECT'   — metadata blob with a redirect key
      'ESRI_JSON'           — ArcGIS/ESRI REST JSON with {attributes, geometry{x,y}}
      'PAGINATED_RESULTS'   — paginated API wrapper (e.g. GBIF) with a 'results' list
      'TABULAR_POINTS'      — flat records; lat/lon auto-detection or manual override
      'UNKNOWN'             — no match
    """
    if isinstance(data, dict):
        # GeoJSON Feature or FeatureCollection
        if data.get("type") in ("FeatureCollection", "Feature") or "geometry" in data:
            return "GEOJSON_DIRECT"
        # ArcGIS / ESRI REST JSON (has geometryType + features list with attributes)
        if data.get("geometryType") in _ESRI_GEOM_TYPES and "features" in data:
            return "ESRI_JSON"
        # Paginated API wrapper: has a 'results' list alongside pagination keys
        # e.g. GBIF: {offset, limit, count, endOfRecords, results: [...]}
        if (
            isinstance(data.get("results"), list)
            and any(k in data for k in ("offset", "limit", "count", "total", "page", "totalPages"))
        ):
            return "PAGINATED_RESULTS"
        # Metadata redirect blob
        if any(k in data for k in REDIRECT_KEYS):
            return "METADATA_REDIRECT"
        # Single flat tabular record — require at least one of lat OR lon to match
        keys = set(data.keys())
        if keys & set(LAT_KEYS) or keys & set(LON_KEYS):
            return "TABULAR_POINTS"

    if isinstance(data, list) and data:
        first = data[0]
        if not isinstance(first, dict):
            return "UNKNOWN"
        # List of GeoJSON Features
        if first.get("type") == "Feature" or "geometry" in first:
            return "GEOJSON_DIRECT"
        # Any flat list of dicts — treat as TABULAR_POINTS regardless of whether
        # lat/lon keys are recognized. tabular_to_gdf() will raise LatLonDetectionError
        # with the full column list if auto-detection fails, enabling the UI dropdown.
        return "TABULAR_POINTS"

    return "UNKNOWN"


def _unwrap_paginated(data: dict) -> list[dict]:
    """Extract the records list from a paginated API response wrapper.

    Handles wrappers like:
      GBIF:        {results: [...], offset, limit, count, endOfRecords}
      Generic:     {results: [...], total, page}
      Alternative: {data: [...], meta: {...}}
    """
    # Most common: 'results' key
    if isinstance(data.get("results"), list):
        return data["results"]
    # Some APIs use 'data' key for the records list
    if isinstance(data.get("data"), list):
        return data["data"]
    # Fallback: find the first list value
    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []





# ---------------------------------------------------------------------------
# Redirect resolution
# ---------------------------------------------------------------------------


def resolve_redirect(
    data: dict,
    headers: Optional[dict] = None,
) -> tuple[dict | list, str]:
    """Follow the metadata API pointer to the real GeoJSON.

    Returns:
        (geojson_data, redirect_url)

    Raises:
        UnreachableURLError: Could not fetch the redirect target.
        UnrecognizedShapeError: No recognizable redirect key found.
    """
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}

    for key in REDIRECT_KEYS:
        url = data.get(key)
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=60, headers=merged_headers)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise UnreachableURLError(
                f"HTTP {resp.status_code} fetching redirect URL {url}: {resp.reason}",
                url=url,
                status_code=resp.status_code,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise UnreachableURLError(
                f"Cannot connect to redirect URL: {url}",
                url=url,
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise UnreachableURLError(
                f"Timed out fetching redirect URL: {url}",
                url=url,
            ) from exc

        try:
            return resp.json(), url
        except ValueError as exc:
            raise ConverterError(
                f"Redirect target {url} returned non-JSON: {exc}"
            ) from exc

    raise UnrecognizedShapeError(
        "JSON looks like a metadata redirect, but none of the known redirect keys "
        f"({', '.join(REDIRECT_KEYS)}) point to a valid URL."
    )


# ---------------------------------------------------------------------------
# Tabular → GeoDataFrame
# ---------------------------------------------------------------------------


def _detect_latlon(first_record: dict) -> tuple[Optional[str], Optional[str]]:
    """Best-effort auto-detection of lat/lon field names from a sample record."""
    lat = next((k for k in LAT_KEYS if k in first_record), None)
    lon = next((k for k in LON_KEYS if k in first_record), None)
    return lat, lon


def tabular_to_gdf(
    records: list[dict],
    lat_field: Optional[str] = None,
    lon_field: Optional[str] = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Build a Point GeoDataFrame from flat lat/lon records.

    If *lat_field* / *lon_field* are not given, auto-detection is attempted.

    Raises:
        LatLonDetectionError: Auto-detection failed; caller should prompt user.
        ConverterError:       Type/conversion error when building geometries.
    """
    if not records:
        raise ConverterError("Tabular data contains zero records.")

    det_lat, det_lon = _detect_latlon(records[0])
    lat_field = lat_field or det_lat
    lon_field = lon_field or det_lon

    if not lat_field or not lon_field:
        available = sorted(records[0].keys())
        raise LatLonDetectionError(
            "Could not auto-detect latitude/longitude fields from common variants. "
            "Please select them manually.",
            available_columns=available,
            detected_lat=lat_field,
            detected_lon=lon_field,
        )

    try:
        geometry = [
            Point(float(r[lon_field]), float(r[lat_field])) for r in records
        ]
    except KeyError as exc:
        raise ConverterError(
            f"Field {exc} not present in all records. "
            f"Check that '{lat_field}' and '{lon_field}' exist in every row."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ConverterError(
            f"Cannot convert lat/lon values to float using fields "
            f"'{lat_field}' / '{lon_field}': {exc}"
        ) from exc

    return gpd.GeoDataFrame(records, geometry=geometry, crs=crs)


# ---------------------------------------------------------------------------
# Column name truncation & collision resolution
# ---------------------------------------------------------------------------


def _check_column_collisions(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, str], list[str]]:
    """Truncate all non-geometry column names to ≤10 chars for ESRI Shapefile.

    Collisions are resolved by appending a numeric suffix so no data is lost.

    Returns:
        (renamed_gdf, truncation_map, warnings)
        - truncation_map: {original_name: final_name} (all columns, changed or not)
        - warnings: list of human-readable collision messages
    """
    truncation_map: dict[str, str] = {}  # original → final
    used: set[str] = set()
    warnings: list[str] = []

    for col in gdf.columns:
        if col == "geometry":
            continue

        # Sanitize: shapefile column names must be ASCII, no spaces
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", col)[:10]
        if not sanitized or sanitized[0].isdigit():
            sanitized = "col_" + sanitized[:6]

        candidate = sanitized
        if candidate in used:
            # Find unique suffix
            suffix_num = 2
            while True:
                suffix = str(suffix_num)
                candidate = sanitized[: 10 - len(suffix)] + suffix
                if candidate not in used:
                    break
                suffix_num += 1
            warnings.append(
                f"Column '{col}' truncates to '{sanitized}', which collides with "
                f"another column. Renamed to '{candidate}' to avoid data loss."
            )

        elif col != candidate:
            # Non-collision truncation (informational, not a warning)
            pass

        used.add(candidate)
        truncation_map[col] = candidate

    rename_map = {k: v for k, v in truncation_map.items() if k != v}
    renamed_gdf = gdf.rename(columns=rename_map) if rename_map else gdf
    return renamed_gdf, truncation_map, warnings


# ---------------------------------------------------------------------------
# GeoJSON → GeoDataFrame helper
# ---------------------------------------------------------------------------


def _flatten_socrata_geolocation(records: list[dict]) -> list[dict]:
    """Flatten Socrata-style nested geolocation fields into top-level lat/lon.

    Socrata APIs (e.g. data.nasa.gov) often embed a GeoJSON Point object under
    a field called 'geolocation', ':@computed_region_*', or 'location':
      {"geolocation": {"type": "Point", "coordinates": [lon, lat]}}

    We extract those coordinates into top-level '_lat' / '_lon' keys so that
    tabular_to_gdf() can auto-detect them.
    """
    GEOLOC_KEYS = ("geolocation", "location", "the_geom", "geo_point_2d")
    out = []
    for r in records:
        row = dict(r)
        for key in GEOLOC_KEYS:
            val = row.get(key)
            if not isinstance(val, dict):
                continue
            # GeoJSON Point embedded in field
            if val.get("type") == "Point" and "coordinates" in val:
                coords = val["coordinates"]
                if len(coords) >= 2:
                    row.setdefault("_lon", coords[0])
                    row.setdefault("_lat", coords[1])
                    break
            # Socrata {latitude: "...", longitude: "..."} dict
            if "latitude" in val and "longitude" in val:
                row.setdefault("_lat", val["latitude"])
                row.setdefault("_lon", val["longitude"])
                break
        out.append(row)
    return out


def _build_gdf_from_geojson(
    data: dict | list,
    crs: str,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from a GeoJSON dict or list of Feature dicts."""
    if isinstance(data, list):
        features = data  # assume list of Feature dicts
    elif data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        # Best-effort: treat as Feature or geometry-bearing dict
        features = [data]

    if not features:
        raise ConverterError("GeoJSON contains zero features.")

    gdf = gpd.GeoDataFrame.from_features(features)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    return gdf


def _build_gdf_from_esri_json(data: dict, crs: str) -> gpd.GeoDataFrame:
    """Convert an ArcGIS/ESRI REST JSON response to a GeoDataFrame.

    ESRI REST features look like:
      {"attributes": {...}, "geometry": {"x": lon, "y": lat}}   (Point)
      {"attributes": {...}, "geometry": {"rings": [...]}}        (Polygon)
      {"attributes": {...}, "geometry": {"paths": [...]}}        (Polyline)
    """
    from shapely.geometry import (
        Point, MultiPoint, Polygon, MultiPolygon,
        LineString, MultiLineString, mapping
    )
    import shapely

    geom_type = data.get("geometryType", "")
    features = data.get("features", [])
    if not features:
        raise ConverterError("ESRI JSON contains zero features.")

    # Detect the CRS from spatialReference if present
    sr = data.get("spatialReference", {})
    wkid = sr.get("wkid") or sr.get("latestWkid")
    detected_crs = f"EPSG:{wkid}" if wkid else crs

    records = []
    geometries = []

    for feat in features:
        attrs = feat.get("attributes", {})
        geom_raw = feat.get("geometry")
        records.append(attrs)

        if geom_raw is None:
            geometries.append(None)
            continue

        try:
            if geom_type == "esriGeometryPoint":
                x, y = geom_raw.get("x"), geom_raw.get("y")
                geometries.append(Point(x, y) if x is not None and y is not None else None)

            elif geom_type == "esriGeometryMultipoint":
                pts = [Point(p[0], p[1]) for p in geom_raw.get("points", [])]
                geometries.append(MultiPoint(pts) if pts else None)

            elif geom_type == "esriGeometryPolygon":
                rings = geom_raw.get("rings", [])
                if not rings:
                    geometries.append(None)
                elif len(rings) == 1:
                    geometries.append(Polygon(rings[0]))
                else:
                    # First ring = exterior, rest = holes (ESRI convention)
                    geometries.append(Polygon(rings[0], rings[1:]))

            elif geom_type == "esriGeometryPolyline":
                paths = geom_raw.get("paths", [])
                if not paths:
                    geometries.append(None)
                elif len(paths) == 1:
                    geometries.append(LineString(paths[0]))
                else:
                    geometries.append(MultiLineString(paths))

            else:
                geometries.append(None)
        except Exception:
            geometries.append(None)

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=detected_crs)
    return gdf




# ---------------------------------------------------------------------------
# Main convert() entry-point
# ---------------------------------------------------------------------------


def _safe_basename(name: str, max_len: int = 50) -> str:
    """Return a filesystem-safe slug from *name*."""
    slug = re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:max_len]
    return slug or "output"


def export_shapefile(gdf: gpd.GeoDataFrame, output_dir: str, safe_basename: str, zip_path: str):
    """Write GeoDataFrame to shapefile(s). If geometries are mixed, split by type."""
    # Shapefiles do not support mixed geometries (e.g. Point and Polygon in the same file)
    geom_types = gdf.geometry.geom_type.dropna().unique()
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        if len(geom_types) <= 1:
            shp_path = os.path.join(output_dir, f"{safe_basename}.shp")
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                part = os.path.join(output_dir, safe_basename + ext)
                if os.path.exists(part):
                    zf.write(part, arcname=safe_basename + ext)
        else:
            for gtype in geom_types:
                sub_gdf = gdf[gdf.geometry.geom_type == gtype]
                if sub_gdf.empty:
                    continue
                gtype_suffix = str(gtype).replace(" ", "")
                sub_name = f"{safe_basename}_{gtype_suffix}"
                shp_path = os.path.join(output_dir, f"{sub_name}.shp")
                sub_gdf.to_file(shp_path, driver="ESRI Shapefile")
                for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    part = os.path.join(output_dir, sub_name + ext)
                    if os.path.exists(part):
                        zf.write(part, arcname=sub_name + ext)


def export_geodatabase(gdf: gpd.GeoDataFrame, output_dir: str, safe_basename: str, zip_path: str):
    """Write GeoDataFrame to a File Geodatabase (.gdb) and zip it."""
    import fiona
    
    # Check for GDB writing support
    driver = None
    if "OpenFileGDB" in fiona.supported_drivers and "w" in fiona.drvsupport.supported_drivers.get("OpenFileGDB", ""):
        driver = "OpenFileGDB"
    elif "FileGDB" in fiona.supported_drivers and "w" in fiona.drvsupport.supported_drivers.get("FileGDB", ""):
        driver = "FileGDB"
        
    if not driver:
        raise NotImplementedError(
            "Your environment does not support writing File Geodatabases (.gdb). "
            "GDAL 3.6+ and Fiona 1.9+ are required with OpenFileGDB write support."
        )

    gdb_dir = os.path.join(output_dir, f"{safe_basename}.gdb")
    geom_types = gdf.geometry.geom_type.dropna().unique()
    
    try:
        if len(geom_types) <= 1:
            gdf.to_file(gdb_dir, driver=driver, layer=safe_basename)
        else:
            for gtype in geom_types:
                sub_gdf = gdf[gdf.geometry.geom_type == gtype]
                if sub_gdf.empty:
                    continue
                gtype_suffix = str(gtype).replace(" ", "")
                sub_name = f"{safe_basename}_{gtype_suffix}"
                sub_gdf.to_file(gdb_dir, driver=driver, layer=sub_name)
    except Exception as exc:
        raise Exception(f"Failed to write Geodatabase: {exc}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for root, _, files in os.walk(gdb_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(gdb_dir))
                zf.write(file_path, arcname=arcname)


def convert(
    source: str,
    output_dir: str,
    base_name: str = "output",
    lat_field: Optional[str] = None,
    lon_field: Optional[str] = None,
    headers: Optional[dict] = None,
    crs: str = "EPSG:4326",
    export_format: str = "shp",
    data: Optional[dict | list] = None,
) -> ConvertResult:
    """Convert a JSON source to a zipped ESRI Shapefile bundle.

    Args:
        source:     URL or local file path. Ignored when *data* is provided.
        output_dir: Directory to write output files into.
        base_name:  Stem used for output filenames (sanitized automatically).
        lat_field:  Override lat column name for TABULAR_POINTS inputs.
        lon_field:  Override lon column name for TABULAR_POINTS inputs.
        headers:    Extra HTTP headers (merged with DEFAULT_HEADERS).
        crs:        Default CRS string if not embedded in source data.
        data:       Pre-parsed JSON dict/list (bypasses load_source; for uploads).

    Returns:
        ConvertResult with zip path and rich metadata.

    Raises:
        UnreachableURLError, UnrecognizedShapeError, LatLonDetectionError,
        ConverterError — all subclasses of ConverterError.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    if data is None:
        data = load_source(source, headers=headers)

    # 2. Classify
    kind = classify(data)
    original_kind = kind
    redirect_url: Optional[str] = None

    # 3. Resolve redirect
    if kind == "METADATA_REDIRECT":
        data, redirect_url = resolve_redirect(data, headers=headers)
        kind = classify(data)

    # 4. Build GeoDataFrame
    if kind == "GEOJSON_DIRECT":
        gdf = _build_gdf_from_geojson(data, crs)

    elif kind == "ESRI_JSON":
        gdf = _build_gdf_from_esri_json(data, crs)

    elif kind == "PAGINATED_RESULTS":
        # Unwrap the results list and re-classify its contents
        records = _unwrap_paginated(data)
        if not records:
            raise ConverterError("Paginated API response contains zero results.")
        inner_kind = classify(records)
        if inner_kind == "GEOJSON_DIRECT":
            gdf = _build_gdf_from_geojson(records, crs)
        else:
            # Treat as tabular (most paginated APIs return flat records)
            records = _flatten_socrata_geolocation(records)
            gdf = tabular_to_gdf(records, lat_field=lat_field, lon_field=lon_field, crs=crs)

    elif kind == "TABULAR_POINTS":
        records: list[dict] = data if isinstance(data, list) else [data]
        # Flatten Socrata-style geolocation objects before lat/lon detection
        records = _flatten_socrata_geolocation(records)
        gdf = tabular_to_gdf(records, lat_field=lat_field, lon_field=lon_field, crs=crs)

    else:
        raise UnrecognizedShapeError(
            f"JSON does not match any known input format "
            f"(GEOJSON_DIRECT, ESRI_JSON, PAGINATED_RESULTS, METADATA_REDIRECT, TABULAR_POINTS). "
            f"Internal classification: '{kind}'."
        )

    # 5. Truncate column names & detect collisions
    gdf, trunc_map, col_warnings = _check_column_collisions(gdf)

    # 6. Write shapefile(s) and bundle into zip
    safe = _safe_basename(base_name)
    if export_format == "gdb":
        zip_path = os.path.join(output_dir, f"{safe}.gdb.zip")
        export_geodatabase(gdf, output_dir, safe, zip_path)
    else:
        zip_path = os.path.join(output_dir, f"{safe}.zip")
        export_shapefile(gdf, output_dir, safe, zip_path)

    # 8. Assemble metadata
    geom_types = [g for g in gdf.geometry.geom_type.unique().tolist() if g]
    geom_type_str = "/".join(geom_types) if geom_types else "Unknown"

    # Only report changed names in truncated_columns
    changed = {orig: final for orig, final in trunc_map.items() if orig != final}

    return ConvertResult(
        zip_path=zip_path,
        input_type=original_kind,
        feature_count=len(gdf),
        geometry_type=geom_type_str,
        columns=[c for c in gdf.columns if c != "geometry"],
        truncated_columns=changed,
        column_warnings=col_warnings,
        crs=str(gdf.crs) if gdf.crs else crs,
        source_name=base_name,
        redirect_url=redirect_url,
        export_format=export_format,
    )


# ---------------------------------------------------------------------------
# GeoPackage support (.gpkg)  — binary spatial format, no JSON pipeline
# ---------------------------------------------------------------------------
# Web Feature Service (WFS) support — remote OGC standard
# ---------------------------------------------------------------------------


def list_wfs_layers(url: str) -> list[str]:
    """Return the list of layer names (typeNames) from a WFS endpoint.

    Uses GDAL's WFS driver by prefixing the URL with 'WFS:'.
    """
    wfs_url = url if url.startswith("WFS:") else f"WFS:{url}"
    try:
        import fiona
        layers = fiona.listlayers(wfs_url)
    except Exception as exc:
        raise ConverterError(f"Cannot connect to WFS endpoint '{url}': {exc}") from exc
    if not layers:
        raise ConverterError(f"WFS endpoint '{url}' returned zero layers.")
    return list(layers)


def convert_wfs(
    url: str,
    output_dir: str,
    base_name: str = "output",
    layer: Optional[str] = None,
    crs: str = "EPSG:4326",
) -> ConvertResult:
    """Download a layer from a WFS endpoint and save as Shapefile.

    Args:
        url:        WFS GetCapabilities or base URL.
        output_dir: Directory to write output files.
        base_name:  Stem for output filenames.
        layer:      WFS typeName to request.
        crs:        Fallback CRS.
    """
    os.makedirs(output_dir, exist_ok=True)
    wfs_url = url if url.startswith("WFS:") else f"WFS:{url}"

    all_layers = list_wfs_layers(url)
    target_layer = layer if (layer and layer in all_layers) else all_layers[0]

    try:
        gdf = gpd.read_file(wfs_url, layer=target_layer)
    except Exception as exc:
        raise ConverterError(f"Cannot download layer '{target_layer}' from WFS: {exc}") from exc

    if len(gdf) == 0:
        raise ConverterError(f"WFS layer '{target_layer}' contains zero features.")

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)

    gdf, trunc_map, col_warnings = _check_column_collisions(gdf)

    safe = _safe_basename(base_name)
    if export_format == "gdb":
        zip_path = os.path.join(output_dir, f"{safe}.gdb.zip")
        export_geodatabase(gdf, output_dir, safe, zip_path)
    else:
        zip_path = os.path.join(output_dir, f"{safe}.zip")
        export_shapefile(gdf, output_dir, safe, zip_path)

    geom_types = [g for g in gdf.geometry.geom_type.unique().tolist() if g]
    changed = {orig: final for orig, final in trunc_map.items() if orig != final}

    return ConvertResult(
        zip_path=zip_path,
        input_type="WFS",
        feature_count=len(gdf),
        geometry_type="/".join(geom_types) if geom_types else "Unknown",
        columns=[c for c in gdf.columns if c != "geometry"],
        truncated_columns=changed,
        column_warnings=col_warnings,
        crs=str(gdf.crs) if gdf.crs else crs,
        source_name=base_name,
        redirect_url=None,
        export_format=export_format,
    )


# ---------------------------------------------------------------------------


def list_gdal_layers(source_path: str) -> list[str]:
    """Return the ordered list of layer names inside a GDAL-compatible file (GPKG, KML, etc).

    Raises:
        ConverterError: If the file cannot be opened or has no layers.
    """
    try:
        import fiona
        fiona.drvsupport.supported_drivers['MVT'] = 'r'
        fiona.drvsupport.supported_drivers['OSM'] = 'r'
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
        fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
        fiona.drvsupport.supported_drivers['GPX'] = 'rw'
        fiona.drvsupport.supported_drivers['DXF'] = 'rw'
        fiona.drvsupport.supported_drivers['TopoJSON'] = 'r'
        fiona.drvsupport.supported_drivers['SQLite'] = 'rw'
        fiona.drvsupport.supported_drivers['GML'] = 'rw'
        fiona.drvsupport.supported_drivers['FlatGeobuf'] = 'rw'
        fiona.drvsupport.supported_drivers['MapInfo File'] = 'rw'
        fiona.drvsupport.supported_drivers['DGN'] = 'rw'
        layers = fiona.listlayers(source_path)
    except Exception as exc:
        raise ConverterError(f"Cannot read file '{source_path}': {exc}") from exc
    if not layers:
        raise ConverterError(f"File '{source_path}' contains no layers.")
    return list(layers)


def convert_gdal_source(
    source_path: str,
    output_dir: str,
    base_name: str = "output",
    layer: Optional[str] = None,
    crs: str = "EPSG:4326",
) -> ConvertResult:
    """Convert one layer of a GDAL file to a zipped ESRI Shapefile bundle.

    Args:
        source_path: Absolute path to the file on disk (e.g., .gpkg, .kml).
        output_dir:  Directory to write output files into.
        base_name:   Stem for output filenames.
        layer:       Layer name to convert. If None, the first layer is used.
        crs:         Fallback CRS if the layer has none.

    Returns:
        ConvertResult with zip path and rich metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve layer
    all_layers = list_gdal_layers(source_path)
    target_layer = layer if (layer and layer in all_layers) else all_layers[0]

    try:
        import fiona
        fiona.drvsupport.supported_drivers['MVT'] = 'r'
        fiona.drvsupport.supported_drivers['OSM'] = 'r'
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
        fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
        fiona.drvsupport.supported_drivers['GPX'] = 'rw'
        fiona.drvsupport.supported_drivers['DXF'] = 'rw'
        fiona.drvsupport.supported_drivers['TopoJSON'] = 'r'
        fiona.drvsupport.supported_drivers['SQLite'] = 'rw'
        fiona.drvsupport.supported_drivers['GML'] = 'rw'
        fiona.drvsupport.supported_drivers['FlatGeobuf'] = 'rw'
        fiona.drvsupport.supported_drivers['MapInfo File'] = 'rw'
        fiona.drvsupport.supported_drivers['DGN'] = 'rw'
        gdf = gpd.read_file(source_path, layer=target_layer)
    except Exception as exc:
        raise ConverterError(
            f"Cannot read layer '{target_layer}' from file: {exc}"
        ) from exc

    if len(gdf) == 0:
        raise ConverterError(f"Layer '{target_layer}' contains zero features.")

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)

    # Column truncation & collision detection
    gdf, trunc_map, col_warnings = _check_column_collisions(gdf)

    # Write shapefile bundle
    safe = _safe_basename(base_name)
    if export_format == "gdb":
        zip_path = os.path.join(output_dir, f"{safe}.gdb.zip")
        export_geodatabase(gdf, output_dir, safe, zip_path)
    else:
        zip_path = os.path.join(output_dir, f"{safe}.zip")
        export_shapefile(gdf, output_dir, safe, zip_path)

    geom_types = [g for g in gdf.geometry.geom_type.unique().tolist() if g]
    changed = {orig: final for orig, final in trunc_map.items() if orig != final}

    return ConvertResult(
        zip_path=zip_path,
        input_type="GPKG",
        feature_count=len(gdf),
        geometry_type="/".join(geom_types) if geom_types else "Unknown",
        columns=[c for c in gdf.columns if c != "geometry"],
        truncated_columns=changed,
        column_warnings=col_warnings,
        crs=str(gdf.crs) if gdf.crs else crs,
        source_name=base_name,
        redirect_url=None,
        export_format=export_format,
    )


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def batch_convert(
    sources: list[dict],
    output_dir: str,
    merge: bool = False,
    headers: Optional[dict] = None,
    crs: str = "EPSG:4326",
) -> BatchResult:
    """Convert multiple sources in one call.

    Args:
        sources: List of dicts, each with:
            - "name":      str  — display name / output base filename
            - "source":    str  — path or URL  (OR)
            - "data":      dict/list — pre-parsed JSON (e.g. from upload)
            - "lat_field": str  (optional override)
            - "lon_field": str  (optional override)
        output_dir: Root directory for all output files.
        merge:      If True, merge all successful GDFs into one shapefile.
        headers:    HTTP headers shared across all requests.
        crs:        Default CRS.

    Returns:
        BatchResult with per-item results and path to the combined zip.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []
    gdfs: list[gpd.GeoDataFrame] = []

    for item in sources:
        name = item.get("name", "output")
        source = item.get("source", "")
        pre_data = item.get("data")
        lat_f = item.get("lat_field")
        lon_f = item.get("lon_field")
        is_gpkg = item.get("is_gpkg", False)
        gpkg_layer = item.get("layer")
        item_dir = os.path.join(output_dir, "items", _safe_basename(name))

        try:
            if is_gpkg:
                result = convert_gdal_source(
                    source_path=source,
                    output_dir=item_dir,
                    base_name=name,
                    layer=gpkg_layer,
                    crs=crs,
                )
            else:
                result = convert(
                    source=source,
                    output_dir=item_dir,
                    base_name=name,
                    lat_field=lat_f,
                    lon_field=lon_f,
                    headers=headers,
                    crs=crs,
                    data=pre_data,
                )
            results.append({"result": result, "name": name})
            if merge:
                safe = _safe_basename(name)
                gdfs.append(gpd.read_file(os.path.join(item_dir, safe + ".shp")))

        except LatLonDetectionError as exc:
            results.append(
                {
                    "error": str(exc),
                    "name": name,
                    "exception": exc,
                    "needs_latlon": True,
                    "available_columns": exc.available_columns,
                }
            )
        except ConverterError as exc:
            results.append({"error": str(exc), "name": name, "exception": exc})
        except Exception as exc:  # noqa: BLE001
            results.append(
                {"error": f"Unexpected error: {exc}", "name": name, "exception": exc}
            )

    success = sum(1 for r in results if "result" in r)
    errors = sum(1 for r in results if "error" in r and not r.get("needs_latlon"))
    needs_ll = sum(1 for r in results if r.get("needs_latlon"))

    batch_zip = os.path.join(output_dir, "batch_output.zip")

    if merge and gdfs:
        # Merge all GeoDataFrames into one shapefile
        merged = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=crs
        )
        merged_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_shp = os.path.join(merged_dir, "merged.shp")
        merged.to_file(merged_shp, driver="ESRI Shapefile")

        with zipfile.ZipFile(batch_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                p = os.path.join(merged_dir, "merged" + ext)
                if os.path.exists(p):
                    zf.write(p, arcname="merged" + ext)
    else:
        # zip-of-zips: each successful conversion's zip goes inside the outer zip
        with zipfile.ZipFile(batch_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for r in results:
                if "result" in r:
                    zf.write(
                        r["result"].zip_path,
                        arcname=os.path.basename(r["result"].zip_path),
                    )

    return BatchResult(
        results=results,
        batch_zip_path=batch_zip,
        total_success=success,
        total_error=errors,
        total_needs_latlon=needs_ll,
    )


import os
from typing import Optional
import geopandas as gpd
from core.exceptions import ConverterError
import requests

def list_ogc_collections(url: str) -> list[str]:
    """Fetch OGC API Features collections."""
    try:
        # GDAL OAPIF driver expects the root URL, not the /collections endpoint
        clean_url = url
        if clean_url.endswith("/collections/"):
            clean_url = clean_url[:-13]
        elif clean_url.endswith("/collections"):
            clean_url = clean_url[:-12]
            
        import fiona
        oapif_url = clean_url if clean_url.startswith("OAPIF:") else f"OAPIF:{clean_url}"
        layers = fiona.listlayers(oapif_url)
        if not layers:
            raise ConverterError(f"OGC API endpoint '{url}' returned zero layers.")
        return list(layers)
    except Exception as exc:
        raise ConverterError(f"Cannot connect to OGC API endpoint '{url}': {exc}") from exc

def convert_ogc(
    url: str,
    output_dir: str,
    base_name: str = "output",
    layer: Optional[str] = None,
    crs: str = "EPSG:4326",
) -> ConvertResult:
    """Download a layer from an OGC API Features endpoint and save as Shapefile."""
    os.makedirs(output_dir, exist_ok=True)
    clean_url = url
    if clean_url.endswith("/collections/"):
        clean_url = clean_url[:-13]
    elif clean_url.endswith("/collections"):
        clean_url = clean_url[:-12]
        
    oapif_url = clean_url if clean_url.startswith("OAPIF:") else f"OAPIF:{clean_url}"

    all_layers = list_ogc_collections(url)
    target_layer = layer if (layer and layer in all_layers) else all_layers[0]

    try:
        gdf = gpd.read_file(oapif_url, layer=target_layer)
    except Exception as exc:
        raise ConverterError(f"Cannot download layer '{target_layer}' from OGC API: {exc}") from exc

    if len(gdf) == 0:
        raise ConverterError(f"OGC API layer '{target_layer}' contains zero features.")

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)

    gdf, trunc_map, col_warnings = _check_column_collisions(gdf)

    safe = _safe_basename(base_name)
    if export_format == "gdb":
        zip_path = os.path.join(output_dir, f"{safe}.gdb.zip")
        export_geodatabase(gdf, output_dir, safe, zip_path)
    else:
        zip_path = os.path.join(output_dir, f"{safe}.zip")
        export_shapefile(gdf, output_dir, safe, zip_path)

    geom_types = [g for g in gdf.geometry.geom_type.unique().tolist() if g]
    changed = {orig: final for orig, final in trunc_map.items() if orig != final}

    return ConvertResult(
        zip_path=zip_path,
        input_type="OGC_API",
        feature_count=len(gdf),
        geometry_type="/".join(geom_types) if geom_types else "Unknown",
        columns=[c for c in gdf.columns if c != "geometry"],
        truncated_columns=changed,
        column_warnings=col_warnings,
        crs=str(gdf.crs) if gdf.crs else crs,
        source_name=base_name,
        redirect_url=None,
        export_format=export_format,
    )

def convert_osm(
    place_name: str,
    tags: dict,
    output_dir: str,
    base_name: str = "output",
    crs: str = "EPSG:4326",
) -> ConvertResult:
    """Fetch OpenStreetMap data using osmnx."""
    import osmnx as ox
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Fetch features
        gdf = ox.features_from_place(place_name, tags=tags)
    except Exception as exc:
        raise ConverterError(f"Failed to fetch OSM data for '{place_name}': {exc}") from exc

    if len(gdf) == 0:
        raise ConverterError(f"OSM query returned zero features for '{place_name}'.")

    # osmnx can return points, lines, polygons in one GDF. 
    # Shapefiles can't mix geometry types!
    # Our geometry_splitter will handle it if we just save it?
    # No, _write_shapefile_bundle doesn't split! We have to split manually or just drop non-polygons?
    # Actually, let's just write the whole GDF. Wait, geopandas to_file fails if mixed geometries.
    # We must split by geometry type for OSM!
    from shapely.geometry import Point, LineString, Polygon, MultiPolygon
    geom_types = gdf.geometry.geom_type
    
    # We'll split it inside _write_shapefile_bundle if we update it, or do it here.
    # For now, let's just keep it simple and write it. If it fails, we catch it.
    
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)

    gdf, trunc_map, col_warnings = _check_column_collisions(gdf)

    safe = _safe_basename(base_name)
    zip_path = os.path.join(output_dir, f"{safe}.zip")
    
    # To handle mixed geometries safely, we can export multiple shapefiles
    if export_format == "gdb":
        zip_path = os.path.join(output_dir, f"{safe}.gdb.zip")
        export_geodatabase(gdf, output_dir, safe, zip_path)
    else:
        zip_path = os.path.join(output_dir, f"{safe}.zip")
        export_shapefile(gdf, output_dir, safe, zip_path)

    types_list = [g for g in gdf.geometry.geom_type.unique().tolist() if g]
    changed = {orig: final for orig, final in trunc_map.items() if orig != final}

    return ConvertResult(
        zip_path=zip_path,
        input_type="OSM",
        feature_count=len(gdf),
        geometry_type="/".join(types_list) if types_list else "Unknown",
        columns=[c for c in gdf.columns if c != "geometry"],
        truncated_columns=changed,
        column_warnings=col_warnings,
        crs=str(gdf.crs) if gdf.crs else crs,
        source_name=base_name,
        redirect_url=None,
        export_format=export_format,
    )



def convert_xyz_tiles(
    url_template: str,
    bbox: tuple[float, float, float, float],
    zoom: int,
    output_dir: str,
    base_name: str = "output",
    crs: str = "EPSG:4326",
) -> ConvertResult:
    """Download XYZ tiles and merge them into a Shapefile bundle."""
    import mercantile
    import requests
    import tempfile
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Calculate tiles
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))
    if not tiles:
        raise ConverterError(f"No tiles found intersecting bbox {bbox} at zoom {zoom}.")
        
    # Cap tiles to prevent infinite downloads
    MAX_TILES = 500
    if len(tiles) > MAX_TILES:
        raise ConverterError(f"Requested {len(tiles)} tiles, which exceeds the safety limit of {MAX_TILES}. Please zoom out or make the bounding box smaller.")
        
    gdfs = []
    
    # We need a temp dir to save downloaded tiles
    tmp_dir = tempfile.mkdtemp()
    
    # Determine extension
    ext = ".mvt"
    if ".pbf" in url_template.lower():
        ext = ".pbf"
    elif ".geojson" in url_template.lower() or ".json" in url_template.lower():
        ext = ".geojson"
    elif ".topojson" in url_template.lower():
        ext = ".topojson"
        
    # Download tiles
    import fiona
    
    first_error_status = None
    successful_downloads = 0
    
    for tile in tiles:
        # replace {x}, {y}, {z}
        tile_url = url_template.replace("{x}", str(tile.x)).replace("{y}", str(tile.y)).replace("{z}", str(tile.z))
        
        try:
            resp = requests.get(tile_url, timeout=10)
            if resp.status_code == 200:
                successful_downloads += 1
                tmp_path = os.path.join(tmp_dir, f"tile_{tile.z}_{tile.x}_{tile.y}{ext}")
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
                
                # read layer
                try:
                    layers = fiona.listlayers(tmp_path)
                    for layer_name in layers:
                        try:
                            gdf = gpd.read_file(tmp_path, layer=layer_name)
                            if len(gdf) > 0:
                                gdf["source_layer"] = layer_name
                                gdfs.append(gdf)
                        except Exception:
                            pass
                except Exception:
                    try:
                        gdf = gpd.read_file(tmp_path)
                        if len(gdf) > 0:
                            gdfs.append(gdf)
                    except Exception:
                        pass
            elif first_error_status is None:
                first_error_status = resp.status_code
        except Exception as e:
            if first_error_status is None:
                first_error_status = str(e)
            
    if successful_downloads == 0:
        raise ConverterError(f"Failed to download any tiles. Server returned: {first_error_status}")
        
    if not gdfs:
        raise ConverterError("Downloaded tiles contained no valid geometries (they might be empty water tiles).")
        
    # concat all
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    if merged_gdf.crs is None:
        merged_gdf = merged_gdf.set_crs(crs)
        
    merged_gdf, trunc_map, col_warnings = _check_column_collisions(merged_gdf)
    
    safe = _safe_basename(base_name)
    zip_path = os.path.join(output_dir, f"{safe}.zip")
    if export_format == "gdb":
        zip_path = os.path.join(output_dir, f"{safe}.gdb.zip")
        export_geodatabase(merged_gdf, output_dir, safe, zip_path)
    else:
        zip_path = os.path.join(output_dir, f"{safe}.zip")
        export_shapefile(merged_gdf, output_dir, safe, zip_path)
    
    geom_types = [str(g) for g in merged_gdf.geometry.geom_type.unique().tolist() if g]
    changed = {orig: final for orig, final in trunc_map.items() if orig != final}
    
    return ConvertResult(
        zip_path=zip_path,
        input_type="XYZ_Tile",
        feature_count=len(merged_gdf),
        geometry_type="/".join(geom_types) if geom_types else "Unknown",
        columns=[c for c in merged_gdf.columns if c != "geometry"],
        truncated_columns=changed,
        column_warnings=col_warnings,
        crs=str(merged_gdf.crs) if merged_gdf.crs else crs,
        source_name=base_name,
        redirect_url=None,
        export_format=export_format,
    )



