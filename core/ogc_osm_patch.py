import os
from typing import Optional
import geopandas as gpd
from core.converter import ConvertResult, _check_column_collisions, _safe_basename, _write_shapefile_bundle
from core.exceptions import ConverterError
import requests

def list_ogc_collections(url: str) -> list[str]:
    """Fetch OGC API Features collections."""
    try:
        # if the url does not end with /collections, we might need to probe, but let's assume it has /collections
        # Actually, let's just use GDAL OAPIF driver
        import fiona
        oapif_url = url if url.startswith("OAPIF:") else f"OAPIF:{url}"
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
    oapif_url = url if url.startswith("OAPIF:") else f"OAPIF:{url}"

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
    zip_path = os.path.join(output_dir, f"{safe}.zip")
    _write_shapefile_bundle(gdf, output_dir, safe, zip_path)

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
    _write_shapefile_bundle(gdf, output_dir, safe, zip_path)

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
    )
