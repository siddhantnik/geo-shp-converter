
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
    for tile in tiles:
        # replace {x}, {y}, {z}
        tile_url = url_template.replace("{x}", str(tile.x)).replace("{y}", str(tile.y)).replace("{z}", str(tile.z))
        
        try:
            resp = requests.get(tile_url, timeout=10)
            if resp.status_code == 200:
                tmp_path = os.path.join(tmp_dir, f"tile_{tile.z}_{tile.x}_{tile.y}{ext}")
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
                
                # read layer
                # if it's MVT/PBF, there might be multiple layers. Let's just grab all layers and concat
                try:
                    layers = fiona.listlayers(tmp_path)
                    for layer_name in layers:
                        try:
                            gdf = gpd.read_file(tmp_path, layer=layer_name)
                            if len(gdf) > 0:
                                # We can optionally add a column for layer name
                                gdf["source_layer"] = layer_name
                                gdfs.append(gdf)
                        except Exception:
                            pass
                except Exception:
                    # Maybe it's a simple geojson without layers
                    try:
                        gdf = gpd.read_file(tmp_path)
                        if len(gdf) > 0:
                            gdfs.append(gdf)
                    except Exception:
                        pass
        except Exception:
            pass
            
    if not gdfs:
        raise ConverterError("No valid geometries found in any downloaded tiles.")
        
    # concat all
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    if merged_gdf.crs is None:
        merged_gdf = merged_gdf.set_crs(crs)
        
    merged_gdf, trunc_map, col_warnings = _check_column_collisions(merged_gdf)
    
    safe = _safe_basename(base_name)
    zip_path = os.path.join(output_dir, f"{safe}.zip")
    _write_shapefile_bundle(merged_gdf, output_dir, safe, zip_path)
    
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
    )
