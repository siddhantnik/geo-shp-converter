
def convert_socrata(
    domain: str,
    dataset_id: str,
    app_token: str,
    soql_where: str,
    max_rows: int,
    output_dir: str,
    base_name: str = "output",
    crs: str = "EPSG:4326",
) -> ConvertResult:
    """Download data from Socrata (SODA) and merge into a Shapefile bundle."""
    import requests
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean domain
    domain = domain.replace("http://", "").replace("https://", "").split("/")[0]
    
    # Base URL
    url = f"https://{domain}/resource/{dataset_id}.json"
    
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token
        
    all_data = []
    limit = 50000
    offset = 0
    
    import math
    max_pages = math.ceil(max_rows / limit)
    
    import logging
    
    for page in range(max_pages):
        params = {
            "$limit": min(limit, max_rows - offset),
            "$offset": offset,
        }
        if soql_where:
            params["$where"] = soql_where
            
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        
        if resp.status_code != 200:
            raise ConverterError(f"Socrata API Error {resp.status_code}: {resp.text}")
            
        data = resp.json()
        if not data:
            break
            
        all_data.extend(data)
        offset += len(data)
        
        if len(data) < params["$limit"]:
            break  # we've reached the end
            
        if offset >= max_rows:
            break

    if not all_data:
        raise ConverterError("Socrata API returned no data.")
        
    # We have all_data (list of dicts). We can just pass it to the same logic used for Tabular JSON!
    # Wait, we can't easily reuse `convert` because `convert` is designed to be the entry point for everything and writes to disk itself.
    # Actually, we CAN reuse the logic of `convert` by just passing the data directly to it!
    # But wait, `convert` is in `converter.py`. We can just call it!
    from core.converter import convert
    
    # Determine lat/lon fields if any
    lat_field = None
    lon_field = None
    
    # Find lat/lon if they exist
    first_row = all_data[0]
    # some socrata datasets have location objects
    # if it's just point data, we let `convert` auto-detect lat/lon!
    # `convert` will automatically try to find 'latitude', 'lat', 'y', etc.
    
    return convert(
        source="",
        output_dir=output_dir,
        base_name=base_name,
        data=all_data,
        lat_field=None,
        lon_field=None,
        crs_override=crs
    )
