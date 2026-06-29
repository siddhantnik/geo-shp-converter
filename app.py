"""
app.py — GeoJSON → Shapefile Converter
Streamlit web UI. Run with: streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import streamlit as st

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from core.converter import (
    ConvertResult,
    _safe_basename,
    convert,
    convert_gdal_source,
    list_gdal_layers,
    convert_wfs,
    list_wfs_layers,
    convert_ogc,
    list_ogc_collections,
    convert_osm,
    convert_xyz_tiles,
    load_from_bytes,
)
from core.exceptions import (
    ConverterError,
    LatLonDetectionError,
    UnreachableURLError,
    UnrecognizedShapeError,
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Shapefile Converter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── Root & Body ───────────────────────────────────────────────────── */
html, body {
    font-family: 'Times New Roman', Times, serif !important;
}
html, body, [class*="css"], label, [data-testid="stWidgetLabel"] p, .stMarkdown p, p, span, code {
    color: #000000 !important;
}
code {
    background: transparent !important;
}

/* ── App background gradient ──────────────────────────────────────── */
.stApp {
    background: #e6e3da !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #e6e3da !important;
    border-right: 1px solid #dee2e6;
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #000000;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Header gradient text ──────────────────────────────────────────── */
.app-title {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    color: #000000;
    margin-bottom: 0.2rem;
}
.app-subtitle {
    color: #000000;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* ── Input type badge pills ────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 8px;
}
.badge-geojson  { background:#e8f5e9; color:#2e7d32; border:1px solid #c8e6c9; }
.badge-redirect { background:#e3f2fd; color:#1565c0; border:1px solid #bbdefb; }
.badge-tabular  { background:#f3e5f5; color:#7b1fa2; border:1px solid #e1bee7; }
.badge-error    { background:#ffebee; color:#c62828; border:1px solid #ffcdd2; }
.badge-unknown  { background:#f5f5f5; color:#616161; border:1px solid #e0e0e0; }
.badge-gpkg     { background:#e0f2f1; color:#00695c; border:1px solid #b2dfdb; }
.badge-wfs      { background:#e8f5e9; color:#2e7d32; border:1px solid #c8e6c9; }

/* ── Result card ───────────────────────────────────────────────────── */
.result-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.25s ease;
}
.result-card:hover {
}
.result-card.success { border-left: 3px solid #3fb950; }
.result-card.error   { border-left: 3px solid #ff7b72; }
.result-card.warning { border-left: 3px solid #f0883e; }

.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #000000;
    margin-bottom: 4px;
}
.card-meta {
    font-size: 0.78rem;
    color: #333333;
    margin-bottom: 12px;
}

/* ── Stat pills ────────────────────────────────────────────────────── */
.stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
.stat-pill {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: #000000;
}
.stat-pill strong { color: #000000; }

/* ── Warning / error banners ───────────────────────────────────────── */
.warn-banner {
    background: rgba(240, 136, 62, 0.1);
    border: 1px solid rgba(240, 136, 62, 0.35);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #f0883e;
    margin-top: 8px;
}
.err-banner {
    background: rgba(255, 123, 114, 0.08);
    border: 1px solid rgba(255, 123, 114, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #ff7b72;
    margin-top: 8px;
}
.info-banner {
    background: rgba(88, 166, 255, 0.08);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #58a6ff;
    margin-top: 8px;
}

/* ── Section divider ───────────────────────────────────────────────── */
.section-heading {
    color: #000000;
    font-size: 0.72rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    font-weight: 600;
    margin: 20px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #dee2e6;
}

/* ── Column table ──────────────────────────────────────────────────── */
.col-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    margin-top: 8px;
}
.col-table th {
    color: #000000;
    text-align: left;
    border-bottom: 1px solid #dee2e6;
    padding: 4px 8px;
    font-weight: 600;
}
.col-table td {
    color: #000000;
    border-bottom: 1px solid #dee2e6;
    padding: 4px 8px;
}
.col-table tr:last-child td { border-bottom: none; }
.col-changed { color: #f0883e; font-weight: 500; }

/* ── All Buttons (Convert, Download, Clear) ────────────────────────── */
div[data-testid="stButton"] > button,
.stDownloadButton > button,
div[data-testid="stButton"] > button *,
.stDownloadButton > button * {
    background: #000000 !important;
    border: 1px solid #000000 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover,
.stDownloadButton > button:hover {
    background: #333333 !important;
    border-color: #333333 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f0ece3;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #d4d0c5;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 6px 20px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #000000 !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
}

/* ── Inputs ────────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background: #ffffff !important;
    border-color: #cccccc !important;
    color: #000000 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #000000 !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1) !important;
}

/* ── File uploader ─────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #333333 !important;
    border: 1.5px dashed #555555 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"] *, [data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #000000 !important;
}

/* ── Sidebar toggle ────────────────────────────────────────────────── */
.stToggle { margin-top: 4px; }

/* ── Scrollable column list ────────────────────────────────────────── */
.col-scroll {
    max-height: 150px;
    overflow-y: auto;
    background: #ffffff;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #000000;
    border: 1px solid #cccccc;
}

/* ── Metric cards ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #cccccc;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #000000; font-size: 1.4rem; }
[data-testid="stMetricLabel"] { color: #000000; font-size: 0.75rem; }

/* ── Selectbox ─────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] {
    background: #ffffff !important;
    border-color: #cccccc !important;
}
.stSelectbox [data-baseweb="select"] span {
    color: #000000 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "output_dir": None,          # set below; a tempdir that persists the session
    "results": [],               # list[dict]: {result|error, name, zip_bytes?, ...}
    "needs_latlon": [],          # list[dict]: sources needing manual lat/lon pick
    "latlon_selections": {},     # {name: {"lat": str, "lon": str}}
    "converted": False,          # whether a conversion has been run
    "pending_sources": [],       # sources queued for retry after lat/lon selection
    "merge_mode": False,
    "gpkg_saved": {},            # {filename: {"path": str, "layers": list[str]}}
    "gpkg_layer_selections": {}, # {filename: [selected_layer, ...]}
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Ensure the temp dir exists (only create once per session)
if st.session_state.output_dir is None:
    st.session_state.output_dir = tempfile.mkdtemp(prefix="geo_shp_")


# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    user_agent = st.text_input(
        "User-Agent",
        value="Mozilla/5.0 (compatible; json2shp/1.0)",
        help="Some APIs return 403 with non-browser UAs. Override here.",
    )

    extra_headers_raw = st.text_area(
        "Extra HTTP headers (JSON)",
        value="{}",
        height=90,
        help='Optional additional headers, e.g. {"Authorization": "Bearer TOKEN"}',
    )

    crs_override = st.text_input(
        "Default CRS",
        value="EPSG:4326",
        help="Used when CRS is absent from the source data.",
    )

    st.markdown("---")
    st.markdown("### Batch Mode")

    merge_mode = st.toggle(
        "Merge all into single shapefile",
        value=False,
        help=(
            "OFF: each source produces its own .zip; all zips are bundled into "
            "one outer zip-of-zips.\n"
            "ON: all features are merged into a single combined shapefile."
        ),
    )
    st.session_state.merge_mode = merge_mode

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#6b7894;line-height:1.6;">'
        "Supports <strong>GEOJSON_DIRECT</strong>, <strong>METADATA_REDIRECT</strong> "
        "(follows pointer URLs), and <strong>TABULAR_POINTS</strong> (flat lat/lon "
        "records).<br><br>EPSG:4326 is assumed unless the source specifies otherwise."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers — parse sidebar settings
# ---------------------------------------------------------------------------


def _parse_headers() -> dict[str, str]:
    try:
        extra = json.loads(extra_headers_raw or "{}")
    except json.JSONDecodeError:
        st.sidebar.error("Extra headers must be valid JSON.")
        extra = {}
    return {"User-Agent": user_agent, **extra}


def _clean_crs(raw: str) -> str:
    cleaned = raw.strip()
    return cleaned if cleaned else "EPSG:4326"


# ---------------------------------------------------------------------------
# Conversion logic (called from Convert button handlers)
# ---------------------------------------------------------------------------


def _run_single(
    *,
    name: str,
    source: str = "",
    data: Optional[dict | list] = None,
    lat_field: Optional[str] = None,
    lon_field: Optional[str] = None,
    is_gdal_source: bool = False,
    is_wfs: bool = False,
    is_ogc: bool = False,
    is_osm: bool = False,
    is_xyz: bool = False,
    bbox: Optional[tuple[float, float, float, float]] = None,
    zoom: int = 12,
    tags: Optional[dict] = None,
    layer: Optional[str] = None,
    headers: dict,
    crs: str,
) -> dict:
    """Run convert() for one source; return a result dict for session state."""
    item_dir = os.path.join(
        st.session_state.output_dir, "items", _safe_basename(name)
    )
    try:
        if is_gdal_source:
            result: ConvertResult = convert_gdal_source(
                source_path=source,
                output_dir=item_dir,
                base_name=name,
                layer=layer,
                crs=crs,
            )
        elif is_wfs:
            result: ConvertResult = convert_wfs(
                url=source,
                output_dir=item_dir,
                base_name=name,
                layer=layer,
                crs=crs,
            )
        elif is_ogc:
            result: ConvertResult = convert_ogc(
                url=source,
                output_dir=item_dir,
                base_name=name,
                layer=layer,
                crs=crs,
            )
        elif is_osm:
            result: ConvertResult = convert_osm(
                place_name=source,
                tags=tags or {},
                output_dir=item_dir,
                base_name=name,
                crs=crs,
            )
        elif is_xyz:
            result: ConvertResult = convert_xyz_tiles(
                url_template=source,
                bbox=bbox or (-180, -90, 180, 90),
                zoom=zoom,
                output_dir=item_dir,
                base_name=name,
                crs=crs,
            )
        else:
            result: ConvertResult = convert(
                source=source,
                output_dir=item_dir,
                base_name=name,
                lat_field=lat_field,
                lon_field=lon_field,
                headers=headers,
                crs=crs,
                data=data,
            )
        with open(result.zip_path, "rb") as fh:
            zip_bytes = fh.read()
        return {"result": result, "name": name, "zip_bytes": zip_bytes}

    except LatLonDetectionError as exc:
        return {
            "needs_latlon": True,
            "name": name,
            "source": source,
            "data": data,
            "available_columns": exc.available_columns,
            "message": str(exc),
            "error": str(exc),
        }
    except UnreachableURLError as exc:
        return {"error": str(exc), "name": name, "error_type": "URL_ERROR"}
    except UnrecognizedShapeError as exc:
        return {"error": str(exc), "name": name, "error_type": "SHAPE_ERROR"}
    except ConverterError as exc:
        return {"error": str(exc), "name": name, "error_type": "CONVERTER_ERROR"}
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Unexpected error: {type(exc).__name__}: {exc}",
            "name": name,
            "error_type": "UNEXPECTED",
        }


def _run_all(sources: list[dict], headers: dict, crs: str) -> None:
    """Convert all sources, populate session state."""
    results = []
    needs_latlon = []

    progress = st.progress(0, text="Starting conversion…")
    total = len(sources)

    for idx, item in enumerate(sources):
        name = item["name"]
        progress.progress(
            (idx) / total,
            text=f"Converting {idx + 1}/{total}: {name}…",
        )

        # Check for previously saved lat/lon selections
        ll = st.session_state.latlon_selections.get(name, {})
        r = _run_single(
            name=name,
            source=item.get("source", ""),
            data=item.get("data"),
            lat_field=item.get("lat_field") or ll.get("lat"),
            lon_field=item.get("lon_field") or ll.get("lon"),
            is_gdal_source=item.get("is_gdal_source", False),
            is_wfs=item.get("is_wfs", False),
            is_ogc=item.get("is_ogc", False),
            is_osm=item.get("is_osm", False),
            is_xyz=item.get("is_xyz", False),
            bbox=item.get("bbox"),
            zoom=item.get("zoom", 12),
            tags=item.get("tags"),
            layer=item.get("layer"),
            headers=headers,
            crs=crs,
        )
        if r.get("needs_latlon"):
            needs_latlon.append(r)
        else:
            results.append(r)

    progress.progress(1.0, text="Done!")
    st.session_state.results = results
    st.session_state.needs_latlon = needs_latlon
    st.session_state.converted = True


def _build_batch_zip_bytes(results: list[dict], merge: bool, crs: str) -> Optional[bytes]:
    """Assemble a combined zip from successful results; return raw bytes."""
    success = [r for r in results if "result" in r]
    if not success:
        return None

    buf = io.BytesIO()

    if merge:
        gdfs = []
        for r in success:
            safe = _safe_basename(r["name"])
            item_dir = os.path.join(
                st.session_state.output_dir,
                "items",
                safe,
            )
            if os.path.exists(item_dir):
                for f in os.listdir(item_dir):
                    if f.endswith(".shp"):
                        gdfs.append(gpd.read_file(os.path.join(item_dir, f)))

        if not gdfs:
            return None

        merged = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=crs
        )
        merged_dir = os.path.join(st.session_state.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        
        # Merge could result in mixed geometries, so we must split them just like single converts
        geom_types = merged.geometry.geom_type.dropna().unique()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            if len(geom_types) <= 1:
                merged.to_file(os.path.join(merged_dir, "merged.shp"), driver="ESRI Shapefile")
                for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    p = os.path.join(merged_dir, "merged" + ext)
                    if os.path.exists(p):
                        zf.write(p, arcname="merged" + ext)
            else:
                for gtype in geom_types:
                    sub_gdf = merged[merged.geometry.geom_type == gtype]
                    if sub_gdf.empty:
                        continue
                    gtype_suffix = str(gtype).replace(" ", "")
                    sub_name = f"merged_{gtype_suffix}"
                    sub_gdf.to_file(os.path.join(merged_dir, sub_name + ".shp"), driver="ESRI Shapefile")
                    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                        p = os.path.join(merged_dir, sub_name + ext)
                        if os.path.exists(p):
                            zf.write(p, arcname=sub_name + ext)
    else:
        # zip-of-zips
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for r in success:
                zf.writestr(
                    f"{_safe_basename(r['name'])}.zip",
                    r["zip_bytes"],
                )

    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI: Header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="app-title">Shapefile Converter</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">'
    "Convert geospatial JSON (GeoJSON, metadata APIs, flat lat/lon tables) "
    "to ESRI Shapefile bundles ready for QGIS — locally, no data leaves your machine."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# UI: Input tabs
# ---------------------------------------------------------------------------

tab_file, tab_url = st.tabs(["📁 File Upload", "🔗 URL Input"])

sources_from_files: list[dict] = []
sources_from_urls: list[dict] = []

with tab_file:
    st.markdown(
        '<div class="section-heading">Upload JSON / GeoJSON files</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["json", "geojson", "gpkg", "kml", "kmz", "gpx", "dxf", "topojson", "sqlite", "gml", "fgb", "tab", "mif", "mid", "dgn", "csv", "tsv", "xlsx", "xls", "py", "xsd", "map", "dat", "id", "mvt", "pbf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        import tempfile
        import subprocess
        
        if "shared_tmp_dir" not in st.session_state:
            st.session_state.shared_tmp_dir = tempfile.mkdtemp()
            
        # First pass: save ALL uploaded files to the shared temporary directory
        for f in uploaded:
            f_path = os.path.join(st.session_state.shared_tmp_dir, f.name)
            if not os.path.exists(f_path):
                f.seek(0)
                with open(f_path, "wb") as out:
                    out.write(f.read())
                f.seek(0)

        name_counts: dict[str, int] = {}
        for f in uploaded:
            # Skip sidecar files so they aren't processed as primary geometries
            if f.name.lower().endswith((".xsd", ".map", ".dat", ".id")):
                continue
                
            stem = Path(f.name).stem
            if stem in name_counts:
                name_counts[stem] += 1
                stem = f"{stem}_{name_counts[stem]}"
            else:
                name_counts[stem] = 1

            if f.name.lower().endswith((".gpkg", ".kml", ".kmz", ".gpx", ".dxf", ".topojson", ".sqlite", ".gml", ".fgb", ".tab", ".mif", ".mid", ".dgn", ".mvt", ".pbf")):
                # GDAL Source: Must be read from disk, so we use the shared temp path
                tmp_path = os.path.join(st.session_state.shared_tmp_dir, f.name)
                
                if f.name not in st.session_state.gpkg_saved:
                    try:
                        layers = list_gdal_layers(tmp_path)
                        st.session_state.gpkg_saved[f.name] = {"path": tmp_path, "layers": layers}
                    except Exception as exc:
                        st.error(f"**{f.name}**: {exc}")
                        continue
                
                saved = st.session_state.gpkg_saved[f.name]
                st.markdown(f"**{f.name}** contains {len(saved['layers'])} layer(s):")
                layer_choice = st.selectbox(
                    "Select layer to convert", 
                    options=saved["layers"], 
                    key=f"gdal_layer_{f.name}",
                    label_visibility="collapsed"
                )
                
                sources_from_files.append({
                    "name": stem, 
                    "source": saved["path"], 
                    "is_gdal_source": True, 
                    "layer": layer_choice
                })
            elif f.name.lower().endswith((".csv", ".tsv", ".xlsx", ".xls")):
                import pandas as pd
                try:
                    if f.name.lower().endswith((".xlsx", ".xls")):
                        df = pd.read_excel(f)
                    else:
                        df = pd.read_csv(f, sep="\t" if f.name.lower().endswith(".tsv") else ",")
                    df = df.fillna("")
                    records = df.to_dict(orient="records")
                    sources_from_files.append({"name": stem, "data": records})
                except Exception as exc:
                    st.error(f"**{f.name}**: Failed to read tabular file - {exc}")
                    continue
            elif f.name.lower().endswith(".py"):
                # Run the Python script and capture standard output as JSON
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, f.name)
                with open(tmp_path, "wb") as out:
                    out.write(f.read())
                
                with st.spinner(f"Executing {f.name}..."):
                    try:
                        res = subprocess.run(
                            [sys.executable, tmp_path], 
                            cwd=tmp_dir,
                            capture_output=True, text=True, check=True
                        )
                        raw_bytes = res.stdout.encode('utf-8')
                        try:
                            parsed = load_from_bytes(raw_bytes, name=f.name)
                            sources_from_files.append({"name": stem, "data": parsed})
                        except ConverterError as exc:
                            # Fallback: Check if the script created a .json or .geojson file in its working dir
                            created_files = [
                                p for p in os.listdir(tmp_dir) 
                                if p.lower().endswith(('.json', '.geojson')) and p != f.name
                            ]
                            if created_files:
                                fallback_path = os.path.join(tmp_dir, created_files[0])
                                with open(fallback_path, "rb") as fallback_f:
                                    parsed = load_from_bytes(fallback_f.read(), name=created_files[0])
                                sources_from_files.append({"name": stem, "data": parsed})
                                st.info(f"**{f.name}** didn't print JSON, but it generated `{created_files[0]}` which was loaded instead.")
                            else:
                                st.error(f"**{f.name}** output parsing failed: {exc}")
                                continue
                    except subprocess.CalledProcessError as exc:
                        st.error(f"**{f.name}** script execution failed:\n```\n{exc.stderr}\n```")
                        continue
            else:
                raw_bytes = f.read()
                try:
                    parsed = load_from_bytes(raw_bytes, name=f.name)
                except ConverterError as exc:
                    st.error(f"**{f.name}**: {exc}")
                    continue

                sources_from_files.append({"name": stem, "data": parsed})

        if sources_from_files:
            st.success(
                f"{len(sources_from_files)} file(s) ready: "
                + ", ".join(f"**{s['name']}**" for s in sources_from_files)
            )

    convert_files_btn = st.button(
        "Convert Files",
        type="primary",
        disabled=not sources_from_files,
        key="btn_convert_files"
    )
with tab_url:
    api_mode = st.radio("API Type", ["Standard URLs (GeoJSON, WFS, OGC)", "OpenStreetMap (Overpass)", "XYZ Vector Tiles"], horizontal=True, label_visibility="collapsed")
    
    if api_mode == "OpenStreetMap (Overpass)":
        st.markdown(
            '<div class="section-heading">Search OpenStreetMap</div>',
            unsafe_allow_html=True,
        )
        st.info("Extract features by providing a city name in 'Location' and key-value OSM tags (e.g. amenity=cafe) in 'Tags' to filter the data.")
        osm_place = st.text_input("Location (e.g., 'Paris, France', 'Manhattan, NY')")
        osm_tags_raw = st.text_input("Tags (e.g., 'amenity=cafe', 'building=yes')")
        
        if osm_place and osm_tags_raw:
            tags_dict = {}
            for t in osm_tags_raw.split(","):
                if "=" in t:
                    k, v = t.split("=", 1)
                    tags_dict[k.strip()] = v.strip()
            
            slug = re.sub(r"[^A-Za-z0-9_\-]", "_", osm_place)[:40]
            sources_from_urls.append({
                "name": slug,
                "source": osm_place,
                "tags": tags_dict,
                "is_osm": True
            })
    elif api_mode == "XYZ Vector Tiles":
        st.markdown(
            '<div class="section-heading">Download Map Tiles</div>',
            unsafe_allow_html=True,
        )
        st.info("Download tiles by providing an XYZ URL with {x},{y},{z} parameters, bounding box coordinates, and a zoom level.")
        xyz_url = st.text_input("Tile URL Template (must contain {z}, {x}, {y})", placeholder="https://basemaps.arcgis.com/arcgis/rest/services/World_Basemap_v2/VectorTileServer/tile/{z}/{y}/{x}.pbf")
        st.markdown("**Bounding Box (WGS84)**")
        col1, col2, col3, col4 = st.columns(4)
        with col1: min_lon = st.number_input("Min Longitude", value=-74.02, step=0.01)
        with col2: min_lat = st.number_input("Min Latitude", value=40.70, step=0.01)
        with col3: max_lon = st.number_input("Max Longitude", value=-73.93, step=0.01)
        with col4: max_lat = st.number_input("Max Latitude", value=40.80, step=0.01)
        zoom = st.slider("Zoom Level", min_value=0, max_value=18, value=12)
        
        if xyz_url and "{x}" in xyz_url and "{y}" in xyz_url and "{z}" in xyz_url:
            slug = re.sub(r"[^A-Za-z0-9_\-]", "_", xyz_url.split("//")[-1])[:40]
            sources_from_urls.append({
                "name": slug,
                "source": xyz_url,
                "bbox": (min_lon, min_lat, max_lon, max_lat),
                "zoom": zoom,
                "is_xyz": True
            })
    else:
        st.markdown(
            '<div class="section-heading">Enter API URLs (one per line)</div>',
            unsafe_allow_html=True,
        )
        st.info("Paste direct URLs (one per line) to GeoJSON, TopoJSON, WFS, or OGC API endpoints to automatically fetch and convert.")
        url_text = st.text_area(
            "URLs",
            placeholder=(
                "https://api.geoboundaries.org/v3.0.0/gbOpen/AFG/ADM0/\n"
                "https://example.com/data.geojson\n"
                "https://demo.pygeoapi.io/master/collections"
            ),
            height=120,
            label_visibility="collapsed",
        )

        if url_text.strip():
            raw_urls = [u.strip() for u in url_text.strip().splitlines() if u.strip()]
            for url in raw_urls:
                # derive a clean name from the URL
                slug = re.sub(r"[^A-Za-z0-9_\-]", "_", url.split("//")[-1])[:40]
                
                if "wfs" in url.lower():
                    if url not in st.session_state.gpkg_saved:
                        try:
                            layers = list_wfs_layers(url)
                            st.session_state.gpkg_saved[url] = {"layers": layers}
                        except Exception as exc:
                            st.error(f"**{url}**: {exc}")
                            continue
                    
                    saved = st.session_state.gpkg_saved[url]
                    st.markdown(f"**WFS Source** contains {len(saved['layers'])} layer(s):")
                    layer_choice = st.selectbox(
                        "Select WFS layer", 
                        options=saved["layers"], 
                        key=f"wfs_layer_{slug}",
                        label_visibility="collapsed"
                    )
                    
                    sources_from_urls.append({
                        "name": slug or "wfs_source", 
                        "source": url, 
                        "is_wfs": True, 
                        "layer": layer_choice
                    })
                elif "/collections" in url.lower() or "oapif" in url.lower():
                    if url not in st.session_state.gpkg_saved:
                        try:
                            layers = list_ogc_collections(url)
                            st.session_state.gpkg_saved[url] = {"layers": layers}
                        except Exception as exc:
                            st.error(f"**{url}**: {exc}")
                            continue
                    
                    saved = st.session_state.gpkg_saved[url]
                    st.markdown(f"**OGC API Source** contains {len(saved['layers'])} collection(s):")
                    layer_choice = st.selectbox(
                        "Select OGC collection", 
                        options=saved["layers"], 
                        key=f"ogc_layer_{slug}",
                        label_visibility="collapsed"
                    )
                    
                    sources_from_urls.append({
                        "name": slug or "ogc_source", 
                        "source": url, 
                        "is_ogc": True, 
                        "layer": layer_choice
                    })
                else:
                    sources_from_urls.append({"name": slug or "url_source", "source": url})

        if sources_from_urls:
            st.success(
                f"{len(sources_from_urls)} URL(s) ready: "
                + ", ".join(f"`{s['source']}`" for s in sources_from_urls)
            )

    convert_urls_btn = st.button(
        "Convert URLs",
        type="primary",
        disabled=not sources_from_urls,
        key="btn_convert_urls",
    )

# ---------------------------------------------------------------------------
# Trigger conversion
# ---------------------------------------------------------------------------

_headers = _parse_headers()
_crs = _clean_crs(crs_override)

if convert_files_btn and sources_from_files:
    st.session_state.pending_sources = sources_from_files
    with st.spinner("Converting…"):
        _run_all(sources_from_files, _headers, _crs)
    st.rerun()

if convert_urls_btn and sources_from_urls:
    st.session_state.pending_sources = sources_from_urls
    with st.spinner("Fetching & converting…"):
        _run_all(sources_from_urls, _headers, _crs)
    st.rerun()

# ---------------------------------------------------------------------------
# UI: Results
# ---------------------------------------------------------------------------

if st.session_state.converted:
    results: list[dict] = st.session_state.results
    needs_latlon: list[dict] = st.session_state.needs_latlon

    st.markdown("---")
    st.markdown(
        '<div class="section-heading">Results</div>',
        unsafe_allow_html=True,
    )

    # Summary metrics
    n_ok = sum(1 for r in results if "result" in r)
    n_err = sum(1 for r in results if "error" in r and not r.get("needs_latlon"))
    n_ll = len(needs_latlon)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Converted", n_ok)
    col_m2.metric("Errors", n_err)
    col_m3.metric("Need Lat/Lon", n_ll)

    st.markdown("")  # spacer

    # ── Per-result cards ──────────────────────────────────────────────────

    def _type_badge(input_type: str) -> str:
        classes = {
            "GEOJSON_DIRECT":    "badge-geojson",
            "METADATA_REDIRECT": "badge-redirect",
            "TABULAR_POINTS":    "badge-tabular",
            "ESRI_JSON":         "badge-redirect",
            "PAGINATED_RESULTS": "badge-tabular",
            "GPKG":              "badge-gpkg",
            "WFS":               "badge-wfs",
        }
        labels = {
            "GEOJSON_DIRECT":    "GeoJSON Direct",
            "METADATA_REDIRECT": "Metadata Redirect",
            "TABULAR_POINTS":    "Tabular Points",
            "ESRI_JSON":         "ESRI / ArcGIS JSON",
            "PAGINATED_RESULTS": "Paginated API",
            "GPKG":              "GeoPackage",
            "WFS":               "Web Feature Service",
        }
        cls = classes.get(input_type, "badge-unknown")
        label = labels.get(input_type, input_type)
        return f'<span class="badge {cls}">{label}</span>'

    for r in results:
        name = r["name"]

        if "result" in r:
            res: ConvertResult = r["result"]

            # Column table HTML
            trunc = res.truncated_columns
            col_rows = ""
            for col in res.columns:
                orig = next(
                    (o for o, t in trunc.items() if t == col), col
                )
                changed = orig != col
                td_cls = ' class="col-changed"' if changed else ""
                orig_cell = f'<td style="color:#8b949e;font-style:italic">{orig}</td>' if changed else ""
                col_rows += (
                    f"<tr><td{td_cls}>{col}</td>"
                    + orig_cell
                    + "</tr>"
                )

            trunc_header = (
                "<th>Original Name</th>" if trunc else ""
            )
            col_table = (
                f'<table class="col-table"><thead><tr>'
                f"<th>Shapefile Column</th>{trunc_header}</tr></thead>"
                f"<tbody>{col_rows}</tbody></table>"
            )

            redirect_info = ""
            if res.redirect_url:
                redirect_info = (
                    f'<div class="info-banner">Followed redirect: '
                    f'<code style="font-size:0.75rem">{res.redirect_url[:80]}</code></div>'
                )

            warn_html = ""
            for w in res.column_warnings:
                warn_html += f'<div class="warn-banner">{w}</div>'

            st.markdown(
                f"""
<div class="result-card success">
  <div class="card-title">
    {name}
    {_type_badge(res.input_type)}
  </div>
  <div class="card-meta">Output: <code>{_safe_basename(name)}.zip</code></div>
  <div class="stat-row">
    <div class="stat-pill">Features <strong>{res.feature_count:,}</strong></div>
    <div class="stat-pill">Geometry <strong>{res.geometry_type}</strong></div>
    <div class="stat-pill">CRS <strong>{res.crs}</strong></div>
    <div class="stat-pill">Columns <strong>{len(res.columns)}</strong></div>
  </div>
  {redirect_info}
  {warn_html}
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander(f"Column details — {name}"):
                st.markdown(col_table, unsafe_allow_html=True)
                if res.truncated_columns:
                    st.caption(
                        f"{len(res.truncated_columns)} column name(s) were truncated "
                        "to fit the 10-character shapefile limit."
                    )

            st.download_button(
                label=f"Download {_safe_basename(name)}.zip",
                data=r["zip_bytes"],
                file_name=f"{_safe_basename(name)}.zip",
                mime="application/zip",
                key=f"dl_{name}",
            )
            st.markdown("")

        elif "error" in r and not r.get("needs_latlon"):
            st.markdown(
                f"""
<div class="result-card error">
  <div class="card-title">{name}</div>
  <div class="err-banner">
    <strong>{r.get("error_type", "Error")}:</strong> {r["error"]}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    # ── Lat/Lon manual selection ──────────────────────────────────────────

    if needs_latlon:
        st.markdown(
            '<div class="section-heading">Manual Lat/Lon Selection Required</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-banner">'
            "The following sources have tabular data but auto-detection could not "
            "identify latitude/longitude fields. Select them manually below, "
            "then click <strong>Retry</strong>."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        all_selected = True
        for item in needs_latlon:
            iname = item["name"]
            cols = item.get("available_columns", [])

            with st.expander(f"{iname} — pick lat/lon fields", expanded=True):
                st.markdown(
                    f'<div class="err-banner">{item["message"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                available = ["(auto-detect)"] + sorted(cols)
                prev = st.session_state.latlon_selections.get(iname, {})

                def _default_idx(opts, saved_key, default_list):
                    saved = prev.get(saved_key, "")
                    if saved and saved in opts:
                        return opts.index(saved)
                    return 0

                c1, c2 = st.columns(2)
                with c1:
                    sel_lat = st.selectbox(
                        "Latitude field",
                        options=available,
                        index=_default_idx(available, "lat", []),
                        key=f"sel_lat_{iname}",
                    )
                with c2:
                    sel_lon = st.selectbox(
                        "Longitude field",
                        options=available,
                        index=_default_idx(available, "lon", []),
                        key=f"sel_lon_{iname}",
                    )

                lat_val = None if sel_lat == "(auto-detect)" else sel_lat
                lon_val = None if sel_lon == "(auto-detect)" else sel_lon
                st.session_state.latlon_selections[iname] = {
                    "lat": lat_val,
                    "lon": lon_val,
                }
                if not lat_val or not lon_val:
                    all_selected = False

        retry_btn = st.button(
            "Retry with selected fields",
            type="primary",
            disabled=not all_selected,
            key="btn_retry_latlon",
        )

        if retry_btn:
            # Re-run only the needs_latlon items with user-selected fields
            retry_sources = []
            for item in needs_latlon:
                iname = item["name"]
                ll = st.session_state.latlon_selections.get(iname, {})
                retry_sources.append(
                    {
                        "name": iname,
                        "source": item.get("source", ""),
                        "data": item.get("data"),
                        "lat_field": ll.get("lat"),
                        "lon_field": ll.get("lon"),
                    }
                )
            with st.spinner("Retrying with selected lat/lon fields…"):
                # Run only the pending items
                _run_all(retry_sources, _headers, _crs)
                # Merge retry results into the pre-existing successful results
                existing_ok = [r for r in results if "result" in r]
                retry_results = st.session_state.results  # from _run_all above
                retry_needs_ll = st.session_state.needs_latlon
                # Combine: keep old successes + new successes/errors from retry
                st.session_state.results = existing_ok + retry_results
                st.session_state.needs_latlon = retry_needs_ll
            st.rerun()

    # ── Batch download ────────────────────────────────────────────────────

    success_results = [r for r in results if "result" in r]
    if len(success_results) > 1 or (len(success_results) >= 1 and st.session_state.merge_mode):
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Batch Download</div>',
            unsafe_allow_html=True,
        )

        mode_label = "merged shapefile" if st.session_state.merge_mode else "zip-of-zips"
        st.markdown(
            f'<div class="info-banner">'
            f"Download all {len(success_results)} successful conversion(s) as a single "
            f"<strong>{mode_label}</strong>."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        batch_bytes = _build_batch_zip_bytes(results, st.session_state.merge_mode, _crs)
        if batch_bytes:
            filename = (
                "merged_shapefile.zip"
                if st.session_state.merge_mode
                else "batch_shapefiles.zip"
            )
            st.download_button(
                label=f"Download All ({filename})",
                data=batch_bytes,
                file_name=filename,
                mime="application/zip",
                key="dl_batch",
            )

    # ── Clear / Reset ─────────────────────────────────────────────────────

    st.markdown("")
    if st.button("Clear results and start over", key="btn_clear"):
        for key in ["results", "needs_latlon", "latlon_selections", "converted", "pending_sources"]:
            st.session_state[key] = _DEFAULTS[key]
        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid rgba(48,54,61,0.5);
            text-align:center; font-size:0.72rem; color:#404862;">
  Shapefile Converter &nbsp;·&nbsp; Stateless local tool &nbsp;·&nbsp;
  All processing happens on your machine &nbsp;·&nbsp;
  Built with Streamlit · GeoPandas · Shapely · Fiona
</div>
""",
    unsafe_allow_html=True,
)
