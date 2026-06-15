# config.py
import matplotlib.colors as mcolors
from datetime import datetime
import pandas as pd

from data.clients.gfs_client import GFSClient, GFS
from utils.constants import CACHE_BASE_DIR

# Network & Server Options
SERVER_PORT = 8650
now_dt = pd.Timestamp(datetime.now())
now_date = "2026-06-08"
fxx_range = 48
clients = [GFSClient]
fx_names = [GFS]

max_workers = 10

date_str = f"{now_dt.year}{now_dt.month}{now_dt.day}"

OUTPUT_HTML = f"{CACHE_BASE_DIR}/folium/weekly_fire_map_{date_str}.html"
CSV_MANIFEST = f"{CACHE_BASE_DIR}/folium/fire_pipeline_manifest_{date_str}.csv"

# Stable Asset CDNs to prevent Leaflet/Jinja2 freezing bugs
TIMEDIMENSION_ASSETS = {
    'js_links': [
        'https://cdn.jsdelivr.net/npm/iso8601-js-period@0.2.1/iso8601.min.js',
        'https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.min.js'
    ],
    'css_links': [
        'https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.min.css'
    ]
}

# Spatial Mapping Extents [West Lon, East Lon, South Lat, North Lat]
MAP_BOUNDS = {
    'west': -135.0, 'east': -40.0, 'south': 24.0, 'north': 65.0
}

bounds = [MAP_BOUNDS["west"], MAP_BOUNDS['east'], MAP_BOUNDS["south"], MAP_BOUNDS['north']]

lons = [MAP_BOUNDS["west"], MAP_BOUNDS['east']]
lats = [MAP_BOUNDS["south"], MAP_BOUNDS['north']]

# VEDA Ingestion Regions
VEDA_REGIONS = {
    "western_us_ca": "-125,32,-114,42",   
    "pacific_nw_bc": "-130,42,-110,60",   
    "boreal_canada": "-120,50,-80,70",    
    "eastern_na": "-90,30,-60,55",
    "central_us": "-115,30,-90,50"
}

VEDA_BASE_URL = "https://openveda.cloud/api/features"

# Model Normalization Configurations
def get_log_norm(vmin, vmax):
    return mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)