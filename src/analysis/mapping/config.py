# config.py
import matplotlib.colors as mcolors
from datetime import datetime
import os
from os.path import join as os_join
import pandas as pd

from analysis.mapping.active_incident_map import ActiveFirePerimeterPipeline
from analysis.mapping.firms_pft_pipeline import FirmsPftLandmaskedPipeline
from analysis.mapping.veda_pft_pipeline import VedaPftLandmaskedPipeline

from data.clients.gfs_client import GFSClient, GFS
from data.clients.rrfs_client import RRFSClient, RRFS
from utils.constants import CACHE_BASE_DIR

def make_n_len_str(value, n=2):
    return f"%0{n}d" % (int(value),)

DEBUG_MODE = False

# Network & Server Options
SERVER_PORT = 8650

now_dt = pd.Timestamp("2026-06-01") if DEBUG_MODE else pd.Timestamp(datetime.now())
forecast_time = "06:00"
date_name = f"{now_dt.year}-{make_n_len_str(now_dt.month)}-{make_n_len_str(now_dt.day)}"
now_date = f"{now_dt.year}-{make_n_len_str(now_dt.month)}-{make_n_len_str(now_dt.day)} {forecast_time}"
date_str = f"{now_dt.year}-{now_dt.month}-{now_dt.day}_06Z"

rave_lookback = 3
fxx_range = 48 if DEBUG_MODE else 72
fxx_freq = 2
plot_freq = 6
clients =  [GFSClient] if DEBUG_MODE else [RRFSClient]
fx_names = [GFS] if DEBUG_MODE else [RRFS]
active_fire_class = ActiveFirePerimeterPipeline
max_workers = 48

cmap_name = 'viridis'

current_dir = os_join(CACHE_BASE_DIR, 'active_fires', 'current')
today_dir = os_join(CACHE_BASE_DIR, 'active_fires', date_name.replace('-','_'))
rave_cache = os_join(today_dir, '.rave_downloads')

html_fname = f"pft_fire_map_{date_str}.html"
active_rave_fn = 'active_rave_timeseries.csv'
active_fire_fname = 'fire_pipeline_manifest.csv'
can_frp_fname = 'fire_predictions_timeseries.csv'
snd_cache_fn = "sounding_pipeline_cache.db"
pft_fname = "pft_data.csv"

os.makedirs(today_dir, exist_ok=True)
os.makedirs(current_dir, exist_ok=True)

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
    'west': -140.0, 'east': -50.0, 'south': 24.0, 'north': 75.0
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

REMOTE_DIR = "/srv/data/web/data-web/research/inspyre/pft"
USERNAME = 'jaredgoldman'
HOSTNAME = 'chaos.atmos.ucla.edu'

# Model Normalization Configurations
def get_log_norm(vmin, vmax):
    return mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)