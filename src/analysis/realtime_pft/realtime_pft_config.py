import datetime
import os
import pandas as pd
from pathlib import Path

from utils.constants import CACHE_BASE_DIR, get_repo_root
from analysis.realtime_pft.rrfs_para_oper_client import RRFSParallelOperClient

rave_lookback = 2
fxx_range = 24
fxx_interval = 1
max_workers = 48

VALID_CYCLES = [0, 6, 12, 18]
pft_oper_client = RRFSParallelOperClient

today_str = datetime.date.today().strftime("%Y_%m_%d")
now_dt = pd.Timestamp(datetime.date.today().strftime("%Y-%m-%d %H:%M:%S"))
manifest_path = os.path.join(CACHE_BASE_DIR, "active_fires", today_str, "fire_pipeline_manifest.csv")
pft_out_dir = Path(CACHE_BASE_DIR) / "pft_outputs"

inspyre_icon_path = f"{get_repo_root()}/src/utils/inspyre_icon.png"

BASE_DIR = os.path.join(CACHE_BASE_DIR, "realtime_pft", today_str)
NC_DIR = os.path.join(BASE_DIR, 'rrfs_soundings_nc')
GRIB_DIR = os.path.join(BASE_DIR, 'rrfs_grib')
RAVE_DIR = os.path.join(BASE_DIR, 'active_rave')
MANIFEST_DIR = os.path.join(BASE_DIR, 'active_fires')

REMOTE_DIR = "/srv/data/web/data-web/research/inspyre/realtime_pft"
USERNAME = 'jaredgoldman'
HOSTNAME = 'chaos.atmos.ucla.edu'