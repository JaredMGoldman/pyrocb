import datetime
import os
from pathlib import Path

from utils.constants import CACHE_BASE_DIR, get_repo_root
from analysis.realtime_pft.rrfs_para_oper_client import RRFSParallelOperClient

fxx_range = 24
fxx_interval = 1
max_workers = 48

pft_oper_client = RRFSParallelOperClient

today_str = datetime.date.today().strftime("%Y_%m_%d")
manifest_path = os.path.join(CACHE_BASE_DIR, "active_fires", today_str, "fire_pipeline_manifest.csv")
pft_out_dir = Path(CACHE_BASE_DIR) / "pft_outputs"

inspyre_icon_path = f"{get_repo_root()}/src/utils/inspyre_icon.png"

REMOTE_DIR = "/srv/data/web/data-web/research/inspyre/realtime_pft"
USERNAME = 'jaredgoldman'
HOSTNAME = 'chaos.atmos.ucla.edu'