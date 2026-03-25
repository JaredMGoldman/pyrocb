import os
from utils.utils import get_repo_root
from pathlib import Path

ML_DATA_ROOT = os.path.join(f"{os.sep}data","lthapa","data2restore","lthapa","ML_daily")

OUTPUTS_DIR = os.path.join(get_repo_root(), "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
ML_FEATS_DIR = os.path.join(MODELS_DIR, "features")
FEATURE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "features")
LOG_DIR = os.path.join(OUTPUTS_DIR, 'logs')
DATA_DIR = os.path.join(get_repo_root(), "src", "data")
CLIENTS_DIR = os.path.join(DATA_DIR,"clients")
CACHE_DIR = os.path.join(CLIENTS_DIR,"cache")
CACHE_BASE_DIR = Path(f"{os.environ.get('HOME')}/data/cache") # Path(f"{os.environ.get('SCRATCH')}/data/cache")
FIRMS_KEY_FNAME = "firms.key"
EARTHACCESS_KEY_NAME = "earthaccess.netrc"