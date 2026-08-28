import os
from pathlib import Path
import subprocess

def get_repo_root() -> Path:
    """
    Return the root directory of the current git repository.

    Works when executed anywhere inside the repo.
    Raises RuntimeError if not in a git repo.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return Path(out)
    except Exception as e:
        raise RuntimeError("Not inside a git repository (git rev-parse failed).") from e

dry_run_cps_0 = [80, 21, 36, 903, 889, 45, 94]
dry_run_cps_1 = [754, 36, 470, 458, 1228]
dry_run_cps = [82, 1623, 975, 10699, 780, 2205, 10127]
dry_run_names = ['Wiley Flat', 'Fish Creek', 'Magruder Ridge', 'Line', 'Flat Top', 'Firestone', 'Wye']
dry_run_date = "09-08-2024"
dry_run_map = {cp: name for cp, name in zip(dry_run_cps, dry_run_names)}


DATA_BASE = os.path.join("/data",f"{os.environ['USER']}")
if not os.path.exists(DATA_BASE):
    print(f"created personal data directory at {DATA_BASE}")
    os.makedirs(DATA_BASE)

DATA_DIR = os.path.join(get_repo_root(), "src", "data")
CP_POLY_PATH = os.path.join(DATA_BASE,"cp_poly.gpkg")
CP_IDX_PATH = os.path.join(DATA_BASE,"cp_na.csv")

CLIENTS_DIR = os.path.join(DATA_DIR,"clients")
CACHE_BASE_DIR = Path("/data/jaredgoldman/cache") # Path(f"{os.environ.get('SCRATCH')}/data/cache")
RAVE_CACHE = Path("/data/jaredgoldman/RAVE")

OUTPUTS_DIR = os.path.join(DATA_BASE, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
ML_FEATS_DIR = os.path.join(MODELS_DIR, "features")
FEATURE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "features")
LOG_DIR = os.path.join(OUTPUTS_DIR, 'logs')

FIRMS_KEY_FNAME = "firms.key"
EARTHACCESS_KEY_NAME = "earthaccess.netrc"

RRFS = 'RRFS'
GFS = 'GFS'
ECMWF = 'ECMWF'
NAM = 'NAM'
ERA5 = 'ERA5'
DEM = 'DEM'
FABDEM = 'FABDEM'