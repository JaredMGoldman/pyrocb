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

dry_run_cps = [80, 21, 36, 903, 889, 45, 94]

OUTPUTS_DIR = os.path.join(get_repo_root(), "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
ML_FEATS_DIR = os.path.join(MODELS_DIR, "features")
FEATURE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "features")
LOG_DIR = os.path.join(OUTPUTS_DIR, 'logs')

DATA_DIR = os.path.join(get_repo_root(), "src", "data")
CP_POLY_PATH = os.path.join(DATA_DIR,"cp_poly.gpkg")
CP_IDX_PATH = os.path.join(DATA_DIR,"cp_na.csv")

CLIENTS_DIR = os.path.join(DATA_DIR,"clients")
CACHE_DIR = os.path.join(CLIENTS_DIR,"cache")
CACHE_BASE_DIR = Path(f"{os.environ.get('HOME')}/data/cache") # Path(f"{os.environ.get('SCRATCH')}/data/cache")

FIRMS_KEY_FNAME = "firms.key"
EARTHACCESS_KEY_NAME = "earthaccess.netrc"