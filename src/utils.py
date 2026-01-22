from __future__ import annotations

import joblib
from pathlib import Path
import subprocess
import sklearn
import time
import re
import matplotlib.pyplot as plt

PLOTS_DIR = "outputs/plots"
MODELS_DIR = "outputs/models"

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


def slugify(name: str) -> str:
    # safe filename
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9\-_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def get_dir(subdir) -> Path:
    d = get_repo_root() / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_plot(
    name: str,
    dpi: int = 200,
    fmt: str = "png",
    add_timestamp: bool = True,
    tight: bool = False,
) -> Path:
    """
    Save the current matplotlib figure to <repo_root>/<subdir>/<name>.<fmt>.
    Returns the saved path.
    """
    plot_dir = get_dir(PLOTS_DIR)
    base = slugify(name)

    if add_timestamp:
        base = f"{base}_{time.strftime('%Y%m%d-%H%M')}"

    path = plot_dir / f"{base}.{fmt}"

    if tight:
        plt.tight_layout()

    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path

def save_model(
    model: sklearn.ensemble,
    name: str,
    fmt: str = "joblib",
    add_timestamp: bool = True,
) -> Path:
    """
    Save the current sklearn model to <repo_root>/<subdir>/<name>.<fmt>.
    Returns the saved path.
    """
    model_dir = get_dir(subdir=MODELS_DIR)
    base = slugify(name)

    if add_timestamp:
        base = f"{base}_{time.strftime('%Y%m%d-%H%M')}"

    path = model_dir / f"{base}.{fmt}"

    joblib.dump(model, path)
    return path