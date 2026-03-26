from __future__ import annotations

from utils.constants import *

import numpy as np
import os
from joblib import dump, load
from pathlib import Path
from shapely.geometry import Polygon
import time
import re
import matplotlib.pyplot as plt
import rasterio
import pyproj
from datetime import datetime, timezone
import uuid
from shapely.ops import transform
import pyproj

def make_run_id() -> str:
    # Example: run_20260226T235901Z_5f2c9c3a0b8c4d8aa2a1a0f5b7b20d3e
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    u = uuid.uuid4().hex
    return f"{ts}_{u}"

def make_cache_dir(base: Path) -> Path:
    run_id = make_run_id()
    pid = os.getpid()
    d = base / f"{run_id}_pid{pid}"
    d.mkdir(parents=True, exist_ok=False)
    return d

def set_env_var(var_name, key_file):
    """
    Sets the environment variable specified by var_name to the value 
    from the key_file
    
    :param var_name: name of environment variable
    :param key_file: path to file containing key value

    :returns value from key file
    """
    with open(key_file, 'r') as f:
        var_val = f.read()
    os.environ[var_name] = var_val
    return var_val


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

def add_timestamp(input_str):
    return f"{input_str}_{time.strftime('%Y%m%d-%H%M')}"

def save_plot(
    name: str,
    dpi: int = 200,
    fmt: str = "png",
    ts_bool: bool = True,
    tight: bool = False,
) -> Path:
    """
    Save the current matplotlib figure to <repo_root>/<subdir>/<name>.<fmt>.
    Returns the saved path.
    """
    plot_dir = get_dir(PLOTS_DIR)
    base = slugify(name)

    if ts_bool:
        base = add_timestamp(base)

    path = plot_dir / f"{base}.{fmt}"

    if tight:
        plt.tight_layout()

    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def save_model(
    model,
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

    dump(model, path)
    print(f"model saved to {path}")
    return path

def load_model(fname):
    path = os.path.join(MODELS_DIR, f"{fname}.joblib")
    return load(path)

def add_lonlat_coords(ds):
    """
    Takes a rasterio formatted xarray Dataset with x and y projection values instead of lat/lon
    and returns a Dataset with lat/lon coordinates in addition to x/y vals
    
    :param ds: Description
    """
    transform = ds.rio.transform()
    ny, nx = ds.sizes["y"], ds.sizes["x"]

    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    proj = pyproj.Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    lon, lat = proj.transform(xs, ys)
    lon = lon.reshape(ds.y.size,ds.x.size)
    lat = lat.reshape(ds.y.size,ds.x.size)
    
    return ds.assign_coords(
        longitude=(("y","x"), lon),
        latitude=(("y","x"), lat)
    )

def buffer_polygon_meters(
    geom: Polygon,
    resolution_m: float,
    *,
    factor: float = 0.5,
    cap_style: int = 1,
    join_style: int = 1,
) -> Polygon:
    """
    Buffer a lon/lat (EPSG:4326) Polygon/MultiPolygon by (factor * resolution_m) meters.

    Parameters
    ----------
    geom : shapely Polygon or MultiPolygon
        Geometry in EPSG:4326 (lon/lat degrees).
    resolution_m : float
        Dataset resolution in meters (e.g., 3000 for 3km grid).
    factor : float
        Multiply resolution_m by this factor for the buffer distance.
        - Use 0.5 to include points whose *centers* are within half a cell of the polygon boundary.
        - Use 1.0 to be more conservative (include full-cell neighborhood).
    cap_style : int
        1=round, 2=flat, 3=square (shapely buffer cap style).
    join_style : int
        1=round, 2=mitre, 3=bevel (shapely buffer join style).

    Returns
    -------
    shapely geometry
        Buffered geometry back in EPSG:4326.
    """
    if geom.is_empty:
        return geom

    dist_m = float(resolution_m) * float(factor)

    # Choose a local UTM zone based on centroid (good default for buffering)
    lon0, lat0 = geom.centroid.x, geom.centroid.y
    zone = int((lon0 + 180) // 6) + 1
    epsg = 32600 + zone if lat0 >= 0 else 32700 + zone  # WGS84 UTM north/south

    # Project to meters, buffer, project back
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    to_ll  = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform

    geom_utm = transform(to_utm, geom)
    geom_buf_utm = geom_utm.buffer(dist_m, cap_style=cap_style, join_style=join_style)
    geom_buf_ll = transform(to_ll, geom_buf_utm)

    return geom_buf_ll