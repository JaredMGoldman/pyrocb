from __future__ import annotations

from utils.constants import *

import numpy as np
import os
import pandas as pd
from joblib import dump, load
from pathlib import Path
from shapely.geometry import Polygon
import time
from typing import Tuple
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
    save_dir: str,
    name: str,
    fmt: str = "joblib",
    add_timestamp: bool = False,
) -> Path:
    """
    Save the current model to <repo_root>/<subdir>/<name>.<fmt>.
    Returns the saved path.
    """ 

    if add_timestamp:
        name = f"{name}_{time.strftime('%Y%m%d-%H%M')}"

    path = f"{save_dir}/{name}.{fmt}"

    dump(model, path)
    print(f"model saved to {path}")
    return path

def save_feature_names(feat_list: list, out_dir: str, exp_name: str):
    """
    saves the list of features for a model to the given directory or the endlosing directory if a
    filename is provided

    :param feat_list: list to save
    :param out_dir: directory to save list to
    :param exp_name: name of experiment to label list
    """
    if os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir)
    path = os.path.join(out_dir, f"{exp_name}_features.txt")
    with open(path, 'w') as f:
        for item in feat_list:
            f.write(f"{item}\n")
    return path

def load_feature_names(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"features file {path} not found")
    with open(path, 'r') as f:
        out_list = [line.strip() for line in f]
    return out_list

def save_features(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                  X_test: pd.DataFrame, y_test: pd.DataFrame, 
                  out_dir: str, exp_name: str) -> str: 
    """
    saves the features for a model run at the location provided

    :param X_train: training features dataframe
    :param y_train: training labels dataframe
    :param X_test: testing features dataframe
    :param y_test: testing labels dataframe
    :param out_dir: directory to save list to
    :param exp_name: name of experiment to label list
    """
    if os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir)

    [df.to_csv(os.path.join(out_dir, fname), index = False) for df, fname in 
        zip([X_train, y_train, X_test, y_test],
            [f"train_data_{exp_name}.csv", f"train_labels_{exp_name}.csv", 
                f"test_data_{exp_name}.csv", f"test_labels_{exp_name}.csv"])]
    return out_dir

def load_features(dir_name, exp_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = pd.read_csv(f"{dir_name}/train_data_{exp_name}.csv")
    y_train = pd.read_csv(f"{dir_name}/train_labels_{exp_name}.csv")
    X_test = pd.read_csv(f"{dir_name}/test_data_{exp_name}.csv")
    y_test = pd.read_csv(f"{dir_name}/test_labels_{exp_name}.csv")
    return X_train, y_train, X_test, y_test

def load_model(fname, fmt = "joblib"):
    path = f"{fname}.{fmt}"
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