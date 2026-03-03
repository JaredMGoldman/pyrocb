from __future__ import annotations

import numpy as np
import pandas as pd
import os
import geopandas as gpd
from os.path import exists
from joblib import dump, Parallel, delayed
from pathlib import Path
from shapely.geometry import Polygon, Point, box
import subprocess
import time
import re
import matplotlib.pyplot as plt
import xarray as xr
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

def add_timestamp(input_str):
    return f"{input_str}_{time.strftime('%Y%m%d-%H%M')}"

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
    return path


def build_one_gridcell(LAT_COR, LON_COR, LAT_CTR, LON_CTR, loc):
    ii=int(loc[0])
    jj=int(loc[1])

    sw = (LON_COR[ii, jj],LAT_COR[ii, jj]) #SW
    se =(LON_COR[ii, jj+1],LAT_COR[ii, jj+1]) #SE
    nw = (LON_COR[ii+1, jj],LAT_COR[ii+1, jj]) #NW
    ne = (LON_COR[ii+1, jj+1],LAT_COR[ii+1, jj+1]) #NE
            
    poly_cell = Polygon([sw,nw,ne,se])

    return LAT_CTR[ii,jj], LON_CTR[ii,jj],ii,jj,poly_cell


def calculate_intersection(poly,dataset_name,bf):
    #load in the merra grid
    grid = xr.open_dataset(dataset_name+'.nc')

    bounds = poly.buffer(bf).to_crs(epsg=4326).bounds

    #first check for rows and cols, filtering near the polygon
    [rows,cols] = np.where((grid.LAT_CTR.values>bounds['miny'].values)&
                    (grid.LAT_CTR.values<bounds['maxy'].values)&
                    (grid.LON_CTR.values>bounds['minx'].values)&
                    (grid.LON_CTR.values<bounds['maxx'].values))
    locs = zip(rows,cols)

    #make a geodataframe (in paralell of the rows and cols)
    results = Parallel(n_jobs=8)(delayed(build_one_gridcell)
                                 (grid['LAT_COR'].values, grid['LON_COR'].values,
                                  grid['LAT_CTR'].values, grid['LON_CTR'].values,loc) 
                                 for loc in locs)

    #format the grid subset into a dataframs
    df_grid=gpd.GeoDataFrame(results)
    df_grid.columns = ['lat', 'lon', 'row', 'col', 'geometry']
    df_grid.set_geometry(col='geometry',inplace=True,crs='EPSG:4326') #need to say it's in lat/lon before transform to LCC
    df_grid=df_grid.to_crs(epsg=3347)

    #intersect the polygon with the grid subset
    intersection = gpd.overlay(df_grid, poly, how='intersection',keep_geom_type=False).drop_duplicates()
    intersection['grid intersection area (ha)'] =intersection['geometry'].area/10000
    intersection['weights'] = intersection['grid intersection area (ha)']/intersection['fire area (ha)'] 
    return intersection


#LAT and LON are 2d arrays
def calculate_grid_cell_corners(LAT, LON):
    #we will assume the very edges of the polygons don't touch the boundary of the domain
    lat_corners = (LAT[0:(LAT.shape[0]-1),  0:(LAT.shape[1])-1] + LAT[1:(LAT.shape[0]), 1:(LAT.shape[1])])/2
    lon_corners = (LON[0:(LAT.shape[0]-1),  0:(LAT.shape[1])-1] + LON[1:(LAT.shape[0]), 1:(LAT.shape[1])])/2
    return lat_corners, lon_corners


def make_file_namelist(time,base_filename):
    filename_list = np.array([])
    times_back_used = np.array([])
    for jj in range(len(time)):
        fname = base_filename.replace('YYYY',time[jj].strftime('%Y')).\
                                replace('MM',time[jj].strftime('%m')).\
                                replace('DD',time[jj].strftime('%d')).\
                                replace('HH',time[jj].strftime('%H')).\
                                replace('JJJ',time[jj].strftime('%j'))
        if exists(fname):
            filename_list = np.append(filename_list,fname)
            times_back_used = np.append(times_back_used,time[jj])
    return filename_list, times_back_used

def generate_df(variables, length):
    df = pd.DataFrame()
    for vv in variables:
        df[vv] = np.zeros(length)
    return df

def parallel_intersection_labels(df, label):
    intersections = Parallel(n_jobs=10)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],label,2000) 
                                 for ii in range(len(df)))

    intersection=gpd.GeoDataFrame(pd.concat(intersections, ignore_index=True))
    intersection.set_geometry(col='geometry')
    intersection = intersection.set_index(['12Z Start Day', 'lat', 'lon'])
    intersection = intersection[~intersection.index.duplicated()]
    intersection_xr = intersection.to_xarray()    
    intersection_xr['weights_mask'] = xr.where(intersection_xr['weights']>0,1, np.nan)
   
    return intersection_xr

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
    
ML_DATA_ROOT = os.path.join(f"{os.sep}data","lthapa","data2restore","lthapa","ML_daily")

OUTPUTS_DIR = os.path.join(get_repo_root(), "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
FEATURE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "features")
DATA_DIR = os.path.join(get_repo_root(), "src", "data")
CLIENTS_DIR = os.path.join(DATA_DIR,"clients")
CACHE_DIR = os.path.join(CLIENTS_DIR,"cache")
CACHE_BASE_DIR = Path(f"{os.environ.get('HOME')}/data/cache") # Path(f"{os.environ.get('SCRATCH')}/data/cache")
FIRMS_KEY_FNAME = "firms.key"