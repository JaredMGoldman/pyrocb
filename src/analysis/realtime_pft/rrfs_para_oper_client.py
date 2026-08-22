import os
import re
import sys
import time
import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import requests
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm as timer

from utils.constants import CACHE_BASE_DIR
from utils.rio_utils import download_file_safe

Geom = Union[Polygon, MultiPolygon]

CURL_HEADERS = (
    "-H 'Cache-Control: no-cache, no-store, must-revalidate' "
    "-H 'Pragma: no-cache' "
    "-H 'Expires: 0' "
    "-H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'"
)


# =============================================================================
# Spatial & Dataset Utility Functions
# =============================================================================

def infer_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    """Infers latitude and longitude coordinate or variable names from dataset."""
    lat_names = ("lat", "latitude", "LAT", "Latitude", "y", "grid_latt")
    lon_names = ("lon", "longitude", "LON", "Longitude", "x", "grid_lont")

    lat_var = next((n for n in lat_names if n in ds.coords or n in ds.variables), None)
    lon_var = next((n for n in lon_names if n in ds.coords or n in ds.variables), None)

    if lat_var is None or lon_var is None:
        raise KeyError(
            f"Could not infer lat/lon names. Found coords={list(ds.coords)} "
            f"and vars={list(ds.variables)[:30]}"
        )
    return lat_var, lon_var


def wrap_lons_to_180(ds: xr.Dataset) -> xr.Dataset:
    """Remaps longitudes from [0, 360] to (-180, 180] if necessary."""
    _, lon_name = infer_lat_lon_names(ds)
    lon = ds[lon_name]
    lonv = lon.values

    ds_is_360 = (np.nanmin(lonv) >= 0) and (np.nanmax(lonv) > 180)
    if ds_is_360:
        newlon = ((lonv + 180) % 360) - 180
        if lon.ndim == 1:
            ds = ds.assign_coords({lon_name: newlon}).sortby(lon_name)
        elif lon.ndim == 2:
            ds = ds.assign_coords({lon_name: (lon.dims, newlon)})
    return ds


def subset_dataset_to_bbox(ds: xr.Dataset, bbox: Tuple[float, float, float, float], buffer_deg: float = 0.15) -> xr.Dataset:
    """Subsets dataset to a bounding box (min_lon, min_lat, max_lon, max_lat) with padding."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_name, lon_name = infer_lat_lon_names(ds)
    
    min_lon -= buffer_deg
    max_lon += buffer_deg
    min_lat -= buffer_deg
    max_lat += buffer_deg

    if ds[lat_name].ndim == 1 and ds[lon_name].ndim == 1:
        return ds.sel(
            {
                lat_name: slice(min_lat, max_lat) if ds[lat_name][0] < ds[lat_name][-1] else slice(max_lat, min_lat),
                lon_name: slice(min_lon, max_lon)
            }
        )
    else:
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values
        mask = (
            (lat_vals >= min_lat) & (lat_vals <= max_lat) &
            (lon_vals >= min_lon) & (lon_vals <= max_lon)
        )
        return ds.where(xr.DataArray(mask, dims=ds[lat_name].dims), drop=True)


# =============================================================================
# Multiprocessing Worker Function
# =============================================================================

def _process_rrfs_sounding_worker(
    grib_path: Path, 
    fires: List[Dict], 
    output_dir: Path, 
    target_vars: List[str]
) -> List[Dict]:
    """
    Worker process: Opens downloaded GRIB2 file ONCE, extracts soundings
    for each fire boundary, and saves compressed NetCDF output files.
    """
    results = []
    if not grib_path.exists():
        return results

    try:
        ds = xr.load_dataset(
            grib_path,
            engine="cfgrib",
            filter_by_keys={'typeOfLevel': 'isobaricInhPa'},
            backend_kwargs={'errors': 'ignore'}
        )
        ds = wrap_lons_to_180(ds)

        available_vars = [v for v in target_vars if v in ds.data_vars]
        if available_vars:
            ds = ds[available_vars]

        for fire in fires:
            fire_id = fire['fire_index_id']
            fire_name = fire.get('fire_name', fire_id)
            out_fname = output_dir / f"rrfs_sounding_{fire_id}_{grib_path.stem}.nc"
            
            if os.path.exists(out_fname):
                results.append({
                    "fire_index_id": fire_id,
                    "fire_name": fire_name,
                    "grib_file": grib_path.name,
                    "output_nc": str(out_fname),
                    "status": "SUCCESS"
                })
                continue

            geom = wkt.loads(fire['wkt_geometry'])
            bounds = geom.bounds  # (minx, miny, maxx, maxy)

            try:
                subset = subset_dataset_to_bbox(ds, bounds, buffer_deg=0.15)
                spatial_dims = [d for d in subset.dims if d != 'isobaricInhPa']
                ds_compressed = subset.stack(spatial_point=spatial_dims)
                
                if available_vars:
                    ds_compressed = ds_compressed.dropna(dim="spatial_point", how="all", subset=available_vars)
                ds_compressed = ds_compressed.reset_index('spatial_point')
                # Add fire metadata directly to the xarray attributes
                ds_compressed.attrs['fire_index_id'] = str(fire_id)
                ds_compressed.attrs['fire_name'] = str(fire_name)

                ds_compressed.to_netcdf(out_fname)

                results.append({
                    "fire_index_id": fire_id,
                    "fire_name": fire_name,
                    "grib_file": grib_path.name,
                    "output_nc": str(out_fname),
                    "status": "SUCCESS"
                })
            except Exception as e:
                results.append({
                    "fire_index_id": fire_id,
                    "fire_name": fire_name,
                    "grib_file": grib_path.name,
                    "error": str(e),
                    "status": "FAILED"
                })

        ds.close()
    except Exception as e:
        print(f"Failed to process file {grib_path.name}: {e}", file=sys.stderr)

    return results


# =============================================================================
# Primary Operations Client
# =============================================================================

class RRFSParallelOperClient:
    """Pipeline to discover, download, and extract fire soundings into NetCDF files."""

    BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/para"
    VALID_CYCLES = ["18", "12", "06", "00"]

    def __init__(
        self, 
        csv_manifest_path: str, 
        download_dir: Optional[str] = None, 
        output_dir: Optional[str] = None,
        target_vars: Optional[List[str]] = None
    ):
        self.csv_path = Path(csv_manifest_path)
        
        today_str = datetime.date.today().strftime("%Y%m%d")
        self.download_dir = Path(download_dir) if download_dir else Path(CACHE_BASE_DIR) / "rrfs_para_downloads" / today_str
        self.output_dir = Path(output_dir) if output_dir else Path(CACHE_BASE_DIR) / "rrfs_soundings_nc" / today_str
        
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_vars = target_vars or ['t', 'r', 'gh', 'u', 'v']
        self._load_manifest()
        self.session = requests.Session()

    def _load_manifest(self):
        """Loads and parses fire manifest CSV."""
        print(f"[*] Loading fire manifest: {self.csv_path}")
        df = pd.read_csv(self.csv_path).dropna(subset=['wkt_geometry', 'fire_index_id'])
        if 'fire_name' not in df.columns:
            df['fire_name'] = df['fire_index_id']
        self.df_manifest = df
        self.serialized_fires = df[['fire_index_id', 'fire_name', 'wkt_geometry']].to_dict(orient='records')

    def find_latest_available_cycle(self, max_lookback_days: int = 2) -> Tuple[str, str]:
        """Queries NOMADS directory listing to locate latest available date and cycle [00, 06, 12, 18] UTC."""
        now = datetime.datetime.utcnow()
        for day_offset in range(max_lookback_days + 1):
            target_date = (now - datetime.timedelta(days=day_offset)).strftime("%Y%m%d")
            
            for cycle in self.VALID_CYCLES:
                check_url = f"{self.BASE_URL}/rrfs.{target_date}/{cycle}/"
                cache_buster_url = f"{check_url}?_={int(time.time())}"
                cmd = f"curl -s -L {CURL_HEADERS} '{cache_buster_url}'"
                
                try:
                    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                    if "rrfs.t" in res.stdout and ".prslev." in res.stdout:
                        print(f"[+] Located latest available cycle: Date={target_date}, Cycle={cycle}Z")
                        return target_date, cycle
                except subprocess.CalledProcessError:
                    continue

        raise FileNotFoundError("Could not find any available RRFS cycles on NOMADS within lookback period.")

    def download_latest_prslev_data(self, fxx_hours: List[int] = [0, 2, 6]) -> Tuple[List[Path], str, str]:
        """Downloads pressure level forecast GRIB2 files for target cycle."""
        target_date, cycle = self.find_latest_available_cycle()
        downloaded_paths = []

        for fxx in fxx_hours:
            fxx_str = f"{fxx:03d}"
            fname = f"rrfs.t{cycle}z.prslev.13km.f{fxx_str}.na.grib2"
            file_url = f"{self.BASE_URL}/rrfs.{target_date}/{cycle}/{fname}"
            local_path = self.download_dir / fname

            if local_path.exists():
                print(f"[*] File already exists locally: {fname}")
                downloaded_paths.append(local_path)
                continue

            print(f"[+] Downloading: {fname}")
            try:
                download_file_safe(file_url, local_path, self.session)
                downloaded_paths.append(local_path)
            except Exception as e:
                print(f"[-] Failed downloading {fname}: {e}", file=sys.stderr)

        return downloaded_paths, target_date, cycle

    def process_fires_in_parallel(self, grib_paths: List[Path], max_workers: int = 4) -> Path:
        """Executes parallel spatial extraction across downloaded GRIB files into NetCDFs."""
        if not grib_paths:
            print("[-] No valid GRIB files available for processing.")
            return self.output_dir / "processing_summary.csv"

        print(f"[*] Extracting soundings for {len(self.serialized_fires)} fires using {max_workers} processes...")
        
        all_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_rrfs_sounding_worker, 
                    path, 
                    self.serialized_fires, 
                    self.output_dir, 
                    self.target_vars
                ): path for path in grib_paths
            }

            for future in timer(as_completed(futures), desc="RRFS NetCDF Extraction", total=len(futures)):
                path = futures[future]
                try:
                    res = future.result()
                    all_results.extend(res)
                except Exception as e:
                    print(f"[-] Exception executing process for {path.name}: {e}")

        summary_df = pd.DataFrame(all_results)
        summary_csv = self.output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[*] Extraction complete! NetCDF manifest saved to: {summary_csv}")
        return summary_csv


if __name__ == "__main__":
    today_str = datetime.date.today().strftime("%Y_%m_%d")
    manifest_path = os.path.join(CACHE_BASE_DIR, "active_fires", today_str, "fire_pipeline_manifest.csv")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(CACHE_BASE_DIR, "active_fires", "2026_08_20", "fire_pipeline_manifest.csv")

    client = RRFSParallelOperClient(csv_manifest_path=manifest_path)
    grib_files, date, cycle = client.download_latest_prslev_data(fxx_hours=[0])
    client.process_fires_in_parallel(grib_files, max_workers=4)