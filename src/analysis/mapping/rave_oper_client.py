import os
import re
import datetime
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import xarray as xr
import rioxarray
from shapely import wkt, buffer, points, contains, distance
from shapely.geometry import Polygon, MultiPolygon
import time
from tqdm import tqdm as timer
from typing import Tuple, Union

from utils.io_utils import CACHE_BASE_DIR

Geom = Union[Polygon, MultiPolygon]

HEADERS = {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
}


def _infer_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat_names: Tuple[str, ...] = ("lat", "latitude", "LAT", "Latitude", "y", "grid_latt")
    lon_names: Tuple[str, ...] = ("lon", "longitude", "LON", "Longitude", "x", "grid_lont")

    lat = next((n for n in lat_names if n in ds.coords or n in ds.variables), None)
    lon = next((n for n in lon_names if n in ds.coords or n in ds.variables), None)
    if lat is None or lon is None:
        raise KeyError(
            f"Could not infer lat/lon names. "
            f"Looked for lat in {lat_names} and lon in {lon_names}. "
            f"Found coords={list(ds.coords)} vars={list(ds.variables)[:30]}"
        )
    return lat, lon


def wrap_lons_to_180(ds: xr.Dataset) -> xr.Dataset:
    """
    If dataset lon is 0..360 but polygon likely uses -180..180, remap to (-180, 180] and sort.
    """
    _, lon_name = _infer_lat_lon_names(ds)
    lon = ds[lon_name]

    lonv = lon.values
    ds_is_360 = (np.nanmin(lonv) >= 0) and (np.nanmax(lonv) > 180)
    if lon.ndim == 1:
        if ds_is_360:
            newlon = ((lonv + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: newlon}).sortby(lon_name)
    elif lon.ndim == 2:
        if ds_is_360:
            newlon = ((lonv + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: (lon.dims, newlon)})
    return ds


def _polygon_mask(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    polygon: Geom,
    fallback_to_nearest: bool = True,
) -> np.ndarray:
    """Boolean mask for points inside a polygon using cell-centers."""
    minx, miny, maxx, maxy = polygon.bounds

    lat_spacing = abs(lat2d[1, 0] - lat2d[0, 0]) if lat2d.shape[0] > 1 else 0.1
    lon_spacing = abs(lon2d[0, 1] - lon2d[0, 0]) if lon2d.shape[1] > 1 else 0.1
    pad = max(lat_spacing, lon_spacing) * 3  # 3-pixel pad

    in_bbox = (
        (lon2d >= minx - pad)
        & (lon2d <= maxx + pad)
        & (lat2d >= miny - pad)
        & (lat2d <= maxy + pad)
    )

    mask = np.zeros(lat2d.shape, dtype=bool)

    if not np.any(in_bbox):
        return mask

    bbox_indices = np.where(in_bbox)
    subset_lats = lat2d[in_bbox]
    subset_lons = lon2d[in_bbox]

    coords = np.column_stack((subset_lons, subset_lats))
    pts_subset = points(coords)

    subset_mask = contains(polygon, pts_subset)

    if fallback_to_nearest and not np.any(subset_mask):
        dists = distance(polygon, pts_subset)
        subset_mask = dists == np.min(dists)

    mask[bbox_indices[0][subset_mask], bbox_indices[1][subset_mask]] = True

    return mask


def _subset_to_polygon(
    ds: xr.Dataset,
    polygon: Geom,
    padding: float = 0.03
) -> xr.Dataset:
    lat_name, lon_name = _infer_lat_lon_names(ds)
    lat = ds[lat_name]
    lon = ds[lon_name]

    if padding > 0:
        polygon = polygon.buffer(padding)
    
    minx, miny, maxx, maxy = polygon.bounds

    # FAST SLICING OPTIMIZATION: Crop bounding box first before heavy 2D masking
    if lat.ndim == 1 and lon.ndim == 1:
        # Determine dimension order for slicing
        lat_slice = slice(miny, maxy) if lat.values[0] < lat.values[-1] else slice(maxy, miny)
        lon_slice = slice(minx, maxx) if lon.values[0] < lon.values[-1] else slice(maxx, minx)
        
        ds_cropped = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
        if ds_cropped.sizes[lat_name] == 0 or ds_cropped.sizes[lon_name] == 0:
            return ds_cropped  # Empty subset

        lats = ds_cropped[lat_name].values
        lons = ds_cropped[lon_name].values
        lon2d, lat2d = np.meshgrid(lons, lats)
        mask = _polygon_mask(lat2d, lon2d, polygon)
        mask_da = xr.DataArray(mask, dims=(lat_name, lon_name))
        return ds_cropped.where(mask_da, drop=True)

    if lat.ndim == 2 and lon.ndim == 2:
        # Bounding Box pre-filter on 2D grids
        lat_vals, lon_vals = lat.values, lon.values
        in_extent = (
            (lon_vals >= minx - 0.1) & (lon_vals <= maxx + 0.1) &
            (lat_vals >= miny - 0.1) & (lat_vals <= maxy + 0.1)
        )
        if not np.any(in_extent):
            return ds.isel({dim: slice(0, 0) for dim in lat.dims})

        y_indices, x_indices = np.where(in_extent)
        y_dim, x_dim = lat.dims[0], lat.dims[1]
        
        ds_cropped = ds.isel({
            y_dim: slice(y_indices.min(), y_indices.max() + 1),
            x_dim: slice(x_indices.min(), x_indices.max() + 1)
        })

        mask = _polygon_mask(ds_cropped[lat_name].values, ds_cropped[lon_name].values, polygon)
        mask_da = xr.DataArray(mask, dims=ds_cropped[lat_name].dims)
        return ds_cropped.where(mask_da, drop=True)

    raise ValueError(f"Unsupported lat/lon shapes: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")


def _process_file_multiprocessing_worker(file_path: Path, fires: list[dict]) -> Tuple[list[dict], dict]:
    """
    Worker function executed in an isolated process.
    """
    results = []
    all_rave_grids = {}
    filename = file_path.name
    
    try:
        timestamp_match = re.search(r'_s(\d{14})', filename)
        if not timestamp_match:
            return results, all_rave_grids
        timestamp = pd.to_datetime(timestamp_match.group(1), format="%Y%m%d%H%M%S")
    except Exception:
        return results, all_rave_grids

    try:
        with xr.open_dataset(file_path) as ds:
            if not ds.rio.crs:
                ds = ds.rio.write_crs("EPSG:4326")
            ds = wrap_lons_to_180(ds)

            frp_var = 'FRP_MEAN' if 'FRP_MEAN' in ds.data_vars else list(ds.data_vars.keys())[0]
            
            for fire in fires:
                fire_id = fire['fire_index_id']
                # Reconstruct Shapely geometry (pre-parsed in dictionary if available)
                geom = fire['geometry'] if isinstance(fire['geometry'], (Polygon, MultiPolygon)) else wkt.loads(fire['wkt_geometry'])
                
                try:
                    clipped = _subset_to_polygon(ds[[frp_var]], geom)
                    
                    if clipped[frp_var].size == 0 or np.all(np.isnan(clipped[frp_var].values)):
                        total_frp = 0.0
                    else:
                        total_frp = np.float32(clipped[frp_var].sum(skipna=True))

                    if fire_id not in all_rave_grids and clipped[frp_var].size > 0:
                        lons = clipped.grid_lont.values.flatten()
                        lats = clipped.grid_latt.values.flatten()
                        # Fast array formatting
                        all_rave_grids[fire_id] = '_'.join([f"{x} {y}" for x, y in zip(lons, lats)])
                        
                    results.append({
                        "fire_index_id": fire_id,
                        "timestamp": timestamp,
                        "total_rave_frp": total_frp
                    })
                except (ValueError, KeyError, rioxarray.exceptions.NoDataInBounds):
                    results.append({
                        "fire_index_id": fire_id,
                        "timestamp": timestamp,
                        "total_rave_frp": 0.0
                    })
                    
    except Exception as e:
        print(f"Error processing file {filename} in worker process: {e}")
        
    return results, all_rave_grids


class RAVEOperClient:
    """
    Pipeline to download NOAA RAVE Hourly NetCDF files and aggregate FRP per fire perimeter.
    """
    BASE_URL = "https://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km"

    def __init__(self, csv_path: str, download_dir: str = "rave_downloads", output_csv: str = "fire_frp_output.csv"):
        self.csv_path = Path(csv_path)
        self.download_dir = Path(download_dir)
        self.output_csv = Path(output_csv)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.df_manifest = pd.read_csv(self.csv_path)
        self._parse_geometries()

    def _parse_geometries(self):
        """Parses WKT geometries once into memory structures."""
        print("Parsing fire geometries...")
        self.df_manifest = self.df_manifest.dropna(subset=['wkt_geometry', 'fire_index_id']).copy()
        
        # Pre-parse Shapely geometries to avoid doing wkt.loads() repeatedly in workers
        serialized = []
        for _, row in self.df_manifest.iterrows():
            serialized.append({
                'fire_index_id': row['fire_index_id'],
                'wkt_geometry': row['wkt_geometry'],
                'geometry': wkt.loads(row['wkt_geometry'])
            })
        self.serialized_fires = serialized

    def get_date_range(self, last_n_days: int, reference_date: datetime.date) -> list[datetime.date]:
        return [reference_date - pd.Timedelta(i, unit='D') for i in range(last_n_days)]

    def _download_file_task(self, url: str, dest_path: str) -> str:
        if os.path.exists(dest_path):
            return dest_path
        
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        return dest_path

    def _download_rave_data(self, target_date: str, out_dir: str) -> list[str]:
        base_url = f"{self.BASE_URL}/{target_date[:4]}/{target_date[4:6]}/"
        cache_buster_url = f"{base_url}?_={int(time.time())}"
        
        os.makedirs(out_dir, exist_ok=True)
        try:
            req = urllib.request.Request(cache_buster_url, headers=HEADERS)
            with urllib.request.urlopen(req) as resp:
                html_content = resp.read().decode('utf-8')
        except Exception as e:
            print(f"Error fetching index for date {target_date}: {e}", file=sys.stderr)
            return []

        pattern = rf'href="([^"]*{re.escape(f"s{target_date}")}[^"]*\.nc)"'
        filenames = list(dict.fromkeys(re.findall(pattern, html_content)))

        if not filenames:
            print(f"No NetCDF files found matching date: {target_date}")
            return []

        print(f"Found {len(filenames)} matching files for {target_date}. Starting downloads...")

        downloaded = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for fname in filenames:
                file_url = f"{base_url.rstrip('/')}/{fname}"
                dest_path = os.path.join(out_dir, fname)
                futures[executor.submit(self._download_file_task, file_url, dest_path)] = dest_path
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    downloaded.append(res)
                except Exception as e:
                    print(f"Failed to download {futures[future]}: {e}", file=sys.stderr)

        return downloaded

    def download_rave_files(self, last_n_days: int, reference_date: datetime.date) -> list[Path]:
        dates = self.get_date_range(last_n_days, reference_date)

        print(f"Scanning NOAA repo for the last {last_n_days} days of RAVE data...")
        downloaded_paths = []
        out_dir = f"{CACHE_BASE_DIR}/rave/{datetime.date.today().strftime('%Y%m%d')}/"
        
        for date_obj in dates:
            yyyymmdd = date_obj.strftime("%Y%m%d")
            files = self._download_rave_data(yyyymmdd, out_dir)
            if files:
                downloaded_paths.extend([Path(fname) for fname in files])
                  
        print(f"Successfully downloaded/verified {len(downloaded_paths)} RAVE files.")
        return downloaded_paths

    def extract_frp_data_parallel_files(self, file_paths: list[Path], max_workers: int = 4):
        all_records = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_file_multiprocessing_worker, path, self.serialized_fires): path 
                for path in file_paths
            }
            no_grids = True
            for future in timer(as_completed(futures), desc="RAVE nc processing", total=len(futures)):
                file_path = futures[future]
                try:
                    records, all_rave_grids = future.result()
                    if records:
                        all_records.extend(records)
                    if all_rave_grids and no_grids:
                        self.df_manifest['rave_grids'] = (
                            self.df_manifest['fire_index_id'].map(all_rave_grids).fillna("")
                        )
                        no_grids = False
                        self.df_manifest.to_csv(self.csv_path, index=False)
                        print(f"[+] Rewrote manifest with grid geometry mapping to {self.csv_path}")
                except Exception as e:
                    print(f"Failed to process file {file_path.name}: {e}")

        if not all_records:
            print("No FRP records extracted.")
            return

        df_output = pd.DataFrame(all_records)
        df_output = df_output.sort_values(by=['fire_index_id', 'timestamp']).reset_index(drop=True)
        df_output.to_csv(self.output_csv, index=False)
        
        print(f"[*] Data processing complete! Saved final metrics to: {self.output_csv}")


if __name__ == "__main__":
    import analysis.mapping.config as config
    
    CSV_MANIFEST = f"{CACHE_BASE_DIR}/active_fires/2026_07_14/fire_pipeline_manifest.csv"
    LOOKBACK_DAYS = 2
    today = config.now_dt
    
    pipeline = RAVEOperClient(
        csv_path=CSV_MANIFEST,
        download_dir="rave_temp_files",
        output_csv="hourly_fire_frp_summary.csv"
    )
    
    downloaded_files = pipeline.download_rave_files(last_n_days=LOOKBACK_DAYS, reference_date=today)
    
    if downloaded_files:
        pipeline.extract_frp_data_parallel_files(downloaded_files, max_workers=25)
    else:
        print("No files were downloaded. Pipeline skipped.")