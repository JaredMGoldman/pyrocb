import os
import re
import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import pandas as pd
import subprocess
import sys
import xarray as xr
import rioxarray
from shapely import wkt
from tqdm import tqdm as timer
from typing import Tuple, Union

from utils.io_utils import CACHE_BASE_DIR

from shapely.geometry import Polygon, MultiPolygon
from shapely import points, contains, distance

Geom = Union[Polygon, MultiPolygon]

curl_body = \
    f"""-H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
    -H 'accept-language: en-US,en;q=0.9,fr;q=0.8' \
    -H 'cache-control: max-age=0' \
    -b 'nmstat=44cf2bba-98cb-af77-e050-319844a0bcbf; _ga_P6H1P1YGHK=GS2.1.s1770318320$o3$g0$t1770318320$j60$l0$h0; _ga_8QRDKZKW09=GS2.1.s1771791152$o1$g1$t1771791382$j60$l0$h0; _ga_PTKGMX2RGX=GS2.1.s1772049972$o4$g0$t1772049972$j60$l0$h0; _ga_4V68PS2QKC=GS2.1.s1772310646$o7$g0$t1772310646$j60$l0$h0; _ga_5C3L2X4VLP=GS2.1.s1772649516$o1$g1$t1772649537$j39$l0$h0; _ga_FVJXY24VSZ=GS2.1.s1773978460$o3$g0$t1773978470$j50$l0$h0; _ga_VNE5GEVT4X=GS2.1.s1775324807$o1$g0$t1775324807$j60$l0$h0; _ga_KB1ED9SLQD=GS2.1.s1777239350$o7$g1$t1777240441$j50$l0$h0; _ga_G2DC4173X1=GS2.1.s1777239441$o5$g1$t1777240477$j30$l0$h0; _ga_8PVW29QMYJ=GS2.2.s1778531157$o10$g0$t1778531157$j60$l0$h0; _ga_WEE2HX9G91=GS2.1.s1778706966$o11$g0$t1778706966$j60$l0$h0; _ga_F0TVX8GTMV=GS2.1.s1778706966$o11$g0$t1778706966$j60$l0$h0; _ga_G1F0K33KY9=GS2.1.s1779141469$o2$g0$t1779141469$j60$l0$h0; _ga_JK69RBSYCC=GS2.1.s1779141469$o2$g0$t1779141469$j60$l0$h0; _ga_8G5H3D09RC=GS2.1.s1781286353$o4$g1$t1781288038$j60$l0$h0; _ga_HS0NRB74WC=GS2.1.s1781461685$o5$g0$t1781461685$j60$l0$h0; _ga_E1Q1BML6E5=GS2.1.s1782164256$o5$g0$t1782164256$j60$l0$h0; _ga_73CXWL3FH9=GS2.1.s1782842269$o4$g1$t1782842302$j27$l0$h0; _ga_BGDP0TBYX2=GS2.1.s1783443043$o3$g1$t1783443283$j60$l0$h0; _ga=GA1.1.488889452.1770153793; _ga_CSLL4ZEK4L=GS2.1.s1784074627$o72$g0$t1784075012$j60$l0$h0; _ga_WFK3K5ECCC=GS2.1.s1784074627$o22$g0$t1784075012$j60$l0$h0' \
    -H 'dnt: 1' \
    -H 'priority: u=0, i' \
    -H 'sec-ch-ua: "Not;A=Brand";v="8", "Chromium";v="150", "Google Chrome";v="150"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'sec-ch-ua-platform: "macOS"' \
    -H 'sec-fetch-dest: document' \
    -H 'sec-fetch-mode: navigate' \
    -H 'sec-fetch-site: none' \
    -H 'sec-fetch-user: ?1' \
    -H 'upgrade-insecure-requests: 1' \
    -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36'"""

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
    """Boolean mask for points inside a polygon using cell-centers.

    If no points are contained within the polygon and fallback_to_nearest is
    True, it returns the closest point(s) to the polygon to avoid empty subsets.
    """
    minx, miny, maxx, maxy = polygon.bounds

    # Calculate grid spacing to add a buffer to the bounding box (ensures fallback point is caught)
    lat_spacing = abs(lat2d[1, 0] - lat2d[0, 0]) if lat2d.shape[0] > 1 else 0.1
    lon_spacing = abs(lon2d[0, 1] - lon2d[0, 0]) if lon2d.shape[1] > 1 else 0.1
    pad = max(lat_spacing, lon_spacing) * 3  # 3-pixel pad

    in_bbox = (
        (lon2d >= minx - pad)
        & (lon2d <= maxx + pad)
        & (lat2d >= miny - pad)
        & (lat2d <= maxy + pad)
    )

    # Initialize a global boolean mask matching the 2D grid shape
    mask = np.zeros(lat2d.shape, dtype=bool)

    if not np.any(in_bbox):
        # If the bounding box doesn't overlap the grid at all, return empty mask
        return mask

    # Slice out only the subset of points inside the padded bounding box
    bbox_indices = np.where(in_bbox)
    subset_lats = lat2d[in_bbox]
    subset_lons = lon2d[in_bbox]

    # We stack lons and lats to shape (N, 2) to construct all points in one vectorized call
    coords = np.column_stack((subset_lons, subset_lats))
    pts_subset = points(coords)

    subset_mask = contains(polygon, pts_subset)

    if fallback_to_nearest and not np.any(subset_mask):
        # Vectorized distance calculation
        dists = distance(polygon, pts_subset)
        subset_mask = dists == np.min(dists)

    # 5. Map the subset calculations back to the original 2D grid shape
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
    
    # Case A: 1D lat/lon
    if lat.ndim == 1 and lon.ndim == 1:
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        lon2d, lat2d = np.meshgrid(lons, lats)
        mask = _polygon_mask(lat2d, lon2d, polygon)
        mask_da = xr.DataArray(mask, dims=(lat_name, lon_name))
        return ds.where(mask_da, drop=True)

    # Case B: 2D lat/lon coords (y,x)
    if lat.ndim == 2 and lon.ndim == 2:
        mask = _polygon_mask(lat.values, lon.values, polygon)
        mask_da = xr.DataArray(mask, dims=lat.dims)
        return ds.where(mask_da, drop=True)

    raise ValueError(f"Unsupported lat/lon shapes: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")

def _process_file_multiprocessing_worker(file_path: Path, fires: list[dict]) -> list[dict]:
    """
    Worker function executed in an isolated process. 
    Opens a single NetCDF file ONCE and calculates FRP for all fire geometries.
    """
    results = []
    filename = file_path.name
    
    # Extract timestamp from filename (e.g., _sYYYYMMDDHHMMSS)
    try:
        timestamp_match = re.search(r'_s(\d{14})', filename)
        if not timestamp_match:
            return results
        timestamp = pd.to_datetime(timestamp_match.group(1), format="%Y%m%d%H%M%S")
    except Exception:
        return results

    try:
        # Open the dataset inside the isolated process
        with xr.open_dataset(file_path) as ds:
            # Set CRS if not defined
            if not ds.rio.crs:
                ds = ds.rio.write_crs("EPSG:4326")
            ds = wrap_lons_to_180(ds)

            # Auto-detect target variable (looking for FRP)
            frp_var = 'FRP_MEAN' if 'FRP_MEAN' in ds.data_vars else list(ds.data_vars.keys())[0]
            
            for fire in fires:
                fire_id = fire['fire_index_id']
                # Reconstruct shapely geometry from WKT (highly portable for multiprocessing)
                geom = wkt.loads(fire['wkt_geometry'])
                
                try:
                    # Clip the single opened file to this specific fire's polygon
                    clipped = _subset_to_polygon(ds[[frp_var]], geom)
                    total_frp = np.float32(clipped[frp_var].sum(skipna=True))
                    results.append({
                        "fire_index_id": fire_id,
                        "timestamp": timestamp,
                        "total_rave_frp": total_frp
                    })
                except (ValueError, rioxarray.exceptions.NoDataInBounds):
                    # Fallback if the fire doesn't overlap the spatial grid of this specific file
                    print(f"no data found for {fire_id} at {timestamp}")
                    results.append({
                        "fire_index_id": fire_id,
                        "timestamp": timestamp,
                        "total_rave_frp": 0.0
                    })
                    
    except Exception as e:
        print(f"Error processing file {filename} in worker process: {e}")
        
    return results

class RAVEOperClient:
    """
    A pipeline to download NOAA RAVE (Rapidly-varying Emissions) Hourly NetCDF files,
    align them with active fire perimeters, and calculate the total hourly FRP (Fire Radiative Power)
    inside each fire boundary using parallel execution.
    """
    
    BASE_URL = "https://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km"

    def __init__(self, csv_path: str, download_dir: str = "rave_downloads", output_csv: str = "fire_frp_output.csv"):
        self.csv_path = Path(csv_path)
        self.download_dir = Path(download_dir)
        self.output_csv = Path(output_csv)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the manifest
        self.df_manifest = pd.read_csv(self.csv_path)
        self._parse_geometries()

    def _parse_geometries(self):
        """Parses WKT geometries from the manifest and filters out invalid ones."""
        print("Parsing fire geometries...")
        self.df_manifest['geometry'] = self.df_manifest['wkt_geometry'].apply(wkt.loads)
        # Drop rows missing geometry or crucial columns
        self.df_manifest = self.df_manifest.dropna(subset=['geometry', 'fire_index_id'])

    def get_date_range(self, last_n_days: int, reference_date: datetime.date) -> list[datetime.date]:
        """Generates a list of date objects for the last N days."""
        return [reference_date - pd.Timedelta(i, unit='D') for i in range(last_n_days)]

    def _download_rave_data(self, target_date, command, out_dir):
        # Define paths and URLs
        base_url = f"https://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km/{target_date[:4]}/{target_date[4:6]}/"

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, shell=True
            )
            html_content = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error fetching HTML index: {e}", file=sys.stderr)
            return
        except FileNotFoundError:
            print(
                "Error: 'curl' command not found in your system PATH.",
                file=sys.stderr,
            )
            return

        # This regex matches any href="filename.nc" containing the target date
        pattern = rf'href="([^"]*{re.escape(f"s{target_date}")}[^"]*\.nc)"'
        filenames = re.findall(pattern, html_content)

        # Deduplicate matching files (Apache listings sometimes repeat links in tables)
        filenames = list(dict.fromkeys(filenames))

        if not filenames:
            print(f"No NetCDF files found matching date: {target_date}")
            return

        print(f"Found {len(filenames)} matching files. Starting downloads...")

        # 3. Spawn wget subprocesses for each file
        downloaded = []
        for filename in filenames:
            file_url = f"{base_url.rstrip('/')}/{filename}"
            if os.path.exists(os.path.join(out_dir, filename)):
                downloaded.append(os.path.join(out_dir, filename))
                continue
            print(f"Downloading: {filename}")

            # -nc (no-clobber) skips download if the file already exists locally
            wget_cmd = ' '.join(["wget", "-nc", "-P", out_dir, file_url])

            try:
                # We run with check=True to raise an exception if a download fails
                subprocess.run(wget_cmd, check=True, 
                               shell = True, 
                               capture_output=True)
                downloaded.append(os.path.join(out_dir, filename))
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {filename}: {e}", file=sys.stderr)
            except FileNotFoundError:
                print(
                    "Error: 'wget' command not found in your system PATH.",
                    file=sys.stderr,
                )
                return None

        print("Process complete.")
        return downloaded

    def download_rave_files(self, last_n_days: int, reference_date: datetime.date) -> list[Path]:
        """
        Downloads RAVE hourly files with a robust, chunked HTML stream-reader 
        to prevent page truncation.
        """
        dates = self.get_date_range(last_n_days, reference_date)

        print(f"Scanning NOAA repo for the last {last_n_days} days of RAVE data...")
        downloaded_paths = []
        out_dir = f"{CACHE_BASE_DIR}/rave/{datetime.date.today().strftime("%Y%m%d")}/"
        for date_obj in dates:
            year = date_obj.strftime("%Y")
            yyyymmdd = date_obj.strftime("%Y%m%d")
            mm = date_obj.strftime("%m")
            curl_header = f"curl '{self.BASE_URL}/{year}/{mm}/'"
            command = curl_header + " " + curl_body 
            files = self._download_rave_data(yyyymmdd, command, out_dir)
            if files:
                downloaded_paths.extend([Path(fname) for fname in files])
                  
        print(f"Successfully downloaded/verified {len(downloaded_paths)} RAVE files.")

        return downloaded_paths

    def _process_single_nc(self, file_path: Path) -> list[dict]:
        """
        Processes a single RAVE NetCDF file, aligning it spatially with each fire polygon 
        to sum total FRP.
        """
        results = []
        filename = file_path.name
        
        # Parse timestamp from filename format (e.g., RAVE-HrlyEmiss-3km_v1r1_blend_sYYYYMMDDHHMMSS...)
        try:
            timestamp_match = re.search(r'_s(\d{14})', filename)
            if not timestamp_match:
                return results
            timestamp_str = timestamp_match.group(1)
            timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M%S")
        except Exception:
            return results

        try:
            with xr.open_dataset(file_path) as ds:
                # Confirm target coordinate systems (assign WGS84 if not defined)
                if not ds.rio.crs:
                    ds = ds.rio.write_crs("EPSG:4326")
                
                # Check for standard target variables (FRP)
                frp_var = 'FRP' if 'FRP' in ds.data_vars else list(ds.data_vars.keys())[0]
                
                # Process each fire perimeter
                for _, row in self.df_manifest.iterrows():
                    fire_id = row['fire_index_id']
                    geom = row['geometry']
                    
                    # Optimization: Filter out if fire bounding box doesn't overlap global/regional dataset bounding box
                    # This avoids slow spatial operations on unrelated geometries
                    try:
                        # Clip dataset to the fire geometry using rioxarray
                        clipped = ds[frp_var].rio.clip([geom], crs="EPSG:4326", all_touched=True)
                        
                        # Sum up all valid FRP values inside the polygon
                        # RAVE uses NaN or fill values for areas with no fire detected
                        total_frp = float(clipped.sum(skipna=True).values)
                        
                        results.append({
                            "fire_index_id": fire_id,
                            "timestamp": timestamp,
                            "total_rave_frp": total_frp
                        })
                    except (ValueError, rioxarray.exceptions.NoDataInBounds):
                        # Geometric non-overlap or empty array after clipping
                        results.append({
                            "fire_index_id": fire_id,
                            "timestamp": timestamp,
                            "total_rave_frp": 0.0
                        })
                        
        except Exception as e:
            print(f"Error processing NetCDF file {filename}: {e}")
            
        return results

    def _parse_geometries(self):
        """Parses manifest data into clean, serializable structures."""
        self.df_manifest = self.df_manifest.dropna(subset=['wkt_geometry', 'fire_index_id'])
        
        # Convert to a lightweight list of dicts to pass easily to worker processes
        self.serialized_fires = self.df_manifest[['fire_index_id', 'wkt_geometry']].to_dict(orient='records')

    def extract_frp_data_parallel_files(self, file_paths: list[Path], max_workers: int = 4):
        """
        Alternative extraction method. 
        Spins up independent worker processes. Each process is dedicated to an individual 
        NetCDF file, looping through all target fires internally.
        """        
        all_records = []

        # We use ProcessPoolExecutor so each worker runs on a native CPU core bypasses the GIL
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit each file to its own process
            futures = {
                executor.submit(_process_file_multiprocessing_worker, path, self.serialized_fires): path 
                for path in file_paths
            }
            
            for future in timer(as_completed(futures), desc="RAVE nc processing", total = len(futures)):
                file_path = futures[future]
                try:
                    records = future.result()
                    if records:
                        all_records.extend(records)
                except Exception as e:
                    print(f"Failed to process file {file_path.name}: {e}")

        if not all_records:
            print("No FRP records extracted.")
            return

        # Export compiled records to CSV
        df_output = pd.DataFrame(all_records)
        df_output = df_output.sort_values(by=['fire_index_id', 'timestamp']).reset_index(drop = True)
        zero_frp_mask = df_output.groupby('fire_index_id')['total_rave_frp'].sum() == 0
        zero_frp_fires = zero_frp_mask[zero_frp_mask].index.tolist()

        print(f"[*] identified and removing {len(zero_frp_fires)} inactive fires")
        df_output = df_output[~df_output.fire_index_id.isin(zero_frp_fires)].reset_index(drop = True)
        df_output.to_csv(self.output_csv, index=False)
        
        print(f"[*] Data processing complete! Saved final metrics to: {self.output_csv}")
        return zero_frp_fires


# --- Execution Hook ---
if __name__ == "__main__":
    import analysis.mapping.config as config
    # Settings
    CSV_MANIFEST = f"{CACHE_BASE_DIR}/active_fires/2026_07_14/fire_pipeline_manifest.csv"
    LOOKBACK_DAYS = 2  # Set how far back you want to fetch RAVE data
    today = config.now_dt
    
    # Initialize the Pipeline Object
    pipeline = RAVEOperClient(
        csv_path=CSV_MANIFEST,
        download_dir="rave_temp_files",
        output_csv="hourly_fire_frp_summary.csv"
    )
    
    downloaded_files = pipeline.download_rave_files(last_n_days=LOOKBACK_DAYS, reference_date=today)
    
    # (Computationally intensive; keep workers close to physical CPU cores)
    if downloaded_files:
        pipeline.extract_frp_data_parallel_files(downloaded_files, max_workers = 25)
    else:
        print("No files were downloaded. Pipeline skipped.")