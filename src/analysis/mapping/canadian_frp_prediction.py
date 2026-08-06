import os
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.wkt import loads
import shutil
import concurrent.futures

import analysis.mapping.config as config


# ==========================================
# PHASE 1: AUTOMATED DATA RETRIEVAL (CANADIAN)
# ==========================================

def download_canadian_forecasts(target_date_str, output_dir):
    """
    Automates downloading the latest and yesterday's Canadian forecasts safely.
    Uses atomic writing and header verification to prevent corrupt .nc files.
    """
    os.makedirs(output_dir, exist_ok=True)
    current_dt = datetime.strptime(target_date_str, "%Y%m%d")
    yesterday_dt = current_dt - timedelta(days=1)
    
    urls = [
        f"https://firesmoke.ca/forecasts/fireweather/{yesterday_dt.strftime('%Y%m%d')}00/fwf-hourly-d02-{yesterday_dt.strftime('%Y%m%d')}06.nc",
        f"https://firesmoke.ca/forecasts/fireweather/{current_dt.strftime('%Y%m%d')}00/fwf-hourly-d02-{current_dt.strftime('%Y%m%d')}06.nc"
    ]
    
    downloaded_files = []
    for url in urls:
        filename = os.path.basename(url)
        dest_path = os.path.join(output_dir, filename)
        if os.path.exists(dest_path):
            print(f"[+] {dest_path} is cached skipping {url} download")
            continue
        tmp_path = dest_path + ".tmp"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )

        try:
            with urllib.request.urlopen(req) as response:
                # Check if the server is returning an HTML webpage instead of raw data
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    print(f"[-] File not yet available on server for {url}. (Server returned HTML instead of NetCDF). Skipping.")
                    # Do not save this file to dest_path
                    continue

                # If it's a valid binary stream, write it directly to the temporary file
                print(f"[+] Server confirmed valid dataset. Downloading content from {url}...")

                with open(tmp_path, 'wb') as tmp_file:
                    shutil.copyfileobj(response, tmp_file)

            # Move the validated temporary file to its final destination path
            os.replace(tmp_path, dest_path)
            print(f"[+] saved {url} to {dest_path}")
            downloaded_files.append(dest_path)

        except Exception as e:
            print(f"[-] Download interrupted or failed for {url}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return downloaded_files

# ==========================================
# PHASE 2: SPATIAL INDEX LOOKUP TRANSFORM
# ==========================================

def build_rave_to_forecast_map(rave_coords, lat_forecast_full, lon_forecast_full):
    """
    Maps each 2D (lon, lat) point in `rave_coords` to its enclosing forecast grid cell (Y, X index).
    
    Parameters:
    -----------
    rave_coords : list of lists or np.ndarray
        List of [lon, lat] pairs for the RAVE lattice.
    lat_forecast_full : np.ndarray
        2D grid (Ny, Nx) or 1D array of forecast latitudes.
    lon_forecast_full : np.ndarray
        2D grid (Ny, Nx) or 1D array of forecast longitudes.
        
    Returns:
    --------
    map_indices : list of tuples
        List of (y_idx, x_idx) tuples corresponding 1-to-1 with `rave_coords`.
    """
    rave_coords_arr = np.array(rave_coords, dtype=np.float32)
    
    # Handle both 2D meshgrids (Ny, Nx) and 1D coordinate vectors
    if lat_forecast_full.ndim == 1 and lon_forecast_full.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_forecast_full, lat_forecast_full)
    else:
        lat_grid, lon_grid = lat_forecast_full, lon_forecast_full

    # Stack coordinates into an (N, 2) array for KDTree space lookup
    fc_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    tree = cKDTree(fc_points)
    
    # Query nearest forecast center for all RAVE points simultaneously
    _, flat_indices = tree.query(rave_coords_arr)
    
    # Unravel flat indices back to 2D (Y, X) forecast grid indices
    y_indices, x_indices = np.unravel_index(flat_indices, lat_grid.shape)
    
    # Pair up into list of (y, x) tuples
    map_indices = list(zip(y_indices, x_indices))
    return map_indices

def extract_remapped_rave_frp(forecast_files, mapped_indices):
    """
    Remaps forecast FRP data directly onto RAVE grid elements for all forecast timestamps.
    
    Parameters:
    -----------
    forecast_files : list of str
        List of filepaths to forecast NetCDF files.
    mapped_indices : list of tuples
        List of (y_idx, x_idx) mapping positions for each RAVE grid element.
        
    Returns:
    --------
    time_forecast : list of np.ndarray
        Timestamps associated with each forecast file.
    frp_remapped_by_file : list of np.ndarray
        FRP values reshaped to (Time, Num_RAVE_Grids) for each forecast file.
    """
    time_forecast = []
    frp_remapped_by_file = []

    # Unzip indices into array slice vectors for vectorized xarray indexing
    y_idxs = np.array([idx[0] for idx in mapped_indices])
    x_idxs = np.array([idx[1] for idx in mapped_indices])

    for fc_file in forecast_files:
        with xr.open_dataset(fc_file) as ds_f:
            ds_f = ds_f.swap_dims({'time': 'Time'})
            time_forecast.append(np.array([pd.to_datetime(t) for t in ds_f.Time.values]))
            
            # Identify spatial dimensions (excluding 'Time')
            spatial_dims = [dim for dim in ds_f['FRP'].dims if dim != 'Time']
            y_dim, x_dim = spatial_dims[0], spatial_dims[1]
            
            # Vectorized lookup across time: extracts FRP for all RAVE grids at once
            # Resulting shape: (Time, Num_RAVE_Grids)
            frp_data = ds_f['FRP'].isel({
                y_dim: xr.DataArray(y_idxs, dims='rave_points'),
                x_dim: xr.DataArray(x_idxs, dims='rave_points')
            }).values
            
            frp_remapped_by_file.append(frp_data)

    return time_forecast, frp_remapped_by_file

# ==========================================
# PHASE 4: REPLICATED INDIVIDUAL WORKER TASK
# ==========================================


def process_single_fire(args):
    """
    Isolated worker execution context handling a single target fire.
    Generates diagnostic files and returns the path to a cached NetCDF file.
    """
    idx, row, context = args
    
    # Unpack clean, pickle-safe variable primitives from context payload
    forecast_files = context['forecast_files']
    lat_forecast_full = context['lat_forecast_full']
    lon_forecast_full = context['lon_forecast_full']
    date_mask_rave_dt = context['date_mask_rave_dt']
    out_dir = context['out_dir']
    cache_dir = context['cache_dir']
    
    fire_name = str(row.get('name', row.get('fire_key', f'target_fire_{idx}'))).replace(" ", "_")
    fire_idx = str(row.get('fire_index_id', 'UNKNOWN'))

    # Parse target RAVE grid lattice coordinates [[lon, lat], ...]
    rave_grids = row['rave_grids']
    rave_grids_arr = [[np.float32(val) for val in elt.split(' ')] for elt in rave_grids.split('_')]
    num_rave_points = len(rave_grids_arr)
    print(f"[*] [PID {os.getpid()}] Processing cross-grid interpolations for target: {fire_name} ({num_rave_points} RAVE grids)")
    
    # Read historical RAVE DataFrame
    rave_df = pd.read_csv(os.path.join(out_dir, config.active_rave_fn))
    this_df = rave_df[rave_df.fire_index_id == fire_idx].sort_values('timestamp')
    frp_timeseries = this_df['total_rave_frp'].to_numpy(copy=True)
    rave_timeline = [pd.to_datetime(t) for t in this_df['timestamp'].values]

    try:
        # Re-locate the precise index step matching pivot targets inside the dataset timeline
        idx_rave_pivot = min(range(len(rave_timeline)), key=lambda i: abs(rave_timeline[i] - date_mask_rave_dt))
    except ValueError:
        print(f"[-] Pivot timestamp {date_mask_rave_dt} error out boundaries for {fire_name}.")
        return None

    # ==============================================================
    # SPATIAL REMAPPING: MAP FORECAST CELLS TO EACH RAVE GRID ELEMENT
    # ==============================================================
    # 1. Calculate map indices matching each RAVE point to its forecast bounding cell
    mapped_indices = build_rave_to_forecast_map(
        rave_coords=rave_grids_arr, 
        lat_forecast_full=lat_forecast_full, 
        lon_forecast_full=lon_forecast_full
    )

    # 2. Extract mapped forecast values for all timestamps
    # Output shapes: time_forecast -> list of 1D arrays; frp_forecast_int_masked -> list of (Time, Num_RAVE_Grids) 2D arrays
    time_forecast, frp_forecast_int_masked = extract_remapped_rave_frp(
        forecast_files=forecast_files, 
        mapped_indices=mapped_indices
    )

    # ==========================================
    # PHASE 5: HYBRID PREDICTION SYNTHESIS STAGE
    # ==========================================
    time_forecast_hybrid = time_forecast[1]
    
    # Shape is now (Time, Num_RAVE_Grids) to accommodate spatial grid elements
    frp_forecast_hybrid = np.full(len(time_forecast_hybrid), np.nan)
    
    try:
        idx_fc1_pivot = list(time_forecast[0]).index(date_mask_rave_dt)
        idx_fc2_pivot = list(time_forecast[1]).index(date_mask_rave_dt)
    except ValueError:
        print(f"[-] Warning: Time axes are misaligned for {fire_name}. Skipping hybrid step.")
        return None

    # Slice the historical data (up to 24 hours prior to pivot)
    historical_rave_sample = frp_timeseries[max(0, idx_rave_pivot-24):idx_rave_pivot]
    actual_history_len = len(historical_rave_sample)

    if actual_history_len == 0:
        print(f"[-] Warning: No historical RAVE data found prior to pivot for {fire_name}. Filling hybrid forecast with NaNs.")
    else:
        # Historical forecast reference sample shape: (History_Len, Num_RAVE_Grids)
        fc1_reference_sample = frp_forecast_int_masked[0][max(0, idx_fc1_pivot-actual_history_len):idx_fc1_pivot]
        
        # --- DAY 1 HOURLY SCALING ---
        fc2_day1_sample = frp_forecast_int_masked[1][idx_fc2_pivot:idx_fc2_pivot+actual_history_len]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            hourly_ratio_day1 = fc2_day1_sample / fc1_reference_sample
            invalid_mask = (
                (fc1_reference_sample < config.MIN_FRP_THRESHOLD) | 
                np.isnan(hourly_ratio_day1) | 
                np.isinf(hourly_ratio_day1)
            )
            hourly_ratio_day1 = np.where(invalid_mask, 1.0, hourly_ratio_day1)
            hourly_ratio_day1 = np.clip(hourly_ratio_day1, a_min=0.0, a_max=config.MAX_ALLOWED_RATIO)
        
        # 1. Average growth ratios spatially across all RAVE grids to get a 1D multiplier per timestamp
        mean_growth_ratio_day1 = np.nanmean(hourly_ratio_day1, axis=1)  # Shape: (History_Len,)

        # 2. Scale the total historical RAVE sample directly by the mean regional growth rate
        day1_projection = historical_rave_sample * mean_growth_ratio_day1  # Shape: (History_Len,)

        # 3. Project Day 1 onto 1D hybrid array starting at pivot + 1
        day1_start_idx = idx_fc2_pivot + 1
        day1_end_idx = min(day1_start_idx + actual_history_len, len(frp_forecast_hybrid))
        target_len = day1_end_idx - day1_start_idx
        
        frp_forecast_hybrid[day1_start_idx:day1_end_idx] = day1_projection[:target_len]
        
        # --- DAY 2 HOURLY SCALING ---
        day2_start_idx = idx_fc2_pivot + 24
        day2_end_idx = min(day2_start_idx + actual_history_len, len(frp_forecast_hybrid))
        day2_slice_len = day2_end_idx - day2_start_idx
        
        if day2_slice_len > 0:
            fc2_day2_sample = frp_forecast_int_masked[1][day2_start_idx:day2_end_idx]
            ref_sample_sliced = fc1_reference_sample[:day2_slice_len]
            hist_sample_sliced = historical_rave_sample[:day2_slice_len]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                hourly_ratio_day2 = fc2_day2_sample / ref_sample_sliced
                invalid_mask_day2 = (
                    (ref_sample_sliced < config.MIN_FRP_THRESHOLD) | 
                    np.isnan(hourly_ratio_day2) | 
                    np.isinf(hourly_ratio_day2)
                )
                hourly_ratio_day2 = np.where(invalid_mask_day2, 1.0, hourly_ratio_day2)
                hourly_ratio_day2 = np.clip(hourly_ratio_day2, a_min=0.0, a_max=config.MAX_ALLOWED_RATIO)
            
            # Average spatially and scale
            mean_growth_ratio_day2 = np.nanmean(hourly_ratio_day2, axis=1)
            frp_forecast_hybrid[day2_start_idx:day2_end_idx] = hist_sample_sliced * mean_growth_ratio_day2

    # ==========================================
    # PHASE 7: LOCAL FILE SYNC AND NETCDF CACHING
    # ==========================================
    # Store output data arrays inside a temporary xarray Dataset object with spatial RAVE dimension
    fc_yesterday_total = np.sum(frp_forecast_int_masked[0], axis=1)
    fc_latest_total = np.sum(frp_forecast_int_masked[1], axis=1)

    ds_cache = xr.Dataset(
        data_vars=dict(
            fc_yesterday=(["time_fc0"], fc_yesterday_total),
            fc_latest=(["time_fc1"], fc_latest_total),
            fc_hybrid=(["time_hybrid"], frp_forecast_hybrid),
            rave_historical=(["time_rave"], frp_timeseries[:idx_rave_pivot+1])
        ),
        coords=dict(
            time_fc0=[t.strftime('%Y-%m-%d %H:%M:%S') for t in time_forecast[0]],
            time_fc1=[t.strftime('%Y-%m-%d %H:%M:%S') for t in time_forecast[1]],
            time_hybrid=[t.strftime('%Y-%m-%d %H:%M:%S') for t in time_forecast_hybrid],
            time_rave=[t.strftime('%Y-%m-%d %H:%M:%S') for t in rave_timeline[:idx_rave_pivot+1]],
            rave_points=np.arange(num_rave_points)
        ),
        attrs=dict(fire_name=fire_name, parent_idx=idx, fire_idx=fire_idx)
    )

    # Dump cleanly to unique local disk tracking target to safely escape sub-process boundary
    cache_filepath = os.path.join(cache_dir, f"cache_arrays_{fire_idx}.nc")
    ds_cache.to_netcdf(cache_filepath)
    ds_cache.close()
    
    return cache_filepath

# ==========================================
# PHASE 3: CORE PREDICTIVE EXECUTIVE PIPELINE
# ==========================================

def execute_predictive_fire_pipeline(target_dt, out_dir = "./ca_frp"):
    """
    Consolidated operational function replacing the original multi-step MATLAB execution loop.
    Integrates the programmatic RAVEClient API to extract data directly.
    """
    os.makedirs(out_dir, exist_ok=True)
    target_date_str = (target_dt - timedelta(days=1)).strftime('%Y%m%d')
    yesterday_str = (target_dt - timedelta(days=2)).strftime('%Y%m%d')
    forecast_files = [
        os.path.join(out_dir, f"fwf-hourly-d02-{yesterday_str}06.nc"),
        os.path.join(out_dir, f"fwf-hourly-d02-{target_date_str}06.nc")
    ]
    
    # Establish running timeframe context anchors
    date_mask_rave_dt = target_dt + timedelta(hours=6) # 06Z forecast mark
    
    fire_polygon_file = os.path.join(out_dir, 'fire_pipeline_manifest.csv')
    
    if not all([os.path.exists(fname) for fname in forecast_files]):
        download_canadian_forecasts(target_date_str, out_dir)
    
    # 3. Read Inbound Pipeline Manifest Inventory Targets
    if not os.path.exists(fire_polygon_file):
        raise FileNotFoundError(f"Missing base operational manifest tracking file: {fire_polygon_file}")
        
    fire_df = pd.read_csv(fire_polygon_file)
    
    if fire_df.empty:
        print("[!] No active targets found matching minimum size conditions (>10 km²).")
        return

    # Read base static spatial framework dimensions of forecasting engines
    try:
        with xr.open_dataset(forecast_files[0]) as fc_sample:
            lat_forecast_full = fc_sample['XLAT'].values
            lon_forecast_full = fc_sample['XLONG'].values
            Nx_fc_max, Ny_fc_max = lat_forecast_full.shape
            # sum over rave polygons
    except Exception as e:
        print(f"[-] failed to process Canadian FRP predictions with Exception {e}")
    # ==========================================
    # PARALLEL EXECUTION MULTIPROCESSING HARNESS
    # ==========================================
    cache_dir = os.path.join(out_dir, ".xarray_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Build a pickle-safe context payload mapping primitives across sub-processes
    context_payload = {
        'forecast_files': forecast_files,
        'lat_forecast_full': lat_forecast_full,
        'lon_forecast_full': lon_forecast_full,
        'Nx_fc_max': Nx_fc_max,
        'Ny_fc_max': Ny_fc_max,
        'date_mask_rave_dt': date_mask_rave_dt,
        'out_dir': out_dir,
        'cache_dir': cache_dir
    }
    
    worker_tasks = [(idx, row, context_payload) for idx, row in fire_df.iterrows()]
    # out_task = [task for task in worker_tasks if task[1].fire_index_id == 'fire_idx_79']
    cached_nc_paths = []
    
    print(f"[*] Dispatching grid interpolation calculations to ProcessPoolExecutor...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            # Map retains tracking index sequence integrity natively
            results = executor.map(process_single_fire, worker_tasks)
            cached_nc_paths = [path for path in results if path is not None]
            
        print(f"[+] All processes completed execution. Synchronizing time series outputs...")
        
        # Initialize a list to hold dataframes for each individual fire
        timeseries_list = []
        
        # Read the stored intermediate NetCDF cache structures sequentially inside the main execution thread
        for nc_path in cached_nc_paths:
            with xr.open_dataset(nc_path) as ds_cache:
                fire_name = ds_cache.attrs['fire_name']
                fire_idx = ds_cache.attrs['fire_idx']
                
                # Collect timelines into dictionaries mapping time string to value
                data_by_type = {}
                all_timestamps = set()
                
                for label, coordinate_dim, dataset_var in [
                    ('fc_yesterday', 'time_fc0', 'fc_yesterday'),
                    ('fc_latest', 'time_fc1', 'fc_latest'),
                    ('fc_hybrid', 'time_hybrid', 'fc_hybrid'),
                    ('rave_historical', 'time_rave', 'rave_historical')
                    ]:
                    time_axis_st = [str(t) for t in ds_cache[coordinate_dim].values]
                    data_values = ds_cache[dataset_var].values
                    
                    data_by_type[label] = pd.Series(data_values, index=time_axis_st)
                    all_timestamps.update(time_axis_st)
                
                # Align data across all unique timestamps for this specific fire
                sorted_timestamps = sorted(list(all_timestamps))
                fire_ts_df = pd.DataFrame(index=sorted_timestamps)
                fire_ts_df['fc_yesterday'] = fire_ts_df.index.map(data_by_type['fc_yesterday'])
                fire_ts_df['fc_latest'] = fire_ts_df.index.map(data_by_type['fc_latest'])
                fire_ts_df['fc_hybrid'] = fire_ts_df.index.map(data_by_type['fc_hybrid'])
                fire_ts_df['rave_historical'] = fire_ts_df.index.map(data_by_type['rave_historical'])
                
                # Set up the multi-key structure: fire_name and time
                fire_ts_df.index.name = 'time'
                fire_ts_df = fire_ts_df.reset_index()
                
                fire_ts_df.insert(0, 'fire_idx', fire_idx)
                fire_ts_df.insert(1, 'fire_name', fire_name)
                
                timeseries_list.append(fire_ts_df)

        # Combine all individual fires into a single master time series dataframe
        if timeseries_list:
            master_ts_df = pd.concat(timeseries_list, ignore_index=True)
            # Set the keys (fire_idx, time) as the multi-index
            master_ts_df.set_index(['fire_idx', 'time'], inplace=True)
            
            # Save to a dedicated file
            predictions_output_csv = os.path.join(out_dir, "fire_predictions_timeseries.csv")
            master_ts_df.to_csv(predictions_output_csv)
            print(f"[+] Multi-key prediction time series saved to: {predictions_output_csv}")
            if os.path.exists(cache_dir):
                print("[*] Cleaning up intermediate cache tracking workspace directories...")
                shutil.rmtree(cache_dir)
                return predictions_output_csv
        else:
            print("[!] No prediction datasets found to compile.")

    except Exception as exc:
        print(f"[-] Critical failure inside pipeline loop executor: {exc}")
        raise exc
    finally:
        # Complete file deletion tracking cleanup of temporary intermediate NetCDF caches
        if os.path.exists(cache_dir):
            print("[*] Cleaning up intermediate cache tracking workspace directories...")
            shutil.rmtree(cache_dir)
            return None

if __name__ == "__main__":
    # Context-aware real-time dynamic analysis harness execution
    target_dt = config.now_dt
    execute_predictive_fire_pipeline(target_dt = target_dt)