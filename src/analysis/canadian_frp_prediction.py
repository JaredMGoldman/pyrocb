import os
import glob
import analysis.mapping.config as config
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.wkt import loads
import shutil
import concurrent.futures

# Import the programmatic internal data API directly
from data.clients.rave_client import RAVEClient
from utils.constants import CACHE_BASE_DIR

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
        tmp_path = dest_path + ".tmp"
        
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            downloaded_files.append(dest_path)
            continue
            
        print(f"[+] Requesting Canadian Forecast: {filename}")
        try:
            with urllib.request.urlopen(url) as response:
                if response.status != 200:
                    print(f"[-] Server returned status {response.status} for {filename}. Skipping.")
                    continue
                
                with open(tmp_path, 'wb') as tmp_file:
                    shutil.copyfileobj(response, tmp_file)
            
            os.replace(tmp_path, dest_path)
            print(f"[+] Download complete: {filename}")
            downloaded_files.append(dest_path)
            
        except Exception as e:
            print(f"[-] Failed to fetch {url}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return downloaded_files

# ==========================================
# PHASE 2: SPATIAL INDEX LOOKUP TRANSFORM
# ==========================================

def map_latlon_obs_to_model(poly_lats, poly_lons, grid_lats, grid_lons):
    """
    Replicates the original 'map_latlon_obs_to_model' bounding logic.
    """
    min_lat, max_lat = np.min(poly_lats), np.max(poly_lats)
    min_lon, max_lon = np.min(poly_lons), np.max(poly_lons)
    
    mask = (grid_lats >= min_lat) & (grid_lats <= max_lat) & (grid_lons >= min_lon) & (grid_lons <= max_lon)
    idx_x, idx_y = np.where(mask)
    
    if len(idx_x) == 0:
        lat_centers = np.abs(grid_lats - np.mean(poly_lats))
        lon_centers = np.abs(grid_lons - np.mean(poly_lons))
        total_dist = lat_centers + lon_centers
        idx_x, idx_y = np.where(total_dist == np.min(total_dist))
        
    return idx_x, idx_y

# ==========================================
# PHASE 4: REPLICATED INDIVIDUAL WORKER TASK
# ==========================================

def plot_frp_predictions(rave_timeline, frp_timeseries, forecast_files,
                         time_forecast, frp_forecast_int_masked, forecast_dates,
                           time_forecast_hybrid, frp_forecast_hybrid, fire_name, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(rave_timeline, frp_timeseries, '-b', label='RAVE Observations')
    
    colors = ['k', 'r']
    for i in range(len(forecast_files)):
        plt.plot(time_forecast[i], frp_forecast_int_masked[i], f"-{colors[i]}", 
                 label=f"Forecast {forecast_dates[i].strftime('%m/%d %H')}Z")
                 
    plt.plot(time_forecast_hybrid, frp_forecast_hybrid, '-g', 
             label=f"Forecast {forecast_dates[1].strftime('%m/%d %H')}Z Hybrid")
    
    plt.xlabel('Date')
    plt.ylabel('Total Fire FRP [MW]')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/frp_rave_vs_prediction_{fire_name.lower().replace(' ','_')}.png", dpi=300)
    print(f'[*] saved plot to {out_dir}/frp_rave_vs_prediction_{fire_name.lower().replace(' ','_')}.png')
    plt.close()

def process_single_fire(args):
    """
    Isolated worker execution context handling a single target fire.
    Generates diagnostic files and returns the path to a cached NetCDF file.
    """
    idx, row, context = args
    
    # Unpack clean, pickle-safe variable primitives from context payload
    forecast_files = context['forecast_files']
    forecast_dates = context['forecast_dates']
    lat_forecast_full = context['lat_forecast_full']
    lon_forecast_full = context['lon_forecast_full']
    Nx_fc_max = context['Nx_fc_max']
    Ny_fc_max = context['Ny_fc_max']
    date_i_dt = context['date_i_dt']
    date_f_dt = context['date_f_dt']
    date_mask_rave_dt = context['date_mask_rave_dt']
    out_dir = context['out_dir']
    cache_dir = context['cache_dir']
    rave_api = context['rave_api']
    plot_fires = context['plot_fires']
    
    fire_name = str(row.get('name', row.get('fire_id', f'target_fire_{idx}'))).replace(" ", "_")
    print(f"[*] [PID {os.getpid()}] Processing cross-grid interpolations for target: {fire_name}")
    
    # Geometrical envelope assignment reconstruction mapping structures
    # --- Geometry Parsing ---
    if 'wkt_geometry' in row and pd.notna(row['wkt_geometry']):
        poly_geom = loads(row['wkt_geometry'])
        
        # Check if geometry is a MultiPolygon or single Polygon
        if poly_geom.geom_type == 'MultiPolygon':
            lon_list, lat_list = [], []
            for part in poly_geom.geoms:
                x, y = part.exterior.xy
                lon_list.extend(x)
                lat_list.extend(y)
            lon_verts = np.array(lon_list)
            lat_verts = np.array(lat_list)
        else:
            # Handle standard single Polygon
            lon_verts, lat_verts = poly_geom.exterior.xy
            lon_verts, lat_verts = np.array(lon_verts), np.array(lat_verts)
    else:
        c_lat, c_lon = row['lat_centroid'], row['lon_centroid']
        lat_verts = np.array([c_lat - 0.25, c_lat - 0.25, c_lat + 0.25, c_lat + 0.25])
        lon_verts = np.array([c_lon - 0.25, c_lon + 0.25, c_lon + 0.25, c_lon - 0.25])
        from shapely.geometry import box
        poly_geom = box(lon_verts.min(), lat_verts.min(), lon_verts.max(), lat_verts.max())

    # Programmatically retrieve RAVE data directly from API subset routines
    try:
        rave_ds = rave_api()._query(
            polygon=poly_geom,
            start=date_i_dt.strftime('%Y-%m-%d %H:%M'),
            end=date_f_dt.strftime('%Y-%m-%d %H:%M'),
            variables=["FRP_MEAN", "FRE"],
            drop_outside=False
        )
    except Exception as e:
        print(f"    [-] RAVE Client query processing error skipped for {fire_name}: {e}")
        return None

    # ==========================================
    # EXTRACT 1D SUMMARY STATISTICS FROM RAVE API
    # ==========================================
    frp_timeseries = rave_ds['FRP_MEAN'].values
    frp_timeseries[frp_timeseries == 0] = np.nan
    rave_timeline = [pd.to_datetime(t) for t in rave_ds['time'].values]

    # Map Canadian Forecasting windows onto target spatial crops bounding box
    fc_x, fc_y = map_latlon_obs_to_model(lat_verts, lon_verts, lat_forecast_full, lon_forecast_full)
    min_fcx, max_fcx = max(np.min(fc_x) - 1, 0), min(np.max(fc_x) + 1, Nx_fc_max - 1)
    min_fcy, max_fcy = max(np.min(fc_y) - 1, 0), min(np.max(fc_y) + 1, Ny_fc_max - 1)

    time_forecast = []
    frp_forecast_int_masked = []

    try:
        # Re-locate the precise index step matching pivot targets inside the dataset timeline
        idx_rave_pivot = min(range(len(rave_timeline)), key=lambda i: abs(rave_timeline[i] - date_mask_rave_dt))
    except ValueError:
        print(f"[-] Pivot timestamp {date_mask_rave_dt} error out boundaries for {fire_name}.")
        return None

    # ===================================================
    # AGGREGATE FORECAST MODELS DIRECTLY ON BOUNDING BOX
    # ===================================================
    for f_idx, fc_file in enumerate(forecast_files):
        with xr.open_dataset(fc_file) as ds_f:
            fc_hours = ds_f['time'].values
            fc_time_axis = np.array([forecast_dates[f_idx] + timedelta(hours=float(th)) for th in fc_hours])
            time_forecast.append(fc_time_axis)
            
            # 1. Get the exact dimension names from the dataset (e.g., 'south_north', 'west_east')
            # 'Time' is usually the first dimension, the other two are spatial
            spatial_dims = [dim for dim in ds_f['FRP'].dims if dim != 'time']
            
            # 2. Use xarray's positional indexer using the exact dimension names
            # This ensures min_fcx always maps to the first spatial dim, and min_fcy to the second
            slicers = {
                spatial_dims[0]: slice(min_fcx, max_fcx + 1),
                spatial_dims[1]: slice(min_fcy, max_fcy + 1)
            }
            crop_ds = ds_f['FRP'].isel(**slicers)
            
            # 3. Sum over the spatial dimensions natively, leaving only the 'Time' dimension intact
            masked_totals = crop_ds.sum(dim=spatial_dims).values
            
            frp_forecast_int_masked.append(masked_totals)
    # ==========================================
    # PHASE 5: HYBRID PREDICTION SYNTHESIS STAGE
    # ==========================================
    time_forecast_hybrid = time_forecast[1]
    frp_forecast_hybrid = np.full(time_forecast_hybrid.shape, np.nan)
    
    try:
        idx_fc1_pivot = list(time_forecast[0]).index(date_mask_rave_dt)
        idx_fc2_pivot = list(time_forecast[1]).index(date_mask_rave_dt)
    except ValueError:
        print(f"[-] Warning: Time axes are misaligned for {fire_name}. Skipping hybrid step.")
        return None

    historical_rave_sample = frp_timeseries[max(0, idx_rave_pivot-24):idx_rave_pivot]
    fc1_reference_sample = frp_forecast_int_masked[0][max(0, idx_fc1_pivot-24):idx_fc1_pivot]
    
    # Fill hybrid sequences safely across available forecast horizons
    scaling_ratio = np.nanmean(frp_forecast_int_masked[1][idx_fc2_pivot:idx_fc2_pivot+24]) / np.nanmean(fc1_reference_sample)
    frp_forecast_hybrid[idx_fc2_pivot:idx_fc2_pivot+24] = historical_rave_sample * scaling_ratio
    
    scaling_ratio_day2 = np.nanmean(frp_forecast_int_masked[1][idx_fc2_pivot+24:idx_fc2_pivot+48]) / np.nanmean(fc1_reference_sample)
    frp_forecast_hybrid[idx_fc2_pivot+24:idx_fc2_pivot+48] = historical_rave_sample * scaling_ratio_day2

    # ==========================================
    # PHASE 6: DIAGNOSTIC GRAPH GENERATION
    # ==========================================
    if plot_fires:
        plot_frp_predictions(rave_timeline, frp_timeseries, 
                            forecast_files, time_forecast, 
                            frp_forecast_int_masked, forecast_dates,
                            time_forecast_hybrid, frp_forecast_hybrid, 
                            fire_name, out_dir)

    # ==========================================
    # PHASE 7: LOCAL FILE SYNC AND NETCDF CACHING
    # ==========================================
    csv_out_path = f"{out_dir}/rave_frp_{fire_name.lower().replace(' ','_')}.csv"
    records_df = pd.DataFrame({
        'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in rave_timeline],
        'rave_frp_mw': np.nan_to_num(frp_timeseries, nan=0.0).astype(int)
    })
    records_df.to_csv(csv_out_path, index=False)
    
    # Store output data arrays inside a temporary xarray Dataset object
    ds_cache = xr.Dataset(
        data_vars=dict(
            fc_yesterday=(["time_fc0"], frp_forecast_int_masked[0]),
            fc_latest=(["time_fc1"], frp_forecast_int_masked[1]),
            fc_hybrid=(["time_hybrid"], frp_forecast_hybrid)
        ),
        coords=dict(
            time_fc0=[t.strftime('%Y-%m-%d %H:%M:%S') for t in time_forecast[0]],
            time_fc1=[t.strftime('%Y-%m-%d %H:%M:%S') for t in time_forecast[1]],
            time_hybrid=[t.strftime('%Y-%m-%d %H:%M:%S') for t in time_forecast_hybrid]
        ),
        attrs=dict(fire_name=fire_name, parent_idx=idx)
    )
    
    # Dump cleanly to unique local disk tracking target to safely escape sub-process boundary
    cache_filepath = os.path.join(cache_dir, f"cache_arrays_{fire_name}_{idx}.nc")
    ds_cache.to_netcdf(cache_filepath)
    ds_cache.close()
    
    return cache_filepath

# ==========================================
# PHASE 3: CORE PREDICTIVE EXECUTIVE PIPELINE
# ==========================================

def execute_predictive_fire_pipeline(target_dt, out_dir = f"{CACHE_BASE_DIR}/ca_frp",
                                     plot_fires = False):
    """
    Consolidated operational function replacing the original multi-step MATLAB execution loop.
    Integrates the programmatic RAVEClient API to extract data directly.
    """
    target_date_str = target_dt.strftime('%Y%m%d')
    
    # Establish running timeframe context anchors
    date_i_dt = target_dt - timedelta(days=3)
    date_f_dt = target_dt + timedelta(days=3)
    date_mask_rave_dt = target_dt + timedelta(hours=6) # 06Z forecast mark
    
    forecast_path = '/data/jaredgoldman/INSPYRE/sample_firesmoke_ca_data/'
    fire_polygon_file = '/data/jaredgoldman/cache/active_fires/current/fire_pipeline_manifest.csv'
    
    os.makedirs(f"{out_dir}", exist_ok=True)
    os.makedirs("csv", exist_ok=True)
    
    download_canadian_forecasts(target_date_str, forecast_path)

    # 2. Map Forecast NetCDF Data Target Handles    
    yesterday_str = (target_dt - timedelta(days=1)).strftime('%Y%m%d')
    forecast_files = [
        os.path.join(forecast_path, f"fwf-hourly-d02-{yesterday_str}06.nc"),
        os.path.join(forecast_path, f"fwf-hourly-d02-{target_date_str}06.nc")
    ]
    forecast_dates = [
        pd.to_datetime(f"{yesterday_str} 06:00:00"),
        pd.to_datetime(f"{target_date_str} 06:00:00")
    ]
    
    # 3. Read Inbound Pipeline Manifest Inventory Targets
    if not os.path.exists(fire_polygon_file):
        raise FileNotFoundError(f"Missing base operational manifest tracking file: {fire_polygon_file}")
        
    fire_df = pd.read_csv(fire_polygon_file)
    fire_df = fire_df[fire_df['peak_area_km2'] > 10].reset_index(drop=True)
    
    if fire_df.empty:
        print("[!] No active targets found matching minimum size conditions (>10 km²).")
        return

    # Instantiate programmatic RAVE API client instance
    rave_api = RAVEClient

    # Read base static spatial framework dimensions of forecasting engines
    with xr.open_dataset(forecast_files[0]) as fc_sample:
        lat_forecast_full = fc_sample['XLAT'].values
        lon_forecast_full = fc_sample['XLONG'].values
        Nx_fc_max, Ny_fc_max = lat_forecast_full.shape

    # ==========================================
    # PARALLEL EXECUTION MULTIPROCESSING HARNESS
    # ==========================================
    cache_dir = os.path.join(out_dir, ".xarray_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Build a pickle-safe context payload mapping primitives across sub-processes
    context_payload = {
        'forecast_files': forecast_files,
        'forecast_dates': forecast_dates,
        'lat_forecast_full': lat_forecast_full,
        'lon_forecast_full': lon_forecast_full,
        'Nx_fc_max': Nx_fc_max,
        'Ny_fc_max': Ny_fc_max,
        'date_i_dt': date_i_dt,
        'date_f_dt': date_f_dt,
        'date_mask_rave_dt': date_mask_rave_dt,
        'out_dir': out_dir,
        'cache_dir': cache_dir,
        'rave_api': rave_api,
        'plot_fires': plot_fires
    }
    
    worker_tasks = [(idx, row, context_payload) for idx, row in fire_df.iterrows()]
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
                
                # Collect timelines into dictionaries mapping time string to value
                data_by_type = {}
                all_timestamps = set()
                
                for label, coordinate_dim, dataset_var in [
                    ('fc_yesterday', 'time_fc0', 'fc_yesterday'),
                    ('fc_latest', 'time_fc1', 'fc_latest'),
                    ('fc_hybrid', 'time_hybrid', 'fc_hybrid')
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
                
                # Set up the multi-key structure: fire_name and time
                fire_ts_df.index.name = 'time'
                fire_ts_df = fire_ts_df.reset_index()
                fire_ts_df.insert(0, 'fire_name', fire_name)
                
                timeseries_list.append(fire_ts_df)

        # Combine all individual fires into a single master time series dataframe
        if timeseries_list:
            master_ts_df = pd.concat(timeseries_list, ignore_index=True)
            # Set the keys (fire_name, time) as the multi-index
            master_ts_df.set_index(['fire_name', 'time'], inplace=True)
            
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