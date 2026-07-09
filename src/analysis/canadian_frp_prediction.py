import os
import glob
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.wkt import loads

# Import the programmatic internal data API directly
from data.clients.rave_client import RAVEClient
from utils.constants import CACHE_BASE_DIR

# ==========================================
# PHASE 1: AUTOMATED DATA RETRIEVAL (CANADIAN)
# ==========================================

def download_canadian_forecasts(target_date_str, output_dir):
    """
    Automates downloading the latest and yesterday's Canadian forecasts.
    Expects target_date_str in 'YYYYMMDD' format.
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
        if not os.path.exists(dest_path):
            print(f"[+] Downloading Canadian Forecast: {filename}")
            try:
                urllib.request.urlretrieve(url, dest_path)
            except Exception as e:
                print(f"[-] Failed to fetch {url}: {e}")
        downloaded_files.append(dest_path)
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
# PHASE 3: CORE PREDICTIVE EXECUTIVE PIPELINE
# ==========================================

def execute_predictive_fire_pipeline(target_date_str, out_dir = f"{CACHE_BASE_DIR}/ca_frp", run_downloads=True):
    """
    Consolidated operational function replacing the original multi-step MATLAB execution loop.
    Integrates the programmatic RAVEClient API to extract data directly.
    
    Parameters:
    -----------
    target_date_str : str
        Format 'YYYYMMDD' (e.g., '20260623')
    """
    forecast_pivot_dt = datetime.strptime(target_date_str, "%Y%m%d")
    
    # Establish running timeframe context anchors
    date_i_dt = forecast_pivot_dt - timedelta(days=3)
    date_f_dt = forecast_pivot_dt + timedelta(days=3)
    date_mask_rave_dt = forecast_pivot_dt + timedelta(hours=6) # 06Z forecast mark
    
    forecast_path = '/data/saide/INSPYRE/sample_firesmoke_ca_data/'
    fire_polygon_file = '/data/jaredgoldman/cache/active_fires/current/fire_pipeline_manifest.csv'
    
    os.makedirs(f"{out_dir}", exist_ok=True)
    os.makedirs("csv", exist_ok=True)
    
    # 1. Trigger Async Canadian Download Engine
    if run_downloads:
        print("[*] Checking near-real-time Canadian forecast downloads...")
        download_canadian_forecasts(target_date_str, forecast_path)

    # 2. Map Forecast NetCDF Data Target Handles
    yesterday_str = (forecast_pivot_dt - timedelta(days=1)).strftime('%Y%m%d')
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
    rave_api = RAVEClient()[cite: 5]

    # Read base static spatial framework dimensions of forecasting engines
    with xr.open_dataset(forecast_files[0]) as fc_sample:
        lat_forecast_full = fc_sample['XLAT'].values
        lon_forecast_full = fc_sample['XLONG'].values
        Nx_fc_max, Ny_fc_max = lat_forecast_full.shape

    # ==========================================
    # PHASE 4: REPLICATE INDIVIDUAL FIRE MAPPING
    # ==========================================
    for idx, row in fire_df.iterrows():
        fire_name = str(row.get('name', row.get('fire_id', f'target_fire_{idx}'))).replace(" ", "_")
        print(f"[*] Processing cross-grid interpolations for target: {fire_name}")
        
        # Geometrical envelope assignment reconstruction mapping structures
        if 'wkt_geometry' in row and pd.notna(row['wkt_geometry']):
            poly_geom = loads(row['wkt_geometry'])
            lon_verts, lat_verts = poly_geom.exterior.xy
            lon_verts, lat_verts = np.array(lon_verts), np.array(lat_verts)
        else:
            c_lat, c_lon = row['lat_centroid'], row['lon_centroid']
            lat_verts = np.array([c_lat - 0.25, c_lat - 0.25, c_lat + 0.25, c_lat + 0.25])
            lon_verts = np.array([c_lon - 0.25, c_lon + 0.25, c_lon + 0.25, c_lon - 0.25])
            from shapely.geometry import box
            poly_geom = box(lon_verts.min(), lat_verts.min(), lon_verts.max(), lat_verts.max())

        # Programmatically retrieve RAVE data directly from API subset routines[cite: 5]
        print(f"    [->] Programmatically fetching RAVE values via RAVEClient inside geometry...")
        try:
            rave_ds = rave_api._query([cite: 5]
                polygon=poly_geom,[cite: 5]
                start=date_i_dt.strftime('%Y-%m-%d %H:%M'),[cite: 5]
                end=date_f_dt.strftime('%Y-%m-%d %H:%M'),[cite: 5]
                variables=["FRP_MEAN", "FRE"],[cite: 5]
                drop_outside=False[cite: 5]
            )
        except Exception as e:
            print(f"    [-] RAVE Client query processing error skipped: {e}")
            continue

        # Extract underlying coordinate grids and maps programmatically handled by RAVEClient[cite: 5]
        lat_RAVE_crop = rave_ds['grid_latt'].values
        lon_RAVE_crop = rave_ds['grid_lont'].values
        
        # Handle 0..360 normalization if RAVE data still relies on raw grids[cite: 5]
        if np.nanmin(lon_RAVE_crop) >= 0 and np.nanmax(lon_RAVE_crop) > 180:
            lon_RAVE_crop = ((lon_RAVE_crop + 180) % 360) - 180[cite: 5]
            
        frp_cube = rave_ds['FRP_MEAN'].values
        fre_cube = rave_ds['FRE'].values
        rave_timeline = [pd.to_datetime(t) for t in rave_ds['time'].values]

        # Calculate time series totals
        frp_timeseries = np.nansum(np.nansum(frp_cube, axis=1), axis=0)
        frp_timeseries[frp_timeseries == 0] = np.nan

        # Map Canadian Forecasting windows onto target spatial crops
        fc_x, fc_y = map_latlon_obs_to_model(lat_verts, lon_verts, lat_forecast_full, lon_forecast_full)
        min_fcx, max_fcx = max(np.min(fc_x) - 1, 0), min(np.max(fc_x) + 1, Nx_fc_max - 1)
        min_fcy, max_fcy = max(np.min(fc_y) - 1, 0), min(np.max(fc_y) + 1, Ny_fc_max - 1)
        
        lon_fc_crop = lon_forecast_full[min_fcx:max_fcx+1, min_fcy:max_fcy+1]
        lat_fc_crop = lat_forecast_full[min_fcx:max_fcx+1, min_fcy:max_fcy+1]

        time_forecast = []
        frp_forecast_int_masked = []

        try:
            # Re-locate the precise index step matching pivot targets inside the dataset timeline
            idx_rave_pivot = min(range(len(rave_timeline)), key=lambda i: abs(rave_timeline[i] - date_mask_rave_dt))
        except ValueError:
            raise ValueError(f"Pivot timestamp {date_mask_rave_dt} is positioned outside the specified boundaries.")

        # Re-verify lookback index boundaries to calculate the operational FRE fire activity mask
        fre_lookback_window = fre_cube[:, :, max(0, idx_rave_pivot-23):idx_rave_pivot+1]
        frp_mask = np.nansum(fre_lookback_window, axis=2) > 0

        for f_idx, fc_file in enumerate(forecast_files):
            with xr.open_dataset(fc_file) as ds_f:
                fc_hours = ds_f['Time'].values
                fc_time_axis = np.array([forecast_dates[f_idx] + timedelta(hours=float(th)) for th in fc_hours])
                time_forecast.append(fc_time_axis)
                
                raw_frp_fc = ds_f['FRP'].values[min_fcx:max_fcx+1, min_fcy:max_fcy+1, :]
                Nt_fc = len(fc_hours)
                
                masked_totals = np.zeros(Nt_fc)
                for t in range(Nt_fc):
                    grid_z = griddata(
                        (lon_fc_crop.ravel(), lat_fc_crop.ravel()), 
                        raw_frp_fc[:, :, t].ravel(), 
                        (lon_RAVE_crop, lat_RAVE_crop), 
                        method='linear'
                    )
                    masked_totals[t] = np.nansum(grid_z[frp_mask])
                
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
            continue

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
        
        plt.savefig(f"figs/frp_rave_vs_prediction_{fire_name}.png", dpi=300)
        plt.close()

        # ==========================================
        # PHASE 7: BROAD DATA OUTPUT SYNCHRONIZATION
        # ==========================================
        csv_out_path = f"csv/rave_frp_{fire_name}.csv"
        records_df = pd.DataFrame({
            'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in rave_timeline],
            'rave_frp_mw': np.nan_to_num(frp_timeseries, nan=0.0).astype(int)
        })
        records_df.to_csv(csv_out_path, index=False)
        
        # Merge calculated arrays into columns within the unified file dataframe
        for label, t_axis, data_v in [
            ('fc_yesterday', time_forecast[0], frp_forecast_int_masked[0]),
            ('fc_latest', time_forecast[1], frp_forecast_int_masked[1]),
            ('fc_hybrid', time_forecast_hybrid, frp_forecast_hybrid)
        ]:
            col_name = f"{fire_name}_{label}"
            val_map = {t.strftime('%Y-%m-%d %H:%M:%S'): val for t, val in zip(t_axis, data_v)}
            fire_df[col_name] = fire_df.get('t_start', pd.Series(dtype=str)).map(val_map)

    # Save data updates back down to your pipeline storage target manifest
    fire_df.to_csv(fire_polygon_file, index=False)
    print(f"[+] All procedures concluded successfully. Master manifest tracking updated at: {fire_polygon_file}")

if __name__ == "__main__":
    # Context-aware real-time dynamic analysis harness execution
    execute_predictive_fire_pipeline(target_date_str="20260623", run_downloads=True)