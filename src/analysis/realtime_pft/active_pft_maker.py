import os
import sys
import datetime
from pathlib import Path
from typing import  List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import xarray as xr
from tqdm import tqdm as timer

from utils.constants import CACHE_BASE_DIR
import analysis.mapping.config as config
from analysis.mapping.pft_gen_parallel import (
    transform_ds_for_parallel,
    sounding_worker,
    pft_worker_direct
)
from analysis.realtime_pft.rrfs_para_oper_client import RRFSParallelOperClient
from analysis.realtime_pft.plot_realtime_pft import generate_realtime_pft_plots


def _process_single_nc_to_pft(
    nc_path_str: str, 
    fire_key: str, 
    fire_name: str, 
    max_plume_temps: List[float]
) -> List[Dict]:
    """
    Worker process: Opens a single subsetted NetCDF file, transforms it to a 
    dictionary payload, generates soundings for all spatial coordinates, 
    and computes PFT values without merged dataset logic.
    """
    results = []
    nc_path = Path(nc_path_str)
    fire_key = fire_key or ds.attrs.get('fire_index_id', '')
    fire_name = fire_name or ds.attrs.get('fire_name', fire_key)
    if not nc_path.exists():
        return results

    try:
        # 1. Open NetCDF file directly
        with xr.open_dataset(nc_path) as ds:
            payload = transform_ds_for_parallel(
                ds, 
                features=['t', 'r', 'gh', 'u', 'v'], 
                target_dim='isobaricInhPa'
            )

        # 2. Iterate through each lat/lon coordinate in payload
        for (lat, lon), point_data in payload.items():
            num_times = len(point_data['time'])
            
            for t_idx in range(num_times):
                # 3. Build Sounding using pft_gen_parallel worker logic
                snd_out = sounding_worker(point_data, (lat, lon), t_idx, fire_key)
                if snd_out is None:
                    continue

                snd, _ = snd_out

                # 4. Calculate PFT across plume top temperatures
                for max_temp in max_plume_temps:
                    try:
                        pft_out = pft_worker_direct(snd_out, max_temp)
                        (valid_time, pft_val, p_temp), _ = pft_out

                        results.append({
                            "fire_name": fire_name,
                            "fire_key": fire_key,
                            "time": valid_time,
                            "pft_value": pft_val,
                            "lat": lat,
                            "lon": lon,
                            "plume_temp": p_temp
                        })
                    except Exception:
                        continue

    except Exception as e:
        print(f"[-] Error calculating PFT for {nc_path.name}: {e}", file=sys.stderr)

    return results


def run_parallel_pft_from_netcdf(
    summary_csv_path: Path, 
    manifest_path: str,
    cycle_date: str, 
    cycle_hour: str, 
    max_workers: int = 4
) -> Path:
    """Processes extracted NetCDF files in parallel and amalgamates results into a CSV file."""
    df_summary = pd.read_csv(summary_csv_path)
    df_manifest = pd.read_csv(manifest_path)
    successful_df = (
        df_summary[df_summary['status'] == 'SUCCESS']
        .drop(columns=['fire_name'], errors='ignore')
        .merge(df_manifest[['fire_index_id', 'name']], on='fire_index_id', how='left')
        .rename(columns={'name' : 'fire_name'})
    )
    if successful_df.empty:
        raise ValueError("No valid NetCDF files recorded in processing summary.")

    max_plume_temps = getattr(config, 'MAX_PLUME_TOP_TS', [0.0, -10.0, -20.0])
    
    tasks = []
    for _, row in successful_df.iterrows():
        tasks.append((
            row['output_nc'], 
            row['fire_index_id'], 
            row.get('fire_name', row['fire_index_id']), 
            max_plume_temps
        ))

    print(f"[*] Computing PFT values across {len(tasks)} NetCDF files using {max_workers} processes...")

    all_records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_nc_to_pft, *task): task[0] for task in tasks
        }

        for future in timer(as_completed(futures), desc="PFT Parallel Processing", total=len(futures)):
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                print(f"[-] Task failure: {e}")

    if not all_records:
        raise RuntimeError("No PFT values were successfully computed from NetCDF soundings.")

    # Convert records into DataFrame and format columns per realtime_pft_algorithm.md
    df_final = pd.DataFrame(all_records)
    df_final = df_final[['fire_name', 'fire_key', 'time', 'pft_value', 'lat', 'lon', 'plume_temp']]
    df_final = df_final.sort_values(by=['fire_key', 'time']).reset_index(drop=True)

    # Format timestamped output CSV name
    
    output_dir = Path(CACHE_BASE_DIR) / "pft_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / f"pft_rrfs_{cycle_date}_{cycle_hour}z.csv"
    df_final.to_csv(output_csv, index=False)

    print(f"[+] Successfully saved amalgamated fire PFT results to: {output_csv}")
    return output_csv


# =============================================================================
# Execution Entry Point
# =============================================================================

if __name__ == "__main__":
    today_str = datetime.date.today().strftime("%Y_%m_%d")
    manifest_path = os.path.join(CACHE_BASE_DIR, "active_fires", today_str, "fire_pipeline_manifest.csv")
    fxx_range = 24
    fxx_interval = 1
    max_workers = 48
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(CACHE_BASE_DIR, "active_fires", "2026_08_20", "fire_pipeline_manifest.csv")

    # 1. Download RRFS pressure level files and extract fire-subsetted NetCDFs
    client = RRFSParallelOperClient(csv_manifest_path=manifest_path)
    grib_paths, cycle_date, cycle_hour = client.download_latest_prslev_data(fxx_hours=range(0, fxx_range + 1,fxx_interval))
    summary_csv = client.process_fires_in_parallel(grib_paths, max_workers=getattr(config, 'max_workers', max_workers))

    # 2. Run PFT processor on NetCDF files in parallel (no dataset merging)
    out_csv = run_parallel_pft_from_netcdf(
        summary_csv_path=summary_csv,
        manifest_path = manifest_path,
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
        max_workers=getattr(config, 'max_workers', 4)
    )
    generate_realtime_pft_plots(out_csv, max_workers=max_workers)