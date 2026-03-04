from __future__ import annotations

import data.clients as clients

import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import os
import pandas as pd
from random import sample
import geopandas as gpd
from shapely import box, contains
from utils import FEATURE_OUTPUT_DIR, DATA_DIR

parser = argparse.ArgumentParser(
                    prog='ParallelDataDownloader',
                    description='Download fire features as a csv')
parser.add_argument('--debug', action = 'store_true', dest = 'debug', default= False,
                    help = "turn debug mode on to run dataset generation in series")
parser.add_argument('-o', '--out', type = str, dest = 'o', 
                    default = "average_polygon_features_parallel.csv",
                    help = f"path to output file relative to {FEATURE_OUTPUT_DIR}")
parser.add_argument('-c', '--cp', type = str, dest = 'c', default = "cp_na.csv",
                    help = f"path to `cp_na.csv` file with fire information relative to {DATA_DIR}")
parser.add_argument('-p', '--poly', type = str, dest = 'p', default = "cp_poly.gpkg",
                    help = f"path to `cp_poly.gpkg` file with fire polygon information relative to {DATA_DIR}")
parser.add_argument('-w', '--num_workers', type = int, dest = 'num_workers', default = 8,
                    help = "maximum number of workers to use")
parser.add_argument('-f', '--flush_n_fires', type = int, dest = 'flush_n_fires', default = 50,
                    help = "frequency that fire features are written to output file")
parser.add_argument('--third', type = int, dest = 'third', default = 0,
                    help = "which third of dataset to process. if value is 0, process the whole dataset.")
# -----------------------------
# Helpers
# -----------------------------
def varname_map(name, var_name):
    if not 'hrrr' in name:
        return f"{name}_{var_name}"
    can_map = { 'r' : 'rh', 't2m' : 't', 'd2m' : 'dpt'}
    us_map = { 'r2' : 'rh', 't2m' : 't', 'd2m' : 'dpt',
              'u10' : 'u', 'v10' : 'v'}
    if 'can' in name:
        if var_name in can_map.keys():
            return f"hrrr_{can_map[var_name]}"
        return f"hrrr_{var_name}"
    if var_name in us_map.keys():
        return f"hrrr_{us_map[var_name]}"
    return f"hrrr_{var_name}"

def is_conus(perim):
    conus_polygon = box(-125.0, 24.0, -66.5, 49.5)
    return contains(conus_polygon, perim)

def skip_fire(start, end, perim):
    skip_lb = pd.Timestamp("2021-01-01") 
    skip_ub = pd.Timestamp("2024-09-19")
    start = pd.Timestamp(start) if type(start) is str else start
    end = pd.Timestamp(end) if type(end) is str else end
    if start <= pd.Timestamp('2019-06-04 00:00:00'):
        return True
    if start >= skip_lb and start < skip_ub \
        or end > skip_lb and end <= skip_lb:
        return True
    if (start - pd.Timedelta(1, 'D')).year < (end + pd.Timedelta(1, 'D')).year:
        return True
    if not is_conus(perim) and end < skip_lb:
        return True
    return False

def time_bins(times):
    unique_dates, inverse = np.unique(times.astype("datetime64[D]"), 
                                      return_inverse=True)

    indices_by_date = {
        date: np.where(inverse == i)[0]
        for i, date in enumerate(unique_dates)
    }
    return indices_by_date

def safe_buffer(poly, buf=0.15):
    # buffer can occasionally create invalid geometries; buffer(0) cleans in many cases
    return poly.buffer(buf, join_style=3).buffer(0)

def find_bad_data(fname):
    df = pd.read_csv(fname)
    import ipdb; ipdb.set_trace()

def compute_daily_features_for_fire(
    cp_idx,
    this_poly_gdf,
    this_cp_df,
    client_query_specs,  # <-- dict/specs, not client instances (see below)
    DEBUG_MODE,
    feature_start_pad="1D",
    feature_end_pad="1D"
    ) -> List[Dict[str, Any]]:
    """
    Runs ALL client queries for one fire and returns list of day_dict rows.
    Designed to be executed in a worker process.
    """

    # --- extract geometry/times ---
    fire_poly = this_poly_gdf["geometry"].values[0]
    fire_tmin = pd.Timestamp(this_cp_df["dtime_min"].values[0])
    fire_tmax = pd.Timestamp(this_cp_df["dtime_max"].values[0])

    # skip logic (uses your existing functions)
    if skip_fire(fire_tmin, fire_tmax, fire_poly):
        return []

    fire_poly = safe_buffer(fire_poly, 0.15)

    start = fire_tmin - pd.Timedelta(feature_start_pad)
    end   = fire_tmax + pd.Timedelta(feature_end_pad)

    def run_one_client_safe(spec):
        if DEBUG_MODE:
            return _run_one_client_safe(spec)
        
        try:
            return _run_one_client_safe(spec)
        except Exception as e:
            name = spec.get("name", "unknown")
            print(f"[WARN] {name} failed: {e}")
            return None

    # --- query clients in parallel (threads: network I/O) ---
    def _run_one_client_safe(spec):
        name = spec["name"]
        vars_ = spec["vars"]

        # conus gating example you had
        if is_conus(fire_poly) and name == "can_hrrr":
            print("fire is conus and model is canadian")
            return None
        elif not is_conus(fire_poly) and name == "us_hrrr":
            print("fire is not conus and model is us")
            return None

        # IMPORTANT: create the client INSIDE the worker (avoid pickling sessions, locks, etc.)
        client_ctor = spec["client_ctor"]
        client_kwargs = spec.get("client_kwargs", {})
        client = client_ctor(**client_kwargs)
        
        ds = client.query(
            polygon=fire_poly,
            start=start,
            end=end,
            variables=vars_,
        )
        dates = time_bins(ds.time.values)
        day_dict = {date : {} for date in dates.keys()}
        for date in dates.keys():
            # compute mean for each var
            idx = dates[date]
            for var_name in ds.data_vars:
                arr = ds[var_name].isel(time=idx).values
                day_dict[date][varname_map(name, var_name)] = float(np.nanmean(arr))
        del ds
        return (name, day_dict)

    ds_list = []
    if DEBUG_MODE:
        for spec in client_query_specs:
            out = run_one_client_safe(spec)
            if out is not None:
                ds_list.append(out)
    else:
    # Thread pool: good for requests/HTTP, Herbie downloads, etc.
        with ThreadPoolExecutor(max_workers=min(6, len(client_query_specs))) as tpex:        
            futures = [tpex.submit(run_one_client_safe, spec) for spec in client_query_specs]
            for f in as_completed(futures):
                out = f.result()
                if out is not None:
                    ds_list.append(out)

    if not ds_list:
        return []

    # --- aggregate per day ---
    dates = np.array(pd.date_range(fire_tmin - pd.Timedelta(1, "D"),
                                   fire_tmax + pd.Timedelta(1, "D"))).astype("datetime64[D]")

    data_per_day: List[Dict[str, Any]] = []

    for date in dates:
        day_dict: Dict[str, Any] = {"cp": cp_idx, "day": date}
        for _, stats in ds_list:
            # Precompute bins once per dataset
            # Assumes time_bins returns dict {datetime64[D]: indices}
            for day, results in stats.items(): 
                if day != date:
                    continue
                # compute mean for each var
                for var_name, avg in results.items():
                    day_dict[var_name] = avg
        data_per_day.append(day_dict)
    
    return data_per_day

def compute_daily_features_for_fire_safe(
        cp_idx,
        this_poly_gdf,
        this_cp_df,
        client_query_specs,  # <-- dict/specs, not client instances (see below)
        DEBUG_MODE,
        feature_start_pad="1D",
        feature_end_pad="1D"
    ) -> List[Dict[str, Any]]:
    if DEBUG_MODE:
        return compute_daily_features_for_fire(
            cp_idx,
            this_poly_gdf,
            this_cp_df,
            client_query_specs,  # <-- dict/specs, not client instances (see below)
            DEBUG_MODE,
            feature_start_pad=feature_start_pad,
            feature_end_pad=feature_end_pad
        )
    try:
        out = compute_daily_features_for_fire(
            cp_idx,
            this_poly_gdf,
            this_cp_df,
            client_query_specs,  # <-- dict/specs, not client instances (see below)
            DEBUG_MODE,
            feature_start_pad=feature_start_pad,
            feature_end_pad=feature_end_pad
        )
        return out
    except Exception as e:
        print(f"[WARN] {cp_idx} fire failed: {e}")
        return None
    
hrrr_headers = [f'hrrr_{feat}' for feat in ['dpt', 'u', 'v', 't', 'rh', 'tp', 'mstav', 'sdwe']]

client_query_specs = [
    {"name": "esi", "client_ctor": clients.ESIClient, "client_kwargs": {}, "vars": ["DFPPM"]},
    {"name": "firms", "client_ctor": clients.FirmsClient, "client_kwargs": {}, "vars": ["frp"]},
    {"name": "us_hrrr", "client_ctor": clients.HRRRClient, "client_kwargs": {}, "vars": [
        ":TMP:2 m",
        ":DPT:2 m",
        ":UGRD:10 m",
        ":VGRD:10 m",
        ":RH:2 m",
        ":MSTAV:",
        ":WEASD:",
    ]},
    {"name": "can_hrrr", "client_ctor": clients.HRRRClient, "client_kwargs": {}, "vars": [
        ":tp:",
        ":r:",
        ":u:1000",
        ":v:1000",
        ":2t:",
        ":2d:",
    ]},
    {"name": "modis", "client_ctor": clients.MODISClient, "client_kwargs": {}, "vars": ["MaxFRP"]},
    {"name": "rave", "client_ctor": clients.RAVEClient, "client_kwargs": {}, "vars": ["FRP_MEAN", "FRP_SD"]},
]

def main(cp: pd.DataFrame, 
         feature_file: str, 
         cp_poly: gpd.GeoDataFrame,
         flush_fires: int = 50,
         max_workers: int = 8,
         DEBUG_MODE: bool = False,
         THIRD: int = 0):
    # -----------------------------
    # Main driver
    # -----------------------------
    # IMPORTANT: don't pass instantiated clients into processes.
    # Pass constructors + kwargs so workers create their own.
    # Pre-split cp indices in the parent
    all_cps = list(cp.cp.unique())
    all_fires = len(all_cps)
    if THIRD == 1:
        cp_ids = all_cps[:int(all_fires/2)+1]
    elif THIRD == 2:
        cp_ids = all_cps[int(all_fires/2):2*int(all_fires/3)+1]
    elif THIRD == 3:
        cp_ids = all_cps[2*int(all_fires/3):]
    else:
        cp_ids = all_cps

    all_fires = len(cp_ids)

    random_cps = sample(cp_ids, all_fires)
    header = {"cp": [], "day": []}
    for val in client_query_specs:
        if 'hrrr' in val['name']:
            continue
        for varname in val['vars']:
            header[varname_map(val["name"], varname)] = []
    for k in hrrr_headers:
        header[k] = []
    pd.DataFrame(header).to_csv(feature_file, index = False)
    written_header = True

    start_time = datetime.now()

    # Chunked write buffer
    buffer_rows: List[Dict[str, Any]] = []
    flush_every_n_fires = flush_fires  # tune this
    max_workers = max_workers # tune: start with 4-8 depending on CPU/network

    done_count = 0
    # Use processes for per-fire parallelism (bigger jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as ppex:
        futures = []
        for cp_idx in random_cps:
            this_poly = cp_poly[cp_poly.cp == cp_idx]
            this_cp = cp[cp.cp == cp_idx]
            if DEBUG_MODE:
                f = compute_daily_features_for_fire(
                        cp_idx,
                        this_poly,
                        this_cp,
                        client_query_specs,
                        DEBUG_MODE=DEBUG_MODE)
                done_count = 0
               
                if f is None:
                    done_count += 1
                    continue
                rows = f
                done_count += 1
                if rows:
                    buffer_rows.extend(rows)

                # periodic flush
                if done_count % flush_every_n_fires == 0:
                    if buffer_rows:
                        df_out = pd.DataFrame(buffer_rows)
                        df_old = pd.read_csv(feature_file)
                        pd.concat([df_old, df_out]).to_csv(feature_file, index = False)
                        buffer_rows.clear()

                    elapsed = datetime.now() - start_time
                    print(f"{int(done_count/all_fires*100)}% fires done in {int(elapsed.total_seconds()/60)} min")
            else:
                futures.append(
                    ppex.submit(
                        compute_daily_features_for_fire,
                        cp_idx,
                        this_poly,
                        this_cp,
                        client_query_specs,
                        DEBUG_MODE,
                    )
                )

        if DEBUG_MODE:
            import ipdb; ipdb.set_trace()

        done_count = 0
        for f in as_completed(futures):
            if f is None:
                done_count += 1
                continue
            try:
                rows = f.result()
            except Exception as e:
                print(f"[WARN] failed to load row with exception {e}")
                done_count += 1
                continue
            done_count += 1

            if rows:
                buffer_rows.extend(rows)

            # periodic flush
            if done_count % flush_every_n_fires == 0:
                if buffer_rows:
                    df_out = pd.DataFrame(buffer_rows)

                    # append mode, write header once
                    df_old = pd.read_csv(feature_file)
                    pd.concat([df_old, df_out]).to_csv(feature_file, index = False)
                    buffer_rows.clear()

                elapsed = datetime.now() - start_time
                print(f"{int(done_count/all_fires*100)}% fires done in {int(elapsed.total_seconds()/60)} min")

    # final flush
    if buffer_rows:
        pd.DataFrame(buffer_rows).to_csv(feature_file, mode="a", header=not written_header, index=False)
        print(f"saved features to {feature_file}")

if __name__ == "__main__":
    args = parser.parse_args()
    # find_bad_data(os.path.join(FEATURE_OUTPUT_DIR, args.o))

    main(cp           = pd.read_csv(os.path.join(DATA_DIR, args.c)), 
         feature_file = os.path.join(FEATURE_OUTPUT_DIR, args.o), 
         cp_poly      = gpd.read_file(os.path.join(DATA_DIR, args.p)),
         flush_fires  = args.flush_n_fires,
         max_workers  = args.num_workers,
         DEBUG_MODE   = args.debug,
         THIRD         = args.third)