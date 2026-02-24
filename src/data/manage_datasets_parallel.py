from __future__ import annotations

import data.clients as clients

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import box, contains


# -----------------------------
# Helpers
# -----------------------------


def is_conus(perim):
    conus_polygon = box(-125.0, 24.0, -66.5, 49.5)
    if contains(conus_polygon, perim):
        return True
    return False

def skip_fire(start, end, perim):
    skip_lb = pd.Timestamp("2021-01-01") 
    skip_ub = pd.Timestamp("2024-09-19")
    start = pd.Timestamp(start) if type(start) is str else start
    end = pd.Timestamp(end) if type(end) is str else end
    if start >= skip_lb and start < skip_ub \
        or end > skip_lb and end <= skip_lb:
        return True
    if (start - pd.Timedelta(1, 'D')).year < (end + pd.Timedelta(1, 'D')).year:
        return True
    if not is_conus(perim) and end < skip_lb:
        return True
    return False

def time_bins(times):
    dates = times.astype("datetime64[D]")

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

def compute_daily_features_for_fire(
    cp_idx,
    this_poly_gdf,
    this_cp_df,
    client_query_specs,  # <-- dict/specs, not client instances (see below)
    feature_start_pad="1D",
    feature_end_pad="1D",
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

    # --- query clients in parallel (threads: network I/O) ---
    def run_one_client(spec):
        name = spec["name"]
        vars_ = spec["vars"]

        # conus gating example you had
        if is_conus(fire_poly) and name == "can_hrrr":
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
        return (name, ds)

    ds_list = []

    # Thread pool: good for requests/HTTP, Herbie downloads, etc.
    with ThreadPoolExecutor(max_workers=min(8, len(client_query_specs))) as tpex:
        futures = [tpex.submit(run_one_client, spec) for spec in client_query_specs]
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

        for name, ds in ds_list:
            # Precompute bins once per dataset
            # Assumes time_bins returns dict {datetime64[D]: indices}
            bins = time_bins(ds.time.values)

            if date not in bins:
                continue

            idx = bins[date]

            # compute mean for each var
            for var_name in ds.data_vars:
                arr = ds[var_name].isel(time=idx).values
                day_dict[f"{name}_{var_name}"] = float(np.nanmean(arr))

        data_per_day.append(day_dict)

    return data_per_day


# -----------------------------
# Main driver
# -----------------------------

feature_file = "/home/jaredgoldman/dev/pyrocb/src/data/average_polygon_features_parallel.csv"
cp = pd.read_csv("data/cp_na.csv")
cp_poly = gpd.read_file("data/cp_poly.gpkg")

# IMPORTANT: don't pass instantiated clients into processes.
# Pass constructors + kwargs so workers create their own.
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
        ":APCP:.*:(?:0-1|[1-9]\\d*-\\d+) hour",
    ]},
    {"name": "can_hrrr", "client_ctor": clients.HRRRClient, "client_kwargs": {}, "vars": [
        ":tp:",
        ":10si:",
        ":r:",
        ":2t:",
        ":sd:",
        ":ssw:",
        ":2d:",
    ]},
    {"name": "modis", "client_ctor": clients.MODISClient, "client_kwargs": {}, "vars": ["MaxFRP", "FireMask"]},
    {"name": "rave", "client_ctor": clients.RAVEClient, "client_kwargs": {}, "vars": ["FRP_MEAN", "FRP_SD", "FRE", "PM25"]},
]

# Pre-split cp indices in the parent
cp_ids = list(cp.cp.unique()[::-1])
all_fires = len(cp_ids)

# Optional: resume behavior by loading existing CSV keys
written_header = False
if pd.io.common.file_exists(feature_file):
    written_header = True

start_time = datetime.now()

# Chunked write buffer
buffer_rows: List[Dict[str, Any]] = []
flush_every_n_fires = 50  # tune this
max_workers = 6           # tune: start with 4-8 depending on CPU/network

# Use processes for per-fire parallelism (bigger jobs)
with ProcessPoolExecutor(max_workers=max_workers) as ppex:
    futures = []
    for cp_idx in cp_ids:
        this_poly = cp_poly[cp_poly.cp == cp_idx]
        this_cp = cp[cp.cp == cp_idx]

        futures.append(
            ppex.submit(
                compute_daily_features_for_fire,
                cp_idx,
                this_poly,
                this_cp,
                client_query_specs,
            )
        )

    done_count = 0
    for f in as_completed(futures):
        rows = f.result()
        done_count += 1

        if rows:
            buffer_rows.extend(rows)

        # periodic flush
        if done_count % flush_every_n_fires == 0:
            if buffer_rows:
                df_out = pd.DataFrame(buffer_rows)

                # append mode, write header once
                df_out.to_csv(feature_file, mode="a", header=not written_header, index=False)
                written_header = True
                buffer_rows.clear()

            elapsed = datetime.now() - start_time
            print(f"{int(done_count/all_fires*100)}% fires done in {int(elapsed.total_seconds()/60)} min")

# final flush
if buffer_rows:
    pd.DataFrame(buffer_rows).to_csv(feature_file, mode="a", header=not written_header, index=False)
    print(f"saved features to {feature_file}")