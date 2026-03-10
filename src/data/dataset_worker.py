from concurrent.futures import ThreadPoolExecutor, as_completed
import faulthandler
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import traceback

from data.parallel_utils import skip_fire, is_conus, safe_buffer, time_bins, varname_map
from logging_utils import log_cp, set_cp, log_client, set_client, configure_queue_logging, init_worker_logging, reset_tokens
from utils import LOG_DIR

def compute_daily_features_for_fire(
    cp_idx,
    this_poly_gdf,
    this_cp_df,
    log_queue,
    client_query_specs,  # <-- dict/specs, not client instances (see below)
    DEBUG_MODE,
    feature_start_pad="1D",
    feature_end_pad="1D"
    ) -> List[Dict[str, Any]]:
    """
    Runs ALL client queries for one fire and returns list of day_dict rows.
    Designed to be executed in a worker process.
    """

    fire_poly = this_poly_gdf["geometry"].values[0]
    fire_tmin = pd.Timestamp(this_cp_df["dtime_min"].values[0])
    fire_tmax = pd.Timestamp(this_cp_df["dtime_max"].values[0])

    if skip_fire(fire_tmin, fire_tmax, fire_poly):
        return []

    fire_poly = safe_buffer(fire_poly, 0.15)

    start = fire_tmin - pd.Timedelta(feature_start_pad)
    end   = fire_tmax + pd.Timedelta(feature_end_pad)

    def run_one_client(spec):
        name = spec["name"]
        vars_ = spec["vars"]
        # conus gating example you had
        if is_conus(fire_poly) and name == "can_hrrr":
            return None
        elif not is_conus(fire_poly) and name == "us_hrrr":
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
    
    def worker_entry(task_payload):
        """
        Top-level process worker wrapper.
        Logs full traceback on failure and returns None instead of crashing the pool.
        """
        # configure_queue_logging(log_queue)
        # init_worker_logging(log_queue)

        crash_dir = Path(f"{LOG_DIR}/worker_crashes")
        crash_dir.mkdir(parents=True, exist_ok=True)
        crash_file = open(crash_dir / f"worker_{os.getpid()}.log", "a")
        faulthandler.enable(file=crash_file, all_threads=True)

        logger = logging.getLogger("worker_entry")

        cp_idx = task_payload.get("cp_idx", "-")
        client_name = task_payload.get("client_name", "worker")

        cp_token = set_cp(cp_idx)
        client_token = set_client(client_name)

        if DEBUG_MODE:
            try:
                result = run_one_client(task_payload)
                return result
            finally:
                reset_tokens(cp_token, client_token)
                try:
                    crash_file.flush()
                    crash_file.close()
                    os.remove(crash_dir / f"worker_{os.getpid()}.log")
                except Exception:
                    pass

        try:
            logger.info("worker started")

            # call your real worker here
            result = run_one_client(task_payload)

            logger.info("worker finished successfully")
            return result

        except KeyboardInterrupt:
            logger.warning("worker interrupted by keyboard interrupt")
            exit()

        except Exception:
            logger.exception("worker failed with traceback:\n%s", traceback.format_exc())
            return None

        finally:
            reset_tokens(cp_token, client_token)
            try:
                crash_file.flush()
                crash_file.close()
                os.remove(crash_dir / f"worker_{os.getpid()}.log")
            except Exception:
                pass
        
    ds_list = []
    if DEBUG_MODE:
        for spec in client_query_specs:
            out = worker_entry(spec)
            if out is not None:
                ds_list.append(out)
    else:
        with ThreadPoolExecutor(max_workers=min(6, len(client_query_specs))) as tpex:        
            futures = [tpex.submit(worker_entry, spec) for spec in client_query_specs]
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