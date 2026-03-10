from __future__ import annotations

from utils.logging_utils import start_log_listener, init_worker_logging
from data.parallel_utils import varname_map
from data.specs import client_query_specs, hrrr_headers
from data.dataset_worker import compute_daily_features_for_fire
from utils.utils import FEATURE_OUTPUT_DIR, DATA_DIR, LOG_DIR

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import geopandas as gpd
import multiprocessing
import logging
import os
import pandas as pd
from pathlib import Path
from random import sample
from typing import Dict, List, Any

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
parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING",
        help="Desired logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)")


def main(cp: pd.DataFrame, 
         feature_file: str, 
         cp_poly: gpd.GeoDataFrame,
         flush_fires: int = 50,
         max_workers: int = 8,
         DEBUG_MODE: bool = False,
         THIRD: int = 0,
         logging_level: int = logging.INFO):
    # -----------------------------
    # Main driver
    # -----------------------------
    # IMPORTANT: don't pass instantiated clients into processes.
    # Pass constructors + kwargs so workers create their own.
    # Pre-split cp indices in the parent
    log_path = Path(os.path.join(LOG_DIR, feature_file.split(os.path.sep)[-1].replace(".csv",".log")))

    log_queue, listener = start_log_listener(log_path, level = logging_level)
    logger = logging.getLogger(feature_file.split(os.path.sep)[-1].replace(".csv",""))
    written_header = False
    all_cps = list(cp[cp['t_min'] + 1 < cp['t_max']].cp.unique())
    if os.path.exists(feature_file):
        processed_cps = pd.read_csv(feature_file).cp.unique()
        all_cps = list(set(all_cps) - set(list(processed_cps)))
        written_header = True
        
    all_fires = len(all_cps)
    logger.info(f"generating data for {all_fires} fires")
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
    if not written_header:
        pd.DataFrame(header).to_csv(feature_file, index = False)
        written_header = True

    start_time = datetime.now()

    # Chunked write buffer
    buffer_rows: List[Dict[str, Any]] = []
    flush_every_n_fires = flush_fires  # tune this
    max_workers = max_workers # tune: start with 4-8 depending on CPU/network

    done_count = 0
    # Use processes for per-fire parallelism (bigger jobs)
    try:
        with ProcessPoolExecutor(max_workers=max_workers,
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker_logging,
                                 initargs=(log_queue,)) as ppex:
            futures = []
            for cp_idx in random_cps:
                this_poly = cp_poly[cp_poly.cp == cp_idx]
                this_cp = cp[cp.cp == cp_idx]
                if DEBUG_MODE:
                    f = compute_daily_features_for_fire(
                            cp_idx,
                            this_poly,
                            this_cp,
                            log_queue,
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
                        logger.info(f"{int(done_count/all_fires*100)}% fires done in {int(elapsed.total_seconds()/60)} min")
                else:
                    futures.append(
                        ppex.submit(
                            compute_daily_features_for_fire,
                            cp_idx,
                            this_poly,
                            this_cp,
                            log_queue,
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
                    logger.error(f"failed to load row with exception {e}")
                    exit()
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
                    logger.info(f"{int(done_count/all_fires*100)}% fires done in {int(elapsed.total_seconds()/60)} min")

        # final flush
        if buffer_rows:
            pd.DataFrame(buffer_rows).to_csv(feature_file, mode="a", header=not written_header, index=False)
            logger.info(f"saved features to {feature_file}")
    finally:
        listener.stop()
        pass
if __name__ == "__main__":
    args = parser.parse_args()
    
    main(cp            = pd.read_csv(os.path.join(DATA_DIR, args.c)), 
         feature_file  = os.path.join(FEATURE_OUTPUT_DIR, args.o), 
         cp_poly       = gpd.read_file(os.path.join(DATA_DIR, args.p)),
         flush_fires   = args.flush_n_fires,
         max_workers   = args.num_workers,
         DEBUG_MODE    = args.debug,
         THIRD         = args.third,
         logging_level = getattr(logging, args.log_level.upper()))