from os.path import join as os_join
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import pandas as pd
from scipy.io import savemat
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree
import tqdm

import analysis.mapping.config as config 
from analysis.mapping.signal_cache import SignalCache
from analysis.mapping.pft_gen_parallel import pft_worker_direct
from utils.constants import CACHE_BASE_DIR


ACTIVE_FIRES_DIR = os_join(CACHE_BASE_DIR, 'active_fires')
plume_temps = [-40.0, -20.0]


def get_fire_geom_list(daily_dir):
    """Loads active fire manifest and converts geometries into Shapely objects."""
    fire_manifest_df = pd.read_csv(os_join(daily_dir, config.active_fire_fname))
    fire_polygons = []
    for f_idx in tqdm.tqdm(fire_manifest_df.fire_index_id.values, desc='preprocessing pft subsets'):
        geom = fire_manifest_df[fire_manifest_df.fire_index_id == f_idx]['wkt_geometry'].item()
        fire_name = fire_manifest_df[fire_manifest_df.fire_index_id == f_idx]['name'].item()

        if not geom:
            continue
        try:
            fire_poly = shapely.wkt.loads(geom)
            fire_polygons.append((fire_name, f_idx, fire_poly))
        except Exception:
            continue
    return fire_polygons


def map_cache_to_fires(cache, active_fires, padding=0.125):
    """
    OPTIMIZATION 1: Single-pass spatial index mapping with degree-based geometry padding.
    
    Builds an STRtree spatial index of all padded active fire geometries and 
    streams the SQLite cache ONCE per directory in O(N log M) time.
    
    Parameters
    ----------
    cache : SignalCache
        The loaded SQLite sounding cache object.
    active_fires : list of tuple
        List of (fire_name, f_idx, fire_poly) tuples.
    padding : float, default 0.125
        Padding in degrees added to the fire geometries (0.125 deg approx. 12-14 km).
    """
    if not active_fires:
        return {}

    # Pre-buffer fire geometries BEFORE building the STRtree
    buffered_active_fires = []
    for fire_name, f_idx, fire_poly in active_fires:
        buffered_poly = fire_poly.buffer(padding) if padding > 0 else fire_poly
        buffered_active_fires.append((fire_name, f_idx, buffered_poly))

    # Build spatial index using the expanded geometries
    fire_polys = [buffered_poly for _, _, buffered_poly in buffered_active_fires]
    tree = STRtree(fire_polys)

    # Dictionary mapping fire index -> list of matching records
    fire_records_map = {f_idx: [] for _, f_idx, _ in active_fires}

    # Stream soundings from SQLite database in a single pass
    for record in tqdm.tqdm(cache.stream_records(), total=cache.count(), desc="Streaming soundings"):
        snd, (key_val, time_val, name) = record
        lat, lon = key_val
        point = Point(lon, lat)

        # Query spatial index for intersecting buffered fire polygons
        matching_indices = tree.query(point, predicate='intersects')

        for idx in matching_indices:
            if hasattr(idx, 'item'):
                idx = idx.item()
            _, f_idx, buffered_poly = buffered_active_fires[idx]
            if buffered_poly.intersects(point) or buffered_poly.contains(point):
                fire_records_map[f_idx].append(record)

    return fire_records_map


def process_fire_pfts_worker(args):
    """
    OPTIMIZATION 2: Worker function for process pool execution.
    
    Calculates PFT values for a single fire across all assigned sounding records and plume temperatures,
    returning spatially averaged hourly PFT records.
    """
    fire_name, fire_idx, records, temps = args

    if not records:
        return []

    this_fire_pfts = {
        'plume_temp': [],
        'time': [],
        'pft': []
    }

    # Calculate PFT for each record and plume temp
    for plume_temp in temps:
        for record in records:
            (time_dt, pft_val, _), _ = pft_worker_direct(record, plume_temp)
            this_fire_pfts['time'].append(time_dt)
            this_fire_pfts['pft'].append(pft_val)
            this_fire_pfts['plume_temp'].append(plume_temp)

    fire_df = pd.DataFrame(this_fire_pfts)
    if fire_df.empty:
        return []

    # Calculate spatial average PFT grouped by time and plume temp
    avg_df = fire_df.groupby(['time', 'plume_temp'], as_index=False)['pft'].mean()

    results = []
    for _, row in avg_df.iterrows():
        results.append({
            'fire_name': fire_name,
            'fire_idx': fire_idx,
            'time': row['time'],
            'plume_temp': row['plume_temp'],
            'pft': row['pft']
        })

    return results

def csv_to_mat(csv_filepath: str, mat_filepath: str, split_columns: bool = True):
    """
    Converts a CSV file to a MATLAB .mat file.

    Parameters:
    -----------
    csv_filepath : str
        Path to the input CSV file.
    mat_filepath : str
        Path to output the resulting .mat file.
    split_columns : bool, default=True
        If True, saves each column as its own MATLAB variable.
        If False, saves all numeric data into a single 2D array named 'data'.
    """
    df = pd.read_csv(csv_filepath)

    if split_columns:
        mat_data = {}
        for col in df.columns:
            # Sanitize column names so they are valid MATLAB variable identifiers
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col).strip())
            if not clean_name or clean_name[0].isdigit():
                clean_name = f"var_{clean_name}"
            
            mat_data[clean_name] = df[col].values
    else:
        # Save as a single matrix 'data' (best for purely numeric CSVs)
        mat_data = {'data': df.to_numpy()}

    savemat(mat_filepath, mat_data)


def main():
    print('calculating active pfts...')
    max_workers = 48

    for daily_dir in os.listdir(ACTIVE_FIRES_DIR):
        this_dir = os_join(ACTIVE_FIRES_DIR, daily_dir)
        if not os.path.exists(os_join(this_dir, config.active_fire_fname)) or \
            not os.path.exists(os_join(this_dir, config.snd_cache_fn)) or \
            os.path.exists(os_join(this_dir, 'pft_for_active_fires.csv')):
            if os.path.exists(os_join(this_dir, 'pft_for_active_fires.csv')):
                csv_to_mat(os_join(this_dir, 'pft_for_active_fires.csv'), 
                           os_join(this_dir, f'{daily_dir}_pft_for_active_fires.mat'))
                print(f"generated mat file for {daily_dir}")
            continue

        print(f"\nProcessing active fires in: {daily_dir}")

        # 1) Load database cache
        cache = SignalCache.load(os_join(this_dir, config.snd_cache_fn))

        # 2) Subset database to active fire boundaries in a single streaming pass
        active_fires = get_fire_geom_list(this_dir)
        if not active_fires:
            continue

        print(f"Mapping sounding cache to {len(active_fires)} active fire geometries...")
        fire_records_map = map_cache_to_fires(cache, active_fires)

        # Build task payload for parallel execution
        tasks = [
            (fire_name, f_idx, fire_records_map.get(f_idx, []), plume_temps)
            for fire_name, f_idx, _ in active_fires
            if fire_records_map.get(f_idx)
        ]

        fire_records = {
            'fire_name': [],
            'fire_idx': [],
            'time': [],
            'plume_temp': [],
            'pft': []
        }

        # 3) Parallel PFT evaluation across active fires
        print(f"Evaluating PFTs across {len(tasks)} fires using {max_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_fire_pfts_worker, task) for task in tasks]

            for future in tqdm.tqdm(as_completed(futures), total=len(tasks), desc="PFT Parallel Processing"):
                fire_results = future.result()
                for row in fire_results:
                    fire_records['fire_name'].append(row['fire_name'])
                    fire_records['fire_idx'].append(row['fire_idx'])
                    fire_records['time'].append(row['time'])
                    fire_records['plume_temp'].append(row['plume_temp'])
                    fire_records['pft'].append(row['pft'])

        # 4) Save hourly averaged PFT values for active fires to CSV
        out_df = pd.DataFrame(fire_records)
        if not out_df.empty:
            out_df.set_index(['fire_idx', 'time', 'plume_temp'], inplace=True)
            out_df.to_csv(os_join(this_dir, "pft_for_active_fires.csv"), index = False)
            print(f"Successfully saved PFT values to {os_join(this_dir, 'pft_for_active_fires.csv')}")


if __name__ == "__main__":
    main()