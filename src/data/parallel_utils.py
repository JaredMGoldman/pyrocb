import logging
import pandas as pd
from shapely import contains, box
import numpy as np

from feature_creation import rave_features, hrrr_features

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

def remove_invalid_idxs(cp_df, logger):
    bad_cps = []
    for cp_idx in cp_df.cp.unique():
        df = cp_df[cp_df.cp == cp_idx]
        if df[hrrr_features].isna().all().all() or df[rave_features].isna().all().all():
            bad_cps.append(cp_idx)
        elif df.day.unique().shape != df.day.shape:
            bad_cps.append(cp_idx)
    logger.info(f"found {len(bad_cps)} invalid fires")
    return cp_df[~cp_df.cp.isin(bad_cps)]
    
def skip_fire(start, end, perim):
    skip_lb = pd.Timestamp("2024-01-01") 
    skip_ub = pd.Timestamp("2024-09-19")
    start = pd.Timestamp(start) if type(start) is str else start
    end = pd.Timestamp(end) if type(end) is str else end
    if start < pd.Timestamp('2019-07-01 00:00:00'):
        logging.warning("[SKIPPED] before 7-01-2019")
        return True
    if start >= skip_lb and start < skip_ub \
        or end > skip_lb and end <= skip_lb:
        logging.warning("[SKIPPED] fire at beginning of 2024")
        return True
    if (start - pd.Timedelta(1, 'D')).year < (end + pd.Timedelta(1, 'D')).year:
        logging.warning("[SKIPPED] fire in two different years")
        return True
    if not is_conus(perim) and end < skip_lb:
        logging.warning("[SKIPPED] fire isn't conus before 2024")
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