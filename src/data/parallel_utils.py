import logging
import pandas as pd
from scipy.interpolate import interp1d
from shapely import contains, box
import numpy as np

from utils.feature_creation import rave_features, hrrr_features

def varname_map(name, var_name):
    if not 'hrrr' in name:
        return f"{name}_{var_name}"
    can_map = { 'r' : 'rh', 't2m' : 't', 'd2m' : 'dpt',
              'u10' : 'u', 'v10' : 'v'}
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
    hrrr_ctr = 0
    rave_ctr = 0
    for cp_idx in cp_df.cp.unique():
        df = cp_df[cp_df.cp == cp_idx]
        if df[hrrr_features].isna().all().all():
            bad_cps.append(cp_idx)
            hrrr_ctr += 1 
        elif df[rave_features].isna().all().all():
            bad_cps.append(cp_idx)
            rave_ctr += 1
        # elif df.day.unique().shape != df.day.shape:
        #     bad_cps.append(cp_idx)
    time_glitches = [cp for cp in cp_df[cp_df['time'].isna()].cp.unique()]
    bad_cps.extend(time_glitches)
    logger.info(f"found {len(bad_cps)} invalid fires")
    import ipdb; ipdb.set_trace()
    return cp_df[~cp_df.cp.isin(bad_cps)]
    
def skip_fire(start, end, perim):
    skip_lb = pd.Timestamp("2024-01-01") 
    start = pd.Timestamp(start) if type(start) is str else start
    end = pd.Timestamp(end) if type(end) is str else end
    if start < pd.Timestamp('2019-07-01 00:00:00'):
        logging.warning("[SKIPPED] before 7-01-2019")
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

def interp_hrrr(csv_fname, idx_name='cp', date_name='time', order=2):
    """
    Interpolates missing HRRR weather data within groups using polynomial interpolation.
    
    Args:
        csv_fname (str): Path to the source CSV.
        hrrr_features (list): List of column names to interpolate.
        idx_name (str): The grouping column (e.g., 'cp').
        date_name (str): The time column.
        order (int): The order of the polynomial (2 or 3 is usually best for weather).
    """
    df = pd.read_csv(csv_fname)
    
    df = df.drop('modis_MaxFRP', axis = 1) if 'modis_MAXFRP' in df.columns else df
    rave_feats =  ['rave_FRP_MEAN', 'rave_FRP_SD']
    
    df[date_name] = pd.to_datetime(df[date_name], format = 'mixed')
    df = df.sort_values([idx_name, date_name])
    
    def interpolate_hrrr(group):
        # Set the time as index temporarily so interpolation is 'time-aware'
        group = group.set_index(date_name)
        # Perform polynomial interpolation
        # limit_direction='both' ensures gaps at the start/end of a group are filled
        try:
            group.loc[:, hrrr_features] = group.loc[:, hrrr_features].interpolate(
                method='polynomial', 
                order=order, 
                limit_direction='both'
            )
        except:
            return pd.DataFrame()
        return group.reset_index()
    def interpolate_rave(group):
        # Set the time as index temporarily so interpolation is 'time-aware'
        group = group.set_index(date_name)
        # Perform polynomial interpolation
        # limit_direction='both' ensures gaps at the start/end of a group are filled
        try:
            group.loc[:, rave_features] = group.loc[:, rave_features].interpolate(
                method='polynomial', 
                order=1, 
                limit_direction='both'
            )
        except:
            return pd.DataFrame()
        return group.reset_index()
    
    df_interp = df.groupby(idx_name, group_keys=False).apply(interpolate_hrrr)
    df_interp = df_interp.groupby(idx_name, group_keys=False).apply(interpolate_rave)

    return df_interp

def persist_daily_value(df, var_name, idx_name='cp', date_name='time'):
    """
    Takes a value from the start of the day and persists it across 
    all rows for that specific day and index group.
    """
    df[date_name] = pd.to_datetime(df[date_name])
    
    # We group by both the ID (cp) and the date (YYYY-MM-DD)
    df['_temp_day'] = df[date_name].dt.date
    
    df = df.sort_values([idx_name, date_name])
    
    # This takes the first non-null value and carries it down
    df[var_name] = df.groupby([idx_name, '_temp_day'], group_keys=False)[var_name].ffill()
    
    # to ensure the whole day is uniform, you can also backfill.
    df[var_name] = df.groupby([idx_name, '_temp_day'])[var_name].bfill()

    return df.drop(columns=['_temp_day'])

if __name__ == '__main__':
    fname = '/home/jaredgoldman/dev/pyrocb/outputs/features/cleaned_subdaily_v2.csv'
    out_fname = '/home/jaredgoldman/dev/pyrocb/outputs/features/cleaned_subdaily_v0.csv'
    
    df_clean = interp_hrrr(fname)
    df_clean = persist_daily_value(df_clean, 'esi_DFPPM')
    df_clean.to_csv(out_fname, index = False)