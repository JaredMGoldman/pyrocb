import pandas as pd
from shapely import contains, box
import numpy as np

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