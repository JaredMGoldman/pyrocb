import data.clients as clients
import pandas as pd
import geopandas as gpd
from shapely import box, contains
import numpy as np
from datetime import datetime

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

client_queries = [
    {"name" : "esi",
     "client" : clients.ESIClient(),
     "vars" : ['DFPPM']},
    {"name" : "firms",
     "client" : clients.FirmsClient(),
     "vars" : ['frp']},
    {"name" : "us_hrrr",
     "client" : clients.HRRRClient(),
     "vars" : [":TMP:2 m",                              # temperature at 2m
               ":DPT:2 m",                              # dewpoint at 2m
               ":UGRD:10 m",                            # u wind component at 10m
               ":VGRD:10 m",                            # v wind component at 10m
               ":RH:2 m",                               # relative humidity at 2m
               ":MSTAV:",                               # moisture availability
               ":WEASD:",                               # water equivalent accumulated snow depth
               ":APCP:.*:(?:0-1|[1-9]\d*-\d+) hour"    # accumulated precipitation over past hr
               ]},     
    {"name" : "can_hrrr",
     "client" : clients.HRRRClient(), # canadian HRRR
     "vars" : [":tp:",      # Total precipitation
                ":10si:",   # 10m wind speed
                ":r:",      # relative humidity
                ":2t:",     # temperature
                ":sd:",     # water accumulated snow depth
                ":ssw:",    # soil moisture content
                ":2d:"      # 2 m dewpoint temperature    
                ],
     "canadian" : True
                },
    {"name" : "modis",
     "client" : clients.MODISClient(),
     "vars" : ["MaxFRP",
               "FireMask"]},
    {"name" : "rave",
     "client" : clients.RAVEClient(),
     "vars" : ["FRP_MEAN", "FRP_SD",
               "FRE", "PM25"]},
]

feature_file = "/home/jaredgoldman/dev/pyrocb/src/data/average_polygon_features.csv"
cp = pd.read_csv('data/cp_na.csv')
cp_poly = gpd.read_file('data/cp_poly.gpkg')

# ignore non-conus fires before skip_ub
fire_data = []
all_fires = len(cp.cp.unique())
start_time = datetime.now()
for i, cp_idx in enumerate(cp.cp.unique()[::-1]):
    this_poly = cp_poly[cp_poly.cp == cp_idx]
    this_cp = cp[cp.cp == cp_idx]

    fire_poly = this_poly['geometry'].values[0]
    fire_area = this_poly['area'].values[0]
    fire_perim = this_poly['perimeter'].values[0] 

    fire_tmin = pd.Timestamp(this_cp['dtime_min'].values[0])
    fire_tmax = pd.Timestamp(this_cp['dtime_max'].values[0])

    skip = skip_fire(fire_tmin, fire_tmax, fire_poly) 
    if skip:
        continue

    ds_list = []
    for params in client_queries:
        # merge all dses by times and average over day and fire burned area
        client = params["client"]
        vars   = params["vars"]
        name   = params["name"]

        fire_poly = fire_poly.buffer(0.15, join_style=3)
        if is_conus(fire_poly) and name == "can_hrrr":
            continue

        ds = client.query(polygon = fire_poly, 
                          start = fire_tmin - pd.Timedelta(1, "D"),
                          end = fire_tmax + pd.Timedelta(1, "D"),
                          variables = vars)
        ds_list.append((name, ds))

    data_per_day = []
    dates = np.array(pd.date_range(fire_tmin - pd.Timedelta(1, 'D'), 
                                   fire_tmax + pd.Timedelta(1,'D'))) \
                                    .astype("datetime64[D]")
    for date in dates:
        day_dict = {'cp' : cp_idx, 'day' : date}
        for name, ds in ds_list:
            for var_name in ds.data_vars:
                times = time_bins(ds.time.values)
                for time in times:
                    d_time = time.astype("datetime64[D]")
                    if d_time != date:
                        continue
                    day_dict[f"{name}_{var_name}"] = np.nanmean(ds.isel(time = times[d_time])[var_name].values)
        data_per_day.append(day_dict)
    
    if i % 300 == 0:
        time_elapsed = datetime.now() - start_time
        print(f"{int(i/all_fires * 100)}% of fires processed in {int(time_elapsed.total_seconds()/60)} minutes")
        pd.DataFrame(data_per_day).to_csv(feature_file, index=False)
        print(f"saved features to {feature_file}")

#   TODO:
    # REMOVE CACHED FILES AFTER EXECUTION
    # PARALLELIZE PROCESS SO IT'S NOT SO DANG SLOW
    # SEE IF FASTHERBIE CAN IMPROVE EXECUTION

# Process:
#   1) filter fire list by date range 
#   2) query clients over polygon and time frame
#   3) filter output datasets per fire

# Determine training regime:
#   - train single model based on all fires (v1)
#   - create conditinoal models initialized based on locations

