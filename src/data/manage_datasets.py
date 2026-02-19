import data.clients as clients
import pandas as pd
import geopandas as gpd
from shapely import box, contains

def skip_fire(start, end, perim):
    conus_polygon = box(-125.0, 24.0, -66.5, 49.5)
    skip_lb = pd.Timestamp("2021-01-01") 
    skip_ub = pd.Timestamp("2024-09-19")
    if start >= skip_lb and start < skip_ub \
        or end > skip_lb and end <= skip_lb:
        return True
    if not contains(conus_polygon, perim) \
        and end < skip_lb:
        return True
    return False

client_queries = [
    {"client" : clients.ESIClient(),
     "vars" : ['DFPPM']},
    {"client" : clients.FirmsClient(),
     "vars" : ['frp']},
    {"client" : clients.HRRRClient(),
     "vars" : [":TMP:2 m",
               ":DPT:2 m",
               ":UGRD:10 m",
               ":VGRD:10 m",
               ":RH:2 m",
               ":MSTAV:",                             # moisture availability
               ":WEASD:",                             # water equivalent accumulated snow depth
               ":APCP:.*:(?:0-1|[1-9]\d*-\d+) hour", # accumulated precipitation over past hr
               ":500 mb:",              # All variables on the 500 mb level.
               ]},     
    {"client" : clients.HRRRClient(), # canadian HRRR
     "vars" : [":tp:",      # Total precipitation
                ":u:10 m",  # u wind component
                ":v:10 m",  # v wind component
                ":r:2 m",   # relative humidity
                ":t:2 m",   # temperature
                ":sd:",     # water accumulated snow depth
                ":ssw:",    # soil moisture content
                ":2d:"      # 2 m dewpoint temperature    
                ]},
    {"client" : clients.MODISClient(),
     "vars" : ["MaxFRP",
               "FireMask"]},
    {"client" : clients.RAVEClient(),
     "vars" : ["FRP_Mean", "FRP_SD",
               "FRE", "PM25"]},
]

cp = pd.read_csv('data/cp.csv')
cp_poly = gpd.read_file('data/cp_poly.gpkg')

# ignore non-conus fires before skip_ub
fire_data = []
for cp_idx in cp.cp.unique():
    this_poly = cp_poly[cp_poly.cp == cp_idx]
    this_cp = cp[cp.cp == cp_idx]

    fire_poly, fire_area, fire_perim = this_poly[['geometry', 'area', 'perimeter']] 
    fire_tmin, fire_tmax = this_cp[['dtime_min', 'dtime_max']]

    skip = skip_fire(fire_tmin, fire_tmax, fire_perim) 
    if skip:
        continue
    for params in client_queries:
        # merge all dses by times and average over day and fire burned area
        client = params["client"]
        vars = params["vars"]
        ds = client.query(polygon = fire_perim, 
                          start = fire_tmin - pd.Timedelta(1, "D"),
                          end = fire_tmax,
                          variables = vars)


# Process:
#   1) filter fire list by date range 
#   2) query clients over polygon and time frame
#   3) filter output datasets per fire

# Determine training regime:
#   - train single model based on all fires (v1)
#   - create conditinoal models initialized based on locations

