import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from utils import parallel_intersection_labels, generate_df, ML_DATA_ROOT
    
def count_heatwave_days(fire_intersection, time_gridmet_file, heatwave_days, var_to_check, gridmet_data_root):
    #load in the gridmet file to check for exceedances
    dt_time = time_gridmet_file.astype(datetime)
    gridmet_filename = gridmet_data_root + "/" + \
                        dt_time.strftime("%Y") + \
                        '/GRIDMET/gridmet_all_' + \
                        dt_time.strftime("%Y-%m-%d") + '.nc'
    
    gridmet_today = xr.open_dataset(gridmet_filename)
    
    gridmet_today_sub = gridmet_today.sel(lat=fire_intersection['lat'].values, lon=fire_intersection['lon'].values)
    
    heatwave_in_polygon = (gridmet_today_sub[var_to_check]==1) & (fire_intersection['weights_mask']==1)

    if heatwave_in_polygon.any():
        heatwave_days_inc = heatwave_days+1
        time_gridmet_file_back = np.datetime64(time_gridmet_file)-np.timedelta64(1,'D')
        return count_heatwave_days(fire_intersection, time_gridmet_file_back,heatwave_days_inc,var_to_check, gridmet_data_root)
    
    else:
        heatwave_days_inc=heatwave_days
        return heatwave_days_inc
    
def heatwave_timeseries(df, gridmet_data_root = "/data/lthapa/data2restore/lthapa"):
    varis = ['day','days_in_high_heatwave', 'days_in_highlow_heatwave'] 
    df_heatwave= generate_df(varis, len(df))
    
    fire_gridmet_intersection_xr = parallel_intersection_labels(df, f'{ML_DATA_ROOT}/WESTUS_GRIDMET_GRID').rename(name_dict = {'12Z Start Day': 'Start_Day'})
    
    for xx in range(len(fire_gridmet_intersection_xr['Start_Day'].values)): #loop over all the days where we have polygons
        poly_time = fire_gridmet_intersection_xr['Start_Day'].values[xx]
        
        intersection_today = fire_gridmet_intersection_xr.sel(Start_Day=poly_time)
        
        days_in_high_heatwave = 0 #start by assuming we are out of the heatwave
        df_heatwave['days_in_high_heatwave'].iloc[xx] = count_heatwave_days(intersection_today, poly_time, 
                                                                            days_in_high_heatwave, 'is_high_heatwave',
                                                                            gridmet_data_root)
        
        days_in_highlow_heatwave = 0 #start by assuming we are out of the heatwave
        df_heatwave['days_in_highlow_heatwave'].iloc[xx] = count_heatwave_days(intersection_today, poly_time, 
                                                                               days_in_highlow_heatwave, 'is_highlow_heatwave',
                                                                               gridmet_data_root)
        
    df_heatwave['day'].iloc[:] = pd.to_datetime(fire_gridmet_intersection_xr['Start_Day'].values)
    
    return df_heatwave