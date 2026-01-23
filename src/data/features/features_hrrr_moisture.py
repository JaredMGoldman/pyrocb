import pandas as pd
import numpy as np
import xarray as xr

from utils import parallel_intersection_labels, make_file_namelist, generate_df

def hrrr_moisture_timeseries(df, 
                             hrrr_data_root = "/data/lthapa/data2restore/lthapa/ML_daily/pygraf/processed_hrrr_hdw_hwp",
                             landmask_data = '/data/lthapa/data2restore/lthapa/ML_daily/pygraf/HRRR_LANDMASK.nc'):
    varis_hrrr_moisture = ['day','soilm_sfc', 'soilm_1cm', 'soilm_4cm', 'soilm_10cm', 'soilm_30cm']
    
    df_hrrr_moisture_weighted = generate_df(varis_hrrr_moisture, len(df))
    df_hrrr_moisture_unweighted = generate_df(varis_hrrr_moisture, len(df))

    fire_hrrr_moisture_intersection_xr = parallel_intersection_labels(df, 'HRRR_GRID')
    
    #load in data associated with the fire
    times = pd.date_range(np.datetime64(df['12Z Start Day'].iloc[0]),
                        np.datetime64(df['12Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    hrrr_moisture_filenames,times_back_used = make_file_namelist(times,f'{hrrr_data_root}/Processed_HRRR_YYYYMMDDHH_MOISTURE.nc')
    landmask = xr.open_dataset(landmask_data)

    dat_hrrr_moisture = xr.open_mfdataset(hrrr_moisture_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    dat_hrrr_moisture = dat_hrrr_moisture.assign_coords({'Time': times_back_used}) #assign coords so we can select in time

    #select the locations and times we want
    hrrr_daily_mean = dat_hrrr_moisture.resample(Time='24H',base=12, label='left').mean(dim='Time') #take the daily mean       
    hrrr_daily_mean_region = hrrr_daily_mean.sel(grid_yt = np.unique(fire_hrrr_moisture_intersection_xr['row'].values),
                                                    grid_xt = np.unique(fire_hrrr_moisture_intersection_xr['col'].values)).sel(
                    Time = pd.to_datetime(fire_hrrr_moisture_intersection_xr['12Z Start Day'].values + ' 12:00:00'), method='nearest')#these should be lined up correctly
    landmask_region = landmask.sel(grid_yt = np.unique(fire_hrrr_moisture_intersection_xr['row'].values),
                                   grid_xt = np.unique(fire_hrrr_moisture_intersection_xr['col'].values))
    landmask_region_masked = landmask_region.where(landmask_region['landmask']!=0)

    df_hrrr_moisture_weighted['day'].iloc[:] = pd.to_datetime(fire_hrrr_moisture_intersection_xr['12Z Start Day'].values)
    df_hrrr_moisture_unweighted['day'].iloc[:] = pd.to_datetime(fire_hrrr_moisture_intersection_xr['12Z Start Day'].values)

    for var in varis_hrrr_moisture[1:]:
        df_hrrr_moisture_weighted[var] =np.nansum((hrrr_daily_mean_region[var].values)*(fire_hrrr_moisture_intersection_xr['weights'].values)*(landmask_region_masked['landmask'].values),axis=(1,2))   
        df_hrrr_moisture_unweighted[var] = np.nanmean((hrrr_daily_mean_region[var].values)*(fire_hrrr_moisture_intersection_xr['weights_mask'].values)*(landmask_region_masked['landmask'].values),axis=(1,2))
    return df_hrrr_moisture_weighted, df_hrrr_moisture_unweighted