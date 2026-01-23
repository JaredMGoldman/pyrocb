import pandas as pd
import numpy as np
import xarray as xr

from utils import parallel_intersection_labels, make_file_namelist, generate_df

def hrrrmet_timeseries(df, hrrr_data_root = '/data/lthapa/data2restore/lthapa/ML_daily/pygraf/processed_hrrr_hdw_hwp' ):
    varis_hrrrmet = ['day','wind_speed', 'vpd_2m']
    
    df_hrrrmet_weighted = generate_df(varis_hrrrmet, len(df))
    df_hrrrmet_unweighted = generate_df(varis_hrrrmet, len(df))

    fire_hrrrmet_intersection_xr = parallel_intersection_labels(df, 'HRRR_GRID')
    
    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df['12Z Start Day'].iloc[0]),
                        np.datetime64(df['12Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    hrrrmet_filenames,times_back_used = make_file_namelist(times,f'{hrrr_data_root}/Processed_HRRR_YYYYMMDDHH_HDW_HWP.nc')

    dat_hrrrmet = xr.open_mfdataset(hrrrmet_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    dat_hrrrmet = dat_hrrrmet.assign_coords({'Time': times_back_used}) #assign coords so we can select in time
    
    #select the locations and times we want
    hrrr_daily_mean = dat_hrrrmet.resample(Time='24H',base=12, label='left').mean(dim='Time') #take the daily mean       
    hrrr_daily_mean_region = hrrr_daily_mean.sel(grid_yt = np.unique(fire_hrrrmet_intersection_xr['row'].values),
                                                    grid_xt = np.unique(fire_hrrrmet_intersection_xr['col'].values)).sel(
                    Time = pd.to_datetime(fire_hrrrmet_intersection_xr['12Z Start Day'].values + ' 12:00:00'), method='nearest')#these should be lined up correctly

    df_hrrrmet_weighted['day'].iloc[:] = pd.to_datetime(fire_hrrrmet_intersection_xr['12Z Start Day'].values)
    df_hrrrmet_unweighted['day'].iloc[:] = pd.to_datetime(fire_hrrrmet_intersection_xr['12Z Start Day'].values)

    for var in varis_hrrrmet[1:]:
        df_hrrrmet_weighted[var] =np.nansum((hrrr_daily_mean_region[var].values)*(fire_hrrrmet_intersection_xr['weights'].values),axis=(1,2))   
        df_hrrrmet_unweighted[var] = np.nanmean((hrrr_daily_mean_region[var].values)*(fire_hrrrmet_intersection_xr['weights_mask'].values),axis=(1,2))
    return df_hrrrmet_weighted, df_hrrrmet_unweighted