import pandas as pd
import geopandas as gpd
import numpy as np
from joblib import Parallel, delayed
import xarray as xr

from utils import calculate_intersection, make_file_namelist, generate_df, ML_DATA_ROOT

#HRRR_WS formulation from, take time mean, then take weighted average. For HDW, multiply the weighted means of VPD and WIND
def hdw_lagged_timeseries(df, hrrr_data_root = f"{ML_DATA_ROOT}/pygraf/processed_hrrr_hdw_hwp"):  #with the wind speed
    varis_hrrr_derived = ['day','hd0w0', 'hd1w0','hd2w0', 'hd3w0']#, 'hd4w0', 'hd5w0',
    df_hdw_weighted = generate_df(varis_hrrr_derived, len(df))
    df_hdw_unweighted = generate_df(varis_hrrr_derived, len(df))
    
    #do the intersection, in parallel
    hrrr_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/HRRR_GRID',3000) 
                                 for ii in range(len(df)))
    
    fire_hrrr_intersection=gpd.GeoDataFrame(pd.concat(hrrr_intersections, ignore_index=True))
    fire_hrrr_intersection.set_geometry(col='geometry')
    
    #loop over all of the days we have intersections
    times_intersect = np.unique(fire_hrrr_intersection['12Z Start Day'].values)
    
    for i, today in enumerate(times_intersect):
        df_hdw_weighted['day'].iloc[i] = today
        df_hdw_unweighted['day'].iloc[i] = today
        #get the time
        df_sub = fire_hrrr_intersection.iloc[np.where(fire_hrrr_intersection['12Z Start Day'].values==today)]
        df_sub = df_sub.set_index(['12Z Start Day', 'row', 'col'])
        df_sub=df_sub[~df_sub.index.duplicated()]
        intersection_sub = df_sub.to_xarray() #polygon and weights for today
        intersection_sub['weights_mask'] =xr.where(intersection_sub['weights']>0,1, np.nan)
        
        times_back = pd.date_range(start=np.datetime64(today)-np.timedelta64(3,'D'), end=np.datetime64(today)+
                                   np.timedelta64(36,'h'),freq='H')
        files_back,times_back_used = make_file_namelist(times_back,f'{hrrr_data_root}/Processed_HRRR_YYYYMMDDHH_HDW_HWP.nc')
        
        # load in all the merra files associated with this lookback window
        dat_hrrr = xr.open_mfdataset(files_back,concat_dim='time',combine='nested',compat='override', coords='all')
        dat_hrrr = dat_hrrr.assign_coords({'time': times_back_used})
        dat_hrrr = dat_hrrr.resample(time='h').asfreq()
        
        day0 = np.datetime64(today)+np.timedelta64(12,'h')
        w0 = dat_hrrr['wind_speed'].sel(
            time=pd.date_range( start=day0, 
                                end=day0+np.timedelta64(23,'h'),
                                freq='H'), 
            grid_yt = np.unique(intersection_sub['row'].values),
            grid_xt = np.unique(intersection_sub['col'].values))
        for day_num in range(4):
            # define the days
            day = day0 if day_num == 0 else np.datetime64(today)+np.timedelta64(12,'h') - np.timedelta64(day_num,'D')
                
            #define the times we will select for VPD
            times = pd.date_range(start=day, end=day+np.timedelta64(23,'h'),freq='H')
            
            hd = dat_hrrr['vpd_2m'].sel(time=times, grid_yt = np.unique(intersection_sub['row'].values),grid_xt = np.unique(intersection_sub['col'].values))
            if day_num > 0:
                hd=hd.assign_coords({'time':w0['time'].values})
            
            hdw0 = hd*w0
            hdw0_daily_mean = hdw0.resample(time='24H', offset="12H", label='left').mean() #take the daily mean        

            df_hdw_weighted.loc[i, (f'hd{day_num}w0')] = np.nansum((hdw0_daily_mean.values)*(intersection_sub['weights'].values))
            df_hdw_unweighted.loc[i, (f'hd{day_num}w0')] = np.nanmean((hdw0_daily_mean.values))
        
    return df_hdw_weighted, df_hdw_unweighted