import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from utils import calculate_intersection, make_file_namelist, generate_df, ML_DATA_ROOT

#HRRR_WS formulation from, take time mean, then take weighted average. For HDW, multiply the weighted means of VPD and WIND
def hwp_timeseries(df, hrrr_data_root = f'{ML_DATA_ROOT}/pygraf/processed_hrrr_hdw_hwp'):  #with the wind speed
    varis_hrrr_derived = ['day','hwp','hwp_1'] #'hd1w0','hd2w0', 'hd3w0', 'hd4w0', 'hd5w0',
    
    #return both!
    df_hwp_weighted = generate_df(varis_hrrr_derived, len(df))
    df_hwp_unweighted = generate_df(varis_hrrr_derived, len(df))

    #do the intersection, in parallel
    hrrr_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/HRRR_GRID',3000) 
                                 for ii in range(len(df)))
    print([hrrr_intersections[jj]['weights'].sum() for jj in range(len(hrrr_intersections))])

    fire_hrrr_intersection=gpd.GeoDataFrame(pd.concat(hrrr_intersections, ignore_index=True))
    fire_hrrr_intersection.set_geometry(col='geometry')
    
    #loop over all of the days we have intersections
    times_intersect = np.unique(fire_hrrr_intersection['12Z Start Day'].values)
    
    for i, today in enumerate(times_intersect):
        #get the time
        df_sub = fire_hrrr_intersection.iloc[np.where(fire_hrrr_intersection['12Z Start Day'].values==today)]
        df_sub = df_sub.set_index(['12Z Start Day', 'row', 'col'])
        df_sub=df_sub[~df_sub.index.duplicated()]
        intersection_sub = df_sub.to_xarray() #polygon and weights for today
        intersection_sub['weights_mask'] =xr.where(intersection_sub['weights']>0,1, np.nan)
        
        times_back = pd.date_range(start=np.datetime64(today)-np.timedelta64(1,'D'), end=np.datetime64(today)+
                                   np.timedelta64(1,'D'),freq='h')
        
        files_back,times_back_used = make_file_namelist(times_back,f'{hrrr_data_root}/Processed_HRRR_YYYYMMDDHH_HDW_HWP.nc')
        
        #load in all the merra files associated with this lookback window
        dat_hrrr = xr.open_mfdataset(files_back,concat_dim='time',combine='nested',compat='override', coords='all')
        dat_hrrr = dat_hrrr.assign_coords({'time': times_back_used})
        
        hrrr_daily_mean = dat_hrrr.resample(time='24H',offset='12H', label='left').mean() #take the daily mean        
        
        hrrr_daily_mean_region = hrrr_daily_mean.sel(grid_yt = np.unique(intersection_sub['row'].values),
                                                    grid_xt = np.unique(intersection_sub['col'].values)) #get the location of the overlaps
        hrrr_daily_mean_region = hrrr_daily_mean_region.where(hrrr_daily_mean_region['hwp']!=0) #mask out zeroes

        df_hwp_weighted['day'].iloc[i] =today
        df_hwp_unweighted['day'].iloc[i] =today

        df_hwp_weighted.loc[i, ('hwp')] =np.nansum((hrrr_daily_mean_region['hwp'].sel(time=np.datetime64(today+ np.timedelta64(12,'h'))).values)*(intersection_sub['weights'].values))  
        df_hwp_weighted.loc[i, ('hwp_1')] =np.nansum((hrrr_daily_mean_region['hwp'].sel(time=np.datetime64(today)+ np.timedelta64(12,'h')-np.timedelta64(1,'D')).values)*(intersection_sub['weights'].values))  
        df_hwp_unweighted.loc[i, ('hwp')] = np.nanmean((hrrr_daily_mean_region['hwp'].sel(time=np.datetime64(today+ np.timedelta64(12,'h'))).values)*(intersection_sub['weights_mask'].values))

        dat_hrrr.close()
    return df_hwp_weighted, df_hwp_unweighted