import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from joblib import Parallel, delayed


from utils import calculate_intersection, make_file_namelist, ML_DATA_ROOT

def esi_timeseries(df, esi_data_dir = "/data/lthapa/data2restore/lthapa"):
    #preallocate space for the output
    df_esi_weighted = pd.DataFrame({'day':np.zeros(len(df)),'ESI':np.zeros(len(df))})
    df_esi_unweighted = pd.DataFrame({'day':np.zeros(len(df)),'ESI':np.zeros(len(df))})
    
    #do the intersection, in parallel
    esi_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/ESI_GRID',5000) 
                                 for ii in range(len(df)))
    
    fire_esi_intersection=gpd.GeoDataFrame(pd.concat(esi_intersections, ignore_index=True))
    fire_esi_intersection.set_geometry(col='geometry')
    
    fire_esi_intersection = fire_esi_intersection.set_index(['12Z Start Day', 'lat', 'lon'])
    
    fire_esi_intersection_xr = fire_esi_intersection.to_xarray()
    fire_esi_intersection_xr['weights_mask'] = xr.where(fire_esi_intersection_xr['weights']>0,1, np.nan)

    #load in esi data associated with the fire
    times = pd.date_range(np.datetime64(df['12Z Start Day'].iloc[0])-np.timedelta64(1,'W'),
                        np.datetime64(df['12Z Start Day'].iloc[len(df)-1])+np.timedelta64(1, 'W')+
                        np.timedelta64(1,'D'))
    esi_filenames, esi_times = make_file_namelist(times,f'{esi_data_dir}/YYYY/ESI/DFPPM_4WK_YYYYJJJ.nc')
    
    #open the esi files
    dat_esi = xr.open_mfdataset(esi_filenames,concat_dim='time',combine='nested',compat='override', coords='all')
    dat_esi = dat_esi.assign_coords({'time': esi_times}) #assign coords so we can resample along time
    dat_esi = dat_esi.where(dat_esi['Band1']!=-9999) #gets rid of the missing values!
    dat_esi_daily = dat_esi.reindex(time=times,method='nearest') #makes the weekly data daily
    dat_esi_daily_sub = dat_esi_daily.sel(lat = fire_esi_intersection_xr['lat'].values, 
                                          lon = fire_esi_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_esi_intersection_xr['12Z Start Day'].values), method='nearest')
                                          
    df_esi_weighted['day'].iloc[:] = pd.to_datetime(fire_esi_intersection_xr['12Z Start Day'].values)
    df_esi_unweighted['day'].iloc[:] = pd.to_datetime(fire_esi_intersection_xr['12Z Start Day'].values)

    varis=['ESI']
    for var in varis:
        df_esi_weighted[var] = np.nansum(fire_esi_intersection_xr['weights'].values*dat_esi_daily_sub['Band1'].values, axis=(1,2))
        df_esi_unweighted[var] = np.nanmean(fire_esi_intersection_xr['weights_mask'].values*dat_esi_daily_sub['Band1'].values, axis=(1,2))
    
    return df_esi_weighted, df_esi_unweighted