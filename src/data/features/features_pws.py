import pandas as pd
import numpy as np
import xarray as xr

from utils import parallel_intersection_labels, ML_DATA_ROOT


def pws_timeseries(df, pws_data = 'data/lthapa/data2restore/lthapa/PWS_6_jan_2021.nc'):
    fire_pws_intersection_xr = parallel_intersection_labels(df, f'{ML_DATA_ROOT}/PWS_GRID')
    
    #load in PWS data associated with the fire (it's only one dataset)  
    #open the PWS files
    dat_pws = xr.open_dataset(pws_data) #map is fixed in time
    
    dat_pws = dat_pws.assign_coords({'time': pd.to_datetime(fire_pws_intersection_xr['12Z Start Day'].values)})
    dat_pws_daily = dat_pws['Band1'].expand_dims({'time': pd.to_datetime(fire_pws_intersection_xr['12Z Start Day'].values)}) #the PWS expanded over all the days
    
    dat_pws_daily_sub = dat_pws_daily.sel(lat = fire_pws_intersection_xr['lat'].values, 
                                          lon = fire_pws_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_pws_intersection_xr['12Z Start Day'].values), method='nearest')
    ndays = len(fire_pws_intersection_xr['12Z Start Day'])
    
    #preallocate space for the output
    df_pws_weighted = pd.DataFrame({'day':np.zeros(ndays),'PWS':np.zeros(ndays)})
    df_pws_unweighted = pd.DataFrame({'day':np.zeros(ndays),'PWS':np.zeros(ndays)})

    df_pws_weighted['day'].iloc[:] = pd.to_datetime(fire_pws_intersection_xr['12Z Start Day'].values)
    df_pws_unweighted['day'].iloc[:] = pd.to_datetime(fire_pws_intersection_xr['12Z Start Day'].values)

    varis=['PWS']
    for var in varis:
        df_pws_weighted[var] = np.nansum(fire_pws_intersection_xr['weights'].values*dat_pws_daily_sub.values, axis=(1,2)) #WEIGHTED AVERAGE
        df_pws_unweighted[var] = np.nanmean(fire_pws_intersection_xr['weights_mask'].values*dat_pws_daily_sub.values, axis=(1,2)) #MASK AND AVERAGE

    return df_pws_weighted, df_pws_unweighted