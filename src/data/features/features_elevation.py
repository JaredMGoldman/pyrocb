import pandas as pd
import geopandas as gpd
import numpy as np
from utils import parallel_intersection_labels, ML_DATA_ROOT
import xarray as xr

def elevation_timeseries(df, path_elevation = f'{ML_DATA_ROOT}/elev_990m_LF2020.nc'):
    fire_elevation_intersection_xr = parallel_intersection_labels(df, f'{ML_DATA_ROOT}/ELEV_GRID_990M')
    
    #load in ELEV data associated with the fire (it's only one dataset)  
    #open the ELEV files
    dat_elevation = xr.open_dataset(path_elevation) #map is fixed in time
    
    dat_elevation = dat_elevation.assign_coords({'time': pd.to_datetime(fire_elevation_intersection_xr['12Z Start Day'].values)})
    data_elevation_mean = dat_elevation['mean_elev'].expand_dims({'time': pd.to_datetime(fire_elevation_intersection_xr['12Z Start Day'].values)}) #the mean elevation expanded over all the days
    data_elevation_std = dat_elevation['std_elev'].expand_dims({'time': pd.to_datetime(fire_elevation_intersection_xr['12Z Start Day'].values)})
    
    dat_elevation_mean_daily_sub = data_elevation_mean.sel(row = fire_elevation_intersection_xr['lat'].values, 
                                          col = fire_elevation_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_elevation_intersection_xr['12Z Start Day'].values), method='nearest')
    
    dat_elevation_std_daily_sub = data_elevation_std.sel(row = fire_elevation_intersection_xr['lat'].values, 
                                          col = fire_elevation_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_elevation_intersection_xr['12Z Start Day'].values), method='nearest')
    
    ndays = len(fire_elevation_intersection_xr['12Z Start Day'])
    
    #preallocate space for the output
    df_elevation_weighted = pd.DataFrame({'day':np.zeros(ndays),'MEAN_ELEV':np.zeros(ndays),'STD_ELEV':np.zeros(ndays)})
    df_elevation_unweighted = pd.DataFrame({'day':np.zeros(ndays),'MEAN_ELEV':np.zeros(ndays),'STD_ELEV':np.zeros(ndays)})
    
    df_elevation_weighted['day'].iloc[:] = pd.to_datetime(fire_elevation_intersection_xr['12Z Start Day'].values)
    df_elevation_unweighted['day'].iloc[:] = pd.to_datetime(fire_elevation_intersection_xr['12Z Start Day'].values)

    #mean elevation
    df_elevation_weighted['MEAN_ELEV'] = np.nansum(fire_elevation_intersection_xr['weights'].values*dat_elevation_mean_daily_sub.values, axis=(1,2))
    df_elevation_unweighted['MEAN_ELEV'] = np.nanmean(fire_elevation_intersection_xr['weights_mask'].values*dat_elevation_mean_daily_sub.values, axis=(1,2))
    
    #std elevation
    df_elevation_weighted['STD_ELEV'] = np.nansum(fire_elevation_intersection_xr['weights'].values*dat_elevation_std_daily_sub.values, axis=(1,2))
    df_elevation_unweighted['STD_ELEV'] = np.nanmean(fire_elevation_intersection_xr['weights_mask'].values*dat_elevation_std_daily_sub.values, axis=(1,2))
    return df_elevation_weighted, df_elevation_unweighted