import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from joblib import Parallel, delayed

from utils import calculate_intersection, ML_DATA_ROOT


def slope_timeseries(df, slope_data = f'{ML_DATA_ROOT}/slope_990m_LF2020.nc'):
    #do the intersection, in parallel
    slope_intersections = Parallel(n_jobs=10)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/SLOPE_GRID_990M',2000) 
                                 for ii in range(len(df)))
    print([slope_intersections[jj]['weights'].sum() for jj in range(len(slope_intersections))])

    
    fire_slope_intersection=gpd.GeoDataFrame(pd.concat(slope_intersections, ignore_index=True))
    fire_slope_intersection.set_geometry(col='geometry')
    fire_slope_intersection = fire_slope_intersection.set_index(['12Z Start Day', 'row', 'col'])
    
    fire_slope_intersection_xr = fire_slope_intersection.to_xarray()
    
    fire_slope_intersection_xr['weights_mask'] = xr.where(fire_slope_intersection_xr['weights']>0,1, np.nan)
       
    #load in SLOPE data associated with the fire (it's only one dataset)  
    #open the SLOPE files
    dat_slope = xr.open_dataset(slope_data) #map is fixed in time
    
    dat_slope = dat_slope.assign_coords({'time': pd.to_datetime(fire_slope_intersection_xr['12Z Start Day'].values)})
    data_slope_mean = dat_slope['mean_slope'].expand_dims({'time': pd.to_datetime(fire_slope_intersection_xr['12Z Start Day'].values)}) #the mean slope expanded over all the days
    data_slope_std = dat_slope['std_slope'].expand_dims({'time': pd.to_datetime(fire_slope_intersection_xr['12Z Start Day'].values)})
    
    dat_slope_mean_daily_sub = data_slope_mean.sel(row = fire_slope_intersection_xr['row'].values, 
                                          col = fire_slope_intersection_xr['col'].values,
                      time = pd.to_datetime(fire_slope_intersection_xr['12Z Start Day'].values), method='nearest')
    
    dat_slope_std_daily_sub = data_slope_std.sel(row = fire_slope_intersection_xr['row'].values, 
                                          col = fire_slope_intersection_xr['col'].values,
                      time = pd.to_datetime(fire_slope_intersection_xr['12Z Start Day'].values), method='nearest')
    
    ndays = len(fire_slope_intersection_xr['12Z Start Day'])
    
    #preallocate space for the output
    df_slope_weighted = pd.DataFrame({'day':np.zeros(ndays),'MEAN_SLOPE':np.zeros(ndays),'STD_SLOPE':np.zeros(ndays)})
    df_slope_unweighted = pd.DataFrame({'day':np.zeros(ndays),'MEAN_SLOPE':np.zeros(ndays),'STD_SLOPE':np.zeros(ndays)})

    df_slope_weighted['day'].iloc[:] = pd.to_datetime(fire_slope_intersection_xr['12Z Start Day'].values)
    df_slope_unweighted['day'].iloc[:] = pd.to_datetime(fire_slope_intersection_xr['12Z Start Day'].values)

    #mean slope
    df_slope_weighted['MEAN_SLOPE'] = np.nansum(fire_slope_intersection_xr['weights'].values*dat_slope_mean_daily_sub.values, axis=(1,2))
    df_slope_unweighted['MEAN_SLOPE'] = np.nanmean(fire_slope_intersection_xr['weights_mask'].values*dat_slope_mean_daily_sub.values, axis=(1,2))
    
    #std slope
    df_slope_weighted['STD_SLOPE'] = np.nansum(fire_slope_intersection_xr['weights'].values*dat_slope_std_daily_sub.values, axis=(1,2))
    df_slope_unweighted['STD_SLOPE'] = np.nanmean(fire_slope_intersection_xr['weights_mask'].values*dat_slope_std_daily_sub.values, axis=(1,2))
    return df_slope_weighted, df_slope_unweighted