import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from joblib import Parallel, delayed

from utils import calculate_intersection, ML_DATA_ROOT


def pop_timeseries(df, pop_data = '/data/lthapa/data2restore/lthapa/static_maps/population_density/gpw_v4_population_density_rev11_2pt5_min.nc'):
    #do the intersection, in parallel
    pop_intersections = Parallel(n_jobs=10)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/POP_GRID',5000) 
                                 for ii in range(len(df)))
    print([pop_intersections[jj]['weights'].sum() for jj in range(len(pop_intersections))])

    
    fire_pop_intersection=gpd.GeoDataFrame(pd.concat(pop_intersections, ignore_index=True))
    fire_pop_intersection.set_geometry(col='geometry')
    fire_pop_intersection = fire_pop_intersection.set_index(['12Z Start Day', 'lat', 'lon'])
    
    fire_pop_intersection_xr = fire_pop_intersection.to_xarray()
    
    fire_pop_intersection_xr['weights_mask'] = xr.where(fire_pop_intersection_xr['weights']>0,1, np.nan)
    
    # load in Pop data associated with the fire (it's only one dataset)  
    # open the Pop files
    dat_pop = xr.open_dataset(pop_data) #map is fixed in time
    
    dat_pop = dat_pop.assign_coords({'time': pd.to_datetime(fire_pop_intersection_xr['12Z Start Day'].values)})
    dat_pop_daily =  dat_pop['Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'].sel(raster=5).expand_dims({'time': pd.to_datetime(fire_pop_intersection_xr['12Z Start Day'].values)}) #the POP expanded over all the days
    
    dat_pop_daily_sub = dat_pop_daily.sel(latitude = fire_pop_intersection_xr['lat'].values, 
                                          longitude = fire_pop_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_pop_intersection_xr['12Z Start Day'].values), method='nearest')
    ndays = len(fire_pop_intersection_xr['12Z Start Day'])
    
    #preallocate space for the output
    df_pop_weighted = pd.DataFrame({'day':np.zeros(ndays),'POP_DENSITY':np.zeros(ndays)})
    df_pop_unweighted = pd.DataFrame({'day':np.zeros(ndays),'POP_DENSITY':np.zeros(ndays)})

    df_pop_weighted['day'].iloc[:] = pd.to_datetime(fire_pop_intersection_xr['12Z Start Day'].values)
    df_pop_unweighted['day'].iloc[:] = pd.to_datetime(fire_pop_intersection_xr['12Z Start Day'].values)

    varis=['POP_DENSITY']
    for var in varis:
        df_pop_weighted[var] = np.nansum(fire_pop_intersection_xr['weights'].values*dat_pop_daily_sub.values, axis=(1,2)) #WEIGHTED AVERAGE
        df_pop_unweighted[var] = np.nanmean(fire_pop_intersection_xr['weights_mask'].values*dat_pop_daily_sub.values, axis=(1,2)) #MASK AND AVERAGE
    return df_pop_weighted, df_pop_unweighted