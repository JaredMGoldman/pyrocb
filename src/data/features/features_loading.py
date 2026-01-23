import pandas as pd
import numpy as np
import geopandas as gpd
from joblib import Parallel, delayed
import xarray as xr

from utils import calculate_intersection, generate_df, ML_DATA_ROOT

def fuel_loading_timeseries(df, fuel_fwi_data = f'{ML_DATA_ROOT}/fuel_fwi_990m.nc'):
    fuel_fwi_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/FUEL_FWI_GRID_990M',2000) 
                                 for ii in range(len(df)))
    
    fire_fuel_fwi_intersection=gpd.GeoDataFrame(pd.concat(fuel_fwi_intersections, ignore_index=True))
    fire_fuel_fwi_intersection = fire_fuel_fwi_intersection.drop(columns='geometry')
    fire_fuel_fwi_intersection = fire_fuel_fwi_intersection.set_index(['12Z Start Day','row', 'col'])
    fire_fuel_fwi_intersection_xr = fire_fuel_fwi_intersection.to_xarray()
    fire_fuel_fwi_intersection_xr['weights_mask'] = xr.where(fire_fuel_fwi_intersection_xr['weights']>0,1, np.nan)

    dat_fuel_fwi = xr.open_dataset(fuel_fwi_data) # map is fixed in time
    dat_fuel_fwi = dat_fuel_fwi.where(dat_fuel_fwi!=0)
    dat_fuel_fwi_daily = dat_fuel_fwi.expand_dims({'time': pd.to_datetime(fire_fuel_fwi_intersection_xr['12Z Start Day'].values)}) #the PWS expanded over all the days

    dat_fuel_fwi_sub_daily = dat_fuel_fwi_daily.sel(row = fire_fuel_fwi_intersection_xr['row'].values, 
                                        col = fire_fuel_fwi_intersection_xr['col'].values, method='nearest')

    # preallocate space for the output
    varis = ['day','Extreme_N', 'VeryHigh_N','High_N', 'Moderate_N', 'Low_N']
    df_loading_weighted = generate_df(varis, len(df))
    df_loading_unweighted = generate_df(varis, len(df))

    df_loading_weighted['day'] = df['12Z Start Day'].values
    df_loading_unweighted['day'] = df['12Z Start Day'].values

    for var in varis[1:]:
        df_loading_weighted[var] = np.nansum(fire_fuel_fwi_intersection_xr['weights'].values*dat_fuel_fwi_sub_daily[var].values, axis=(1,2))
        df_loading_unweighted[var] = np.nanmean(fire_fuel_fwi_intersection_xr['weights_mask'].values*dat_fuel_fwi_sub_daily[var].values, axis=(1,2))

    return df_loading_weighted, df_loading_unweighted