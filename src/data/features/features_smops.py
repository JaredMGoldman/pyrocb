import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from joblib import Parallel, delayed

from utils import calculate_intersection, make_file_namelist, generate_df, ML_DATA_ROOT

def smops_timeseries(df, data_root = '/data/lthapa/data2restore/lthapa'):
    varis_smops = ['day','Blended_SM']
    df_smops_weighted = generate_df(varis_smops, len(df))
    df_smops_unweighted = generate_df(varis_smops, len(df))
    
    smops_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/SMOPS_GRID',25000) 
                                 for ii in range(len(df)))

    fire_smops_intersection=gpd.GeoDataFrame(pd.concat(smops_intersections, ignore_index=True))
    fire_smops_intersection.set_geometry(col='geometry')  
    fire_smops_intersection = fire_smops_intersection.set_index(['12Z Start Day', 'row', 'col'])
    fire_smops_intersection=fire_smops_intersection[~fire_smops_intersection.index.duplicated()]

    fire_smops_intersection_xr = fire_smops_intersection.to_xarray()
    fire_smops_intersection_xr['weights_mask'] = xr.where(fire_smops_intersection_xr['weights']>0,1, np.nan)
    
    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df['12Z Start Day'].iloc[0]),
                        np.datetime64(df['12Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    smops_filenames,times_back_used = make_file_namelist(times,f'{data_root}/YYYY/SMOPS/NPR_SMOPS_CMAP_DYYYYMMDD.nc')

    dat_smops = xr.open_mfdataset(smops_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    dat_smops = dat_smops.assign_coords({'Time': times_back_used}) #assign coords so we can select in time
    
    #select the locations and times we want
    dat_smops_sub = dat_smops.isel(Latitude = fire_smops_intersection_xr['row'].values.astype(int), 
                    Longitude = fire_smops_intersection_xr['col'].values.astype(int)).sel(
                    Time = pd.to_datetime(fire_smops_intersection_xr['12Z Start Day'].values))#these should be lined up correctly

    df_smops_weighted['day'].iloc[:] = pd.to_datetime(fire_smops_intersection_xr['12Z Start Day'].values)
    df_smops_unweighted['day'].iloc[:] = pd.to_datetime(fire_smops_intersection_xr['12Z Start Day'].values)

    for var in varis_smops[1:]:
        dat_smops_sub[var]=dat_smops_sub[var].where(dat_smops_sub[var] != -0.0999) #mask out the ocean, there is no soil moisture here
        df_smops_weighted[var] = np.nansum(fire_smops_intersection_xr['weights'].values*dat_smops_sub[var].values, axis=(1,2))
        df_smops_unweighted[var] = np.nanmean(fire_smops_intersection_xr['weights_mask'].values*dat_smops_sub[var].values, axis=(1,2)) #MASK AND AVERAGE
    
    return df_smops_weighted, df_smops_unweighted