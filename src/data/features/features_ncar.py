import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from joblib import Parallel, delayed
from utils import calculate_intersection, make_file_namelist, generate_df, ML_DATA_ROOT

#ncar grid was 1000m = 1km = grid resolution
# fwi grid was 10000m=10km = grid resolution
def ncar_timeseries(df, data_root = '/data/lthapa/data2restore/lthapa'):
    varis_ncar = ['day','FMCG2D','FMCGLH2D']
    df_ncar_weighted = generate_df(varis_ncar, len(df))
    df_ncar_unweighted = generate_df(varis_ncar, len(df))

    ncar_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/NCAR_MOISTURE_GRID',1000) 
                                 for ii in range(len(df)))
    
    fire_ncar_intersection=gpd.GeoDataFrame(pd.concat(ncar_intersections, ignore_index=True))
    fire_ncar_intersection.set_geometry(col='geometry')  
    fire_ncar_intersection = fire_ncar_intersection.set_index(['12Z Start Day', 'row', 'col'])
    fire_ncar_intersection=fire_ncar_intersection[~fire_ncar_intersection.index.duplicated()]

    fire_ncar_intersection_xr = fire_ncar_intersection.to_xarray()
    fire_ncar_intersection_xr['weights_mask'] = xr.where(fire_ncar_intersection_xr['weights']>0,1, np.nan)
    
    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df['12Z Start Day'].iloc[0]),
                        np.datetime64(df['12Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    ncar_filenames,times_back_used = make_file_namelist(times,f'{data_root}/YYYY/FMC/fmc_YYYYMMDD_20Z.nc')

    try:
        dat_ncar = xr.open_mfdataset(ncar_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    except:
        df_ncar_weighted['day'].iloc[:] = pd.to_datetime(fire_ncar_intersection_xr['12Z Start Day'].values)
        df_ncar_unweighted['day'].iloc[:] = pd.to_datetime(fire_ncar_intersection_xr['12Z Start Day'].values)
        return df_ncar_weighted, df_ncar_unweighted
    dat_ncar = dat_ncar.assign_coords({'Time': times_back_used}) #assign coords so we can select in time
    dat_ncar = dat_ncar.reindex(Time=times,method='nearest') #makes the data daily and fills in any gaps
    dat_ncar = dat_ncar.where(dat_ncar!=0) #masks out the 0s for the ocean
    
    #select the locations and times we want
    dat_ncar_sub = dat_ncar.isel(south_north = fire_ncar_intersection_xr['row'].values.astype(int), 
                                 west_east = fire_ncar_intersection_xr['col'].values.astype(int)).sel(
                                 Time = pd.to_datetime(fire_ncar_intersection_xr['12Z Start Day'].values))#these should be lined up correctly
    
    df_ncar_weighted['day'].iloc[:] = pd.to_datetime(fire_ncar_intersection_xr['12Z Start Day'].values)
    df_ncar_unweighted['day'].iloc[:] = pd.to_datetime(fire_ncar_intersection_xr['12Z Start Day'].values)

    for var in varis_ncar[1:]:
        df_ncar_weighted[var] = np.nansum(fire_ncar_intersection_xr['weights'].values*dat_ncar_sub[var].values, axis=(1,2))
        df_ncar_unweighted[var] = np.nansum(fire_ncar_intersection_xr['weights'].values*dat_ncar_sub[var].values, axis=(1,2))

    try: 
        # this day is messed up, fill it in with NANS
        df_ncar_weighted.iloc[df_ncar_weighted['day']=='2020-09-09'] = [pd.date_range(np.datetime64('2020-09-09'),np.datetime64('2020-09-09')+np.timedelta64(0,'D')),
                                                            np.nan,np.nan]
        df_ncar_unweighted.iloc[df_ncar_unweighted['day']=='2020-09-09'] = [pd.date_range(np.datetime64('2020-09-09'),np.datetime64('2020-09-09')+np.timedelta64(0,'D')),
                                                            np.nan,np.nan]
    except:
        pass
    return df_ncar_weighted, df_ncar_unweighted