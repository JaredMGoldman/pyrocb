import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from joblib import Parallel, delayed


from utils import calculate_intersection, make_file_namelist, generate_df, ML_DATA_ROOT

def imerg_fwi_timeseries(df, fwi_data_root = "/data/lthapa/data2restore/lthapa"):
    varis = ['day','IMERG.FINAL.v6_DC','IMERG.FINAL.v6_DMC','IMERG.FINAL.v6_FFMC',
             'IMERG.FINAL.v6_ISI','IMERG.FINAL.v6_BUI','IMERG.FINAL.v6_FWI',
             'IMERG.FINAL.v6_DSR'] 
    df_imerg_weighted = generate_df(varis, len(df))
    df_imerg_unweighted = generate_df(varis, len(df))
    
    fwi_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/IMERG_FWI_GRID',10000) 
                                 for ii in range(len(df)))

    fire_fwi_intersection=gpd.GeoDataFrame(pd.concat(fwi_intersections, ignore_index=True))
    fire_fwi_intersection.set_geometry(col='geometry')    
    fire_fwi_intersection = fire_fwi_intersection.set_index(['12Z Start Day', 'lat', 'lon'])

    fire_fwi_intersection=fire_fwi_intersection[~fire_fwi_intersection.index.duplicated()]
    
    fire_fwi_intersection_xr = fire_fwi_intersection.to_xarray()
    fire_fwi_intersection_xr['weights_mask'] = xr.where(fire_fwi_intersection_xr['weights']>0,1, np.nan)

    #load in FWI data associated with the fire
    times = pd.date_range(np.datetime64(df['UTC Day'].iloc[0]),
                        np.datetime64(df['UTC Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    fwi_filenames,times_back_used = make_file_namelist(times,f'{fwi_data_root}/YYYY/FWI_IMERG/WESTUS_FWI.IMERG.FINAL.v6.Daily.Default.YYYYMMDD.nc')
    
    dat_fwi = xr.open_mfdataset(fwi_filenames,concat_dim='time',combine='nested',compat='override', coords='all')
    dat_fwi = dat_fwi.assign_coords({'time': times_back_used}) #assign coords so we can resample along time
    dat_fwi = dat_fwi.resample(time='1H').pad() #make the data hourly, so we can define the day as 12z-12z instead of 0z-0z
    dat_fwi_mean = dat_fwi.resample(time='24H',offset="12H" ,label='left').mean() #take the daily mean         
    
    #select the locations and times we want
    dat_fwi_sub = dat_fwi_mean.sel(lat = fire_fwi_intersection_xr['lat'].values, 
                    lon = fire_fwi_intersection_xr['lon'].values).sel(
                    time = pd.to_datetime(fire_fwi_intersection_xr['12Z Start Day'].values + np.timedelta64(12, 'h')), method='nearest')#these should be lined up correctly

    df_imerg_weighted['day'].iloc[:] = pd.to_datetime(fire_fwi_intersection_xr['12Z Start Day'].values)
    df_imerg_unweighted['day'].iloc[:] = pd.to_datetime(fire_fwi_intersection_xr['12Z Start Day'].values)

    for var in varis[1:]:
        df_imerg_weighted[var] = np.nansum(fire_fwi_intersection_xr['weights'].values*dat_fwi_sub[var].values, axis=(1,2)) #weighted average
        df_imerg_unweighted[var] = np.nanmean(fire_fwi_intersection_xr['weights_mask'].values*dat_fwi_sub[var].values,axis=(1,2)) #mask and average
    return df_imerg_weighted, df_imerg_unweighted
