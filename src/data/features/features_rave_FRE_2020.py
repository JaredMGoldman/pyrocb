import pandas as pd
import numpy as np
import xarray as xr

from utils import parallel_intersection_labels, make_file_namelist, generate_df

def rave_timeseries2020(df, data_root = '/data/lthapa/data2restore/lthapa'):
    varis = ['day','FRE']#, 'FRP_SD', 'FRE']#, 'CO2', 'CO', 'SO2', 'OC','BC', 'PM25', 'NOx', 'NH3','TPM', 'VOCs', 'CH4'] #don't need 'area', it's the area of each cell
    df_rave_weighted = generate_df(varis, len(df))
    df_rave_unweighted = generate_df(varis, len(df))

    fire_rave_intersection_xr = parallel_intersection_labels(df, 'RAVE_GRID_3KM')
    
    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df['12Z Start Day'].iloc[0]),
                        np.datetime64(df['12Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    rave_filenames,_ = make_file_namelist(times,
                                            f'{data_root}/YYYY/RAVE/MM/RAVE-HrlyEmiss-3km-CONUS_v1r1_blend_sYYYYMMDD.nc')                                                 
    
    #print(rave_filenames)
    dat_rave = xr.open_mfdataset(rave_filenames,concat_dim='time',combine='nested',compat='override', coords='all')

    dat_rave = dat_rave.resample(time='24H',base=12).sum(dim='time') #take the daily sum
    
    #select the locations and times we want
    dat_rave_sub = dat_rave.isel(grid_yt = fire_rave_intersection_xr['row'].values.astype(int), 
                    grid_xt = fire_rave_intersection_xr['col'].values.astype(int)).sel(
                    time = pd.to_datetime(fire_rave_intersection_xr['12Z Start Day'].values+
                                         'T12:00:00.000000000'))#these should be lined up correctly

    df_rave_weighted['day'].iloc[:] = pd.to_datetime(fire_rave_intersection_xr['12Z Start Day'].values)
    df_rave_unweighted['day'].iloc[:] = pd.to_datetime(fire_rave_intersection_xr['12Z Start Day'].values)

    for var in varis[1:]:
        df_rave_weighted[var] = np.nansum(fire_rave_intersection_xr['weights'].values*dat_rave_sub[var].values, axis=(1,2))
        df_rave_unweighted[var] = np.nansum(fire_rave_intersection_xr['weights_mask'].values*dat_rave_sub[var].values,axis=(1,2))
    
    return df_rave_weighted, df_rave_unweighted 