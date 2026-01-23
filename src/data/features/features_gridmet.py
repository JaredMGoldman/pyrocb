import pandas as pd
import numpy as np
import xarray as xr
from utils import parallel_intersection_labels, make_file_namelist, generate_df

def gridmet_timeseries(df, gridmet_data_root = '/data/lthapa/data2restore/lthapa'):
    varis_gridmet = ['day','burning_index_g','energy_release_component-g',
                     'potential_evapotranspiration_alfalfa','potential_evapotranspiration_grass',
                     'dead_fuel_moisture_1000hr','dead_fuel_moisture_100hr', 'precipitation_amount',
                     'max_relative_humidity', 'min_relative_humidity','specific_humidity',
                     'surface_downwelling_shortwave_flux_in_air', 'wind_from_direction',
                     'min_air_temperature', 'max_air_temperature', 'mean_vapor_pressure_deficit', 
                     'wind_speed']
    
    df_gridmet_weighted = generate_df(varis_gridmet, len(df))
    df_gridmet_unweighted = generate_df(varis_gridmet, len(df))

    #do the intersection, in parallel
    fire_gridmet_intersection_xr = parallel_intersection_labels(df, "GRIDMET_GRID")

    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df['12Z Start Day'].iloc[0]),
                        np.datetime64(df['12Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    gridmet_filenames,times_back_used = make_file_namelist(times,f'{gridmet_data_root}/YYYY/GRIDMET/gridmet_all_YYYY-MM-DD.nc')

    dat_gridmet = xr.open_mfdataset(gridmet_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    dat_gridmet = dat_gridmet.assign_coords({'Time': times_back_used}) #assign coords so we can select in time
    
    #select the locations and times we want
    dat_gridmet_sub = dat_gridmet.isel(lat = fire_gridmet_intersection_xr['row'].values, 
                    lon = fire_gridmet_intersection_xr['col'].values).sel(
                    Time = pd.to_datetime(fire_gridmet_intersection_xr['12Z Start Day'].values))#these should be lined up correctly

    df_gridmet_weighted['day'].iloc[:] = pd.to_datetime(fire_gridmet_intersection_xr['12Z Start Day'].values)
    df_gridmet_unweighted['day'].iloc[:] = pd.to_datetime(fire_gridmet_intersection_xr['12Z Start Day'].values)

    for var in varis_gridmet[1:]:
        df_gridmet_weighted[var] = np.nansum(fire_gridmet_intersection_xr['weights'].values*dat_gridmet_sub[var].values, axis=(1,2))
        df_gridmet_unweighted[var] = np.nanmean(fire_gridmet_intersection_xr['weights_mask'].values*dat_gridmet_sub[var].values, axis=(1,2)) #MASK AND AVERAGE
    
    return df_gridmet_weighted, df_gridmet_unweighted