import pandas as pd
import geopandas as gpd
import numpy as np
from joblib import Parallel, delayed
import xarray as xr

from utils import calculate_intersection, make_file_namelist, generate_df, ML_DATA_ROOT

#HRRR_WS formulation from, take time mean, then take weighted average. For HDW, multiply the weighted means of VPD and WIND
def hdw_lagged_timeseries(df, hrrr_data_root = f"{ML_DATA_ROOT}/pygraf/processed_hrrr_hdw_hwp"):  #with the wind speed
    varis_hrrr_derived = ['day','hd0w0', 'hd1w0','hd2w0', 'hd3w0']#, 'hd4w0', 'hd5w0',
    df_hdw_weighted = generate_df(varis_hrrr_derived, len(df))
    df_hdw_unweighted = generate_df(varis_hrrr_derived, len(df))
    
    #do the intersection, in parallel
    hrrr_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],f'{ML_DATA_ROOT}/HRRR_GRID',3000) 
                                 for ii in range(len(df)))
    
    fire_hrrr_intersection=gpd.GeoDataFrame(pd.concat(hrrr_intersections, ignore_index=True))
    fire_hrrr_intersection.set_geometry(col='geometry')

    #loop over all of the days we have intersections
    times_intersect = np.unique(fire_hrrr_intersection['12Z Start Day'].values)
    
    # all todays for this incident
    todays = np.array(times_intersect, dtype="datetime64[ns]")

    # you use: t_min = today + 12h - 3D, t_max = today + 12h + 23h
    global_t_min = todays.min() + np.timedelta64(12, "h") - np.timedelta64(3, "D")
    global_t_max = todays.max() + np.timedelta64(12, "h") + np.timedelta64(23, "h")

    # build union hourly range
    times_back_all = pd.date_range(start=global_t_min, end=global_t_max, freq="h")
    files_all, times_all_used = make_file_namelist(times_back_all, f'{hrrr_data_root}/Processed_HRRR_YYYYMMDDHH_HDW_HWP.nc')

    # open ONCE
    dat_all = xr.open_mfdataset(
        files_all,
        concat_dim="time",
        combine="nested",
        compat="override",
        coords="all")
    wind_var = "wind_speed" if "wind_speed" in dat_all.data_vars else "wind_max"
    vpd_var  = "vpd_2m" if "vpd_2m" in dat_all.data_vars else "vpd_max"
    
    dat_all = dat_all[[wind_var,vpd_var]] \
            .assign_coords(time=times_all_used).resample(time='h').asfreq() \
            .stack(cell=("grid_yt", "grid_xt"))

    for i, today in enumerate(times_intersect):
        df_hdw_weighted.loc[i, 'day'] = today
        df_hdw_unweighted.loc[i, 'day'] = today
        #get the time
        df_sub = fire_hrrr_intersection[fire_hrrr_intersection['12Z Start Day']==today]
        cells = list(zip(df_sub["row"], df_sub["col"]))

        day0 = np.datetime64(today) + np.timedelta64(12,'h')

        t_min = np.datetime64(today) + np.timedelta64(12, "h") - np.timedelta64(3, "D")
        t_max = np.datetime64(today) + np.timedelta64(12, "h") + np.timedelta64(23, "h")
        
        dat_hrrr_stacked = dat_all.sel(time=slice(t_min, t_max))  # cheap view
        
        weights = df_sub["weights"].values
        w0 = dat_hrrr_stacked[wind_var].sel(
            time=pd.date_range( start=day0, 
                                end=day0+np.timedelta64(23,'h'),
                                freq='h')).sel(cell = cells).load()

        # Pull the subset ONCE and load into memory
        vpd_sub = dat_hrrr_stacked[vpd_var] \
            .sel(cell = cells).load()
        for day_num in range(4):
            # define the days
            day = day0 if day_num == 0 else np.datetime64(today)+np.timedelta64(12,'h') - np.timedelta64(day_num,'D')
                
            #define the times we will select for VPD
            times = pd.date_range(start=day, end=day+np.timedelta64(23,'h'),freq='h')
            
            hd = vpd_sub.sel(time=times)
            if day_num > 0:
                hd=hd.assign_coords({'time':w0['time'].values})
            
            hdw0 = hd*w0
            hdw0_daily_mean = hdw0.resample(time='24h', offset="12h", label='left').mean() #take the daily mean        

            df_hdw_weighted.loc[i, (f'hd{day_num}w0')] = np.nansum((hdw0_daily_mean.values)* weights)
            df_hdw_unweighted.loc[i, (f'hd{day_num}w0')] = np.nanmean((hdw0_daily_mean.values))
        
    return df_hdw_weighted, df_hdw_unweighted