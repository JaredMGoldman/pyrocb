import geopandas as gpd
import numpy as np
import pandas as pd
from data.features import *
from utils import get_repo_root, add_timestamp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime

data_dir = '/data/lthapa/data2restore/lthapa/ML_daily'
path_poly = f'{data_dir}/fire_polygons/'
suffix_poly = '_VIIRS_daily_12Z_Day_Start.geojson'
structures_file  = '/data/lthapa/data2restore/lthapaYYYY_STRUCTURES.xlsx'
resources_file = '/data/lthapa/data2restore/lthapa/YYYY+_PROCESSED_RESOURCES.xlsx'

heatwave_dict = {
    'heatwave' : {
        'func' : heatwave_timeseries,
        'gridmet_data_root' : '/data/lthapa/data2restore/lthapa'}}

args_dict_weighting = {
    'elevation' : {
        'func' : elevation_timeseries,
        'path_elevation' : '/data/lthapa/data2restore/lthapa/ML_daily/elev_990m_LF2020.nc'},
    'esi' : {
        'func' : esi_timeseries,
        'esi_data_dir' : '/data/lthapa/data2restore/lthapa'},
    'fwi' : {
        'func' : imerg_fwi_timeseries,
        'fwi_data_root' : '/data/lthapa/data2restore/lthapa'},
    'gridmet' : {
        'func' : gridmet_timeseries,
        'gridmet_data_root' : '/data/lthapa/data2restore/lthapa'},
    'hdw' : {
        'func' : hdw_lagged_timeseries,
        'hrrr_data_root' : f'{ML_DATA_ROOT}/pygraf/processed_hrrr_hdw_hwp'},
    'hdw500' : {
        'func' : hdw500_lagged_timeseries,
        'hrrr_data_root' : f'{ML_DATA_ROOT}/pygraf/processed_hrrr_hdw500'},
    'hrrrmet' : {
        'func' : hrrrmet_timeseries,
        'hrrr_data_root' : f'{ML_DATA_ROOT}/pygraf/processed_hrrr_hdw_hwp'},
    'hrrr_moisture' : {
        'func' : hrrr_moisture_timeseries,
        'hrrr_data_root' : f'{ML_DATA_ROOT}/pygraf/processed_hrrr_moisture',
        'landmask_data' : f'{ML_DATA_ROOT}/pygraf/HRRR_LANDMASK.nc'},
    'hwp' : {
        'func' : hwp_timeseries,
        'hrrr_data_root' : f'{ML_DATA_ROOT}/pygraf/processed_hrrr_hdw_hwp'},
    'loading' : {
        'func' : fuel_loading_timeseries,
        'fuel_fwi_data' : f'{ML_DATA_ROOT}/fuel_fwi_990m.nc'},
    'ncar' : {
        'func' : ncar_timeseries,
        'data_root' : '/data/lthapa/data2restore/lthapa'},
    'population' : {
        'func' : pop_timeseries,
        'pop_data' : '/data/lthapa/data2restore/lthapa/static_maps/population_density/gpw_v4_population_density_rev11_2pt5_min.nc' },
    'pws' : {
        'func' : pws_timeseries,
        'pws_data' : '/data/lthapa/data2restore/lthapa/PWS_6_jan_2021.nc'},
    'rave_2020' : {
        'func' : rave_timeseries2020,
        'data_root' :'/data/lthapa/data2restore/lthapa'},
    'rave_fre' : {
        'func' : rave_timeseries_fre,
       'data_root' :'/data/lthapa/data2restore/lthapa'},
    'rave' : {
        'func' : rave_timeseries,
        'data_root' :'/data/lthapa/data2restore/lthapa'},
    'slope' : {
        'func' : slope_timeseries,
        'slope_data' : f'{ML_DATA_ROOT}/slope_990m_LF2020.nc'},
    'smops' : {
        'func' : smops_timeseries,
        'data_root' : '/data/lthapa/data2restore/lthapa'}
    } 

weighted_funcs = {k: v["func"] for k, v in args_dict_weighting.items()}
weighted_kwargs = {k: {kk: vv for kk, vv in v.items() if kk != "func"} for k, v in args_dict_weighting.items()}

def _run_one_feature_proc(args):
    k, df_fire_local, kwargs = args
    t0 = datetime.now()
    fw, fu = weighted_funcs[k](df_fire_local, **kwargs)  # NOTE: this must be importable at module scope
    secs = (datetime.now() - t0).total_seconds()
    return k, fw, fu, secs

if __name__ == "__main__":
    years = [2019, 2020, 2021]
    features_dict = {}
    timing_dict = {k : 0 for k in args_dict_weighting.keys()}
    timing_dict['heatwave'] = 0
    timing_dict['resources'] = 0
    timing_dict_init = timing_dict.copy()
    timing_file_name = f'{add_timestamp("feature_loading_time_report")}.csv'
    timing_file_path = os.path.join(get_repo_root(),'outputs',timing_file_name)
    header = "feature,time,irwin_id,year\n"
    with open(timing_file_path, 'x') as f:
        f.write(header)
    for year in years:    
        # load fire polygons
        fire_daily_pth = f"{path_poly}ClippedFires{year}{suffix_poly}"
        fire_daily = gpd.read_file(fire_daily_pth)
        fire_daily=fire_daily.drop(columns=['Current Overpass'])
        fire_daily = fire_daily.drop(np.where(fire_daily['geometry']==None)[0])
        fire_daily['fire area (ha)'] = fire_daily['geometry'].area/10000 #hectares. from m2
        fire_daily.set_geometry(col='geometry', inplace=True) #designate the geometry column
        fire_daily = fire_daily.rename(columns={'Current Day':'UTC Day', 'Local Day': '12Z Start Day'})

        irwinIDs = np.unique(fire_daily['irwinID'].values)    

        for irwin_id in irwinIDs:
            timing_dict = timing_dict_init.copy()
            features_dict[irwin_id] = {}
            df_fire = fire_daily[fire_daily['irwinID']==irwin_id] #this is what gets fed to the feature selection code
            
            days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
            df_fire = df_fire[(days>=np.datetime64(str(year)+'-07-01'))&
                                (days<=np.datetime64(str(year+1)+'-01-01'))].reset_index(drop=True)
            
            ## Resources Section ##
            # structures
            """
            CURRENTLY (1-23-26) 2019 DATA (at least) LACKS ALL NECESSARY COLUMNS LEADING TO ISSUES DURING THE 
            FEATURE GEN SCRIPT. VERIFY DATA IS CORRECTLY FORMATTED.

            structures_all = pd.read_excel(resources_file.replace('YYYY', str(year))).iloc[:,1:19]
            if year==2021:
                structures_all = structures_all[['REPORT_FROM_DATE','REPORT_TO_DATE','IRWIN_IDENTIFIER']] #,'QTY_DESTROYED','QTY_DAMAGED','QTY_THREATENED_72']
            else:
                structures_all = structures_all[['REPORT_FROM_DATE','REPORT_TO_DATE','PCT_CONTAINED_COMPLETED',
                                'INCIDENT_NAME', 'IRWIN_IDENTIFIER']] # 'QTY_DESTROYED','QTY_DAMAGED','QTY_THREATENED_72']
            structures = structures_timeseries(df_fire, structures_all)
            structures = pd.concat([structures, pd.DataFrame({'irwinID':[irwin_id]*len(structures)})], axis=1)
            features_dict[irwin_id]["structures"] = structures
            """
            # resources 
            start_r = datetime.now()
            resources_all = pd.read_excel(resources_file.replace('YYYY', str(year))).iloc[:,1:19]
            resources_all = resources_all[['INC209R_IDENTIFIER','REPORT_FROM_DATE','REPORT_TO_DATE','PCT_CONTAINED_COMPLETED',
                                'INCIDENT_NAME', 'IRWIN_IDENTIFIER','CODE_NAME','RESOURCE_QUANTITY','RESOURCE_PERSONNEL',
                                'crew_quantity', 'crew_personnel','engine_quantity', 'engine_personnel', 'air_quantity', 
                                'air_personnel','construction_quantity', 'construction_personnel','overhead_personnel']]
            resources = resources_timeseries(df_fire, resources_all)
            resources = pd.concat([resources, pd.DataFrame({'irwinID':[irwin_id]*len(resources)})], axis=1)
            features_dict[irwin_id]["resources"] = resources
            timing_dict['resources'] += (datetime.now() - start_r).total_seconds()

            # heatwave
            for k, v in heatwave_dict.items():
                start_h = datetime.now()
                func = v.pop('func')
                feature = func(df_fire, **v)
                feature = pd.concat([feature, pd.DataFrame({'irwinID':[irwin_id]*len(feature)})], axis=1) #add IrwinID
                features_dict[irwin_id][f"{k}"] = feature
                heatwave_dict[k]['func'] = func
                timing_dict['heatwave'] += (datetime.now() - start_h).total_seconds()
            
            ### begin weighted/unweighted features ###
            tasks = [(k, df_fire, weighted_kwargs[k]) for k in weighted_funcs.keys()]
            max_workers = min(4, len(tasks), (os.cpu_count() or 4))

            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_run_one_feature_proc, t) for t in tasks]
                for fut in as_completed(futs):
                    results.append(fut.result())
            
            for k, feature_weighted, feature_unweighted, secs in results:
                feature_weighted = pd.concat(
                    [feature_weighted, pd.DataFrame({"irwinID": [irwin_id] * len(feature_weighted)})],
                    axis=1,
                )
                feature_unweighted = pd.concat(
                    [feature_unweighted, pd.DataFrame({"irwinID": [irwin_id] * len(feature_unweighted)})],
                    axis=1,
                )
                features_dict[irwin_id][f"{k}_weighted"] = feature_weighted
                features_dict[irwin_id][f"{k}_unweighted"] = feature_unweighted
                timing_dict[k] += secs

            # for k, v in args_dict_weighting.items():
            #     print(f"loading {k} features...")
            #     start = datetime.now()
            #     func = v.pop('func')
            #     feature_weighted, feature_unweighted = func(df_fire, **v)
            #     feature_weighted = pd.concat([feature_weighted, pd.DataFrame({'irwinID':[irwin_id]*len(feature_weighted)})], axis=1)
            #     feature_unweighted = pd.concat([feature_unweighted, pd.DataFrame({'irwinID':[irwin_id]*len(feature_unweighted)})], axis=1)
                
            #     features_dict[irwin_id][f"{k}_weighted"] = feature_weighted
            #     features_dict[irwin_id][f"{k}_unweighted"] = feature_unweighted
            #     args_dict_weighting[k]['func'] = func

            #     timing_dict[k] += (datetime.now() - start).total_seconds()

            for k, v in timing_dict.items():
                header = f"{header}{k},{v},{irwin_id},{year}\n"

            with open(timing_file_path, 'w') as f:
                f.write(header)

    print("finished feature loading")