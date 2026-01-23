feature_subsets = {'raw_weather_gridmet':["vpd_gridmet","wind_speed_gridmet","dvpd_gridmet","dwind_speed_gridmet"],
                    'temperatures_gridmet':["max_air_temperature","min_air_temperature","temp_range"], 
                   'raw_weather_hrrr':["vpd_hrrrmet","wind_speed_hrrrmet","dvpd_hrrrmet","dwind_speed_hrrrmet"],
                   'heatwave':["days_in_high_heatwave","days_in_highlow_heatwave","min_air_temperature","max_air_temperature"],
                   'derived_weather_hwp': ["hwp","dhwp"],
                   'derived_weather_hdw': ["dhd0w0","dhd1w0", "dhd2w0", "dhd3w0","hd0w0", "hd1w0", "hd2w0", "hd3w0"],
                   'derived_weather_cffrds': [ "dIMERG.FINAL.v6_FWI","dIMERG.FINAL.v6_BUI", "IMERG.FINAL.v6_FWI","IMERG.FINAL.v6_BUI"],
                   'derived_weather_nfdrs':[ "dburning_index_g","denergy_release_component-g","burning_index_g","energy_release_component-g"],
                   'stability': ["PFT", "chi","dchi", "dPFT"],
                   'living_moisture':['FMCGLH2D'],
                   'noaa_dead_moisture': ['Blended_SM'],
                   'ncar_dead_moisture': ['FMCG2D'],
                   'cffdrs_moisture':['IMERG.FINAL.v6_FFMC','IMERG.FINAL.v6_DMC','IMERG.FINAL.v6_DC'],
                   'nfdrs_moisture':['dead_fuel_moisture_100hr','dead_fuel_moisture_1000hr'],
                   'hrrr_moisture':['soilm_sfc','soilm_1cm', 'soilm_4cm', 'soilm_10cm', 'soilm_30cm'],
                   'pws':["PWS"],
                   'esi':["ESI"],
                   'loading': ["Low_N", "Moderate_N", "High_N", "VeryHigh_N", "Extreme_N"],
                   'terrain_slope': ["MEAN_SLOPE","STD_SLOPE"],
                   'terrain_elevation': ["MEAN_ELEV","STD_ELEV"],
                   'DOY': ['DOY'],
                   'containment': ["percent_contained_1"],
                   'resource_personnel': ["crew_personnel_1","engine_personnel_1","air_personnel_1","construction_personnel_1","overhead_personnel_1"],
                   'resource_quantity': ["crew_quantity_1", "engine_quantity_1",  "air_quantity_1", "construction_quantity_1"],
                   'pop': ["POP_DENSITY"],
                   'structures': ["structures_destroyed_1", "structures_damaged_1","structures_threatened_72_1"],
                   'persistence': ["FRE_1"]}

no_persistence = []
for v in feature_subsets.values():
    no_persistence += list(v)
no_persistence = list(set(no_persistence) - set(feature_subsets['persistence']))
feature_subsets['features_no_persistence'] = no_persistence