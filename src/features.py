def generate_sensitivity_list(feat_dict, category_selected_list, additional_to_append):
    sub = []
    for item in category_selected_list: #add the categories
        sub = sub+feat_dict[item]

    for item2 in additional_to_append: #add the other variables, for instance if we want all categories and then HWP
        sub = sub+[item2]
    
    return sub

features_by_category = {'raw_weather_gridmet':["vpd_gridmet","wind_speed_gridmet","dvpd_gridmet","dwind_speed_gridmet"],
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
feature_subsets = {}
category_list = features_by_category.keys()
#all features
not_allowed=[]
feature_subsets['features_all'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

#no persistence
not_allowed = ['persistence']
feature_subsets['features_no_persistence'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])


#no weather
not_allowed = ['raw_weather_gridmet','temperature_gridmet','raw_weather_hrrr','heatwave','derived_weather_hwp','derived_weather_hdw',
                 'derived_weather_cffrds','derived_weather_nfdrs']
feature_subsets['features_no_surface_weather'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

feature_subsets['features_only_raw_gridmet'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["vpd_gridmet","wind_speed_gridmet","dvpd_gridmet","dwind_speed_gridmet"])


feature_subsets['features_only_raw_hrrrmet'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["vpd_hrrrmet","wind_speed_hrrrmet","dvpd_hrrrmet","dwind_speed_hrrrmet"])
feature_subsets['features_only_hd0w0'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["hd0w0","dhd0w0"])
feature_subsets['features_only_hd1w0'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["hd1w0","dhd1w0"])
feature_subsets['features_only_hd2w0'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["hd2w0","dhd2w0"])
feature_subsets['features_only_hd3w0'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["hd3w0","dhd3w0"])
feature_subsets['features_only_hwp'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["hwp","dhwp"])
feature_subsets['features_only_cffdrs'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["dIMERG.FINAL.v6_FWI","dIMERG.FINAL.v6_BUI", "IMERG.FINAL.v6_FWI","IMERG.FINAL.v6_BUI"])
feature_subsets['features_only_nfdrs'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [ "dburning_index_g","denergy_release_component-g","burning_index_g","energy_release_component-g"])
feature_subsets['features_only_heatwave'] = generate_sensitivity_list(features_by_category, 
                    [var for var in category_list if var not in not_allowed], 
                    ["days_in_high_heatwave","days_in_highlow_heatwave","max_air_temperature","min_air_temperature","temp_range"])
#NO_HEATWAVE
not_allowed = ['heatwave']
feature_subsets['features_no_heatwave'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])
feature_subsets['features_only'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])
feature_subsets['features_no_heatwave'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])
feature_subsets['features_no_heatwave'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])
feature_subsets['features_no_heatwave'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])
feature_subsets['features_no_heatwave'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])

#no stability
not_allowed = ['stability']
feature_subsets['features_no_stability'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

feature_subsets['features_only_pft'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ['PFT','dPFT'])
feature_subsets['features_only_chi'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ['chi','dchi'])


#no living_moisture
not_allowed = ['living_moisture']
feature_subsets['features_no_living_moisture'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

#no dead moisture, put categories back in 
not_allowed = ['noaa_dead_moisture' 'ncar_dead_moisture', 'cffdrs_moisture', 'nfdrs_moisture', 'hrrr_sm']
feature_subsets['features_no_dead_moisture'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

feature_subsets['features_only_noaa_sm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['Blended_SM'])
feature_subsets['features_only_ncar_sm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['FMCG2D'])
feature_subsets['features_only_cffdrs_sm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['IMERG.FINAL.v6_FFMC','IMERG.FINAL.v6_DMC','IMERG.FINAL.v6_DC'])
feature_subsets['features_only_nfdrs_sm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['dead_fuel_moisture_100hr','dead_fuel_moisture_1000hr'])
feature_subsets['features_only_hrrr_sm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['soilm_sfc','soilm_1cm', 'soilm_4cm', 'soilm_10cm', 'soilm_30cm'])

#no dead moisture, put individual moistures back in
not_allowed = ['noaa_dead_moisture' 'ncar_dead_moisture', 'cffdrs_moisture', 'nfdrs_moisture']
feature_subsets['features_only_cffdrs_ffmc'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['IMERG.FINAL.v6_FFMC'])
feature_subsets['features_only_cffdrs_dmc'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['IMERG.FINAL.v6_DMC'])
feature_subsets['features_only_cffdrs_dc'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], ['IMERG.FINAL.v6_DC'])
feature_subsets['features_only_nfdrs_100'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], 
                                ['dead_fuel_moisture_100hr'])
feature_subsets['features_only_nfdrs_1000'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], 
                                ['dead_fuel_moisture_1000hr'])

feature_subsets['features_only_hrrr_sfc'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], 
                                ['soilm_sfc'])
feature_subsets['features_only_hrrr_1cm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], 
                                ['soilm_1cm'])
feature_subsets['features_only_hrrr_4cm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], 
                                ['soilm_4cm'])
feature_subsets['features_only_hrrr_10cm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], 
                                ['soilm_10cm'])
feature_subsets['features_only_hrrr_30cm'] = generate_sensitivity_list(features_by_category, 
                            [var for var in category_list if var not in not_allowed], 
                                ['soilm_30cm'])
#no pws
not_allowed = ['pws']
feature_subsets['features_no_pws'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

#no esi
not_allowed = ['esi']
feature_subsets['features_no_esi'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

#no esi
not_allowed = ['loading']
feature_subsets['features_no_loading'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

feature_subsets['features_only_LowN'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ['Low_N'])
feature_subsets['features_only_ModerateN'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ['Moderate_N'])
feature_subsets['features_only_HighN'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ['High_N'])
feature_subsets['features_only_VeryHighN'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ['VeryHigh_N'])
feature_subsets['features_only_ExtremeN'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ['Extreme_N'])

#no terrain
not_allowed = ['terrain_slope', 'terrain_elevation']
feature_subsets['features_no_terrain'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])
feature_subsets['features_only_slope'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["MEAN_SLOPE","STD_SLOPE"])
feature_subsets['features_only_elevation'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], ["MEAN_ELEV","STD_ELEV"])

#no DOY
not_allowed = ['DOY']
feature_subsets['features_no_doy'] = generate_sensitivity_list(features_by_category, [var for var in category_list if var not in not_allowed], [])

# no population
not_allowed = ['pop']
feature_subsets['features_no_population'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])
# no containment
not_allowed = ['containment']
feature_subsets['features_no_containment'] = generate_sensitivity_list(features_by_category, 
                                  [var for var in category_list if var not in not_allowed], [])

# no structures
not_allowed = ['structures']
feature_subsets['features_no_structures'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])
feature_subsets['features_only_destroyed_structures'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["structures_destroyed_1"])
feature_subsets['features_only_damaged_structures'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["structures_damaged_1"])
feature_subsets['features_only_threatened_structures'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["structures_threatened_72_1"])

#no resource quantity
not_allowed=['resource_quantity']
feature_subsets['features_no_resource_quantity'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])

feature_subsets['features_only_crew_quantity'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["crew_quantity_1"])
feature_subsets['features_only_engine_quantity'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["engine_quantity_1"])
feature_subsets['features_only_aircraft_quantity'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["air_quantity_1"])
feature_subsets['features_only_construction_quantity'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["construction_quantity_1"])

#no resource personnel
not_allowed=['resource_personnel']
feature_subsets['features_no_resource_personnel'] = generate_sensitivity_list(features_by_category, 
                                    [var for var in category_list if var not in not_allowed], [])

feature_subsets['features_only_crew_personnel'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["crew_personnel_1"])
feature_subsets['features_only_engine_personnel'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["engine_personnel_1"])
feature_subsets['features_only_aircraft_personnel'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["air_personnel_1"])
feature_subsets['features_only_construction_personnel'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["construction_personnel_1"])
feature_subsets['features_only_overhead_personnel'] = generate_sensitivity_list(features_by_category, 
                [var for var in category_list if var not in not_allowed], ["overhead_personnel_1"])                   


