import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from herbie import Herbie
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import metpy.calc as mpcalc
from metpy.units import units
import os
import pandas as pd
from random import random
import requests
import re
from shapely import Point, distance
from joblib import dump, load
import multiprocessing as mp
from random import random
import shutil
from tqdm import tqdm
from typing import Dict
import xarray as xr

from pyrometeopy.bufkit import Profile, Surface, Sounding, parse_file
from pyrometeopy.formulas import g
from pyrometeopy import fire_plumes


from utils.constants import DATA_DIR, CP_IDX_PATH, PLOTS_DIR, CACHE_BASE_DIR, dry_run_cps
from utils.rio_utils import download_file_safe

# Close to Saskatchawan Fire: 
#   CWJH: Southend Sask
#   CYYL: Lynn Lake
#   56.210556	-102.0075
#       cp: 80
#   Practically Closest: The Pas - CYQD
#       - ['2025-07-16'] -> ['2025-08-11']

# Close to Dragon Bravo Fire:
#   GCN: Grand Canyon Station
#   NRMA3: Bright Angel Research Station
#   36.310001	-112.035278
#       cp: 36
#   Practically Closest: flagstaff - KFGZ
#       - ['2025-07-18'] -> ['2025-08-13']

# Close to Monroe Canyon
#   RIFU1: Richfield Radio KSVC
#       cp: 36
#   38.6098615	-111.9815255
#       - ['2025-07-18'] -> ['2025-08-13']

def download_soundings():
    def find_nearest_sounding(fire_pos: Point, soundings: Dict):
        def _sounding_point(snd):
            lat = snd[0].profile.lat
            lon = snd[0].profile.lon
            return Point(lon, lat)
        distances = {}
        for fname, snd in soundings.items():
            try:
                distances[fname] = distance(fire_pos, _sounding_point(snd))
            except Exception as e:
                pass
        min_distance_fn = sorted(distances)[0]
        min_distance = distances[min_distance_fn]
        min_distance_pt = _sounding_point(soundings[min_distance_fn])
        print(f"min distance from {fire_pos}: {min_distance} at {min_distance_fn} ({min_distance_pt})")
        return min_distance_fn 

    url = 'https://mtarchive.geol.iastate.edu/2025/07/27/bufkit/06/hrrr/'
    bufkit_dir = os.path.join(DATA_DIR, 'bufkits')
    os.makedirs(bufkit_dir, exist_ok=True)

    session = requests.Session()
    r = session.get(url, timeout=30)
    r.raise_for_status()
            # parse links to .hdf files
    fnames =  sorted(set(re.findall(r'href="([^"]+\.buf)"', r.text)))
    for fname in tqdm(fnames):
        if fname in os.listdir(bufkit_dir):
            continue
        r = session.get(f"{url}/{fname}")
        with open(f"{bufkit_dir}/{fname}", 'w') as f:
            f.write(r.text)

    soundings = {fname: parse_file(os.path.join(bufkit_dir, fname)) for fname in os.listdir(bufkit_dir)}

    cp_idx = pd.read_csv(CP_IDX_PATH)

    cp_info = {}
    for cp in dry_run_cps:
        this_fire = cp_idx[cp_idx.cp == int(cp)]
        fire_lat = this_fire.lat_mean
        fire_lon = this_fire.lon_mean
        fire_dtime_min = this_fire.dtime_min.values
        fire_dtime_max = this_fire.dtime_max.values
        print(f"processing fire #{cp}")
        station_fn = find_nearest_sounding(Point(fire_lon, fire_lat), soundings)
        cp_info[str(cp)] = {'station' : station_fn,
                            'dtime_min' : fire_dtime_min,
                            'dtime_max' : fire_dtime_max }
        print(f"{fire_dtime_min} -> {fire_dtime_max}\n")

def plot_sounding(sounding_data, output_folder = os.path.join(PLOTS_DIR, 'sounding_pfts'), fire_name = "Canyon Fire"):
    # store PFT values for this file
    times = []
    pfts = []
    dt_blowups = []
    dz_blowups = []

    print(f"Calculating PFT values for {len(sounding_data)} soundings...")
    os.makedirs(output_folder, exist_ok = True)

    # blow up analysis which will say how much heat is needed from the fire for the atmosphere to blow up
    for i, snd in enumerate(sounding_data):
        times.append(pd.to_datetime(snd.surface.time[0], format='%y%m%d/%H%M'))
        pft_val = fire_plumes.pft(snd, moisture_ratio=15.0)
        pfts.append(np.float64(pft_val) if pft_val else 650)

        # where data is being read in
        blow_up = fire_plumes.blow_up_analysis(snd, moisture_ratio=15.0)
        if blow_up:
            dt_blowups.append(blow_up.dt_lmib_blow_up if blow_up.dt_lmib_blow_up else np.nan)
            dz_blowups.append(blow_up.dz_lmib_blow_up if blow_up.dz_lmib_blow_up else np.nan)
        else:
            dt_blowups.append(np.nan)
            dz_blowups.append(np.nan)

    # 3 plot figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,14), sharex=True)
    fig.suptitle(f'PyroCb Potential Analysis - {fire_name}',
                    fontsize = 14, fontweight = 'bold')


    # PFT plots
    sorted_pairs = sorted(zip(times, pfts))
    sorted_times, sorted_pfts = map(list, zip(*sorted_pairs))
    ax1.plot(sorted_times, sorted_pfts, 'o-', color='red', linewidth=2, markersize=6)
    ax1.axhline(y=300, color='orange', linestyle='--', linewidth=2, 
                label='Moderate Risk (100-300 GW)')
    ax1.axhline(y=100, color='darkred', linestyle='--', linewidth=2, 
                label='Very High Risk (<100 GW)')
    
    ax1.set_ylabel('PFT (GW)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Pyrocumulonimbus Firepower Threshold', 
                fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # blow-up temps
    sorted_pairs = sorted(zip(times, dt_blowups))
    sorted_times, sorted_dt_bu = map(list, zip(*sorted_pairs))
    ax2.plot(sorted_times, sorted_dt_bu, 's-', color='darkgreen', linewidth=2, markersize=6)
    ax2.set_ylabel(r'$\Delta$T ($^\circ$C)', fontsize=12, fontweight='bold')
    ax2.set_title('Heating Required for Blow-Up', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # blow up height
    sorted_pairs = sorted(zip(times, dz_blowups))
    sorted_times, sorted_dz_bu = map(list, zip(*sorted_pairs))
    ax3.plot(sorted_times, sorted_dz_bu, '^-', color='purple', linewidth=2, markersize=6)
    ax3.set_xlabel('(UTC)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Δz (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Blow-Up Size', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Formatting Loop (Apply to all axes at once)
    date_fmt = mdates.DateFormatter('%m/%d %HZ')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(date_fmt)
    # save the plots with their own names from the file number
    # numbers are the time and date of the sounding
    output_filename = os.path.join(output_folder, f"{fire_name.replace(' ','_').lower()}_pft.png")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()  
    
    print(f"Plot saved in '{output_filename}'")

def sounding_level_worker(iter_num, dtime, fire_loc, lead_time):
    try:
        h = Herbie(date = dtime, 
                model = "ifs", 
                product = 'oper', 
                fxx = 0,
                overwrite = True,
                save_dir = os.path.join(CACHE_BASE_DIR, f'ecmwf_sounding_{iter_num}_{lead_time}'))
        search_str = ":(t|u|v|w|z|gh|r|eqpt|cc|li|kx|tcwv|cape|cin|skt|vsw|stl2|sf|cp|ro|q|ptype|e|cbh|vis|lcc|mcc|hcc|pt|2t|2d|sot|10v|10u|mucape|msl|tp):"
        dses = h.xarray(search_str)
    except:
        try:
            shutil.rmtree(os.path.join(CACHE_BASE_DIR,f'ecmwf_sounding_{iter_num}_{lead_time}'))
        except:
            return None
        del h
        return None
    lat_nearest = dses[0].latitude.values[np.argmin(np.abs(dses[0].latitude.values - fire_loc.y))]
    lon_nearest = dses[0].longitude.values[np.argmin(np.abs(dses[0].longitude.values - fire_loc.x))]
    ds_subs = [ds.sel(latitude=lat_nearest).sel(longitude=lon_nearest) for ds in dses]
    ds_subs = [ds.drop_attrs() for ds in ds_subs]
    ds = xr.merge(ds_subs, compat = 'override')
    pressure = ds.isobaricInhPa.values * units.hPa
    temp = ds.t.values * units.K
    dwpt = mpcalc.dewpoint_from_relative_humidity(temp, ds.r.values * units.percent)
    heights = mpcalc.geopotential_to_height(units.Quantity(ds.gh.values * -g, 'm^2/s^2'))
    sfc_height = mpcalc.geopotential_to_height(units.Quantity(ds.z.values, 'm^2/s^2'))
    us = ds.u.values * units('m/s')
    vs = ds.v.values * units('m/s')

    ip_bool = float(np.squeeze(ds.ptype.values) == 8)
    snow_bool = float(np.squeeze(ds.ptype.values) == 5)
    rain_bool = float(np.squeeze(ds.ptype.values) == 1)
    frz_rain_bool = float(np.squeeze(ds.ptype.values) == 3)
    storm_motion = mpcalc.bunkers_storm_motion(pressure, us, vs, heights)        
    
    # Surface qualities
    sfc = Surface((90095),       # 6 digit station number
        tuple([dtime.strftime('%y%m%d/%H%M')]),                    # valid time
        tuple([ds.msl.values]),            # mean sea level pressure hPa
        tuple([pressure[0]]),              # station pressure hPa
        tuple([ds.skt.values]),            # skin temperature C
        tuple([ds.sot.sel(soilLayer=1).values]),         # layer 1 soil temperature(K) TODO: not in initial output (1)
        tuple([ds.sot.sel(soilLayer=2).values]),         # layer 2 soil temperature (k) TODO: not in initial output (2)
        tuple([0.0]),                      # 1-hour snowfall (kg/m^2) TODO: not in initial output
        tuple([0.0]),                      # percent soil moisture availability TODO: make this real
        tuple([ds.tp.values]),             # 1-hour total precipitation (mm)
        tuple([ds.tp.values*random()]),    # 1 hour convective precipitation (mm) TODO: make this real
        tuple([random()*0.5]),             # low cloud coverage (%) TODO: no data in herbie form
        tuple([random()*0.5]),             # mid cloud coverage (%) TODO: no data in herbie form
        tuple([random()*0.5]),             # high cloud coverage (%) TODO: no data in herbie form
        tuple([0.0]),                      # snow ratio from explicit cloud scheme (%) TODO: make real
        tuple([ds.u10.values]),            # 10 meter u-wind (m/s)
        tuple([ds.v10.values]),            # 10 meter v-wind (m/s)
        tuple([ds.ro.values]),             # 1-hour accumulated runoff (mm) TODO: make this real          
        tuple([0.0]),                      # 1-hour accumulated baseflow ground water runoff (mm) TODO: make this real
        tuple([ds.t2m.values]),            # 2-meter temperature (C)
        tuple([ds.q.values[0]]),           # 2-meter specific humidity
        tuple([snow_bool]),                # True/False
        tuple([frz_rain_bool]),            # True/False
        tuple([ip_bool]),                  # True/False
        tuple([rain_bool]),                # True/False
        tuple([np.array(storm_motion[0][0])]),       # U-component storm motion (m/s)
        tuple([np.array(storm_motion[0][1])]),       # V-component storm motion (m/s)
        tuple([np.array(mpcalc.storm_relative_helicity(heights, us, vs, 
                                        depth = 1 * units.km, 
                                        storm_u = storm_motion[0][0], 
                                        storm_v = storm_motion[0][1])[2])]),      # storm relative helicity (m^2/s^2)
        tuple([random()]),           # 1-hour surface evaporation TODO: make real
        tuple([800]),                      # Cloud base pressure (hPa) TODO: make real
        tuple([15*random()]),              # Visibility (km) TODO: make real
        tuple([ds.d2m.values])          # 2-meter dew point (C)
    )
    # calc lcl and showalter index
    p_850 = 850 * units.hPa
    t_850 = ds.sel(isobaricInhPa=850).t.values * units.K
    td_850 = ds.sel(isobaricInhPa=850).t.values * units.K

    p_500 = 500 * units.hPa
    t_500_env = ds.sel(isobaricInhPa=500).t.values * units.K

    p_lcl, t_lcl = mpcalc.lcl(p_850, t_850, td_850)

    parcel_temps = mpcalc.moist_lapse(p_500, t_lcl, p_lcl)
    showalter_index = t_500_env - parcel_temps
    wind_dir = mpcalc.wind_direction(us,vs)
    wind_speed = mpcalc.wind_speed(us,vs)
    # calc sweat index
    sweat_idx = mpcalc.sweat_index(pressure,
                                    temp,
                                    dwpt,
                                    wind_speed,
                                    wind_dir)

    # calc bulk Richardson Number
    prof = mpcalc.parcel_profile(pressure, temp[0].to('degC'), dwpt[0].to('degC')).to('degC')

    cape, cin = mpcalc.cape_cin(pressure, temp, dwpt, prof)
    u_shr_6k, v_shr_6k = mpcalc.bulk_shear(pressure, us, vs, 
                                            height = heights, 
                                            depth = 6*units.km)
    u_shr_500, v_shr_500 = mpcalc.bulk_shear(pressure, us, vs, 
                                                height=heights, 
                                                depth=0.5*units.km)
    
    du = u_shr_6k - u_shr_500
    dv = v_shr_6k - v_shr_500
    shear_magnitude = mpcalc.wind_speed(du, dv).to('m/s').magnitude

    if shear_magnitude > 0.1:
        brn = cape.magnitude / (0.5 * shear_magnitude**2)
    else:
        brn = 0
    profile = Profile( # Location Data
        tuple(["INSPYR"]),                                         # station id, e.g. KMSO
        tuple([90095]),                                           # station number, usually USAF id number
        tuple([lat_nearest]),                                      # Latitude
        tuple([lon_nearest]),                                      # Longitude
        tuple([np.array(sfc_height)]),                                             # Station elevation in meters TODO fill in dummy

        # Time Data
        tuple([str(dtime)]),                                            # valid time of the sounding
        tuple([lead_time]),                                                # lead time in hours from model initialization

        # Indexes
        tuple([np.array(showalter_index)]),                                  # Showalter index
        tuple([np.array(mpcalc.lifted_index(pressure, temp, prof))]),        # Lifted Index
        tuple([np.array(sweat_idx)]),                                        # SWEAT Index
        tuple([np.array(mpcalc.k_index(pressure,
                        temp.to('degC'),
                        dwpt))]),                             # K index
        tuple([np.array(p_lcl)]),                                            # LCL in mb
        tuple([ds.tcwv.values]),                                   # precipitable water in inches
        tuple([np.array(mpcalc.total_totals_index(pressure,
                                    temp,
                                    dwpt))]),   # Total Totals index
        tuple([ds.mucape.values]),                                 # CAPE J/Kg
        tuple([np.array(t_lcl)]),                                            # Potential Temperature at LCL, kelvin
        tuple([np.array(cin)]),                                              # CIN J/Kg
        tuple([np.array(mpcalc.el( pressure,
                    temp.to('degC'),
                    dwpt)[0])]),                         # Equilibrium level, mb
        tuple([np.array(mpcalc.lfc(pressure,
                        temp,
                        dwpt)[0])]),                                         # Level of free convection, mb
        tuple([brn]),                                                      # Bulk Richardson Number

        # Profiles
        tuple(np.array(pressure)),                                         # Pressure in mb
        tuple(np.array(temp.to('degC'))),                                  # Temp in deg C
        tuple(np.array(mpcalc.wet_bulb_temperature( pressure, 
                                        temp, 
                                        dwpt).to('degC'))),                  # Wet bulb temp in deg C
        tuple(np.array(dwpt.to('degC'))),                                             # Dew point in deg C 
        tuple(np.array(mpcalc.potential_temperature(pressure, temp))),     # equivalent potential temp in K
        tuple(np.array(wind_dir)),                                         # wind direction
        tuple(np.array(wind_speed.to('knots'))),                           # wind speed in knots
        tuple(np.array(us)),                                               # west to east wind in m/s
        tuple(np.array(vs)),                                               # south to north wind in m/s
        tuple(mpcalc.vertical_velocity_pressure(ds.w.values * units('m/s'), 
                                                pressure, temp)),          # vertical velocity in Pa/sec
        tuple(np.random.randn(len(us)) * 1/len(us)),                       # Cloud fraction in percent TODO: make real
        tuple(np.array(heights)),                                          # Height in meters
    )
    try:
        shutil.rmtree(os.path.join(CACHE_BASE_DIR,f'ecmwf_sounding_{iter_num}_{lead_time}'))
    except:
        pass
    del h
    del ds
    return Sounding(profile, sfc)

def parallel_sounding_levels(fire_loc, fire_time, fire_num):
    # results = []
    # for i, dtime in enumerate(pd.date_range(fire_time[0], fire_time[1], freq = '6H')):
    #     results.append(sounding_level_worker(fire_num, dtime, fire_loc, i))
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=10, mp_context=ctx) as executor:
        results = []
        futures = []
        
        for i, dtime in enumerate(pd.date_range(fire_time[0], fire_time[1], freq = '6H')):
            futures.append(executor.submit(sounding_level_worker, fire_num, dtime, fire_loc, i))
        
        # Collect results as they complete
        for future in as_completed(futures):
            results.append(future.result())
    soundings = [res for res in results if type(res) is Sounding]
    return soundings

def ecmwf_querying(fire_loc, dtime_min, dtime_max, iter_num):
    soundings = []
    for i, dtime in enumerate(pd.date_range(dtime_min, dtime_max, freq = '6H')):
        try:
            h = Herbie(date = dtime, 
                    model = "ifs", 
                    product = 'oper', 
                    fxx = 0,
                    overwrite = True,
                    save_dir = os.path.join(CACHE_BASE_DIR, f'ecmwf_sounding_{iter_num}'))
            search_str = ":(t|u|v|w|z|gh|r|eqpt|cc|li|kx|tcwv|cape|cin|skt|vsw|stl2|sf|cp|ro|q|ptype|e|cbh|vis|lcc|mcc|hcc|pt|2t|2d|sot|10v|10u|mucape|msl|tp):"
            dses = h.xarray(search_str)
        except:
            shutil.rmtree(os.path.join(CACHE_BASE_DIR,f'ecmwf_sounding_{iter_num}', 'ifs'))
            del h
        lat_nearest = dses[0].latitude.values[np.argmin(np.abs(dses[0].latitude.values - fire_loc.y))]
        lon_nearest = dses[0].longitude.values[np.argmin(np.abs(dses[0].longitude.values - fire_loc.x))]
        ds_subs = [ds.sel(latitude=lat_nearest).sel(longitude=lon_nearest) for ds in dses]
        ds_subs = [ds.drop_attrs() for ds in ds_subs]
        ds = xr.merge(ds_subs, compat = 'override')
        pressure = ds.isobaricInhPa.values * units.hPa
        temp = ds.t.values * units.K
        dwpt = mpcalc.dewpoint_from_relative_humidity(temp, ds.r.values * units.percent)
        heights = mpcalc.geopotential_to_height(units.Quantity(ds.gh.values * -g, 'm^2/s^2'))
        sfc_height = mpcalc.geopotential_to_height(units.Quantity(ds.z.values, 'm^2/s^2'))
        us = ds.u.values * units('m/s')
        vs = ds.v.values * units('m/s')

        ip_bool = float(np.squeeze(ds.ptype.values) == 8)
        snow_bool = float(np.squeeze(ds.ptype.values) == 5)
        rain_bool = float(np.squeeze(ds.ptype.values) == 1)
        frz_rain_bool = float(np.squeeze(ds.ptype.values) == 3)
        storm_motion = mpcalc.bunkers_storm_motion(pressure, us, vs, heights)        
        
        # Surface qualities
        sfc = Surface((90095),       # 6 digit station number
            tuple([dtime.strftime('%y%m%d/%H%M')]),                    # valid time
            tuple([ds.msl.values]),            # mean sea level pressure hPa
            tuple([pressure[0]]),              # station pressure hPa
            tuple([ds.skt.values]),            # skin temperature C
            tuple([ds.sot.sel(soilLayer=1).values]),         # layer 1 soil temperature(K) TODO: not in initial output (1)
            tuple([ds.sot.sel(soilLayer=2).values]),         # layer 2 soil temperature (k) TODO: not in initial output (2)
            tuple([0.0]),                      # 1-hour snowfall (kg/m^2) TODO: not in initial output
            tuple([0.0]),                      # percent soil moisture availability TODO: make this real
            tuple([ds.tp.values]),             # 1-hour total precipitation (mm)
            tuple([ds.tp.values*random()]),    # 1 hour convective precipitation (mm) TODO: make this real
            tuple([random()*0.5]),             # low cloud coverage (%) TODO: no data in herbie form
            tuple([random()*0.5]),             # mid cloud coverage (%) TODO: no data in herbie form
            tuple([random()*0.5]),             # high cloud coverage (%) TODO: no data in herbie form
            tuple([0.0]),                      # snow ratio from explicit cloud scheme (%) TODO: make real
            tuple([ds.u10.values]),            # 10 meter u-wind (m/s)
            tuple([ds.v10.values]),            # 10 meter v-wind (m/s)
            tuple([ds.ro.values]),             # 1-hour accumulated runoff (mm) TODO: make this real          
            tuple([0.0]),                      # 1-hour accumulated baseflow ground water runoff (mm) TODO: make this real
            tuple([ds.t2m.values]),            # 2-meter temperature (C)
            tuple([ds.q.values[0]]),           # 2-meter specific humidity
            tuple([snow_bool]),                # True/False
            tuple([frz_rain_bool]),            # True/False
            tuple([ip_bool]),                  # True/False
            tuple([rain_bool]),                # True/False
            tuple([np.array(storm_motion[0][0])]),       # U-component storm motion (m/s)
            tuple([np.array(storm_motion[0][1])]),       # V-component storm motion (m/s)
            tuple([np.array(mpcalc.storm_relative_helicity(heights, us, vs, 
                                            depth = 1 * units.km, 
                                            storm_u = storm_motion[0][0], 
                                            storm_v = storm_motion[0][1])[2])]),      # storm relative helicity (m^2/s^2)
            tuple([random()]),           # 1-hour surface evaporation TODO: make real
            tuple([800]),                      # Cloud base pressure (hPa) TODO: make real
            tuple([15*random()]),              # Visibility (km) TODO: make real
            tuple([ds.d2m.values])          # 2-meter dew point (C)
        )
        # calc lcl and showalter index
        p_850 = 850 * units.hPa
        t_850 = ds.sel(isobaricInhPa=850).t.values * units.K
        td_850 = ds.sel(isobaricInhPa=850).t.values * units.K

        p_500 = 500 * units.hPa
        t_500_env = ds.sel(isobaricInhPa=500).t.values * units.K

        p_lcl, t_lcl = mpcalc.lcl(p_850, t_850, td_850)

        parcel_temps = mpcalc.moist_lapse(p_500, t_lcl, p_lcl)
        showalter_index = t_500_env - parcel_temps
        wind_dir = mpcalc.wind_direction(us,vs)
        wind_speed = mpcalc.wind_speed(us,vs)
        # calc sweat index
        sweat_idx = mpcalc.sweat_index(pressure,
                                       temp,
                                       dwpt,
                                       wind_speed,
                                       wind_dir)

        # calc bulk Richardson Number
        prof = mpcalc.parcel_profile(pressure, temp[0].to('degC'), dwpt[0].to('degC')).to('degC')

        cape, cin = mpcalc.cape_cin(pressure, temp, dwpt, prof)
        u_shr_6k, v_shr_6k = mpcalc.bulk_shear(pressure, us, vs, 
                                               height = heights, 
                                               depth = 6*units.km)
        u_shr_500, v_shr_500 = mpcalc.bulk_shear(pressure, us, vs, 
                                                 height=heights, 
                                                 depth=0.5*units.km)
        
        du = u_shr_6k - u_shr_500
        dv = v_shr_6k - v_shr_500
        shear_magnitude = mpcalc.wind_speed(du, dv).to('m/s').magnitude

        if shear_magnitude > 0.1:
            brn = cape.magnitude / (0.5 * shear_magnitude**2)
        else:
            brn = 0
        profile = Profile( # Location Data
            tuple(["INSPYR"]),                                         # station id, e.g. KMSO
            tuple([90095]),                                           # station number, usually USAF id number
            tuple([lat_nearest]),                                      # Latitude
            tuple([lon_nearest]),                                      # Longitude
            tuple([np.array(sfc_height)]),                                             # Station elevation in meters TODO fill in dummy

            # Time Data
            tuple([str(dtime)]),                                            # valid time of the sounding
            tuple([i]),                                                # lead time in hours from model initialization

            # Indexes
            tuple([np.array(showalter_index)]),                                  # Showalter index
            tuple([np.array(mpcalc.lifted_index(pressure, temp, prof))]),        # Lifted Index
            tuple([np.array(sweat_idx)]),                                        # SWEAT Index
            tuple([np.array(mpcalc.k_index(pressure,
                            temp.to('degC'),
                            dwpt))]),                             # K index
            tuple([np.array(p_lcl)]),                                            # LCL in mb
            tuple([ds.tcwv.values]),                                   # precipitable water in inches
            tuple([np.array(mpcalc.total_totals_index(pressure,
                                        temp,
                                        dwpt))]),   # Total Totals index
            tuple([ds.mucape.values]),                                 # CAPE J/Kg
            tuple([np.array(t_lcl)]),                                            # Potential Temperature at LCL, kelvin
            tuple([np.array(cin)]),                                              # CIN J/Kg
            tuple([np.array(mpcalc.el( pressure,
                        temp.to('degC'),
                        dwpt)[0])]),                         # Equilibrium level, mb
            tuple([np.array(mpcalc.lfc(pressure,
                          temp,
                          dwpt)[0])]),                                         # Level of free convection, mb
            tuple([brn]),                                                      # Bulk Richardson Number

            # Profiles
            tuple(np.array(pressure)),                                         # Pressure in mb
            tuple(np.array(temp.to('degC'))),                                  # Temp in deg C
            tuple(np.array(mpcalc.wet_bulb_temperature( pressure, 
                                          temp, 
                                          dwpt).to('degC'))),                  # Wet bulb temp in deg C TODO CONVERT
            tuple(np.array(dwpt.to('degC'))),                                             # Dew point in deg C 
            tuple(np.array(mpcalc.potential_temperature(pressure, temp))),     # equivalent potential temp in K
            tuple(np.array(wind_dir)),                                         # wind direction
            tuple(np.array(wind_speed.to('knots'))),                           # wind speed in knots
            tuple(np.array(us)),                                               # west to east wind in m/s
            tuple(np.array(vs)),                                               # south to north wind in m/s
            tuple(mpcalc.vertical_velocity_pressure(ds.w.values * units('m/s'), 
                                                    pressure, temp)),          # vertical velocity in Pa/sec
            tuple(np.random.randn(len(us)) * 1/len(us)),                       # Cloud fraction in percent TODO: make real
            tuple(np.array(heights)),                                          # Height in meters
        )
        soundings.append(Sounding(profile, sfc))
        shutil.rmtree(os.path.join(CACHE_BASE_DIR,f'ecmwf_sounding_{iter_num}', 'ifs'))
        del h
        del ds
    return soundings

def sounding_worker(fire_loc, fire_span, fire_name, iter_num):
    # sounding_data = ecmwf_querying(fire_loc, fire_span[0], fire_span[1], iter_num)
    sounding_data = parallel_sounding_levels(fire_loc, fire_span, iter_num)
    return sounding_data
    plot_sounding(sounding_data, fire_name=fire_name)
    
if __name__ == '__main__':
    load_sounding = True
    fire_locs = [Point(-102.0075, 56.210556), Point(-112.035278, 36.310001), Point(-111.9815255, 38.6098615)]
    fire_times =  [('2025-07-16','2025-08-11'), ('2025-07-18', '2025-08-13'), ('2025-07-18','2025-08-13')]
    fire_names = ["Lynn Lake", "Dragon Bravo", "Monroe Canyon"]
    
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as executor:
        results = []
        futures = []
        for i in range(3):
            if load_sounding:
                results.append(load(f'soundings_{fire_names[i].replace(" ","_")}.joblib'))
                continue
            futures.append(executor.submit(sounding_worker, fire_locs[i], fire_times[i], fire_names[i], i))
            
        # Collect results as they complete
        if not load_sounding:
            for future in as_completed(futures):
                results.append(future.result())

            [dump(res, f'soundings_{fire_names[i].replace(" ","_")}.joblib') for i, res in enumerate(results)]

    for snds, fire_name in zip(results, fire_names):
        plot_sounding(snds, fire_name=fire_name)
    print('all done!')
