from data.clients.rrfs_client import RRFSClient
from data.clients.nam_client import NAMClient
from utils.constants import PLOTS_DIR, CACHE_BASE_DIR

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from metpy.calc import lcl, dewpoint_from_relative_humidity, geopotential_to_height
from metpy.units import units
import numpy as np
import os
import pandas as pd
from pyrometeopy.fire_plumes import blow_up_analysis, pft
from pyrometeopy.bufkit import Profile, Sounding, Surface

import xarray as xr

def plot_sounding(sounding_data, output_folder = os.path.join(PLOTS_DIR, 'sounding_pfts'), 
                  fire_names = ["Canyon Fire"], plot_name = "Forecast Comparison"):

    os.makedirs(output_folder, exist_ok = True)
    fire_times = []
    fire_pfts = []
    fire_dt_blowups = []
    fire_dz_blowups = []
    for soundings in sounding_data:
        times = []
        pfts = []
        dt_blowups = []
        dz_blowups = []
        print(f"Calculating PFT values for {len(soundings)} soundings...")
        for i, snd in enumerate(soundings):
            times.append(pd.to_datetime(snd.profile.time[0], format='%y%m%d/%H%M'))
            pft_val = pft(snd, moisture_ratio=15.0, fire_elevation=0)
            pfts.append(np.float64(pft_val) if pft_val else 650)

            # where data is being read in
            blow_up = blow_up_analysis(snd, moisture_ratio=15.0, elevation=0)
            if blow_up:
                dt_blowups.append(blow_up.dt_lmib_blow_up if blow_up.dt_lmib_blow_up else np.nan)
                dz_blowups.append(blow_up.dz_lmib_blow_up if blow_up.dz_lmib_blow_up else np.nan)
            else:
                dt_blowups.append(np.nan)
                dz_blowups.append(np.nan)
        fire_times.append(times)
        fire_pfts.append(pfts)
        fire_dt_blowups.append(dt_blowups)
        fire_dz_blowups.append(dz_blowups)

    # 3 plot figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,14), sharex=True)
    fig.suptitle(f'PyroCb Potential Analysis - {plot_name}',
                    fontsize = 14, fontweight = 'bold')


    # PFT plots
    for i, fire_name in enumerate(fire_names):
        times = fire_times[i][:-2]
        pfts = fire_pfts[i][:-2]
        sorted_pairs = sorted(zip(times, pfts))
        sorted_times, sorted_pfts = map(list, zip(*sorted_pairs))
        ax1.plot(sorted_times, sorted_pfts, 'o-', linewidth=2, markersize=6, label=fire_name)
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
    for i, fire_name in enumerate(fire_names):
        times = fire_times[i][:-2]
        dt_blowups = fire_dt_blowups[i][:-2]
        sorted_pairs = sorted(zip(times, dt_blowups))
        sorted_times, sorted_dt_bu = map(list, zip(*sorted_pairs))
        ax2.plot(sorted_times, sorted_dt_bu, 's-', linewidth=2, markersize=6, label=fire_name)
    ax2.set_ylabel(r'$\Delta$T ($^\circ$C)', fontsize=12, fontweight='bold')
    ax2.set_title('Heating Required for Blow-Up', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # blow up height
    for i, fire_name in enumerate(fire_names):
        times = fire_times[i][:-2]
        dt_blowups = fire_dz_blowups[i][:-2]
        sorted_pairs = sorted(zip(times, dz_blowups))
        sorted_times, sorted_dz_bu = map(list, zip(*sorted_pairs))
        ax3.plot(sorted_times, sorted_dz_bu, '^-', linewidth=2, markersize=6, label=fire_name)
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
        ax.legend(loc=0)
    # save the plots with their own names from the file number
    # numbers are the time and date of the sounding
    output_filename = os.path.join(output_folder, f"{plot_name.replace(' ','_').lower()}_pft.png")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()  
    
    print(f"Plot saved in '{output_filename}'")

def calc_soundings(ds):
    unused_vars_prf =  ['stid', 'stnm', 'lat', 'lon', 
                    'elevation', 'leadTime', 'show', 
                    'li', 'swet', 'kinx', 'pwat', 
                    'totl', 'cape', 'lclt', 'cin',  
                    'wbt', 'thetaE', 'windDir', 
                    'eql', 'lfc', 'brch',
                    'windSpd', 'omega', 'cloud']
    unused_vars_sfc = [
            'station', 'time', 'pmsl', 'pres','skin_temp',
            'soil_temp1', 'soil_temp2', 'snow', 'soil_moist',
            'precip', 'conv_precip', 'lcld', 'mcld', 'hcld',
            'snow_ratio', 'uWind', 'vWind', 'runoff',
            'baseflow', 'q_2', 'snow_pres', 'fzra_pres', 
            'ip_pres', 'rain_pres', 'u_storm', 'v_storm',
            'helicity', 'evap', 'cloud_base_p', 'visibility', 
            ]
    unused_kwargs_prf = {varname : tuple([None]) for varname in unused_vars_prf}
    unused_kwargs_sfc = {varname : None for varname in unused_vars_sfc}
    soundings = []
    for time in ds.valid_time.values:
        this_ds = ds.sel(valid_time=time)
        dpt = np.array(dewpoint_from_relative_humidity( this_ds.t.values * units.kelvin,
                                 this_ds.r.values / 100.0).to(units.degC))
        
        lcl_vals = np.array(lcl(this_ds.isobaricInhPa.values * units.hPa, 
                                 this_ds.t.values * units.K,
                                 dpt * units.degC)[0])
        t_vals = np.array((this_ds.t.values * units.K).to(units.degC))
        profile = Profile(time = tuple([time]),
                        lcl = tuple(lcl_vals),
                        pressure = tuple(this_ds.isobaricInhPa.values),
                        temp = tuple(t_vals),
                        dewpoint = tuple(dpt),
                        uWind = tuple(this_ds.u.values),
                        vWind = tuple(this_ds.v.values),
                        hgt = tuple(this_ds.gh.values),
                        **unused_kwargs_prf)
        sfc = Surface(  temp = t_vals[0],
                        dewpoint = dpt[0],
                        **unused_kwargs_sfc)
        snd = Sounding(profile, sfc)
        soundings.append(snd)
    return soundings

if __name__ == "__main__":
    client = NAMClient()
    forecast_path = f"{CACHE_BASE_DIR}/ecmwf/05-16-26-fx48.nc"
    this_time = "2026-05-11 00:00"
    fire_name = "ECMWF 5_16_26 fx 48"
    lat = 34.05
    lon = -118.24
    fxx = 2
    forecast_names = ["ECMWF", "GFS", "NAM"]
    forecast_paths = [f"{CACHE_BASE_DIR}/{d_source.lower()}/05-16-26-fx48.nc" for d_source in forecast_names]
    dses = []
    for forecast_path in forecast_paths:
        if os.path.exists(forecast_path):
            ds = xr.load_dataset(forecast_path)
        else:
            ds = client.query(this_time, lat, lon, fxx)
        dses.append(ds)

    soundings = [calc_soundings(ds) for ds in dses]
    plot_sounding(soundings, fire_names = forecast_names)

