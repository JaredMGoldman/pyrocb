from data.clients.rrfs_client import RRFSClient
from data.clients.nam_client import NAMClient
from data.clients.gfs_client import GFSClient
from data.clients.gfs_historical_client import GFSHistClient
from data.clients.rrfs_client import RRFSClient
from data.clients.ecmwf_client import ECMWFClient
from utils.constants import PLOTS_DIR, NAM, GFS, ECMWF, RRFS, CACHE_BASE_DIR

from concurrent.futures import ProcessPoolExecutor, as_completed
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.path import shapely_to_path
from datetime import datetime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from metpy.calc import lcl, dewpoint_from_relative_humidity
from metpy.units import units
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pickle import dumps, loads
from pyrometeopy.fire_plumes import pft
from pyrometeopy.bufkit import Profile, Sounding, Surface
from scipy.interpolate import griddata
from shapely.ops import unary_union
import tqdm
import warnings
import xarray as xr

warnings.filterwarnings("ignore")

# def transform_ds_for_parallel(
#     ds_paths, features=["t", "r", "gh", "u", "v"], target_dim="isobaricInhPa"
# ):
#     """Memory-safe transformer that streams large weather datasets into sounding dictionaries

#     without blowing up RAM limits.
#     """
#     # 1. Open lazily using Dask chunking over spatial dimensions
#     # This prevents the dataset from loading into RAM instantly
#     print("Opening files with lazy spatial chunking...")
#     ds = xr.open_mfdataset(
#         ds_paths,
#         dim="valid_time",
#         combine="nested",
#         parallel=True,
#         chunks={
#             "latitude": 5,
#             "longitude": 5,
#         },  # Keeps memory footprints per chunk tiny
#         coords="minimal",
#         compat="override",
#     )

#     # Core structural dimensions fix
#     if "valid_time" in ds.coords and "valid_time" not in ds.dims:
#         ds = (
#             ds.swap_dims({"time": "valid_time"})
#             if "time" in ds.dims
#             else ds.set_index(valid_time="valid_time")
#         )

#     lat_name = "latitude" if "latitude" in ds.coords else "lat"
#     lon_name = "longitude" if "longitude" in ds.coords else "lon"

#     # 2. Pull metadata vectors only (lightweight primitives)
#     time_vector = ds.valid_time.values
#     level_vector = ds[target_dim].values
#     lat_vals = ds[lat_name].values
#     lon_vals = ds[lon_name].values

#     # 3. Handle coordinate matrices
#     if lat_vals.ndim == 1 and lon_vals.ndim == 1:
#         lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
#     else:
#         lat_2d, lon_2d = lat_vals, lon_vals

#     # 4. Generate the valid physical mask using a fast lazy evaluation
#     # .compute() is ONLY executed on a single 2D spatial slice to isolate active ocean/land cells
#     print("Locating valid sounding points...")
#     spatial_sample = (
#         ds[features[0]]
#         .isel(valid_time=0, **{target_dim: 0})
#         .drop_vars(["valid_time", target_dim], errors="ignore")
#         .compute()
#     )

#     y_indices, x_indices = np.where(~np.isnan(spatial_sample.values))
#     total_points = len(y_indices)
#     print(f"Identified {total_points} active spatial points to extract.")

#     parallel_ready_payload = {}

#     # 5. Stream the points out sequentially or in small blocks
#     # By isolating coordinate slices FIRST on the lazy xarray object,
#     # we only load the exact 1D/2D time-series vectors we need into memory.
#     for i, (y_idx, x_idx) in enumerate(zip(y_indices, x_indices)):
#         lat_val = float(lat_2d[y_idx, x_idx])
#         lon_val = float(lon_2d[y_idx, x_idx])
#         coord_key = (lat_val, lon_val)

#         point_features = {"time": time_vector, "levels": level_vector}

#         # Select the spatial pixel lazily across all times and levels
#         # Calling .compute() here ONLY pulls down the data for this singular point grid cell
#         point_ds = ds.isel(**{lat_name: y_idx, lon_name: x_idx}).compute()

#         for f in features:
#             point_features[f] = point_ds[f].values

#         parallel_ready_payload[coord_key] = point_features

#         if i % 500 == 0 and i > 0:
#             print(f"Processed sounding {i}/{total_points}...")

#     return parallel_ready_payload

def transform_ds_for_parallel(ds, features = ['t', 'r', 'gh', 'u', 'v'], 
                              target_dim='isobaricInhPa'):
    """
    Transforms a heavy, lazy xarray dataset into a lightweight, 
    fully-picklable dictionary of raw NumPy arrays optimized for parallel pipelines.
    """
    # 1. Determine spatial coordinate names dynamically
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    lat_coord = ds[lat_name]
    lon_coord = ds[lon_name]
    
    # 2. Extract shared non-spatial coordinate vectors cleanly as primitives
    time_vector = ds.valid_time.values
    level_vector = ds[target_dim].values

    # 3. Handle Transpose Order - explicitly tracking spatial coordinates
    main_keys = ['valid_time', target_dim]
    if not 'valid_time' in ds.dims:
        ds = ds.expand_dims('valid_time')
    spatial_dims = [d for d in ds.dims if not d in main_keys]
    required_order = main_keys + spatial_dims
    
    # Extract data arrays into raw, in-memory NumPy blocks 
    # Shape layout: (valid_time, isobaricInhPa, Y, X) or (valid_time, isobaricInhPa, lat, lon)
    feature_arrays = {}
    for f in features:
        feature_arrays[f] = ds[f].transpose(*required_order).values
    
    # 4. Handle Grid Coordinate Geometry Matrix Generation
    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        # Rectilinear: build explicit 2D coordinate arrays matching the spatial shape
        # np.meshgrid output needs to match data array's spatial axis layout
        lon_2d, lat_2d = np.meshgrid(lon_coord.values, lat_coord.values)
        if spatial_dims == [lat_name, lon_name]:
            lats_matrix, lons_matrix = lat_2d, lon_2d
        else:
            lats_matrix, lons_matrix = lat_2d.T, lon_2d.T
    else:
        # Curvilinear (e.g. RRFS / NAM Nest): coordinates are already 2D meshes
        lats_matrix = lat_coord.values
        lons_matrix = lon_coord.values

    # 5. Locate where the data contains valid physical values to prune out dead NaN margins
    # Uses the first time slice and first level of your first feature as a template mask
    spatial_sample = feature_arrays[features[0]][0, 0, :, :]
    y_indices, x_indices = np.where(~np.isnan(spatial_sample))
    
    parallel_ready_payload = {}

    # 6. Build the data payload dictionary
    for y_idx, x_idx in zip(y_indices, x_indices):
        lat_val = float(lats_matrix[y_idx, x_idx])
        lon_val = float(lons_matrix[y_idx, x_idx])
        coord_key = (lat_val, lon_val)
        
        # CRITICAL FIX: Instantiate a FRESH dictionary block for every single coordinate pair.
        # This prevents reference overwriting from destroying upstream data profiles.
        point_features = {
            'time': np.array([time_vector]),
            'levels': level_vector
        }
        
        # Extract the continuous temporal and vertical slice via fast NumPy index slicing
        # [:, :, y_idx, x_idx] pulls all times and levels for this specific point location
        for f in features:
            point_features[f] = feature_arrays[f][:, :, y_idx, x_idx]
            
        parallel_ready_payload[coord_key] = point_features

    return parallel_ready_payload


def parse_to_dataframe(raw_data):
    """Converts the nested tuple structure into a clean Pandas DataFrame."""
    records = []
    for (time, val), ((lat, lon), data_name) in raw_data:
        records.append({
            'time': pd.to_datetime(time),
            'lat': lat,
            'lon': lon,
            'data_name': data_name,
            'value': val
        })
    return pd.DataFrame(records)

def plot_spatiotemporal_data(df, output_dir="./plots", save_df = True, plot_data = False):
    """
    Groups data by data_name and timestamp, interpolates sparse/unstructured points
    onto a continuous dense grid, and renders them as a smooth meteorological mesh plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Format data into a DataFrame
    if save_df:
        df.to_csv(f"{output_dir}/pft_vals.csv", index = False)
    if plot_data:
        # 2. CRITICAL: Calculate Global Limits
        try:
            vmin = df['value'].min()
            vmax = df['value'].max()
        except:
            return None
        clean_values = df['value'][np.isfinite(df['value'])]
        vmin = clean_values.min()

        # Set the maximum color stretch to the 99th percentile of your entire dataset
        vmax = np.percentile(clean_values, 99)
        vmin = max(vmin, 1.0) 
        
        if vmin == vmax:
            vmax += 1.0 
            
        global_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        # global_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        print(f"Global Colorbar Limits Established: Min = {vmin:.2f}, Max = {vmax:.2f}")
        
        # 3. Group by the unique combinations of data source and timestamp
        grouped = df.groupby(['data_name', 'time'])

        resolution = '50m'
        shpfilename = shapereader.natural_earth(
            resolution=resolution, category='cultural', name='admin_0_countries'
        )
        reader = shapereader.Reader(shpfilename)
        countries = reader.records()
        
        # Extract and combine the geometric polygons for the US and Canada
        target_geoms = []
        for country in countries:
            name = country.attributes['NAME']
            if name in ['United States of America', 'Canada']:
                target_geoms.append(country.geometry)
                
        # Merge the US and Canada borders into a single, unified land mass polygon
        unified_land_mask = unary_union(target_geoms)
        
        for (data_name, timestamp), group in grouped:
            raw_lats = group['lat'].values
            raw_lons = group['lon'].values
            raw_values = group['value'].values
            
            # FILTER OUT NON-FINITE (NaN / Inf) VALUES (Fixes your previous triangulation issue)
            finite_mask = np.isfinite(raw_values)
            lats = raw_lats[finite_mask]
            lons = raw_lons[finite_mask]
            values = raw_values[finite_mask]
            
            if len(values) < 4:  # Griddata needs at least 4 points for bi-linear/cubic interpolation
                print(f"Skipping {data_name} at {timestamp}: Not enough valid spatial points to interpolate.")
                continue
                
            fig = plt.figure(figsize=(12, 10))
            # Moving the central latitude higher to account for Canada's massive northern footprint
            ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-98, central_latitude=50))
            
            # Set a wide spatial extent that captures all of the US (including Alaska) and Canada
            # Bounds format: [West Longitude, East Longitude, South Latitude, North Latitude]
            extent = [-170, -50, 24, 75]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Add background features (oceans will remain visible around the cropped land)
            # ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=1)
            # ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=1)
            
            # Add geographic layers
            # ─── METEOROLOGICAL CONTINUOUS INTERPOLATION STEP ───
            # Create a dense regular target grid over your map viewport bounds
            # 500x400 grid cells provides an ultra-smooth, continuous gradient definition
            grid_lons = np.linspace(extent[0], extent[1], 500)
            grid_lats = np.linspace(extent[2], extent[3], 400)
            grid_x, grid_y = np.meshgrid(grid_lons, grid_lats)
            
            # Interpolate the sparse scattered points onto our dense target map grid
            # 'linear' provides smooth meteorology; use 'cubic' for extra gradient softness
            grid_z = grid_data = griddata(
                points=(lons, lats), 
                values=values, 
                xi=(grid_x, grid_y), 
                method='linear'
            )

            ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
            ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
            
            # 4. Render the data as a smooth continuous pixel mesh grid
            # 'gouraud' applies bilinear blending between adjacent pixels for a perfect continuous feel
            mesh = ax.pcolormesh(grid_x, grid_y, grid_z, norm=global_norm,
                                transform=ccrs.PlateCarree(), cmap='autumn', 
                                shading='gouraud', zorder=2, alpha=0.85)
            
            # feature_patch = ShapelyFeature([unified_land_mask], ccrs.PlateCarree())
            # ax.add_feature(feature_patch, facecolor='none', edgecolor='none')

            # Project the geometry to the map's native LambertConformal projection paths
            projected_land_geom = ax.projection.project_geometry(unified_land_mask, ccrs.PlateCarree())
            
            # 3. Convert the projected Shapely geometry directly into a Matplotlib Path
            matplotlib_path = shapely_to_path(projected_land_geom)
            
            # 4. Wrap it in a standard PathPatch container
            clipping_patch = mpatches.PathPatch(matplotlib_path, facecolor='none', edgecolor='none')
            
            # Explicitly add it to the axes so it gets registered in the Matplotlib artist loop
            ax.add_patch(clipping_patch)
            
            # 3. Securely clip the weather mesh layer to this specific patch object
            mesh.set_clip_path(clipping_patch)
                
            # Adding borders with a higher zorder ensures geopolitical lines sit on top of the color fill
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='dimgray', zorder=4)
            ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='darkgray', zorder=4)
                
            # 5. Add the Standardized Colorbar
            cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
            cbar.set_label(f'PFT Value (Continuous Scale: {vmin:.1f} to {vmax:.1f})')
            
            time_str = timestamp.strftime('%Y-%m-%d %H:%M UTC')
            plt.title(f"{data_name} Forecast | Valid: {time_str}", pad=15, weight='bold')
            
            # 6. Save and Clear memory
            safe_time = timestamp.strftime('%Y%m%d_H%H%M')
            os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
            filename = f"{output_dir}/{data_name}/{data_name}_{safe_time}.png"
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig) 
            print(f"Saved: {filename}")

def sounding_worker(ds, key_val, t_idx, name):
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

    times = ds['time']
    levels = ds['levels']
    try:
        t = ds['t'][t_idx,:]
        r = ds['r'][t_idx,:]
        u = ds['u'][t_idx,:]
        v = ds['v'][t_idx,:]
        gh = ds['gh'][t_idx,:]
        if np.isnan(gh).all():
            return None
        dpt = np.array(dewpoint_from_relative_humidity( t * units.kelvin,
                                r / 100.0).to(units.degC))
        
        lcl_vals = np.array(lcl(levels * units.hPa, 
                                t * units.K,
                                dpt * units.degC)[0])
        t_vals = np.array((t * units.K).to(units.degC))
        profile = Profile(time = tuple([times[t_idx]]),
                        lcl = tuple(lcl_vals),
                        pressure = tuple(levels),
                        temp = tuple(t_vals),
                        dewpoint = tuple(dpt),
                        uWind = tuple(u),
                        vWind = tuple(v),
                        hgt = tuple(gh),
                        **unused_kwargs_prf)
        sfc = Surface(  temp = t_vals[0],
                        dewpoint = dpt[0],
                        **unused_kwargs_sfc)
        snd = Sounding(profile, sfc)
    except:
        return None
    return (snd, (key_val, times[t_idx], name))

def calc_soundings(ds_dict, max_workers):
    all_soundings = []
    with ProcessPoolExecutor(max_workers = max_workers, mp_context=mp.get_context('fork')) as ppex:
        futures = []
        tot_futures = 0
        all_args = []
        for name, ds in ds_dict.items():
            all_soundings = []
            keys = [key_val  for key_val in ds.keys()]
            for key_val in keys:
                for t_idx in range(len(ds[keys[0]]['time'])):
                    tot_futures += 1
                    spec_args = (ds[key_val], key_val, t_idx, name)
                    all_args.append(spec_args)

        futures.extend([ppex.submit(sounding_worker, *args) for args in all_args])
        for f in tqdm.tqdm(as_completed(futures), total = tot_futures, desc = "sounding formatting"):
            out = f.result()
            if out is not None:
                all_soundings.append(out)
    return all_soundings

def query_worker(client, date, lat, lon, fxx):
    return client().parallel_query(date = date, lat = lat, lon = lon, fxx = fxx)

def pull_data(date, lat, lon, fxx_range, clients, max_workers):
    dses = []
    with ProcessPoolExecutor(max_workers = max_workers) as ppex:
        futures = []
        specs = { 'client':  None,'date' : date, 'lat': lat, 
                  'lon' : lon, 'fxx': 0}
        tot_futures = 0
        for client in clients:
            for fxx in range(fxx_range+1):
                tot_futures += 1
                specs['fxx'] = fxx
                specs['client'] = client
                futures.append(ppex.submit(query_worker, specs['client'], specs['date'], specs['lat'], specs['lon'], specs['fxx']))
        for f in tqdm.tqdm(as_completed(futures), total = tot_futures, desc = "downloading data"):
            out = f.result()
            if out is not None:
                dses.append(out)
    return dses

def process_single_file_to_dict(fname, features, target_dim):
    """Opens a single file, extracts its arrays, and returns a raw dictionary."""
    with xr.open_dataset(fname) as ds:
        # Pass the single-file dataset into your existing transformation function
        return transform_ds_for_parallel(
            ds, features, target_dim
        )


def merge_parallel_payloads(payload_list, features):
    """Merges a list of single-file payloads, concatenating the time axis

    at the NumPy layer.
    """
    merged = {}

    for payload in payload_list:
        for coord_key, point_data in payload.items():
            if coord_key not in merged:
                # First time seeing this pixel location; seed it
                merged[coord_key] = point_data
            else:
                # Coordinate already exists; append along the time dimension (axis 0)
                merged[coord_key]["time"] = np.concatenate(
                    [merged[coord_key]["time"], point_data["time"]]
                )

                for f in features:
                    merged[coord_key][f] = np.concatenate(
                        [merged[coord_key][f], point_data[f]], axis=0
                    )

    # Optional: Sort each point's time axis if files aren't read in chronological order
    for coord_key in merged:
        sort_idx = np.argsort(merged[coord_key]["time"])
        merged[coord_key]["time"] = merged[coord_key]["time"][sort_idx]
        for f in features:
            merged[coord_key][f] = merged[coord_key][f][sort_idx, :, ...]

    return merged


def group_client_dses(
    dses,
    client_names,
    features=["t", "r", "gh", "u", "v"],
    target_dim="isobaricInhPa",
):
    out_dses = {name: [] for name in client_names}

    # dses is assumed to be a list of tuples: (file_path, client_name)
    for fname, name in dses:
        out_dses[name].append(fname)

    # Process each client
    for name in client_names:
        files = out_dses[name]
        print(f"Processing {len(files)} files in parallel for client: {name}")

        # Map files across worker processes to read NetCDF disks concurrently
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_single_file_to_dict, f, features, target_dim
                )
                for f in files
            ]
            single_payloads = [f.result() for f in futures]

        # Combine the independent pieces at the fast NumPy layer
        out_dses[name] = merge_parallel_payloads(single_payloads, features)

    return out_dses

# def group_client_dses(dses, client_names):
#     out_dses = {name : [] for name in client_names}
#     base_dir = os.path.join(CACHE_BASE_DIR, 'pft_processing')
#     os.makedirs(base_dir, exist_ok=True)
#     for ds, name in dses:
#         out_dses[name].append(ds)
#     for name in client_names:
#         dses = [xr.open_dataset(fname) for fname in out_dses[name]]
#         ds = xr.concat(dses, dim='valid_time')
#         out_dses[name] = transform_ds_for_parallel(ds)
#     return out_dses

def pft_worker(sounding):
    (snd, (key_val, time, name)) = loads(sounding)
    time = pd.to_datetime(snd.profile.time[0], format='%y%m%d/%H%M')
    pft_val = pft(snd, moisture_ratio=10.0, fire_elevation=0) # TODO: parameterize fire elevation
    return ((time, pft_val), (key_val, name))

def calc_pfts(soundings, max_workers):
    pfts = []
    with ProcessPoolExecutor(max_workers = max_workers) as ppex:
        futures = [ppex.submit(pft_worker, dumps(snd)) for snd in soundings]
        for f in tqdm.tqdm(as_completed(futures), total = len(soundings), desc = "pft calculation"):
            out = f.result()
            if out is not None:
                pfts.append(out)
    return pfts

if __name__ == "__main__":
    max_workers = 48
    fxx_range = 24
    clients = [GFSHistClient] #, GFSClient, NAMClient, ECMWFClient]
    this_time = "2026-05-31 00:00"
    all_times = []
    days = ["%02d" % (int(i),) for i in range(1,32)]
    for year in [f"{i}" for i in range(2015,2020)]:
        for month in ["06", "07", "08"]:
            for day in days:
                if day == "31" and month == "06":
                    continue
                this_time = f"{year}-{month}-{day}"
                all_times.append(this_time)
    lat = [24.846565, 71.300793]
    lon = [-166.992188, -52.031250]
    
    forecast_names = [GFS] #, GFS, NAM, ECMWF]
    for dtime in tqdm.tqdm(all_times, desc = "all times"):
        try:
            if os.path.exists(os.path.join(PLOTS_DIR, 'pfts', dtime, 'pft_vals.csv')):
                print(f"already processed: {os.path.join(PLOTS_DIR, 'pfts', dtime, 'pft_vals.csv')}")
                continue
            print(f"evaluating time: {dtime}")
            start_time = datetime.now()
            

            dses = pull_data(dtime, lat, lon, fxx_range, clients, max_workers)
            
            client_dses = group_client_dses(dses, forecast_names)
            data_time = datetime.now()
            data_stopwatch = data_time-start_time
            print(f"\nfinished pulling data in {data_stopwatch} secs")

            soundings = calc_soundings(client_dses, max_workers)
            
            sounding_time = datetime.now()
            sounding_stopwatch = sounding_time - data_time
            print(f"\nfinished formatting soundings in {sounding_stopwatch} secs")
            pfts = calc_pfts(soundings, max_workers)

            pft_time = datetime.now()
            pft_stopwatch = pft_time - sounding_time
            print(f"\nfinished pft calculatoin in {pft_stopwatch} secs")


            df = parse_to_dataframe(pfts)
            plot_spatiotemporal_data(df, os.path.join(PLOTS_DIR, 'pfts', dtime))
            plot_time = datetime.now()
            plot_stopwatch = plot_time - pft_time
            
            print("\nTIME SUMMARY")
            print("__________________________________")
            print(f"| DATA ACCESS   | {data_stopwatch} |")
            print(f"| SOUNDING FMT  | {sounding_stopwatch} |")
            print(f"| PFT TIME      | {pft_stopwatch} |")
            print(f"| PLOT TIME     | {plot_stopwatch} |")
            print(f"| TOTAL TIME    | {plot_time - start_time} |")
            print("----------------------------------")
        except:
            continue