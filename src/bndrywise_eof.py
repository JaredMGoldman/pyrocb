import numpy as np
import os
import pandas as pd
import geopandas as gpd
from utils.io_utils import FEATURE_OUTPUT_DIR, DATA_DIR, PLOTS_DIR
from eofs.standard import Eof
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils.feature_creation import all_features

def snap_to_bndry(gdf, var_name, boundary_index, date_col='day'):
    if boundary_index == 'name':
        path = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_1_states_provinces.zip"
        bdrys = gpd.read_file(path)
        boundaries = bdrys[bdrys['iso_a2'].isin(['US', 'CA'])][['name', 'geometry']]
    elif boundary_index == 'CLIMDIV':
        url = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/CONUS_CLIMATE_DIVISIONS.shp.zip"
        
        bdrys = gpd.read_file(url)
        boundaries = bdrys[['CLIMDIV', 'geometry']]
        can_url = "https://agriculture.canada.ca/atlas/data_donnees/nationalEcologicalFramework/data_donnees/geoJSON/ez/nef_ca_ter_ecozone_v2_2.geojson"
        can_zones = gpd.read_file(can_url).to_crs(boundaries.crs)
        can_zones[boundary_index] = can_zones.ECOZONE_ID.astype(str) + '_' + can_zones.OBJECTID.astype(str)

        boundaries = pd.concat([boundaries, can_zones], ignore_index=True)

    boundaries = boundaries.to_crs(gdf.crs)
    joined = gpd.sjoin(gdf, boundaries, how="inner", predicate="intersects")
    state_daily_stats = joined.groupby([date_col, boundary_index])[var_name].mean().reset_index()

    final_gdf = boundaries.merge(state_daily_stats, on=boundary_index, how='inner')
    return final_gdf

def choreoplath_lcc(gdf_eof, eof_name = 'eof1', var_name = 'mean_FRP', thresh_val = 1e-7):
    plot_crs = ccrs.LambertConformal(central_longitude=-100, central_latitude=45)
    data_crs = ccrs.PlateCarree() 

    gdf_eof_filtered = gdf_eof[np.abs(gdf_eof[eof_name]) > thresh_val].copy()

    plt.figure(figsize=(10, 8), dpi=300) # High DPI for journal submission
    ax = plt.axes(projection=plot_crs)

    # Set the extent for CONUS + Southern Canada
    ax.set_extent([-125, -70, 23, 65], crs=data_crs)

    # Add geographical context
    ax.add_feature(cfeature.LAND, facecolor='#f9f9f9')
    ax.add_feature(cfeature.OCEAN, facecolor='#e3eaf2')
    ax.add_feature(cfeature.LAKES, facecolor='#e3eaf2', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')

    # Plot the State-Averaged EOF
    # We use 'final_map.to_crs' to match the Cartopy projection
    gdf_eof_filtered.to_crs(plot_crs.proj4_init).plot(
        column=eof_name,
        ax=ax,
        cmap='PuOr', # Purple-Orange is colorblind-friendly and great for EOFs
        edgecolor='black',
        linewidth=0.4,
        legend=True,
        legend_kwds={'label': "Mean EOF Amplitude", 'orientation': "horizontal", 'shrink': 0.7, 'pad': 0.05},
        missing_kwds={"color": "white", "edgecolor": "lightgrey", "label": "No Data"}
    )

    # Add gridlines with labels (Critical for fire science spatial context)
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.title("Regional EOF Patterns of Fire Weather Drivers", fontsize=12, pad=20)
    os.makedirs(f"{PLOTS_DIR}/data_science/eofs/{var_name}/", exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/data_science/eofs/{var_name}/choreoplath_lcc_{eof_name}_{var_name}.png")
    plt.close()

def scree_plot(variance_frac, var_name, nmodes = 5):
    neigs = min(nmodes, len(variance_frac))
    _, ax = plt.subplots(figsize=(8, 5))
    
    variance_pct = variance_frac[:neigs] * 100
    eigs = np.arange(1, len(variance_pct) + 1)
    
    ax.scatter(eigs, variance_pct, color='k', alpha=0.7)
        
    ax.set_xlabel('EOF Eigenval Number')
    ax.set_ylabel('Variance Explained')
    ax.set_title(f'Scree Plot: {var_name} Variance')
    ax.set_xticks(eigs)
    
    os.makedirs(f"{PLOTS_DIR}/data_science/scree/{var_name}", exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/data_science/scree/{var_name}/{var_name}_scree_plot.png")

def pc_plot(pcs, time_index, var_name, nmodes = 5):
    modes = min(nmodes, pcs.shape[1])
    _, axes = plt.subplots(modes, 1, figsize=(12, modes * 3), sharex=True)    
    for i in range(modes):
        ax = axes[i]

        pc_series = pcs[:, i]
        standardized_pc = (pc_series - np.mean(pc_series)) / np.std(pc_series)
        
        ax.plot(time_index, standardized_pc, color='black', linewidth=1)
        
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)

        ax.set_ylabel(f'PC {i+1}')
        ax.set_title(f'Temporal Evolution of EOF Mode {i+1} for {var_name}')

    plt.xlabel('Date')

    os.makedirs(f"{PLOTS_DIR}/data_science/pcs/{var_name}", exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/data_science/pcs/{var_name}/{var_name}_pcs_plot.png")

if __name__ == "__main__":
    neofs = 5
    seed = 42
    boundary_index = 'CLIMDIV'
    film_only = False

    data_fname = "cleaned_data.csv"
    rng = np.random.default_rng(seed)

    data_path = os.path.join(FEATURE_OUTPUT_DIR, data_fname)
    df = pd.read_csv(data_path)
    gdf = gpd.read_file(os.path.join(DATA_DIR, "cp_poly.gpkg"))

    merged_gdf = gdf.join(df.set_index('cp'),on = 'cp',how = 'inner',rsuffix='gdf')
    merged_gdf['day'] = pd.to_datetime(merged_gdf['day'], format = 'mixed')

    for var_name in all_features:
        if var_name in ["rave_FRP_MEAN", "rave_FRP_SD", "modis_MaxFRP"]:
            this_col = merged_gdf[var_name]
            col_mean = this_col.mean(skipna = True)
            col_std = this_col.std(skipna = True)
            
            na_mask = this_col.isna()

            merged_gdf.loc[na_mask, var_name] = np.zeros(na_mask.sum())
        else:
            this_col = merged_gdf[var_name]
            col_mean = this_col.mean(skipna = True)
            col_std = this_col.std(skipna = True)
            
            na_mask = this_col.isna()

            rand_vals = rng.normal(loc = col_mean, scale = col_std, size = na_mask.sum())
            merged_gdf.loc[na_mask, var_name] = rand_vals
        
        print(f"\ngenerating {var_name} eofs")

        this_gdf = snap_to_bndry(merged_gdf, var_name, boundary_index)
        if film_only:
            from utils.feature_film import create_fire_timelapse
            os.makedirs(f"{PLOTS_DIR}/data_science/films/", exist_ok = True)
            create_fire_timelapse(this_gdf,var_name, f"{PLOTS_DIR}/data_science/films/{var_name}_timelapse.mp4")
            continue
        pivot_df = this_gdf.pivot(index='day', columns=boundary_index, values=var_name)

        anomalies = pivot_df - pivot_df.mean()
        anomalies = anomalies.fillna(0)

        data_matrix = anomalies.values

        solver = Eof(data_matrix)

        eofs = solver.eofs(neofs=5)           
        pcs = solver.pcs()             
        variance_frac = solver.varianceFraction()
        spatial_index = pivot_df.columns
        scree_plot(variance_frac, var_name)
        pc_plot(pcs, pd.to_datetime(sorted(this_gdf.day.unique()), format = 'mixed'), var_name)
        
        for eof in range(neofs):
            print(f"plottiing {var_name}: eof{eof+1}...")
            eof_values = eofs[eof, :]
            mapping = dict(zip(spatial_index, eof_values))

            map_gdf = this_gdf[[boundary_index, 'geometry']].drop_duplicates(boundary_index)
        
            map_gdf[f'eof{eof+1}'] = map_gdf[boundary_index].map(mapping)

            choreoplath_lcc(map_gdf, f'eof{eof+1}', var_name)
        