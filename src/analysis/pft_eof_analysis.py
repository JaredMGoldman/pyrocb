import pandas as pd
import xarray as xr
import os
from utils.constants import PLOTS_DIR
import xeofs as xf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

def calculate_and_plot_enso_lags(model, pc_scores = None, num_modes=3):
    """
    Calculates the correlation between each month of the Niño 3.4 index 
    and the annual magnitude of the top PCA modes, plotting the lag grid.
    """
    # --- Step 1: Extract PC Scores and Compute Annual Magnitude ---
    pft_times = pd.to_datetime(pc_scores.coords['time'].values)
    
    # Store PC scores in a dataframe
    df_pcs = pd.DataFrame(index=pft_times)
    for i in range(num_modes):
        df_pcs[f'PC_{i+1}'] = pc_scores.sel(mode=i+1).values
        
    # Group by year and calculate the magnitude (RMS) of the anomalies for that year
    # This captures how active the mode was that year, regardless of positive/negative phase
    df_annual_pc = df_pcs.groupby(df_pcs.index.year).apply(lambda x: np.sqrt((x**2).mean()))
    
    # --- Step 2: Fetch and Structure Monthly Niño 3.4 ---
    url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.csv"
    nino_df = pd.read_csv(url, sep=r'\s+', header=0)
    # Keep only the years that overlap with your dataset (2015-2020)
    target_years = df_annual_pc.index
    nino_df['dt'] = pd.to_datetime(nino_df['Date,'])
    nino_df['YR'] = nino_df['dt'].dt.year
    nino_df['MON'] = nino_df['dt'].dt.month
    nino_df = nino_df[nino_df['YR'].isin(target_years)]
    
    # Pivot Niño data so rows are Years and columns are Months (1-12)
    # This sets up our "Lag months"
    nino_pivot = nino_df.pivot(index='YR', columns='MON', values='NINA34')
    
    # Ensure years match up perfectly
    nino_pivot = nino_pivot.reindex(target_years)

    # --- Step 3: Compute Cross-Correlation Matrix ---
    # Rows will be PC Modes, Columns will be Months of the Year
    correlation_matrix = np.zeros((num_modes, 12))
    
    for mode_idx in range(num_modes):
        pc_col = f'PC_{mode_idx+1}'
        for month in range(1, 12 + 1):
            # Calculate Pearson correlation coefficient between the annual PC magnitude 
            # and the Niño index value of a specific month across all years
            corr = df_annual_pc[pc_col].corr(nino_pivot[month])
            correlation_matrix[mode_idx, month-1] = corr

    # Convert to Dataframe for easy plotting
    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_corr = pd.DataFrame(correlation_matrix, 
                           index=[f'PC Mode {i+1}' for i in range(num_modes)], 
                           columns=months_names)

    # --- Step 4: Display the Lag in an Intuitive Way ---
    plt.figure(figsize=(12, 2 * num_modes + 1))
    
    # Using a diverging color palette (coolwarm) so positive/negative correlations stand out
    ax = sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                     vmin=-1, vmax=1, linewidths=1, cbar_kws={'label': 'Correlation Coefficient (r)'})
    
    # Intuitively highlight the highest magnitude lag month for each series
    for idx in range(num_modes):
        # Find absolute max correlation row-by-row
        row_values = correlation_matrix[idx, :]
        max_lag_idx = np.argmax(np.abs(row_values))
        
        # Draw a subtle rectangle around the peak predictor month
        rect = plt.Rectangle((max_lag_idx, idx), 1, 1, fill=False, edgecolor='black', lw=3, linestyle='-')
        ax.add_patch(rect)
        
    plt.title("ENSO Lag Analysis", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Month of the Niño 3.4 Index", fontsize=11, labelpad=10)
    plt.ylabel("PFT PC Modes", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'pfts', 'pft_mode_lags.png'))
    print(f"saved to: {os.path.join(PLOTS_DIR, 'pfts', 'pft_mode_lags.png')}")

def plot_pca_time_series_by_year(pc_scores, model, num_modes=3):
    """
    Plots PC scores over time, breaking the lines during winter gaps
    so JJA data chunks stand alone without strange bridging lines.
    """
    timestamps = pd.to_datetime(pc_scores.coords['time'].values)
    variance_fractions = model.explained_variance_ratio() * 100

    
    fig, axes = plt.subplots(num_modes, 1, figsize=(14, 3 * num_modes), sharex=True)
    if num_modes == 1:
        axes = [axes]
        
    for i in range(num_modes):
        ax = axes[i]
        mode_idx = i + 1
        
        mode_scores = pc_scores.sel(mode=mode_idx).values
        var_explained = variance_fractions.sel(mode=mode_idx).values
        
        # Convert to a DataFrame to easily group by year
        df_mode = pd.DataFrame({'scores': mode_scores}, index=timestamps)
        
        # --- THE FIX: Loop through each year and plot it as an independent segment ---
        for year, group in df_mode.groupby(df_mode.index.year):
            ax.plot(group.index, group['scores'], color='darkblue', linewidth=2)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'PC Mode {mode_idx} Time Series (Explains {var_explained:.2f}% of Variance)', 
                     fontsize=12, weight='bold', loc='left')
        ax.set_ylabel('Amplitude (Scores)')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
    plt.xlabel('Timestamp / Timeline', fontsize=12, weight='bold')
    plt.tight_layout()
    fname = "pca_analysis_cute.png"
    plt.savefig(os.path.join(PLOTS_DIR, 'pfts', fname))
    print(f"saved {os.path.join(PLOTS_DIR, 'pfts', fname)}")
    plt.close()


def analyze_jja_eof_north_america(ds, data_var='pft_val', interp_order=1, num_modes=5):
    """
    Handles fragmented JJA-only data by detrending within years,
    running a stacked EOF, restoring the time dimension for downstream code,
    and plotting across North America.
    """
    # -----------------------------------------------------------------
    # 0. Preserve the Original Time Coordinate for Downstream Code
    # -----------------------------------------------------------------
    original_time_coords = ds.time.coords['time']

    # -----------------------------------------------------------------
    # 1. Fill NaNs Spatially (Within the active JJA slices)
    # -----------------------------------------------------------------
    interp_method = 'linear' if interp_order == 1 else 'cubic'
    print(f"Filling NaNs spatially...")
    filled_da = spatial_interpolate_dataset(ds, data_var, interp_method)[data_var]

    # Apply Land Mask
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(filled_da)
    land_only_da = filled_da.where(land_mask == 0)

    # -----------------------------------------------------------------
    # 2. Season-Aware Detrending (Fixes the "Strange Plots" Issue)
    # -----------------------------------------------------------------
    print("Detrending within each summer season individually...")
    
    # Define a helper function to detrend an individual year's chunk
    def detrend_season(group):
        if len(group.time) > 1:
            trend = group.polyfit(dim='time', deg=1)
            fit = xr.polyval(group.time, trend.polyfit_coefficients)
            return group - fit
        return group

    # Group by year, apply detrending to just that summer, then combine back
    detrended_da = land_only_da.groupby('time.year').map(detrend_season)

    # -----------------------------------------------------------------
    # 3. Restructure Time for xeofs (Collapsing the Winter Gaps)
    # -----------------------------------------------------------------
    print("Collapsing winter gaps for EOF processing...")
    
    years = detrended_da['time.year'].values
    day_indices = np.arange(len(detrended_da.time)) % 12 
    
    m_index = pd.MultiIndex.from_arrays([years, day_indices], names=('year', 'day_idx'))
    
    restructured_da = detrended_da.assign_coords(time=m_index)
    stacked_da = restructured_da.rename({'time': 'sample'}).transpose('sample', 'lat', 'lon')

    # -----------------------------------------------------------------
    # 4. Perform EOF Analysis with xeofs
    # -----------------------------------------------------------------
    print("Computing EOF solver on stacked seasonal data...")
    model = xf.single.EOF(n_modes=num_modes, use_coslat=True, standardize=True)
    model.fit(stacked_da, dim='sample')
    
    eofs = model.components()
    variance_fractions = model.explained_variance_ratio() * 100 

    # -----------------------------------------------------------------
    # EXTRA CRITICAL STEP: Restore Original Linear 'time' to the Scores Array
    # -----------------------------------------------------------------
    # Replaces xeofs MultiIndex 'sample' coordinate with your original continuous datetime timeline
    raw_scores = model.scores() 
    
    # 2. Swap out the 'sample' MultiIndex for your clean continuous datetime timeline
    pc_scores = raw_scores.rename({'sample': 'time'}).assign_coords(time=original_time_coords)
    # -----------------------------------------------------------------
    # 5. Plotting (North America Frame)
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13))
    grid = plt.GridSpec(3, 2, hspace=0.4, wspace=0.25)
    map_proj = ccrs.PlateCarree()
    na_extent = [-170.0, -50.0, 22.0, 75.0]

    for i in range(num_modes):
        row = i // 2
        col = i % 2
        if i >= 5: break
            
        ax = fig.add_subplot(grid[row, col], projection=map_proj)
        ax.set_extent(na_extent, crs=map_proj)
        
        ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=1)
        ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)

        mode_da = eofs.sel(mode=i+1)
        mode_da.plot(ax=ax, transform=map_proj, cmap='RdBu_r', add_colorbar=True, zorder=2)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='black', zorder=3)
        
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none'
        )
        ax.add_feature(states_provinces, linewidth=0.5, edgecolor='gray', zorder=3)

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4, zorder=4)
        gl.top_labels = False; gl.right_labels = False

        ax.set_title(f'PFT EOF Mode {i+1} ({variance_fractions.sel(mode=i+1).values:.2f}%)')

    # --- Plot Scree Plot ---
    ax_scree = fig.add_subplot(grid[2, 1])
    modes = np.arange(1, num_modes + 1)
    vf_values = variance_fractions.values
    
    ax_scree.bar(modes, vf_values, color='forestgreen', edgecolor='black', alpha=0.7, label='Individual')
    
    ax_scree.set_title('Scree Plot (Variance Explained)')
    ax_scree.set_xlabel('EOF Mode')
    ax_scree.set_ylabel('Percentage of Variance (%)')
    ax_scree.set_xticks(modes)
    ax_scree.grid(True, linestyle='--', alpha=0.6)
    ax_scree.legend()

    plt.suptitle(f"JJA PFT EOF Variance Analysis", fontsize=16, weight='bold')
    
    fname = "pft_eof_jja_north_america.png"
    plt.savefig(os.path.join(PLOTS_DIR, 'pfts', fname), bbox_inches='tight')
    print(f"Saved to: {os.path.join(PLOTS_DIR, 'pfts', fname)}")
    
    return model, eofs, pc_scores


def spatial_interpolate_dataset(ds, data_var="pft_val", method="linear"):
    """Spatially interpolates a multi-dimensional dataset time-slice by time-slice,

    guaranteeing 2D array masking alignment regardless of extra hidden dimensions.
    """
    # 1. Extract pure 1D coordinate arrays to ensure meshgrids are strictly 2D (lat, lon)
    raw_lons = np.atleast_1d(ds.lon.values)
    raw_lats = np.atleast_1d(ds.lat.values)
    grid_lon, grid_lat = np.meshgrid(raw_lons, raw_lats)

    interpolated_time_slices = []

    # 2. Iterate through time steps
    for t in ds.time:
        # Select the time slice
        slice_da = ds[data_var].sel(time=t)

        # --- THE SAFEST STRIP: Force the slice down to strictly 2D (lat, lon) ---
        # .squeeze() drops any single-value dimensions (e.g., level=1 or expver=7)
        # If there are genuine extra dimensions, we take the first element [0] as a fallback
        if len(slice_da.dims) > 2:
            # Squeeze out dimensions with length 1
            slice_da = slice_da.squeeze()

            # If it's STILL greater than 2D, force-select the first index of non-spatial dimensions
            extra_dims = [
                d for d in slice_da.dims if d not in ["lat", "lon", "latitude", "longitude"]
            ]
            if extra_dims:
                selectors = {dim: 0 for dim in extra_dims}
                slice_da = slice_da.isel(**selectors)

        # Transpose to a predictable layout: (lat, lon)
        slice_da = slice_da.transpose("lat", "lon")

        # 3. Pull pure numpy arrays. Stripping the xarray wrapper stops dimensional broadcast bugs.
        slice_matrix = slice_da.values

        # Generate local 2D meshes mapping exactly to this matrix footprint shape
        lon_mesh, lat_mesh = np.meshgrid(slice_da.lon.values, slice_da.lat.values)

        # 4. Create the mask. valid_mask is now guaranteed to match lon_mesh/lat_mesh perfectly.
        valid_mask = (~np.isnan(slice_matrix)) & (slice_matrix != 0)

        # If this specific slice is completely empty, skip interpolation and preserve the NaNs
        if not np.any(valid_mask):
            # Fall back to matching the standard grid structure
            dummy_grid = np.full(grid_lon.shape, np.nan)
            interpolated_slice = xr.DataArray(
                dummy_grid,
                coords=[("lat", raw_lats), ("lon", raw_lons)],
                name=data_var,
            )
            interpolated_time_slices.append(interpolated_slice)
            continue

        # 5. Extract coordinate pairs
        points = np.column_stack((lon_mesh[valid_mask], lat_mesh[valid_mask]))
        values = slice_matrix[valid_mask]

        # 6. Run griddata interpolation
        grid_z = griddata(
            points=points,
            values=values,
            xi=(grid_lon, grid_lat),
            method=method,
        )

        # Fallback to 'nearest' to clean up remaining edge NaNs left behind by 'linear'
        if np.isnan(grid_z).any():
            grid_z_nearest = griddata(
                points, values, (grid_lon, grid_lat), method="nearest"
            )
            grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

        # Rebuild clean 2D slice
        interpolated_slice = xr.DataArray(
            grid_z, coords=[("lat", raw_lats), ("lon", raw_lons)], name=data_var
        )
        interpolated_time_slices.append(interpolated_slice)

    # 7. Reconstruct the full 3D timeline cleanly
    fixed_da = xr.concat(interpolated_time_slices, dim=ds.time)

    new_ds = ds.copy()
    new_ds[data_var] = fixed_da
    return new_ds

def csv_to_xarray(csv_path):
    """
    Reads a CSV generated from the parse_to_dataframe format 
    and converts it to an xarray Dataset with time, lat, and lon dimensions.
    """
    # 1. Read the CSV, ensuring 'time' is parsed correctly as a datetime object
    try:
        df = pd.read_csv(csv_path, parse_dates=['time'])
    except:
        print(f"couldn't process {csv_path}")
        return None
    
    # 2. Set the multi-index. 
    # xarray will use these index names to construct the dataset dimensions.
    # Note: If your CSV contains multiple 'data_name' types per coordinate, 
    # you should include 'data_name' in the index as well.
    df = df.set_index(['time', 'lat', 'lon'])
    
    # 3. Convert the pandas DataFrame to an xarray Dataset
    ds = df.to_xarray()
    
    # 4. Rename 'value' to your requested data variable name 'pft_val'
    ds = ds.rename({'value': 'pft_val'})
    
    # Optional: If you don't need 'data_name' as a coordinate variable anymore,
    # you can drop it or leave it as a variable depending on your needs.
    # ds = ds.drop_vars('data_name')
    
    return ds

if __name__ == "__main__":
    fnames = [os.path.join(PLOTS_DIR, 'pfts', dir_name, 'pft_vals.csv') for dir_name in os.listdir(os.path.join(PLOTS_DIR, 'pfts'))]
    fnames = [fname for fname in fnames if os.path.exists(fname)]
    dses = [csv_to_xarray(fname) for fname in fnames]
    dses = [ds for ds in dses if not ds is None]
    merged_ds = xr.concat(dses, dim='time' )
    uniform_dates = pd.to_datetime(merged_ds["time"].values, errors="coerce")
    merged_ds = merged_ds.assign_coords(time=uniform_dates).sortby('time')
    num_modes = 5
    
    model, components, pc_scores = analyze_jja_eof_north_america(merged_ds, 'pft_val', num_modes = num_modes)
    plot_pca_time_series_by_year(pc_scores, model=model, num_modes=num_modes)
    calculate_and_plot_enso_lags(model, num_modes = num_modes, pc_scores = pc_scores)
