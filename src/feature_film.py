import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def create_fire_timelapse(final_gdf, var_name, output_fn="fire_timelapse.mp4"):
    # 1. Setup the Projection (LCC for North America)
    plot_crs = ccrs.LambertConformal(central_longitude=-100, central_latitude=45)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=plot_crs)
    ax.set_extent([-125, -70, 23, 65], crs=data_crs)

    # 2. Add static features
    ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)

    # 3. Get unique sorted dates
    days = sorted(final_gdf['day'].unique())
    
    # Define a consistent color scale across the whole film
    vmin = final_gdf[var_name].min()
    vmax = final_gdf[var_name].max()
    cmap = 'YlOrRd' # Classic fire intensity palette

    # Create the initial plot (Day 0)
    day_0_data = final_gdf[final_gdf['day'] == days[0]]
    poly_collection = day_0_data.to_crs(plot_crs.proj4_init).plot(
        ax=ax, column=var_name, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.2
    )

    # Add a persistent colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, ax=ax, orientation='horizontal', label=f'Mean {var_name}', shrink=0.6, pad=0.05)
    
    title = ax.set_title(f"Fire Intensity: {days[0].strftime('%Y-%m-%d')}")

    # 4. The Update Function
    def update(frame):
        current_day = days[frame]
        day_data = final_gdf[final_gdf['day'] == current_day]
        
        # Clear previous polygons and replot for this frame
        ax.collections.clear()
        day_data.to_crs(plot_crs.proj4_init).plot(
            ax=ax, column=var_name, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.2
        )
        import ipdb; ipdb.set_trace()
        
        title.set_text(f"Fire Intensity: {current_day.strftime('%Y-%m-%d')}")
        return ax.collections

    # 5. Build and Save
    ani = animation.FuncAnimation(fig, update, init_func=poly_collection, frames=len(days), interval=100) # 100ms per frame
    import ipdb; ipdb.set_trace()
    # Requires 'ffmpeg' installed on your system
    ani.save(output_fn, writer='ffmpeg', fps=10, dpi=150)
    plt.close()
    print(f"Film saved as {output_fn}")