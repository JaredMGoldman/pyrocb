import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import save_plot, slugify

MTBS_DIR = '/data/lthapa/data2restore/lthapa/mtbs/2020/'
ALL_FIRES_FN = '/data/lthapa/data2restore/lthapa/ML_daily/unique_fires_with_area_and_irwin_192021.csv'
FIRE_DAILY_FN = '/data/lthapa/data2restore/lthapa/ML_daily/fire_polygons/ClippedFires2020_VIIRS_daily_12Z_Day_Start.geojson'

def combined_extent(*gdfs: gpd.GeoDataFrame, buffer_frac: float = 0.05):
    """
    Return (minx, maxx, miny, maxy) for all input GeoDataFrames, with optional fractional buffer.
    """
    minx = min(gdf.total_bounds[0] for gdf in gdfs)
    miny = min(gdf.total_bounds[1] for gdf in gdfs)
    maxx = max(gdf.total_bounds[2] for gdf in gdfs)
    maxy = max(gdf.total_bounds[3] for gdf in gdfs)

    dx = maxx - minx
    dy = maxy - miny
    pad_x = dx * buffer_frac if dx > 0 else 1.0
    pad_y = dy * buffer_frac if dy > 0 else 1.0

    return (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)

def load_mtbs(mtbs_dir):
    """
    Docstring for load_mtbs
    
    :param mtbs_dir: Description
    """
    mtbs_bdrys = gpd.GeoDataFrame()
    for file in os.listdir(mtbs_dir):
        if file.endswith('_burn_bndy.shp'):
            gdf_file = gpd.read_file(mtbs_dir+file)
            mtbs_bdrys = pd.concat([mtbs_bdrys, gdf_file], axis=0)

    mtbs_bdrys_out = mtbs_bdrys.dissolve(by='irwinID').reset_index()
    return mtbs_bdrys_out

def plot_polygon(fire_id, fire_name, mtbs_bdrys, daily_fires, crs = 'EPSG:4326'):
    """
    Plots polygon and saves to output/plots dir
    
    :param fire_id: irwinID of fire to plot
    :param fire_name: name of fire to plot to be used in figure title and save name
    :param mtbs_bdrys: mtbs boundary information as a geojson
    :param daily_fires: viirs geojson dataframe with daily frp information
    :param crs: crs plot type specified for plot
    """
    fig = plt.figure(figsize=(10,14))
    ax= fig.add_subplot(111,projection=ccrs.PlateCarree())
    bdry = mtbs_bdrys.loc[mtbs_bdrys['irwinID']== fire_id].to_crs(crs)
    this_fire = daily_fires.loc[daily_fires['irwinID']== fire_id].to_crs(crs)
    this_fire.plot(column='NEW FRP',cmap='OrRd', ax=ax)
    bdry.boundary.plot(ax=ax, edgecolor='k')
    plt.title(fire_name, fontsize=24)
    minx, maxx, miny, maxy = combined_extent(bdry, this_fire, buffer_frac=0.03)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,\
                linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    save_plot(f"{slugify(fire_name)}_polygon")

def find_common_fires(all_fires_fn, mtbs_bdrys, fire_daily):
    """
    Docstring for find_common_fires
    
    :param all_fires_fn: Description
    :param mtbs_bdrys: Description
    :param fire_daily: Description
    """
    all_fires = pd.read_csv(all_fires_fn)
    fire_dict = {all_fires[all_fires['Fire Name']==fire_name]['irwinID'].unique()[0].replace("['","").replace("']",""): fire_name.strip() for fire_name in all_fires['Fire Name'].unique()}
    common_fires = [irwin_id for irwin_id in fire_dict.keys() if len(mtbs_bdrys[mtbs_bdrys['irwinID'] == irwin_id]['irwinID'].values)>0]
    common_fires = [irwin_id for irwin_id in common_fires if len(fire_daily[fire_daily['irwinID'] == irwin_id]['irwinID'].values)>0]
    return common_fires, fire_dict

if __name__ == "__main__":
    daily_fires = gpd.read_file(FIRE_DAILY_FN)
    mtbs_bdrys = load_mtbs(MTBS_DIR)
    common_fires, fire_dict = find_common_fires(ALL_FIRES_FN, mtbs_bdrys, daily_fires)
    for fire_id in common_fires:
        fire_name = fire_dict[fire_id]
        print(f"plotting {fire_name}...")
        plot_polygon(fire_id, fire_name, mtbs_bdrys, daily_fires)
        print(f"{fire_name} plotted")