import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from tqdm import tqdm
import cv2
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def worker_func(frame_data, var_name, plot_crs, data_crs, frame_date, vmin, vmax):
    fig = Figure(figsize=(12, 8))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1, projection=plot_crs)
    ax.set_extent([-125, -70, 23, 65], crs=data_crs)

    ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
    
    cmap = 'YlOrRd'
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, ax=ax, orientation='horizontal', label=f'Mean {var_name}', shrink=0.6, pad=0.05)
    
    ax.set_title(f"Fire Intensity: {frame_date.strftime('%Y-%m-%d')}")
    frame_data.plot(ax=ax, column=var_name, cmap='YlOrRd', edgecolor='black', linewidth=0.2)
    plt.tight_layout()
    
    canvas.draw()
    image_array = np.asarray(canvas.buffer_rgba(), dtype='uint8')
    image_array = image_array[:, :, :3]
    return (frame_date, image_array)

def create_fire_timelapse(final_gdf, var_name, output_fn="fire_timelapse.mp4", fps = 24):
    plot_crs = ccrs.LambertConformal(central_longitude=-100, central_latitude=45)
    data_crs = ccrs.PlateCarree()
    final_gdf = final_gdf.to_crs(plot_crs.proj4_init)

    days = sorted(final_gdf['day'].unique())
    
    futures = []
    images = []
    vmin = final_gdf[var_name].min()
    vmax = final_gdf[var_name].max()

    with ProcessPoolExecutor(max_workers = 50, mp_context=mp.get_context('spawn')) as ppex:
        for frame in range(len(days)):
            futures.append(ppex.submit(worker_func,
                                        final_gdf[final_gdf.day == days[frame]], 
                                        var_name, plot_crs, data_crs, 
                                        days[frame], vmin, vmax))
        for f in tqdm(as_completed(futures),desc = 'generating images', total = len(futures)):
            images.append(f.result())

    ims = [image[1] for image in sorted(images)]
    print("finished processing images...")
    height, width, layers = ims[0].shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_fn, fourcc, fps, size)

    for img_array in tqdm(ims, desc ='writing to video'):
        bgr_frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    print(f"Video saved as {output_fn}")
    