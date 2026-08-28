import s3fs
import xarray as xr
from xrspatial import slope
import fabdem
import rioxarray
import os
import pandas as pd
import numpy as np

from data.clients.base_client import BaseClient
from utils.constants import FABDEM, CACHE_BASE_DIR


class FABDEMClient(BaseClient):
    def __init__(self, *args, **kwargs):
        self.data_source = FABDEM
        self.cache_dir = os.path.join(CACHE_BASE_DIR, FABDEM.lower())
        super().__init__(*args, cache_dir = self.cache_dir, **kwargs)

    def _query(self, polygon):
        fname = "fabdem"
        local_file = os.path.join(self.save_dir, f'{fname}.tif')
        fabdem.download(polygon.bounds, local_file, cache=self.save_dir)
        da = rioxarray.open_rasterio(local_file)
        
        da_metric = da.rio.reproject("EPSG:3857")
        
        ds = da_metric.to_dataset(name="elevation")
        
        ds = ds.rename({
            'x': 'longitude',
            'y': 'latitude'
        })

        elevation_da = ds['elevation'].squeeze()
        
        slope_degrees = slope(elevation_da)
        
        # xarray-spatial flags nodata positions as NaN
        slope_percent = np.tan(np.deg2rad(slope_degrees)) * 100
        
        average_regional_grade = float(slope_percent.mean(skipna=True))
        return average_regional_grade

    def _to_cycle_hr(self, fxx):
        return "%03d" % (int(fxx),)
    
if __name__ == "__main__":
    import shapely
    client = FABDEMClient()
    this_time = "2026-05-11 00:00"
    california_bbox = shapely.box(-115.4, 41.5, -114.1, 42.0)

    avg_grade = client.query(california_bbox)
    import ipdb; ipdb.set_trace()