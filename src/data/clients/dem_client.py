import s3fs
import xarray as xr
from xrspatial import slope
import rioxarray
import os
import pandas as pd
import numpy as np

from data.clients.base_client import BaseClient
from utils.constants import DEM, CACHE_BASE_DIR


class DEMClient(BaseClient):
    def __init__(self, bucket = "copernicus-dem-30m", *args, **kwargs):
        self.bucket = bucket
        self.data_source = DEM
        self.cache_dir = os.path.join(CACHE_BASE_DIR, DEM.lower())
        self.s3 = s3fs.S3FileSystem(anon=True)
        super().__init__(*args, cache_dir = self.cache_dir, **kwargs)


    def _query(self, polygon):
        # lat/lon in 89 - 90, 177-178
        # 1. Extract the total spatial extent of the input polygon
        # returns: (min_lon, min_lat, max_lon, max_lat)
        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        # If a polygon touches -177.4, the lower-left corner of that tile is -178.
        # We add 1 to the 'max' range stop to ensure the boundary edges are inclusive.
        lat_start = int(np.floor(min_lat))
        lat_end = int(np.floor(max_lat)) + 1

        lon_start = int(np.floor(min_lon))
        lon_end = int(np.floor(max_lon)) + 1

        fnames = []
        # 3. Iterate through every 1-degree tile matrix intersection
        for lat in range(lat_start, lat_end):
            for lon in range(lon_start, lon_end):
                if lat >= 0:
                    lat_str = f"N{lat:02d}"
                else:
                    lat_str = f"S{abs(lat):02d}"

                if lon >= 0:
                    lon_str = f"E{lon:03d}"
                else:
                    lon_str = f"W{abs(lon):03d}"

                fname = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
                fnames.append(fname)
        print(f"processing {len(fnames)} files")
        dses = []
        for fname in fnames:
            print(f"downloading {fname}")
            file_path = f'{self.bucket}/{fname}/{fname}.tif'
            local_file = os.path.join(self.save_dir, f'{fname}.tif')
            self.s3.get(file_path, local_file)
            da = rioxarray.open_rasterio(local_file)
            ds = da.to_dataset(name="elevation")
            ds = ds.rename({
                'x' : 'longitude',
                'y' : 'latitude'
            })
            dses.append(ds)
        merged_ds = xr.combine_by_coords(dses)
        target = self._subset_dataset([min_lat, max_lat], [min_lon, max_lon], merged_ds)
        elevation_da = target['elevation'].squeeze()
        slope_degrees = slope(elevation_da)
        slope_percent = np.tan(slope_degrees * np.pi / 180.0) * 100
        average_regional_grade = float(slope_percent.mean(skipna=True))
        import ipdb; ipdb.set_trace()
        return average_regional_grade

    def _to_cycle_hr(self, fxx):
        return "%03d" % (int(fxx),)
    
if __name__ == "__main__":
    import shapely
    client = DEMClient()
    this_time = "2026-05-11 00:00"
    california_bbox = shapely.box(-115.4, 41.5, -114.1, 42.0)

    ds = client.query(california_bbox)
    import ipdb; ipdb.set_trace()