import s3fs
import xarray as xr
import os
import pandas as pd

from data.clients.base_client import BaseClient
from utils.constants import RRFS, CACHE_BASE_DIR


class RRFSClient(BaseClient):
    # rrfs_a/rrfs.20260114/12
    # rrfs.t00z.prslev.3km.f002.na.grib2
    def __init__(self, bucket = "noaa-rrfs-pds", *args, **kwargs):
        self.bucket = bucket
        self.data_source = RRFS
        self.cache_dir = os.path.join(CACHE_BASE_DIR, RRFS.lower())
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.base_path = "rrfs_a"
        self.target_vars = ['t', 'gh', 'u', 'v', 'r']
        os.makedirs(self.cache_dir, exist_ok = True)
        super().__init__(*args, cache_dir = self.cache_dir,**kwargs)

    def _query(self, date, lat, lon, fxx_range):
        forecasts = [forecasts.append(self._query_worker(date, lat, lon, fxx)[0]) for fxx in range(fxx_range+1)]
        ds_merged = xr.concat(forecasts, dim='valid_time')
        return ds_merged
    
    def _query_worker(self, date, lat, lon, fxx):
        date_ts = pd.to_datetime(date)
        date_str = date_ts.strftime('%Y%m%d') 
        time_str = date_ts.strftime('%H')
        fxx_str = self._to_cycle_hr(fxx)
        fname = f"rrfs.t{time_str}z.prslev.3km.f{fxx_str}.na.grib2"
        file_path = f"{self.bucket}/{self.base_path}/rrfs.{date_str}/{time_str}/{fname}"
        local_file = os.path.join(self.cache_dir, fname)
        if os.path.exists(local_file):
            pass
        else:
            self.s3.get(file_path, local_file)
        ds = xr.open_dataset(
                local_file, 
                engine="cfgrib",
                filter_by_keys={'typeOfLevel': 'isobaricInhPa',
                                'shortName': self.target_vars
                                },
                )
        ds_indexed = ds.set_xindex(["latitude", "longitude"], xr.indexes.NDPointIndex)
        target = self._subset_dataset(lat, lon, ds_indexed)
        out_fname = os.path.join(self.cache_dir, fname.replace('grib2','nc'))
        target.to_netcdf(out_fname)
        return (out_fname, RRFS)

    def _to_cycle_hr(self, fxx):
        return "%03d" % (int(fxx),)
    
if __name__ == "__main__":
    client = RRFSClient()
    this_time = "2026-05-11 00:00"
    lat = 34.05
    lon = -118.24
    fxx = 4
    ds = client.query(this_time, lat, lon, fxx)
    import ipdb; ipdb.set_trace()