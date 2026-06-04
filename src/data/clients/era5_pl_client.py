import s3fs
import xarray as xr
import os
import pandas as pd

from data.clients.base_client import BaseClient
from utils.constants import ERA5, CACHE_BASE_DIR


class ERA5PLClient(BaseClient):
    # rrfs_a/rrfs.20260114/12
    # rrfs.t00z.prslev.3km.f002.na.grib2
    def __init__(self, bucket = "nsf-ncar-era5", *args, **kwargs):
        self.bucket = bucket
        self.data_source = ERA5
        self.cache_dir = os.path.join(CACHE_BASE_DIR, ERA5.lower())
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.base_path = "rrfs_a"

        self.param_map = {
            't' : 130, 
            'z' : 129, 
            'u' : 131, 
            'v' : 132, 
            'r' : 157
        }
        self.rename_dict = {
            'T' : 't',
            'Z' : 'gh',
            'U' : 'u',
            'V' : 'v',
            'R' : 'r'
        }
        os.makedirs(self.cache_dir, exist_ok = True)
        super().__init__(*args, cache_dir = self.cache_dir,**kwargs)

    def _query(self, date, lat, lon, fxx_range):
        forecasts = [self._query_worker(date, lat, lon, fxx)[0] for fxx in range(fxx_range+1)]
        ds_merged = xr.concat(forecasts, dim='valid_time')
        return ds_merged
    
    def _query_worker(self, date, lat, lon, fxx):
        ts = pd.to_datetime(date)

        if fxx >= 24:
            hr_del = int(fxx/24) * 24
            ts = ts + pd.Timedelta(hr_del, 'h')
            fxx = fxx - hr_del
        

        date_ts = pd.to_datetime(date)
        month_str = self._make_n_len_str(date_ts.month, 2)
        day_str = self._make_n_len_str(date_ts.day, 2)
        var_dses = []
        for varname, var_id in self.param_map.items():
            if varname in ['u', 'v']:
                namespace = 'll025uv'
            else:
                namespace = 'll025sc'
            fname = f"e5.oper.an.pl.128_{self._make_n_len_str(var_id, 3)}_{varname}.{namespace}.{date_ts.year}{month_str}{day_str}00_{date_ts.year}{month_str}{day_str}23.nc"
            file_path = f"{self.bucket}/e5.oper.an.pl/{date_ts.year}{month_str}/{fname}"
            local_file = os.path.join(self.cache_dir, fname)
            if os.path.exists(local_file):
                pass
            else:
                try:
                    self.s3.get(file_path, local_file)
                except:
                    import ipdb; ipdb.set_trace()
            ds = xr.open_dataset(
                    local_file
                    )
            ds = ds.assign_coords(longitude=(ds.longitude.values + 180) % 360 - 180).sortby('longitude')
            target = self._subset_dataset(lat, lon, ds)
            var_dses.append(target)
            
        merged_ds = xr.merge(var_dses)
        ds = merged_ds.rename_vars(self.rename_dict)
        out_fname = f"era5_{date_ts.year}{month_str}{day_str}00_{date_ts.year}{month_str}{day_str}23.nc"
        out_path = os.path.join(self.cache_dir, out_fname)
        ds.to_netcdf(out_path)
        import ipdb; ipdb.set_trace()
        return (out_path, ERA5)

    def _to_cycle_hr(self, fxx):
        return "%03d" % (int(fxx),)
    
if __name__ == "__main__":
    client = ERA5PLClient()
    this_time = "1950-05-11 00:00"
    lat = [24.846565, 71.300793]
    lon = [-166.992188, -52.031250]
    fxx = 4
    ds = client.query(this_time, lat, lon, fxx)
    import ipdb; ipdb.set_trace()