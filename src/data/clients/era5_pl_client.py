import s3fs
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import xarray as xr
import os
import pandas as pd

from data.clients.base_client import BaseClient
from utils.constants import ERA5, CACHE_BASE_DIR


class ERA5PLClient(BaseClient):
    # rrfs_a/rrfs.20260114/12
    # rrfs.t00z.prslev.3km.f002.na.grib2
    def __init__(self, bucket = "nsf-ncar-era5", fxx_freq = 3, *args, **kwargs):
        self.bucket = bucket
        self.data_source = ERA5
        self.cache_dir = os.path.join(CACHE_BASE_DIR, ERA5.lower())
        self.base_path = "rrfs_a"
        self.fxx_freq = 3

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
            'R' : 'r',
            'time'  : 'valid_time'
        }
        os.makedirs(self.cache_dir, exist_ok = True)
        super().__init__(*args, cache_dir = self.cache_dir,**kwargs)

    def _query(self, date, lat, lon, fxx_range):
        forecasts = self._query_worker(date, lat, lon, list(range(fxx_range)))[0]
        ds_merged = xr.concat([xr.open_dataset(forecast , engine = 'netcdf4') for forecast in forecasts], dim='valid_time')
        return ds_merged
    
    def __snap_to_date(self, timestamp, fxx):
        hr_del = int(fxx/24) * 24
        ts = timestamp + pd.Timedelta(hr_del, 'h')
        fxx = fxx - hr_del
        return ts, fxx
    
    def _download_single_variable(self, item, month_str, day_str, ts):
        """Worker function to handle a single file download with its own S3 context."""
        varname, var_id = item
        
        # 1. Determine namespace and file names exactly as before
        if varname in ['u', 'v']:
            namespace = 'll025uv'
        else:
            namespace = 'll025sc'
            
        fname = f"e5.oper.an.pl.128_{self._make_n_len_str(var_id, 3)}_{varname}.{namespace}.{ts.year}{month_str}{day_str}00_{ts.year}{month_str}{day_str}23.nc"
        file_path = f"{self.bucket}/e5.oper.an.pl/{ts.year}{month_str}/{fname}"
        local_file = os.path.join(self.cache_dir, fname)
        
        # Cache guard check
        if os.path.exists(local_file):
            return local_file
            
        local_s3 = s3fs.S3FileSystem(anon=True) 
        
        try:
            local_s3.get(file_path, local_file)
            return local_file
        except Exception as e:
            print(f"Failed {fname}: {str(e)}")
            return None

    def download_data_parallel(self, ts, max_workers=5):
        """Parallelized manager replacing your original sequential loop."""
        month_str = f"{ts.month:02d}"
        day_str = f"{ts.day:02d}"
        
        # Using ThreadPoolExecutor since downloading data is heavily Network IO-bound
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures = {
                executor.submit(self._download_single_variable, item, month_str, day_str, ts): item[0]
                for item in self.param_map.items()
            }
            
            # Track and report progress as workers wrap up
            var_fnames = []
            for future in as_completed(futures):
                varname = futures[future]
                result = future.result()
                if not result is None:
                    var_fnames.append(future.result())
                else:
                    raise RuntimeError(f"failed to download {varname} data for {ts}")
            wait(futures.keys())
        return var_fnames

    def _query_worker(self, date, lat, lon, fxx):
        ts = pd.to_datetime(date)        
        fxx_map = {}
        if type(fxx) is list:
            for fxx_val in fxx:
                ts_key, this_fxx = self.__snap_to_date(ts, fxx_val)
                if ts_key in fxx_map:
                    fxx_map[ts_key].append(this_fxx)
                else:
                    fxx_map[ts] = [this_fxx]
        else:
            ts, fxx = self.__snap_to_date(ts, fxx)
            fxx_map = {ts : [fxx]}

        for ts, fxxs in fxx_map.items():
            month_str = self._make_n_len_str(ts.month, 2)
            day_str = self._make_n_len_str(ts.day, 2)
            var_dses = []
            var_fnames = self.download_data_parallel(ts)
            for var_fname in var_fnames:
                ds = xr.open_dataset(
                        var_fname
                        )
                ds = ds.assign_coords(longitude=(ds.longitude.values + 180) % 360 - 180).sortby('longitude')
                target = self._subset_dataset(lat, lon, ds, pool_n = 2).sortby('level', ascending = False)
                var_dses.append(target)
            [os.remove(var_fname) for var_fname in var_fnames]
                
            merged_ds = xr.merge(var_dses)
            ds = merged_ds.rename_vars(self.rename_dict)

            out_paths = []
            for i, fxx_val in enumerate(fxxs):
                if i % self.fxx_freq != 0:
                    continue
                out_fname = f"era5_{ts.year}{month_str}{day_str}00_{ts.year}{month_str}{day_str}{self._make_n_len_str(fxx_val, 2)}00.nc"
                out_path = os.path.join(self.cache_dir, out_fname)

                ds.sel(valid_time=ts + pd.Timedelta(fxx_val, 'hour')).to_netcdf(out_path)
                out_paths.append(out_path)
        return (out_paths, ERA5)

    def _to_cycle_hr(self, fxx):
        return "%03d" % (int(fxx),)
    
if __name__ == "__main__":
    client = ERA5PLClient()
    this_time = "1960-05-11 00:00"
    lat = [24.846565, 71.300793]
    lon = [-166.992188, -52.031250]
    fxx = 4
    out = client.query(this_time, lat, lon, fxx)
    import ipdb; ipdb.set_trace()