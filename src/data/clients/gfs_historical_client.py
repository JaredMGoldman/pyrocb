import xarray as xr
import os
import pandas as pd
from pathlib import Path

from data.clients.base_client import BaseClient
from utils.constants import CACHE_BASE_DIR, GFS
from utils.rio_utils import download_file_safe, url_exists



class GFSHistClient(BaseClient):
    def __init__(self, *args, **kwargs):
        self.base_path = "https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/"
        self.data_source = GFS
        self.cache_dir = os.path.join(CACHE_BASE_DIR, 'gfs')
        self.target_vars = ['t', 'r', 'gh', 'u', 'v']
        self.fxx_freq = 3
        os.makedirs(self.cache_dir, exist_ok = True)
        super().__init__(*args, cache_dir = self.cache_dir,**kwargs)

    def _query(self, date, lat, lon, fxx_range):
        forecasts = [self._query_worker(date, lat, lon, fxx) for fxx in range(fxx_range+1)]
        forecasts = [forecast for forecast in forecasts if not forecast is None]
        ds_merged = xr.concat([xr.open_dataset(forecast) for forecast, _ in forecasts ], dim='valid_time')
        return ds_merged
    
    def _query_worker(self, date, lat, lon, fxx):
        if fxx % self.fxx_freq != 0:
            # print(f"invalid forecast hour for {self.__class__.__name__}: {fxx}")
            return None
        
        ts = pd.to_datetime(date)

        if fxx >= 24:
            hr_del = int(fxx/24) * 24
            ts = ts + pd.Timedelta(hr_del, 'h')
            fxx = fxx - hr_del
        if fxx % 6 == 0:
            hour = fxx
            fxx = 0
        else:
            hour = int(fxx / 6) * 6
            fxx = 3
        

        month = f"{ts.year}{self._make_n_len_str(ts.month, 2)}"
        day = f"{month}{self._make_n_len_str(ts.day, 2)}"
        fx_base = os.path.join(self.base_path, month, day)
        out_dir = os.path.join(self.cache_dir, day)
        os.makedirs(out_dir, exist_ok = True)
        
        fname = f"gfsanl_3_{day}_{self._make_n_len_str(hour, 2)}00_{self._make_n_len_str(fxx, 3)}.grb"
        url = os.path.join(fx_base, fname)
        if not url_exists(url):
            fname = fname.replace("grb", "grb2")
            url = os.path.join(fx_base, fname)
            if not url_exists(url):
                fname = fname.replace("gfsanl_3", "gfsanl_4")
                url = os.path.join(fx_base, fname)
        local_path = os.path.join(out_dir, fname)
        if os.path.exists(local_path):
            pass
        else:
            try:
                download_file_safe(url, Path(local_path), self._session)
            except Exception as e:
                print(f"{e}")
                return None
        ds = xr.open_dataset(local_path,
                                filter_by_keys={'typeOfLevel': 'isobaricInhPa', #})
                                                'shortName': self.target_vars})
        ds = ds.assign_coords({'longitude': (ds.longitude.values + 180) % 360 - 180}).sortby('longitude')
        target = self._subset_dataset(lat, lon, ds)
        out_fname = os.path.join(self.cache_dir, fname.replace('grb','nc'))
        target.to_netcdf(out_fname)
        # print(f"finished processing {fname}")
        return (out_fname, GFS)
    
if __name__ == "__main__":
    client = GFSHistClient()
    this_time = "2014-06-01 00:00"
    lat = 34.05
    lon = -118.24
    fxx = 36
    ds = client.query(this_time, lat, lon, fxx)
    ds.to_netcdf(f"{CACHE_BASE_DIR}/gfs/05-16-26-fx48.nc")
    import ipdb; ipdb.set_trace()
        

    