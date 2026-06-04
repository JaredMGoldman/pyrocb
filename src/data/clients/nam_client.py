import xarray as xr
import os
import pandas as pd
from pathlib import Path

from data.clients.base_client import BaseClient
from utils.constants import CACHE_BASE_DIR, NAM
from utils.rio_utils import download_file_safe

class NAMClient(BaseClient):
    def __init__(self, *args, **kwargs):
        self.base_path = "https://www.ncei.noaa.gov/data/north-american-mesoscale-model/access/forecast"
        self.data_source = NAM
        self.cache_dir = os.path.join(CACHE_BASE_DIR, 'nam4k')
        self.target_vars = ['t', 'r', 'gh', 'u', 'v']
        os.makedirs(self.cache_dir, exist_ok = True)
        super().__init__(*args, cache_dir = self.cache_dir,**kwargs)

    def _query(self, date, lat, lon, fxx_range):
        forecasts = [forecasts.append(self._query_worker(date, lat, lon, fxx)) for fxx in range(fxx_range+1)]
        ds_merged = xr.concat(forecasts, dim='valid_time')
        return ds_merged
    
    def _query_worker(self, date, lat, lon, fxx):
        ts = pd.to_datetime(date)
        month = f"{ts.year}{self._make_n_len_str(ts.month, 2)}"
        day = f"{month}{self._make_n_len_str(ts.day, 2)}"
        fx_base = os.path.join(self.base_path, month, day)
        out_dir = os.path.join(self.cache_dir, day)
        os.makedirs(out_dir, exist_ok = True)
        
        fname = f"nam_218_{day}_{self._make_n_len_str(ts.hour, 2)}00_{self._make_n_len_str(fxx, 3)}.grb2"
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
                            filter_by_keys={'typeOfLevel': 'isobaricInhPa',
                                            'shortName': self.target_vars})
        ds = ds.assign_coords({'longitude': (("y", "x"), (ds.longitude.values + 180) % 360 - 180)})
        ds_indexed = ds.set_xindex(["latitude", "longitude"], xr.indexes.NDPointIndex)
        target = self._subset_dataset(lat, lon, ds_indexed)
        out_fname = os.path.join(self.cache_dir, fname.replace('grib2','nc'))
        target.to_netcdf(out_fname)
        return (out_fname, NAM)
    
if __name__ == "__main__":
    client = NAMClient()
    this_time = "2026-05-16 00:00"
    lat = 34.05
    lon = -118.24
    fxx = 48
    ds = client.query(this_time, lat, lon, fxx)
    ds.to_netcdf(f"{CACHE_BASE_DIR}/nam4k/05-16-26-fx48.nc")
    import ipdb; ipdb.set_trace()
        

    