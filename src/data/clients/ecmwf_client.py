import xarray as xr
import os
import pandas as pd
from pathlib import Path

from data.clients.base_client import BaseClient
from utils.constants import CACHE_BASE_DIR, ECMWF
from ecmwf.opendata import Client

class ECMWFClient(BaseClient):
    def __init__(self, *args, **kwargs):
        self.base_path = "https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast/"
        self.data_source = ECMWF
        self.cache_dir = os.path.join(CACHE_BASE_DIR, 'ecmwf')
        self.target_vars = ['t', 'r', 'gh', 'u', 'v']
        self.ecmwf_client = Client(source="aws")
        os.makedirs(self.cache_dir, exist_ok = True)
        super().__init__(*args, cache_dir = self.cache_dir,**kwargs)

    def _query(self, date, lat, lon, fxx_range):
        forecasts = [forecasts.append(self._query_worker(date, lat, lon, fxx)) for fxx in range(fxx_range+1)]
        ds_merged = xr.concat(forecasts, dim='valid_time')
        return ds_merged
    
    def _query_worker(self, date, lat, lon, fxx):
        if fxx % 3 != 0:
            print(f"invalid forecast hour for {self.__class__.__name__}: {fxx}")
            return None
        ts = pd.to_datetime(date)
        month = f"{ts.year}{self._make_n_len_str(ts.month, 2)}"
        day = f"{month}{self._make_n_len_str(ts.day, 2)}"
        out_dir = os.path.join(self.cache_dir, day)
        os.makedirs(out_dir, exist_ok = True)
        
        fname = f"ifs_{day}_{self._make_n_len_str(ts.hour, 2)}00_{self._make_n_len_str(fxx, 3)}.grb2"
        local_path = os.path.join(out_dir, fname)
        if os.path.exists(local_path):
            pass
        else:
            self.ecmwf_client.retrieve(
                    date=day,
                    time=ts.hour,
                    step=fxx,
                    stream="oper",
                    type="fc",
                    param=self.target_vars,
                    target=local_path,
                )

        try:
            ds = xr.open_dataset(local_path,
                                filter_by_keys={'typeOfLevel': 'isobaricInhPa',
                                                'shortName': self.target_vars})
        except:
            return None
        ds = ds.assign_coords({'longitude': (ds.longitude.values + 180) % 360 - 180}).sortby('longitude')
        target = self._subset_dataset(lat, lon, ds)
        target = target.drop_attrs().load()
        out_fname = os.path.join(self.cache_dir, fname.replace('grib2','nc'))
        target.to_netcdf(out_fname)
        print(f"finished processing {fname}")
        return (out_fname, ECMWF)
    
if __name__ == "__main__":
    client = ECMWFClient()
    this_time = "2026-05-16 00:00"
    lat = 34.05
    lon = -118.24
    fxx = 48
    ds = client.query(this_time, lat, lon, fxx)
    import ipdb; ipdb.set_trace()
    ds.to_netcdf(f"{CACHE_BASE_DIR}/ecmwf/05-16-26-fx48.nc")
    import ipdb; ipdb.set_trace()
        

    