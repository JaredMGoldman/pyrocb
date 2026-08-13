import xarray as xr
import os
import pandas as pd

from data.clients.rrfs_client import RRFSClient
from utils.constants import RRFS, CACHE_BASE_DIR
from utils.rio_utils import download_file_safe, Path


class RRFSClientParallel(RRFSClient):
    # rrfs_a/rrfs.20260114/12
    # rrfs.t00z.prslev.3km.f002.na.grib2
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/para/"
    
    def _query_worker(self, date, lat, lon, fxx):
        date_ts = pd.to_datetime(date)
        date_str = date_ts.strftime('%Y%m%d') 
        time_str = self._make_n_len_str(date_ts.strftime('%H'),2)
        fxx_str = self._to_cycle_hr(fxx)
        fname = f"rrfs.t{time_str}z.prslev.13km.f{fxx_str}.na.grib2"
        file_path = f"{self.base_path}/rrfs.{date_str}/{time_str}/{fname}"
        local_file = os.path.join(self.save_dir, fname)

        if os.path.exists(local_file):
            pass
        else:
            download_file_safe(file_path, Path(local_file), self._session)
        ds = xr.open_dataset(
                local_file, 
                engine="cfgrib",
                filter_by_keys={'typeOfLevel': 'isobaricInhPa',
                                'shortName': self.target_vars
                                },
                )
        
        target = self.subset_na_and_coarsen_2d(ds, pool_n = 1)
        target = self._subset_dataset(lat, lon, target, pool_n = 1)
        ds_compressed = target.stack(spatial_point=("y", "x"))
        ds_compressed = ds_compressed.dropna(dim="spatial_point", how="all", subset=self.target_vars).reset_index('spatial_point')
        out_fname = os.path.join(os.path.join(CACHE_BASE_DIR, RRFS.lower()), fname.replace('grib2','nc'))
        ds_compressed.to_netcdf(out_fname)
        return (out_fname, RRFS)
    
if __name__ == "__main__":
    import analysis.mapping.config as config

    client = RRFSClient()
    lat = config.lats #[32, 34.05]
    lon = config.lons # [-120, -118.24]
    fxx = 2
    this_time = config.now_date
    ds = client.query(this_time, lat, lon, fxx)
    import ipdb; ipdb.set_trace()