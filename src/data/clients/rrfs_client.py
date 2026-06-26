import s3fs
import xarray as xr
import os
import pandas as pd
from cartopy.io import shapereader

from data.clients.base_client import BaseClient
from utils.constants import RRFS, CACHE_BASE_DIR


class RRFSClient(BaseClient):
    # rrfs_a/rrfs.20260114/12
    # rrfs.t00z.prslev.3km.f002.na.grib2
    def __init__(self, bucket = "noaa-rrfs-pds", *args, **kwargs):
        self.bucket = bucket
        self.data_source = RRFS
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.base_path = "rrfs_a"
        self.target_vars = ['t', 'gh', 'u', 'v', 'r']
        super().__init__(*args, cache_dir = os.path.join(CACHE_BASE_DIR, RRFS.lower()),**kwargs)

    def _query(self, date, lat, lon, fxx_range):
        forecasts = [xr.open_dataset(self._query_worker(date, lat, lon, fxx)[0]) for fxx in range(fxx_range+1)]
        ds_merged = xr.concat(forecasts, dim='valid_time')
        return ds_merged
    
    def _query_worker(self, date, lat, lon, fxx):
        date_ts = pd.to_datetime(date)
        date_str = date_ts.strftime('%Y%m%d') 
        time_str = date_ts.strftime('%H')
        fxx_str = self._to_cycle_hr(fxx)
        fname = f"rrfs.t{time_str}z.prslev.3km.f{fxx_str}.na.grib2"
        file_path = f"{self.bucket}/{self.base_path}/rrfs.{date_str}/{time_str}/{fname}"
        local_file = os.path.join(self.save_dir, fname)
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
        # ds_indexed = ds.set_xindex(["latitude", "longitude"], xr.indexes.NDPointIndex)
        
        target = self.subset_na_and_coarsen_2d(ds, pool_n = 3)
        target = self._subset_dataset(lat, lon, target, pool_n = 1)
        ds_compressed = target.stack(spatial_point=("y", "x"))
        ds_compressed = ds_compressed.dropna(dim="spatial_point", how="all", subset=self.target_vars).reset_index('spatial_point')
        out_fname = os.path.join(os.path.join(CACHE_BASE_DIR, RRFS.lower()), fname.replace('grib2','nc'))
        ds_compressed.to_netcdf(out_fname)
        return (out_fname, RRFS)


    def subset_na_and_coarsen_2d(self, ds, decimal_precision=3, pool_n=2):
        import numpy as np
        import regionmask
        import geopandas as gpd
        """
        Masks a 2D dataset down strictly to the North American landmass footprint,
        snaps coordinates to a high-density regular precision layout, and applies 
        structural coarsening.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Your original dataset with 2D coordinates ('latitude', 'longitude') spanning dimensions ('y', 'x')
        decimal_precision : int
            Decimal rounding target (3 decimals provides ~110m resolution mapping).
        pool_n : int
            The spatial downsampling matrix factor.
        """
        ds_working = ds.copy()
        y_dim, x_dim = ds_working.latitude.dims

        # ==========================================================
        # STEP 1: GENERATE VECTOR LAND MASK FOR NORTH AMERICA
        # ==========================================================
        # 1. Fetch World Countries vector geometries from Natural Earthshp
        shpfilename = shapereader.natural_earth(
            resolution='110m', category='cultural', name='admin_0_countries'
        )
        world = gpd.read_file(shpfilename)

        # Filter for the North American continent footprint
        # Note: Natural Earth uses 'North America' in the 'CONTINENT' attribute (all caps or camel case depending on version)
        na_vector = world[world['CONTINENT'].str.upper() == 'NORTH AMERICA']
        
        # 3. Create an xarray-compatible 2D mask matching your exact 2D coordinate mesh grid
        # regionmask looks directly at the 2D latitude/longitude arrays to flag pixels
        na_mask = regionmask.mask_geopandas(
            na_vector, 
            lon_or_obj=ds_working.longitude, 
            lat=ds_working.latitude
        )
        
        # na_mask returns matching index values where land is present, and NaN everywhere else.
        # We turn this into a clear True/False boolean mask array.
        is_na_land = na_mask.notnull()

        # ==========================================================
        # STEP 2: MASK AND TRUNCATE THE EXPENSIVE DATA
        # ==========================================================
        # Apply the mask. drop=True tells xarray to completely shave away empty columns/rows 
        # outside the bounding box of North America, dramatically shrinking the dataset size right away.
        ds_subset = ds_working.where(is_na_land, drop=True)

        if ds_subset.sizes[y_dim] == 0 or ds_subset.sizes[x_dim] == 0:
            raise ValueError("[-] Masking error: No matching dataset pixels fell inside the North American vector.")

        # ==========================================================
        # STEP 3: QUANTIZE COORDINATES TO REDUCE UNIQUE COMBINATIONS
        # ==========================================================
        # Snap the remaining 2D float values to a dense regular threshold matrix layout
        ds_subset['latitude'] = xr.DataArray(
            np.round(ds_subset.latitude.values, decimal_precision), 
            dims=(y_dim, x_dim)
        )
        ds_subset['longitude'] = xr.DataArray(
            np.round(ds_subset.longitude.values, decimal_precision), 
            dims=(y_dim, x_dim)
        )

        # ==========================================================
        # STEP 4: APPLY MATRIX COARSENING
        # ==========================================================
        if pool_n > 1:
            # Verify dataset slice dimensions are large enough to satisfy pool stride factors
            if ds_subset.sizes[y_dim] >= pool_n and ds_subset.sizes[x_dim] >= pool_n:
                coarsen_dict = {y_dim: pool_n, x_dim: pool_n}
                # Trim drops trailing edge fractions that can't cleanly fit inside a pool_n block block
                ds_coarsened = ds_subset.coarsen(coarsen_dict, boundary="trim").mean()
            else:
                print(f"[-] Warning: Masked bounding box footprint is smaller than pooling block target ({pool_n}). Skipping coarsening.")
                ds_coarsened = ds_subset
        else:
            ds_coarsened = ds_subset

        return ds_coarsened

    def _to_cycle_hr(self, fxx):
        return "%03d" % (int(fxx),)
    
if __name__ == "__main__":
    from analysis.mapping.config import lats, lons

    client = RRFSClient()
    this_time = "2026-06-20 00:00"
    lat = lats #[32, 34.05]
    lon = lons # [-120, -118.24]
    fxx = 2
    ds = client.query(this_time, lat, lon, fxx)
    import ipdb; ipdb.set_trace()