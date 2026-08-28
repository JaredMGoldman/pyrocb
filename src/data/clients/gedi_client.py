from datetime import timedelta
import numpy as np
import os
import pandas as pd
import earthaccess
import xarray as xr

from data.clients.base_client import BaseClient
from utils.constants import CACHE_BASE_DIR, CLIENTS_DIR, EARTHACCESS_KEY_NAME

class GEDIClient(BaseClient):
    def __init__(self, *args, **kwargs):
        self.map_key_path: str  = os.path.join(CLIENTS_DIR,
                                  EARTHACCESS_KEY_NAME)
    
        self.timeout: int = 60
        self.max_retries: int = 5
        self.backoff_factor: float = 0.8
        self.map_key: str = ""
        self.dataset_short_name = "GEDI_L4A_AGB_Density_V2_1_2056"
        self.time_lb = pd.Timestamp("2019-04-17")
        self.time_ub = pd.Timestamp("2025-07-09")
        if os.path.exists(self.map_key_path):
                with open(self.map_key_path, 'r') as f:
                    map_key = f.read()
                    os.environ['EARTHDATA_TOKEN'] = map_key
        super().__init__(*args, cache_dir=os.path.join(CACHE_BASE_DIR, 'gedi'), **kwargs)
    
    def query_out_of_range(
        self, polygon, start, end, 
        variables = ['agbd'], download=False,
        max_days_back = 60, step_days = 3
    ):
        starts = []
        ends = []
        time_del = pd.Timedelta(365, 'D')
        end_val = end
        start_val = start
        if start < self.time_lb: 
            while end_val + time_del < self.time_ub:
                start_val += time_del 
                end_val += time_del
                if start_val > self.time_lb:
                    starts.append(start_val)
                    ends.append(end_val)
        elif end > self.time_ub:
            while start_val - time_del > self.time_lb:
                start_val -= time_del 
                end_val -= time_del 
                if end_val < self.time_ub:
                    starts.append(start_val)
                    ends.append(end_val)
        all_dses = []
        for this_start, this_end in zip(starts, ends):
            try:
                ds = self.__query(polygon, this_start, this_end, variables, max_days_back, step_days, download) 
                all_dses.append(ds)
            except Exception as e:
                print(e)
                continue
        if len(all_dses) == 0:
            raise RuntimeError(f"[{self.__class__.__name__}] Failed to find valid data in previous {max_days_back} days for {start}")
        merged_ds = xr.concat(all_dses, dim = 'time')
        mean_ds = merged_ds.mean(dim = 'time')
        midpoint_timestamp = pd.to_datetime(start) + (
                pd.to_datetime(end) - pd.to_datetime(start)
            ) / 2
        out_ds = mean_ds.expand_dims(time=[midpoint_timestamp])
        return out_ds

    def _query(
        self, polygon, start, end, 
        variables = ['agbd'], download=False,
        max_days_back = 60, step_days = 3
    ):
        """Searches and accesses NASA GEDI L4A V2.1 Aboveground Biomass Density data

        using the earthaccess library based on spatial and temporal parameters.

        Parameters:
        -----------
        bbox : tuple
            Spatial bounding box formatted as (lower_left_lon, lower_left_lat,
            upper_right_lon, upper_right_lat).
            Example: (-125.0, 32.0, -114.0, 42.0) for California.
        start_date : str
            Start date of temporal extent formatted as 'YYYY-MM-DD'.
        end_date : str
            End date of temporal extent formatted as 'YYYY-MM-DD'.
        download : bool, optional
            If True, downloads the HDF5 granules to disk. If False, opens them
            natively as lazy file streams.
        output_dir : str, optional
            The local directory where files will be saved if download=True.

        Returns:
        --------
        list
            A list of open file streams (if download=False) or a list of local
            filepath strings (if download=True).
        """
        print("Authenticating with NASA Earthdata...")
        auth = earthaccess.login(strategy="environment")

        if not auth.authenticated:
            print("Interactive login required...")
            auth = earthaccess.login(strategy="interactive")

        
        print(f"Searching for granules in {self.dataset_short_name}...")
        print(f"Spatial Extent: {polygon}")
        print(f"Temporal Extent: {start} to {end}")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt < self.time_lb or end_dt > self.time_ub:
            return self.query_out_of_range(polygon, start_dt, end_dt, variables, download, max_days_back, step_days)
        return self.__query(polygon, start_dt, end_dt, variables, max_days_back, step_days, download)

    def __query(self, polygon, start_dt, end_dt, variables, max_days_back, step_days, download):
        accumulated_shift = 0
        granules = []
        while len(granules) == 0 and accumulated_shift <= max_days_back:
            # Calculate the new shifted time window
            current_start = (start_dt - timedelta(days=accumulated_shift)).strftime('%Y-%m-%d')
            current_end = (end_dt - timedelta(days=accumulated_shift)).strftime('%Y-%m-%d')

            granules = earthaccess.search_data(
                short_name=self.dataset_short_name,
                bounding_box=polygon,
                temporal=(current_start, current_end),
            )

            if len(granules) == 0:
                current_start = (start_dt + timedelta(days=accumulated_shift)).strftime('%Y-%m-%d')
                current_end = (end_dt + timedelta(days=accumulated_shift)).strftime('%Y-%m-%d')
                
                granules = earthaccess.search_data(
                    short_name=self.dataset_short_name,
                    bounding_box=polygon,
                    temporal=(current_start, current_end),
                )
            
            accumulated_shift += step_days
        
        if len(granules) == 0:
            raise RuntimeError(f"[{self.__class__.__name__}] Failed to find valid data in previous {max_days_back} days from {start_dt}")

        if download:
            # Save files locally
            print(f"Downloading granules to local cache: {self.save_dir}")
            files = earthaccess.download(granules, local_path=self.save_dir)
        else:
            # Stream files lazily without downloading (highly recommended for large spatial cuts)
            print("Opening streaming file links natively from the NASA cloud...")
            files = earthaccess.open(granules)
        dses = []
        for file in files:
            ds = xr.open_dataset(file, group="BEAM0000")[variables]
            ds = self._format_time_dim(ds, features = variables).drop(variables)

            dses.append(ds)
            
        merged_ds = xr.concat(dses, dim='time')
        return merged_ds

    def _format_time_dim(self, ds, features=["agbd"], resolution_deg = 0.01):
        """Restructures a 1D shot dataset by adding an explicit time dimension,

        promoting lat and lon to official coordinates alongside shot and time.
        """
        if "phony_dim_0" in ds.dims:
            ds = ds.rename_dims({"phony_dim_0": "shot"})
        
        epoch = pd.to_datetime("2018-01-01 00:00:00")

        time_values = epoch + pd.to_timedelta(ds["delta_time"].values, unit="s")
        granule_timestamp = pd.Series(time_values).median()

        lat_name = "lat_lowestmode" if "lat_lowestmode" in ds else "lat"
        lon_name = "lon_lowestmode" if "lon_lowestmode" in ds else "lon"

        lats = ds[lat_name].values
        lons = ds[lon_name].values
        shot_ids = ds["shot"].values

        # Clean out any spatial or missing data flags
        # GEDI uses -9999 for data drops; we filter these out to protect the grid average
        valid_mask = (~np.isnan(lats)) & (~np.isnan(lons))
        if len(valid_mask) == 0 or not np.any(valid_mask):
            return xr.Dataset()

        lat_min, lat_max = np.floor(lats[valid_mask].min()), np.ceil(
            lats[valid_mask].max()
        )
        lon_min, lon_max = np.floor(lons[valid_mask].min()), np.ceil(
            lons[valid_mask].max()
        )

        lat_axis = np.arange(lat_min, lat_max, resolution_deg, dtype=np.float32)
        lon_axis = np.arange(lon_min, lon_max, resolution_deg, dtype=np.float32)

        # Initialize a clean Pandas DataFrame for fast, structured 2D grouping
        df = pd.DataFrame(
            {"lat": lats[valid_mask], "lon": lons[valid_mask], "shot": shot_ids[valid_mask]}
        )

        # Categorize coordinate points into their respective spatial index bins
        df["lat_bin"] = pd.cut(df["lat"], bins=lat_axis, labels=lat_axis[:-1])
        df["lon_bin"] = pd.cut(df["lon"], bins=lon_axis, labels=lon_axis[:-1])

        grid_vars = {}

        for f in features:
            if f not in ds:
                continue

            raw_feature = ds[f].values[valid_mask]
            # Temporarily drop missing data markers so they don't corrupt the mean calculation
            raw_feature_clean = np.where(raw_feature == -9999, np.nan, raw_feature)

            df[f] = raw_feature_clean

            # Group by 2D coordinates and average out multiple laser impacts hitting the same pixel
            pivoted = (
                df.groupby(["lat_bin", "lon_bin"], observed=False)[f]
                .mean()
                .unstack(fill_value=np.nan)
            )

            # Re-index the pivot table matrix to guarantee it matches our full axis limits
            pivoted = pivoted.reindex(index=lat_axis[:-1], columns=lon_axis[:-1])

            # Expand the dimensions by inserting a new front axis for the Time dimension
            # Converts a 2D grid shape (lat, lon) into a 3D matrix shape (1, lat, lon)
            grid_vars[f] = (["time", "lat", "lon"], pivoted.values[np.newaxis, :])
            grid_vars[f"total_{f}"] = (["time"], [np.nansum(raw_feature_clean)])
            grid_vars[f"mean_{f}"] = (["time"], [np.nanmean(raw_feature_clean)])

        # We pass the shot list vector as a non-dimensional coordinate linked to nothing
        # so your pipeline can still audit the exact original sensor shot footprints
        grid_ds = xr.Dataset(
            data_vars=grid_vars,
            coords={
                "time": [granule_timestamp],
                "lat": lat_axis[:-1],
                "lon": lon_axis[:-1]
            },
        )

        # Strip empty spatial rows/columns outside the true satellite flight path margins
        grid_ds = grid_ds.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

        return grid_ds


if __name__ == "__main__":
    # Define bounds for an active wildfire zone example (e.g., California region)
    california_bbox = (-115.4, 41.5, -114.1, 42.0)
    time_start = "2026-04-01"
    time_end = "2026-04-04"
    client = GEDIClient()

    # Fetch the open data streams (lazy loading)
    ds = client.query(
        polygon=california_bbox, 
        start=time_start, 
        end=time_end, 
        download=False
    )
    import ipdb; ipdb.set_trace()

