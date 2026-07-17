from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import re

import numpy as np
import os
import pandas as pd
import requests
import xarray as xr

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely import points, contains
import shutil

from utils.io_utils import buffer_polygon_meters 
from utils.constants import CACHE_BASE_DIR, RAVE_CACHE
from utils.rio_utils import open_netcdf_safe_cached
from data.clients.base_client import BaseClient

Geom = Union[Polygon, MultiPolygon]

cached_file_lb = pd.Timestamp("07-01-2019")
cached_file_ub = pd.Timestamp("09-30-2024")
cached_file_base = RAVE_CACHE # "/home/jaredgoldman/data/RAVE", "/u/scratch/j/jgoldman/data/RAVE"

@dataclass
class RAVEClient(BaseClient):
    """
    RAVE hourly emissions (3km) client:
      - lists month directories for a date range
      - downloads needed NetCDF files
      - subsets variables and clips to polygon
      - returns xr.Dataset concatenated over time
    """
    base_url: str = "http://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km"
    sampling_freq: str = "2h"
    timeout_s: int = 120

    # common coordinate name fallbacks
    lat_names: Tuple[str, ...] = ("lat", "latitude", "LAT", "Latitude", "y", "grid_latt")
    lon_names: Tuple[str, ...] = ("lon", "longitude", "LON", "Longitude", "x", "grid_lont")
    time_names: Tuple[str, ...] = ("time", "valid_time", "datetime", "date")

    _fname_re: re.Pattern = re.compile(
        r"s(\d{4})(\d{2})(\d{2})", re.VERBOSE
    )

    def __init__(self, *args, **kwargs):
        super().__init__(cache_dir = os.path.join(CACHE_BASE_DIR,'rave'), **kwargs)

    # ----------------------------
    # Public API
    # ----------------------------
        
    def _query(
        self,
        polygon: Geom,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        drop_outside: bool = True,
        bbox_first: bool = True,
        prefer_latest_vr: bool = True,
        keep_attrs: bool = True,
    ) -> xr.Dataset:
        """
        Parameters
        ----------
        polygon : shapely Polygon/MultiPolygon in EPSG:4326 (lon/lat)
        start, end : date-like (inclusive window)
        variables : list of data_vars to keep (None -> keep all)
        drop_outside : mask outside polygon and drop pixels when possible
        bbox_first : bbox slice before polygon mask (usually much faster)
        prefer_latest_vr : if multiple files overlap same start time, keep highest v#r#
        keep_attrs : keep global attrs during concat

        Returns
        -------
        xr.Dataset concatenated on "time"
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        # Collect candidate file URLs by scanning only the needed YYYY/MM dirs
        is_cached = start_ts >= cached_file_lb and end_ts <= cached_file_ub
        polygon = buffer_polygon_meters(polygon, resolution_m=3000, factor = 1.0)

        if is_cached:
            files = self._get_cached_fnames(start_ts, end_ts) 
        else:
            files = self._collect_files_for_range(start_ts, end_ts)

        if not files:
            raise FileNotFoundError("No RAVE files found overlapping the requested time window.")

        dsets: List[xr.Dataset] = []
        ds = None
        for meta in files:
            if is_cached:
                local = meta["path"]
                local_fname = local.split(os.path.sep)[-1]
                cached_fname = os.path.join(self.save_dir, local_fname)
                shutil.copy(local, cached_fname)
                self.cached_files.append(cached_fname)
                ds = open_netcdf_safe_cached(self.base_url, cached_fname, self._session)
            else:
                local = self._download(url = meta['url'])
                ds = open_netcdf_safe_cached(meta['url'], local, self._session)
                self.cached_files.append(local)
                
            # variable subset
            if variables is not None:
                missing = [v for v in variables if v not in ds.data_vars]
                if missing:
                    raise KeyError(f"Missing variables in {local.name}: {missing}")
                ds = ds[variables]

            # ensure time dim exists/standardized
            ds = self._ensure_time(ds, pd.Timestamp(year = meta['year'], month = meta['month'], day = meta['day']))

            # make sure longitude convention matches polygon if needed
            ds = self._maybe_wrap_longitudes_to_180(ds, polygon)

            # clip
            ds = self._subset_to_polygon(ds, polygon, drop_outside=drop_outside, bbox_first=bbox_first)
            if 'FRP_SD' in ds.data_vars:
                ds['FRP_SD'] = ds['FRP_SD'] ** 2
                ds = ds.sum(dim = ('grid_yt', 'grid_xt'))
                ds['FRP_SD'] = np.sqrt(ds['FRP_SD'])
            else:
                ds = ds.sum(dim = ('grid_yt', 'grid_xt'))
            dsets.append(ds)

        if not dsets:
            raise RuntimeError("No datasets remained after subsetting.")

        ds_out = xr.merge(dsets).load()
        return ds_out

    # ----------------------------
    # Listing + parsing
    # ----------------------------
    def _month_dir_url(self, year: int, month: int) -> str:
        return f"{self.base_url.rstrip('/')}/{year:04d}/{month:02d}/"

    def _list_month_filenames(self, year: int, month: int, hrs: pd.DatetimeIndex) -> List[str]:
        """
        OSPO pub directories are simple HTML indexes; grab hrefs ending in .nc
        """
        url = self._month_dir_url(year, month)
        r = self._session.get(url, timeout=self.timeout_s)
        names = re.findall(r'href="([^"]+\.nc)"', r.text, flags=re.IGNORECASE)
        r.raise_for_status() 
        filtered_names = []
        for hr in hrs:
            if hr.year != year or hr.month != month:
                continue
            filtered_names.extend([{ 'url' : self._month_dir_url(year, month) + fname, 
                                'year': year,
                                'month' : month,
                                'day' : hr.day} \
                                    for fname in names if f"s{year:04d}{month:02d}{hr.day:02d}{hr.hour:02d}" in fname])
        return filtered_names
    
    def _get_cached_fnames(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict]:
        date_range = pd.date_range(start_ts, end_ts, freq='d')
        fnames = []
        for date in date_range:
            year, month, day = int(date.year), int(date.month), int(date.day)
            this_dir = os.path.join(cached_file_base, str(year))
            fnames.extend([fname for fname in os.listdir(this_dir) if f"s{year:04d}{month:02d}{day:02d}" in fname])
        
        out = [{ 'path' : os.path.join(this_dir, fname), 
                    'year': year,
                    'month' : month,
                    'day' : day} for fname in fnames]
        return out

    def _collect_files_for_range(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict]:
        # consider months overlapping the window
        hrs = pd.date_range(start_ts, end_ts, freq=self.sampling_freq)
        months = np.unique([hr.month for hr in hrs])
        years = np.unique([hr.year for hr in hrs])
        fnames = []
        for year in years:
            for month in months:
                try:
                    fnames.extend(self._list_month_filenames(year, month, hrs))
                except requests.HTTPError:
                    # month directory may not exist (e.g., outside archive range)
                    continue
                
        return fnames

    # ----------------------------
    # Dataset helpers
    # ----------------------------
    def _ensure_time(self, ds: xr.Dataset, start_time: pd.Timestamp) -> xr.Dataset:
        for tname in self.time_names:
            if tname in ds.dims:
                if tname != "time":
                    ds = ds.rename({tname: "time"})
                return ds

        # no time dim -> add singleton time from filename start timestamp
        return ds.expand_dims(time=[np.datetime64(start_time.to_datetime64())])

    def _infer_lat_lon_names(self, ds: xr.Dataset) -> Tuple[str, str]:
        lat = next((n for n in self.lat_names if n in ds.coords or n in ds.variables), None)
        lon = next((n for n in self.lon_names if n in ds.coords or n in ds.variables), None)
        if lat is None or lon is None:
            raise KeyError(
                f"Could not infer lat/lon names. "
                f"Looked for lat in {self.lat_names} and lon in {self.lon_names}. "
                f"Found coords={list(ds.coords)} vars={list(ds.variables)[:30]}"
            )
        return lat, lon

    def _maybe_wrap_longitudes_to_180(self, ds: xr.Dataset, polygon: Geom) -> xr.Dataset:
        """
        If dataset lon is 0..360 but polygon likely uses -180..180, remap to (-180, 180] and sort.
        """
        lat_name, lon_name = self._infer_lat_lon_names(ds)
        lon = ds[lon_name]

        # decide based on polygon bounds
        poly_minx, _, poly_maxx, _ = polygon.bounds
        polygon_is_180 = (poly_minx < 0) or (poly_maxx <= 180)

        lonv = lon.values
        ds_is_360 = (np.nanmin(lonv) >= 0) and (np.nanmax(lonv) > 180)
        if lon.ndim == 1:
            if polygon_is_180 and ds_is_360:
                newlon = ((lonv + 180) % 360) - 180
                ds = ds.assign_coords({lon_name: newlon}).sortby(lon_name)
        elif lon.ndim == 2:
            if polygon_is_180 and ds_is_360:
                newlon = ((lonv + 180) % 360) - 180
                ds = ds.assign_coords({lon_name: (lon.dims, newlon)})

        return ds

    def _subset_to_polygon(
        self,
        ds: xr.Dataset,
        polygon: Geom,
        *,
        drop_outside: bool,
        bbox_first: bool,
    ) -> xr.Dataset:
        lat_name, lon_name = self._infer_lat_lon_names(ds)
        lat = ds[lat_name]
        lon = ds[lon_name]
        
        # Case A: 1D lat/lon
        if lat.ndim == 1 and lon.ndim == 1:
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            lon2d, lat2d = np.meshgrid(lons, lats)
            mask = self._polygon_mask(lat2d, lon2d, polygon)
            mask_da = xr.DataArray(mask, dims=(lat_name, lon_name))
            return ds.where(mask_da, drop=True)

        # Case B: 2D lat/lon coords (y,x)
        if lat.ndim == 2 and lon.ndim == 2:
            mask = self._polygon_mask(lat.values, lon.values, polygon)
            mask_da = xr.DataArray(mask, dims=lat.dims)
            return ds.where(mask_da, drop=True)

        raise ValueError(f"Unsupported lat/lon shapes: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")

    @staticmethod
    def _polygon_mask(lat2d: np.ndarray, lon2d: np.ndarray, polygon: Geom) -> np.ndarray:
        """
        Boolean mask for points inside polygon, using cell-center point-in-polygon.

        If shapely>=2 is installed, this will use vectorized contains; otherwise falls back to loop.
        """
        try:
            pts = points(lon2d, lat2d)
            return contains(polygon, pts)
        except Exception:
            pass

        # fallback: prepared geometry + Python loop (slower but dependable)
        mask = np.zeros(lat2d.shape, dtype=bool)
        for j in range(lat2d.shape[0]):
            for i in range(lat2d.shape[1]):
                mask[j, i] = polygon.contains( Point(lon2d[j, i], lat2d[j, i]))  
        return mask
    
if __name__ == "__main__":
    from shapely.geometry import box
    from utils.constants import CP_POLY_PATH
    import geopandas as gpd

    poly = box(-120.0, 36.0, -119.75, 36.25)  # CA-ish box
    # gdf = gpd.read_file(CP_POLY_PATH)
    # poly = gdf[gdf.cp == 36].geometry.values[0]
    client = RAVEClient()
    ds = client.query(
        polygon=poly,
        start="2025-07-30 00:00",
        end="2025-07-30 12:00",
        variables= ["FRP_MEAN", "FRP_SD"],
    )
    import ipdb; ipdb.set_trace()
    print(ds)
    print(ds.time.values[:3])

    # can_poly = box(-105.5, 50.5, -104.0, 51.5)
    # ds = client.query(
    #     polygon=can_poly,
    #     start="2010-10-01 00:00",
    #     end="2019-10-01 12:00",
    #     variables=["PM25", "FRP_MEAN", "FRE"],  # replace with actual names in your files
    # )

    # print("can", ds)
    # print("can", ds.time.values[:3])