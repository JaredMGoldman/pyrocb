from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import shutil
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep

Geom = Union[Polygon, MultiPolygon]


@dataclass
class NCARFMCClient:
    """
    Download NCAR GDEX FMC netCDF files (fmc_YYYYMMDD_20Z.nc) for a date range,
    subset to a polygon, and return an xarray.Dataset.

    Assumptions:
      - polygon is lon/lat (EPSG:4326)
      - files are daily with timestamp 20Z in name
    """
    base_url: str = "https://osdf-director.osg-htc.org/ncar/gdex/d583133/fmc_nc"
    cache_dir: Union[str, Path] = "gdex_cache/fmc_nc"
    timeout_s: int = 120

    # common coordinate name fallbacks
    lat_names: Tuple[str, ...] = ("lat", "latitude", "y", "gridlat", "LAT", "Latitude", "XLAT_M")
    lon_names: Tuple[str, ...] = ("lon", "longitude", "x", "gridlon", "LON", "Longitude", "XLONG_M")
    time_names: Tuple[str, ...] = ("Time", "time", "valid_time", "datetime", "date")

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    # -----------------------
    # Public API
    # -----------------------
    def query(
        self,
        polygon: Geom,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        *,
        concat_dim: str = "time",
        drop_outside: bool = True,
        bbox_first: bool = True,
    ) -> xr.Dataset:
        """
        Parameters
        ----------
        polygon : shapely Polygon/MultiPolygon in lon/lat
        start, end : inclusive date range
        variables : subset of data_vars to keep (None -> keep all)
        concat_dim : name of time dimension to use
        drop_outside : mask outside polygon and drop empty rows/cols where possible
        bbox_first : slice by polygon bounds before masking (much faster)

        Returns
        -------
        xr.Dataset concatenated over time with requested variables and clipped spatially.
        """
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        dates = pd.date_range(start_ts, end_ts, freq="D")

        # Build all URLs once (“look ahead”), then download/open
        local_files = [self._download_one(d) for d in dates]

        dsets = []
        for d, fp in zip(dates, local_files):
            ds = xr.open_dataset(fp)
            if 'Times' in ds.data_vars:
                ds = ds.drop_vars('Times')
            for var in ['latitude', 'longitude']:
                if var in ds.data_vars:
                    ds = ds.set_coords([var])

            # Optional variable subset (only data_vars)
            if variables is not None:
                missing = [v for v in variables if v not in ds.data_vars]
                if missing:
                    raise KeyError(f"Requested variables not found in {Path(fp).name}: {missing}")
                ds = ds[variables]

            # Ensure we have a time dimension
            ds = self._ensure_time(ds, d, concat_dim=concat_dim)

            # Spatial subset to polygon
            ds = self._subset_to_polygon(
                ds,
                polygon,
                drop_outside=drop_outside,
                bbox_first=bbox_first,
            )

            dsets.append(ds)

        if not dsets:
            raise FileNotFoundError("No files loaded for requested range.")

        out = xr.concat(dsets, dim=concat_dim)
        return out

    # -----------------------
    # Download logic
    # -----------------------
    @staticmethod
    def _fname(d: pd.Timestamp) -> str:
        return f"fmc_{d.strftime('%Y%m%d')}_20Z.nc"

    def _url(self, d: pd.Timestamp) -> str:
        return f"{self.base_url.rstrip('/')}/{self._fname(d)}"

    def _local_path(self, d: pd.Timestamp) -> Path:
        p = self.cache_dir / str(d.year) / self._fname(d)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _download_one(self, d: pd.Timestamp, chunk_size: int = 1 << 20) -> Path:
        """
        Download one FMC file for date `d`.

        - For dates <= 2023-08-09 (DOY 221, inclusive), uses the original GDEX endpoint:
            .../fmc_YYYYMMDD_20Z.nc
        - For dates >= 2023-08-10 (DOY 222 and later), uses THREDDS endpoint:
            https://tds.rap.ucar.edu/thredds/fileServer/wsap/fmc/conus2250m/conus_YYYYDDD.nc.gz
        and decompresses to a .nc file.

        Returns local path to the decompressed .nc file.
        """

        # Normalize to date (ignore time)
        d = pd.Timestamp(d).normalize()

        # Switch point: after 221st day of 2023 => DOY 222+
        switch = pd.Timestamp("2023-03-26")  # 2023 DOY 86
        gap = pd.Timestamp("2021-10-26") # last available NCAR product
        use_thredds = d >= switch
        if not use_thredds and d>= gap:
            raise ValueError(f"No NCAR Fuel Moisture Content data available from 10-26-2021 -> 03-26-2023. {d} falls in this range...")

        # Build URL + local paths
        if use_thredds:
            year = d.year
            doy = f"{int(d.strftime('%j')):03d}"
            url = f"https://tds.rap.ucar.edu/thredds/fileServer/wsap/fmc/conus2250m/conus_{year}{doy}.nc"
        else:
            url = self._url(d)
        out = self._local_path(d)

        if out.exists() and out.stat().st_size > 0:
            return out

        with self._session.get(url, stream=True, timeout=self.timeout_s) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        return out

    # -----------------------
    # Dataset helpers
    # -----------------------
    def _ensure_time(self, ds: xr.Dataset, d: pd.Timestamp, concat_dim: str) -> xr.Dataset:
        """
        If the dataset already has a time-like dim, keep it.
        Otherwise add a singleton time dimension based on filename date (at 20Z).
        """
        # If it has any known time dim, rename to concat_dim if needed
        for tname in self.time_names:
            if tname in ds.dims:
                if tname != concat_dim:
                    return ds.rename({tname: concat_dim})
                return ds

        # else create time coord (use 20Z)
        t = np.datetime64((d + pd.Timedelta(hours=20)).to_datetime64())
        return ds.expand_dims({concat_dim: [t]})

    def _infer_lat_lon_names(self, ds: xr.Dataset) -> Tuple[str, str]:
        lat = next((n for n in self.lat_names if n in ds.coords or n in ds.data_vars), None)
        lon = next((n for n in self.lon_names if n in ds.coords or n in ds.data_vars), None)

        if lat is None or lon is None:
            raise KeyError(
                f"Could not infer lat/lon names. "
                f"Looked for lat in {self.lat_names} and lon in {self.lon_names}. "
                f"Found coords={list(ds.coords)}"
            )
        return lat, lon

    def _subset_to_polygon(
        self,
        ds: xr.Dataset,
        polygon: Geom,
        *,
        drop_outside: bool,
        bbox_first: bool,
    ) -> xr.Dataset:
        """
        Works for common cases:
          - 1D lat/lon dims (lat(y), lon(x))  -> bbox slice + optional mask
          - 2D lat/lon coords on (y,x) grid   -> bbox mask + polygon mask
        """
        lat_name, lon_name = self._infer_lat_lon_names(ds)
        lat = ds[lat_name].squeeze()
        lon = ds[lon_name].squeeze()

        minx, miny, maxx, maxy = polygon.bounds

        # Case A: 1D lat/lon
        if lat.ndim == 1 and lon.ndim == 1:
            # bbox slice first (fast)
            if bbox_first:
                # handle ascending/descending coords robustly
                lat_slice = slice(miny, maxy) if float(lat[0]) < float(lat[-1]) else slice(maxy, miny)
                lon_slice = slice(minx, maxx) if float(lon[0]) < float(lon[-1]) else slice(maxx, minx)

                ds = ds.sel({lat_name: lat_slice, lon_name: lon_slice})

            if not drop_outside:
                return ds

            # polygon mask on 2D mesh of coords (only for the subset)
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            lon2d, lat2d = np.meshgrid(lons, lats)
            mask = self._polygon_mask(lat2d, lon2d, polygon)

            # apply mask across all vars (broadcast to dims)
            mask_da = xr.DataArray(mask, dims=(lat_name, lon_name))
            return ds.where(mask_da, drop=True)

        # Case B: 2D lat/lon (common for model grids)
        if lat.ndim == 2 and lon.ndim == 2:
            y_dim, x_dim = lat.dims

            if bbox_first:
                bbox_mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
                keep_y = bbox_mask.any(dim=x_dim)
                keep_x = bbox_mask.any(dim=y_dim)

                ds = ds.isel({y_dim: np.where(keep_y.values)[0], x_dim: np.where(keep_x.values)[0]})
                lat = ds[lat_name].squeeze()
                lon = ds[lon_name].squeeze()

            if not drop_outside:
                return ds
            
            mask = self._polygon_mask(lat.values, lon.values, polygon)
            mask_da = xr.DataArray(mask, dims=lat.dims)
            return ds.where(mask_da, drop=True)
        
        if lat.ndim == 3 and lon.ndim == 3:
            time, y_dim, x_dim = lat.dims

            if bbox_first:
                bbox_mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
                keep_y = bbox_mask.any(dim=x_dim)
                keep_x = bbox_mask.any(dim=y_dim)

                ds = ds.isel({y_dim: np.where(keep_y.values)[0], x_dim: np.where(keep_x.values)[0]})
                lat = ds[lat_name]
                lon = ds[lon_name]

            if not drop_outside:
                return ds

            mask = self._polygon_mask(lat.values, lon.values, polygon)
            mask_da = xr.DataArray(mask, dims=lat.dims)
            return ds.where(mask_da, drop=True)

        raise ValueError(
            f"Unsupported lat/lon shapes: {lat_name}.ndim={lat.ndim}, {lon_name}.ndim={lon.ndim}"
        )

    @staticmethod
    def _polygon_mask(lat2d: np.ndarray, lon2d: np.ndarray, polygon: Geom) -> np.ndarray:
        """
        Boolean mask for points inside polygon, using Point-in-polygon on cell centers.
        """
        ppoly = prep(polygon)
        mask = np.zeros(lat2d.shape, dtype=bool)

        for j in range(lat2d.shape[0]):
            mask[j, :] = [ppoly.contains(Point(float(lon2d[j, i]), float(lat2d[j, i])))
                          for i in range(lat2d.shape[1])]
        return mask
    
if __name__ == "__main__":
    from shapely.geometry import box

    poly = box(-119.05, 33.60, -117.50, 34.85)  # LA bbox

    client = NCARFMCClient(
        base_url="https://osdf-director.osg-htc.org/ncar/gdex/d583133/fmc_nc",
        cache_dir="data/gdex_fmc",
    )

    ds = client.query(
        polygon=poly,
        start="2024-08-01",
        end="2024-08-07",
        variables=None,   # <-- replace with the real variable names in the netCDF
    )

    print(ds)