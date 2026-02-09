# https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00994
# until 2024

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re

import numpy as np
import pandas as pd
import requests
import xarray as xr

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep


Geom = Union[Polygon, MultiPolygon]


@dataclass
class SMOPSClient:
    """
    Download NOAA STAR SMOPS CDR NetCDF files for a date range, subset to a polygon,
    and return an xarray.Dataset.

    Files live under:
      https://www.star.nesdis.noaa.gov/pub/smcd/lvb/LAT/SMDM/SMOPScdr/YYYY/
    File naming (example):
      SMOPS-CDR_v1r0_sYYYYMMDD.*.nc
    """
    base_url: str = "https://www.star.nesdis.noaa.gov/pub/smcd/lvb/LAT/SMDM/SMOPScdr"
    cache_dir: Union[str, Path] = "star_cache/smops_cdr"
    timeout_s: int = 120

    # common coordinate name fallbacks
    lat_names: Tuple[str, ...] = ("lat", "latitude", "LAT", "Latitude", "y")
    lon_names: Tuple[str, ...] = ("lon", "longitude", "LON", "Longitude", "x")
    time_names: Tuple[str, ...] = ("time", "valid_time", "datetime", "date")

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    # -------------------------
    # Public API
    # -------------------------
    def query(
        self,
        polygon: Geom,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        *,
        drop_outside: bool = True,
        bbox_first: bool = True,
        allow_multiple_per_day: bool = False,
    ) -> xr.Dataset:
        """
        Parameters
        ----------
        polygon : shapely Polygon/MultiPolygon in lon/lat (EPSG:4326)
        start, end : date-like (inclusive)
        variables : list of data_vars to keep (None -> all data_vars)
        drop_outside : apply polygon mask and drop outside pixels
        bbox_first : bbox slice before polygon mask (much faster)
        allow_multiple_per_day : if True, keep all files matching a day; if False, pick 1 per day.

        Returns
        -------
        xr.Dataset with dims (time, y, x) or (time, lat, lon) depending on file.
        """
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        # List files year-by-year once, then filter by date span (“look ahead”)
        urls = self._collect_urls_for_range(start_ts, end_ts, allow_multiple_per_day=allow_multiple_per_day)
        if not urls:
            raise FileNotFoundError("No SMOPS CDR files found for requested date range.")

        # Download everything needed
        local_files = [self._download_url(url) for url in urls]

        # Open and subset
        dsets: List[xr.Dataset] = []
        for fp in local_files:
            ds = xr.open_dataset(fp)

            # Variable subset
            if variables is not None:
                missing = [v for v in variables if v not in ds.data_vars]
                if missing:
                    raise KeyError(f"Requested variables not found in {Path(fp).name}: {missing}")
                ds = ds[variables]

            # Ensure/standardize time
            file_date = self._date_from_filename(Path(fp).name)
            ds = self._ensure_time(ds, file_date)

            # Normalize lon range if needed for polygon filtering
            ds = self._maybe_wrap_longitudes_to_180(ds)

            # Spatial subset
            ds = self._subset_to_polygon(ds, polygon, drop_outside=drop_outside, bbox_first=bbox_first)

            dsets.append(ds)

        if not dsets:
            raise FileNotFoundError("No datasets loaded after filtering/subsetting.")

        out = xr.concat(dsets, dim="time")
        return out

    # -------------------------
    # Directory listing + URL selection
    # -------------------------
    def _year_dir_url(self, year: int) -> str:
        return f"{self.base_url.rstrip('/')}/{year}/"

    def _list_year_filenames(self, year: int, md_list) -> List[str]:
        """
        STAR directory pages are simple HTML listings. We'll regex out hrefs ending in .nc
        """
        # return 
        url = self._year_dir_url(year)
        r = self._session.get(url, timeout=self.timeout_s)
        r.raise_for_status()

        # Extract linked .nc files
        names = re.findall(r'href="([^"]+\.nc)"', r.text, flags=re.IGNORECASE)
        out_names = []
        for month, day in md_list:
            out_names.extend([f"{url}/{name}" for name in names if f"s{year:04d}{month:02d}{day:02d}0000000" in name])

        return sorted(set(out_names))

    @staticmethod
    def _date_from_filename(fname: str) -> pd.Timestamp:
        """
        Extract sYYYYMMDD from: SMOPS-CDR_v1r0_sYYYYMMDD.*.nc
        """
        m = re.search(r"_s(\d{8})0000000", fname)
        if not m:
            raise ValueError(f"Could not parse date from filename: {fname}")
        return pd.Timestamp(m.group(1))

    def _collect_urls_for_range(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        *,
        allow_multiple_per_day: bool,
    ) -> List[str]:
        ymds = {}
        for date in pd.date_range(start_ts,end_ts,freq ='D'):
            if date.year in ymds.keys():
                ymds[date.year].append((date.month, date.day))
            else:
                ymds[date.year] = [(date.month, date.day)]
        keep: List[Tuple[pd.Timestamp, str]] = []
        for y, md_list in ymds.items():
            # could consolidate by combining same years
            names = self._list_year_filenames(y,md_list)
            keep.extend(names)
        
        return keep

    # -------------------------
    # Download + cache
    # -------------------------
    def _download_url(self, url: str, chunk_size: int = 1 << 20) -> Path:
        fname = url.split("/")[-1]
        # cache by year inferred from fname date
        out = self.cache_dir / fname
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.exists() and out.stat().st_size > 0:
            return out

        with self._session.get(url, stream=True, timeout=self.timeout_s) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return out

    # -------------------------
    # Dataset helpers
    # -------------------------
    def _ensure_time(self, ds: xr.Dataset, d: pd.Timestamp) -> xr.Dataset:
        # If a time-like dim exists, try to rename to "time"
        for tname in self.time_names:
            if tname in ds.dims:
                if tname != "time":
                    ds = ds.rename({tname: "time"})
                return ds

        # else add singleton time coord from filename date
        return ds.expand_dims(time=[np.datetime64(d.to_datetime64())])

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

    def _maybe_wrap_longitudes_to_180(self, ds: xr.Dataset) -> xr.Dataset:
        """
        If longitudes look like 0..360 and polygon likely uses -180..180,
        map lon to (-180, 180] and sort.
        """
        lat_name, lon_name = self._infer_lat_lon_names(ds)
        lon = ds[lon_name]

        if lon.ndim == 1:
            lonv = lon.values
            if np.nanmax(lonv) > 180 and np.nanmin(lonv) >= 0:
                newlon = ((lonv + 180) % 360) - 180
                ds = ds.assign_coords({lon_name: newlon}).sortby(lon_name)
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

        minx, miny, maxx, maxy = polygon.bounds

        # Case 1: 1D lat/lon
        if lat.ndim == 1 and lon.ndim == 1:
            if bbox_first:
                lat_slice = slice(miny, maxy) if float(lat[0]) < float(lat[-1]) else slice(maxy, miny)
                lon_slice = slice(minx, maxx) if float(lon[0]) < float(lon[-1]) else slice(maxx, minx)
                ds = ds.sel({lat_name: lat_slice, lon_name: lon_slice})

            if not drop_outside:
                return ds

            # polygon mask on the subset grid
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            lon2d, lat2d = np.meshgrid(lons, lats)
            mask = self._polygon_mask(lat2d, lon2d, polygon)
            mask_da = xr.DataArray(mask, dims=(lat_name, lon_name))
            return ds.where(mask_da, drop=True)

        # Case 2: 2D lat/lon coords on (y, x)
        if lat.ndim == 2 and lon.ndim == 2:
            y_dim, x_dim = lat.dims

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

        raise ValueError(f"Unsupported lat/lon shapes: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")

    @staticmethod
    def _polygon_mask(lat2d: np.ndarray, lon2d: np.ndarray, polygon: Geom) -> np.ndarray:
        """
        Boolean mask for points inside polygon using point-in-polygon on cell centers.
        (Simple and reliable; if you need speed, I can swap in shapely>=2 vectorized masking.)
        """
        ppoly = prep(polygon)
        mask = np.zeros(lat2d.shape, dtype=bool)
        for j in range(lat2d.shape[0]):
            mask[j, :] = [
                ppoly.contains(Point(float(lon2d[j, i]), float(lat2d[j, i])))
                for i in range(lat2d.shape[1])
            ]
        return mask
    
if __name__ == "__main__":
    from shapely.geometry import box

    client = SMOPSClient(cache_dir="data/smops_cdr")

    poly = box(-119.05, 33.60, -117.50, 34.85)  # LA-ish bbox polygon

    ds = client.query(
        polygon=poly,
        start="2023-07-01",
        end="2023-07-10",
        variables=["sm"],  # <-- replace with real var names in the file
        allow_multiple_per_day=False,
    )

    print(ds)