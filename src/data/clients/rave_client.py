from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import re

import numpy as np
import pandas as pd
import requests
import xarray as xr

from shapely.geometry import Polygon, MultiPolygon
from shapely.prepared import prep

Geom = Union[Polygon, MultiPolygon]


@dataclass
class RAVEHrlyEmiss3kmClient:
    """
    RAVE hourly emissions (3km) client:
      - lists month directories for a date range
      - downloads needed NetCDF files
      - subsets variables and clips to polygon
      - returns xr.Dataset concatenated over time
    """
    base_url: str = "https://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km"
    cache_dir: Union[str, Path] = "rave_cache/RAVE-HrlyEmiss-3km"
    timeout_s: int = 120

    # common coordinate name fallbacks
    lat_names: Tuple[str, ...] = ("lat", "latitude", "LAT", "Latitude", "y", "grid_latt")
    lon_names: Tuple[str, ...] = ("lon", "longitude", "LON", "Longitude", "x", "grid_latt")
    time_names: Tuple[str, ...] = ("time", "valid_time", "datetime", "date")

    _fname_re: re.Pattern = re.compile(
        r"s(\d{4})(\d{2})(\d{2})", re.VERBOSE
    )

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    # ----------------------------
    # Public API
    # ----------------------------
    def query(
        self,
        polygon: Geom,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        *,
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
        files = self._collect_files_for_range(start_ts, end_ts)

        if not files:
            raise FileNotFoundError("No RAVE files found overlapping the requested time window.")


        dsets: List[xr.Dataset] = []
        for meta in files:
            local = self._download(url = meta['url'], year=meta["year"], month=meta["month"])
            ds = xr.open_dataset(local)

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

            dsets.append(ds)

        if not dsets:
            raise RuntimeError("No datasets remained after subsetting.")

        out = xr.concat(dsets, dim="time", combine_attrs="override" if keep_attrs else "drop")
        return out

    # ----------------------------
    # Listing + parsing
    # ----------------------------
    def _month_dir_url(self, year: int, month: int) -> str:
        return f"{self.base_url.rstrip('/')}/{year:04d}/{month:02d}/"

    def _list_month_filenames(self, year: int, month: int, day: int) -> List[str]:
        """
        OSPO pub directories are simple HTML indexes; grab hrefs ending in .nc
        """
        url = self._month_dir_url(year, month)
        r = self._session.get(url, timeout=self.timeout_s)
        names = re.findall(r'href="([^"]+\.nc)"', r.text, flags=re.IGNORECASE)
        r.raise_for_status() 
        
        filtered_names = [{ 'url' : self._month_dir_url(year, month) + fname, 
                            'year': year,
                            'month' : month,
                            'day' : day} \
                                  for fname in names if f"s{year:04d}{month:02d}{day:02d}" in fname]
        return filtered_names

    def _collect_files_for_range(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict]:
        # consider months overlapping the window
        start_month = pd.Timestamp(start_ts.year, start_ts.month, 1)
        end_month = pd.Timestamp(end_ts.year, end_ts.month, 1)
        months = pd.date_range(start_month, end_month, freq="MS")

        for m in months:
            year, month, day = int(m.year), int(m.month), int(m.day)
            try:
                fnames = self._list_month_filenames(year, month, day)
            except requests.HTTPError:
                # month directory may not exist (e.g., outside archive range)
                continue

        return fnames

    # ----------------------------
    # Download + cache
    # ----------------------------
    def _download(self, url: str, year: int, month: int, chunk_size: int = 1 << 20) -> Path:
        fname = url.split("/")[-1]
        out = self.cache_dir / f"{year:04d}" / f"{month:02d}" / fname
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

        if lon.ndim == 1:
            lonv = lon.values
            ds_is_360 = (np.nanmin(lonv) >= 0) and (np.nanmax(lonv) > 180)
            if polygon_is_180 and ds_is_360:
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

        # Case A: 1D lat/lon
        if lat.ndim == 1 and lon.ndim == 1:
            if bbox_first:
                lat_slice = slice(miny, maxy) if float(lat[0]) < float(lat[-1]) else slice(maxy, miny)
                lon_slice = slice(minx, maxx) if float(lon[0]) < float(lon[-1]) else slice(maxx, minx)
                ds = ds.sel({lat_name: lat_slice, lon_name: lon_slice})

            if not drop_outside:
                return ds

            lats = ds[lat_name].values
            lons = ds[lon_name].values
            lon2d, lat2d = np.meshgrid(lons, lats)
            mask = self._polygon_mask(lat2d, lon2d, polygon)
            mask_da = xr.DataArray(mask, dims=(lat_name, lon_name))
            return ds.where(mask_da, drop=True)

        # Case B: 2D lat/lon coords (y,x)
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
        Boolean mask for points inside polygon, using cell-center point-in-polygon.

        If shapely>=2 is installed, this will use vectorized contains; otherwise falls back to loop.
        """
        try:
            import shapely  # type: ignore

            # shapely>=2 vectorized path
            if getattr(shapely, "__version__", "0").startswith("2"):
                from shapely import points, contains  # type: ignore

                pts = points(lon2d, lat2d)
                return contains(polygon, pts)
        except Exception:
            pass

        # fallback: prepared geometry + Python loop (slower but dependable)
        ppoly = prep(polygon)
        mask = np.zeros(lat2d.shape, dtype=bool)
        for j in range(lat2d.shape[0]):
            for i in range(lat2d.shape[1]):
                mask[j, i] = ppoly.contains(
                    type("P", (), {"x": float(lon2d[j, i]), "y": float(lat2d[j, i])})()  # minimal point-like
                )
        return mask
    
if __name__ == "__main__":
    from shapely.geometry import box

    poly = box(-122.0, 36.0, -118.0, 39.0)  # CA-ish box

    client = RAVEHrlyEmiss3kmClient(cache_dir="data/rave_3km")

    ds = client.query(
        polygon=poly,
        start="2024-10-01 00:00",
        end="2024-10-01 12:00",
        variables=["PM25", "FRP_MEAN", "FRE"],  # replace with actual names in your files
    )

    print(ds)
    print(ds.time.values[:3])