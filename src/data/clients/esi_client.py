from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence, Union, List

import numpy as np
import os
import pandas as pd
from rasterio.features import geometry_mask
import re
import requests
import xarray as xr
import rioxarray  # requires rasterio + GDAL
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from utils import CACHE_BASE_DIR, make_cache_dir, buffer_polygon_meters
from data.clients.base_client import BaseClient
from rio_utils import validate_tif_download
import shutil

Geom = Union[Polygon, MultiPolygon]


@dataclass
class ESIClient(BaseClient):
    """
    Query SERVIR ESI 4-week GeoTIFFs by polygon + date range and return an xarray.Dataset.

    Base directory format:
      https://gis1.servirglobal.net/data/esi/4WK/YYYY/

    File format assumed:
      {VAR}_4WK_YYYYDDD.tif
      Example: DFPPM_4WK_2017008.tif
    """
    base_url: str = "http://gis1.servirglobal.net"
    remote_data_dir: str = "data/esi/4WK"
    cached_files = []
    timeout_s: int = 120

    def __init__(self, *args, **kwargs):
        super().__init__(cache_dir = os.path.join(CACHE_BASE_DIR, 'esi'), **kwargs)

    # ----------------------------
    # Public API
    # ----------------------------
    def query(self,
        polygon: Geom,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        variables: Sequence[str] = ("DFPPM",),
        clip: bool = True,
        drop: bool = True,
        ) -> xr.Dataset:
        try:
            return self._query(
                polygon, start, end, variables,
                clip = clip, drop = drop)
        except Exception as e:
            self._remove_cached_files()
            self.logger.error(f"[ERROR] ESI failed: {e}")
            raise RuntimeError(f"[ERROR] ESI failed: {e}")

    def _query(
        self,
        polygon: Geom,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        variables: Sequence[str] = ("DFPPM",),
        *,
        clip: bool = True,
        drop: bool = True,
    ) -> xr.Dataset:
        """
        Parameters
        ----------
        polygon : shapely Polygon/MultiPolygon in EPSG:4326 (lon/lat)
        start, end : date-like
        variables : list of variable prefixes (e.g., ["DFPPM"]) mapping to filenames like
                    "{var}_4WK_YYYYDDD.tif"
        clip : if True, clip each raster to polygon
        drop : if True, drop pixels outside polygon (mask + crop)

        Returns
        -------
        xr.Dataset with dims: time, y, x
        data_vars: one per entry in `variables`
        """
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            self.logger.error("end must be >= start")
            raise ValueError("end must be >= start")

        # Snap each day to the 1+7k DOY bins, then unique them
        snapped_dates = self._snapped_dates(start_ts, end_ts)

        per_time_dsets = []
        for d, doys in snapped_dates.items():
            per_var = []
            for var in variables:
                tif = self._download_tif(d)
                self.cached_files.append(tif)
                da = self._open_tif_as_dataarray(tif, var_name=var)

                if clip:
                    polygon = buffer_polygon_meters(polygon, resolution_m=4000, factor = 1.0)
                    da = self._clip_dataarray_to_polygon(da, polygon, drop=drop)

                # ensure we have y/x dims (rioxarray uses y/x)
                da = da.squeeze(drop=True)

                per_var.append(da.to_dataset(name=var))
                
            ds_day = xr.merge(per_var).expand_dims(time=doys)
            per_time_dsets.append(ds_day)

        if not per_time_dsets:
            self.logger.error("No datasets were loaded for the requested time range.")
            raise FileNotFoundError("No datasets were loaded for the requested time range.")

        ds = xr.merge(per_time_dsets).load()
        self._remove_cached_files()
        return ds

    # ----------------------------
    # Internals
    # ----------------------------
    def _snapped_dates(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[pd.Timestamp]:
        days = pd.date_range(start_ts, end_ts, freq="D")
        snapped = {}
        for d in days:
            snapped_doy = self._snap_to_1_plus_7k(d)
            if snapped_doy in snapped.keys():
                snapped[snapped_doy].append(d)
            else:
                snapped[snapped_doy] = [d]
        return snapped

    @staticmethod
    def _snap_to_1_plus_7k(d: pd.Timestamp) -> pd.Timestamp:
        doy = d.timetuple().tm_yday
        snapped_doy = 1 + ((doy - 1) // 7 ) * 7
        return pd.Timestamp(year=d.year, month=1, day=1) + pd.Timedelta(days=snapped_doy - 1)

    @staticmethod
    def _doy_str(d: pd.Timestamp) -> str:
        return f"{d.timetuple().tm_yday:03d}"

    def _remote_url(self, d: pd.Timestamp) -> str:
        year = d.year
        doy = self._doy_str(d)
        fnames = self._list_dir(year)
        return [f"{self.base_url}{fname}" for fname in fnames if f"_{year}{doy}.tif" in fname][0]
    
    def _list_dir(self, year: int) -> list[str]:
        url = f"{self.base_url}/{self.remote_data_dir}/{year}"
        r = self._session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return sorted(set(re.findall(r'"([^"]+\.tif)"', r.text)))

    def _local_path(self, d: pd.Timestamp) -> Path:
        year = d.year
        doy = self._doy_str(d)
        fname = f"ESI_4WK_{year}{doy}.tif"
        p = self.save_dir / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _download_tif(self, d: pd.Timestamp) -> Path:
        url = self._remote_url(d)
        out_path = self._local_path(d)

        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        return validate_tif_download(url, out_path, self._session)

    @staticmethod
    def _open_tif_as_dataarray(tif_path: Path, var_name: str) -> xr.DataArray:
        # rioxarray returns dims (band, y, x); we drop band.
        da = rioxarray.open_rasterio(tif_path, masked=True).squeeze("band", drop=True)
        da.name = var_name
        da_with_nan = da.where(da != - 9999)
        return da_with_nan

    @staticmethod
    def _clip_dataarray_to_polygon(da: xr.DataArray, polygon: Geom, drop: bool = True) -> xr.DataArray:
        # rioxarray expects a GeoDataFrame/GeoSeries geometry with CRS.
        gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

        # Reproject polygon into raster CRS if needed
        if da.rio.crs is not None and str(da.rio.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs(da.rio.crs)

        mask = geometry_mask(
            geometries=[geom.__geo_interface__ for geom in gdf.geometry],
            out_shape=(da.sizes["y"], da.sizes["x"]),
            transform=da.rio.transform(),
            invert=True,         
            all_touched=True,
        )
        return da.where(mask)
    
if __name__ == "__main__":
    from shapely.geometry import box

    client = ESIClient()

    poly = box(-119.05, 33.60, -117.50, 34.85)  # LA bbox
    ds = client.query(
        polygon=poly,
        start="2020-09-09",
        end="2020-09-11",
        variables=["DFPPM"],   # add more prefixes if they exist in that directory
    )

    print(ds)
    print(ds["DFPPM"].shape)