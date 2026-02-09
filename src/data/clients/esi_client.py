from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr
import rioxarray  # requires rasterio + GDAL
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


Geom = Union[Polygon, MultiPolygon]


@dataclass
class ESI4WKClient:
    """
    Query SERVIR ESI 4-week GeoTIFFs by polygon + date range and return an xarray.Dataset.

    Base directory format:
      https://gis1.servirglobal.net/data/esi/4WK/YYYY/

    File format assumed:
      {VAR}_4WK_YYYYDDD.tif
      Example: DFPPM_4WK_2017008.tif
    """
    base_url: str = "https://gis1.servirglobal.net/data/esi/4WK"
    cache_dir: Union[str, Path] = "esi_cache"
    timeout_s: int = 120

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
            raise ValueError("end must be >= start")

        # Snap each day to the 1+7k DOY bins, then unique them
        snapped_dates = self._snapped_dates(start_ts, end_ts)

        per_time_dsets = []
        for d in snapped_dates:
            per_var = []
            for var in variables:
                tif = self._download_tif(var, d)
                da = self._open_tif_as_dataarray(tif, var_name=var)

                if clip:
                    da = self._clip_dataarray_to_polygon(da, polygon, drop=drop)

                # ensure we have y/x dims (rioxarray uses y/x)
                da = da.squeeze(drop=True)

                per_var.append(da.to_dataset(name=var))

            ds_day = xr.merge(per_var).expand_dims(time=[np.datetime64(d.date())])
            per_time_dsets.append(ds_day)

        if not per_time_dsets:
            raise FileNotFoundError("No datasets were loaded for the requested time range.")

        ds = xr.concat(per_time_dsets, dim="time")
        return ds

    # ----------------------------
    # Internals
    # ----------------------------
    def _snapped_dates(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[pd.Timestamp]:
        days = pd.date_range(start_ts, end_ts, freq="D")
        snapped = {self._snap_to_1_plus_7k(d) for d in days}
        return sorted(snapped)

    @staticmethod
    def _snap_to_1_plus_7k(d: pd.Timestamp) -> pd.Timestamp:
        doy = d.timetuple().tm_yday
        snapped_doy = 1 + ((doy - 1) // 7 + 1) * 7
        return pd.Timestamp(year=d.year, month=1, day=1) + pd.Timedelta(days=snapped_doy - 1)

    @staticmethod
    def _doy_str(d: pd.Timestamp) -> str:
        return f"{d.timetuple().tm_yday:03d}"

    def _remote_url(self, var: str, d: pd.Timestamp) -> str:
        year = d.year
        doy = self._doy_str(d)
        fname = f"{var}_4WK_{year}{doy}.tif"
        return f"{self.base_url}/{year}/{fname}"

    def _local_path(self, var: str, d: pd.Timestamp) -> Path:
        year = d.year
        doy = self._doy_str(d)
        fname = f"{var}_4WK_{year}{doy}.tif"
        p = self.cache_dir / str(year) / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _download_tif(self, var: str, d: pd.Timestamp) -> Path:
        url = self._remote_url(var, d)
        out_path = self._local_path(var, d)

        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        with self._session.get(url, stream=True, timeout=self.timeout_s) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return out_path

    @staticmethod
    def _open_tif_as_dataarray(tif_path: Path, var_name: str) -> xr.DataArray:
        # rioxarray returns dims (band, y, x); we drop band.
        da = rioxarray.open_rasterio(tif_path, masked=True).squeeze("band", drop=True)
        da.name = var_name
        return da

    @staticmethod
    def _clip_dataarray_to_polygon(da: xr.DataArray, polygon: Geom, drop: bool = True) -> xr.DataArray:
        # rioxarray expects a GeoDataFrame/GeoSeries geometry with CRS.
        gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

        # Reproject polygon into raster CRS if needed
        if da.rio.crs is not None and str(da.rio.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs(da.rio.crs)

        return da.rio.clip(gdf.geometry, gdf.crs, drop=drop)
    
if __name__ == "__main__":
    from shapely.geometry import box

    client = ESI4WKClient(cache_dir="esi_cache")

    poly = box(-119.05, 33.60, -117.50, 34.85)  # LA bbox
    ds = client.query(
        polygon=poly,
        start="2017-01-01",
        end="2017-02-01",
        variables=["DFPPM"],   # add more prefixes if they exist in that directory
    )

    print(ds)
    print(ds["DFPPM"].shape)