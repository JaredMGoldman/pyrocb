from __future__ import annotations

from datetime import date
from io import StringIO
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from utils.io_utils import FIRMS_KEY_FNAME, CLIENTS_DIR, set_env_var, buffer_polygon_meters
# 375m accuracy

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import earthaccess as ea
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import Polygon, MultiPolygon
from urllib3.util.retry import Retry


FirmsSources = Literal[
    "LANDSAT_NRT",
    "MODIS_NRT",
    "MODIS_SP",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT",
    "VIIRS_SNPP_SP",
]

Geom = Union[Polygon, MultiPolygon]

class FirmsClient:
    map_key_path: str  = os.path.join(CLIENTS_DIR,
                                  FIRMS_KEY_FNAME)
    
    base_url: str = "https://firms.modaps.eosdis.nasa.gov"
    timeout: int = 60
    max_retries: int = 5
    backoff_factor: float = 0.8
    map_key: str = ""
    if os.path.exists(map_key_path):
            with open(map_key_path, 'r') as f:
                map_key = f.read()
    else:
        raise ValueError(f"Save your FIRMS key to {map_key_path}")

    @classmethod
    def from_env(cls, env_var: str = "FIRMS_MAP_KEY") -> "FirmsClient":
        return cls(map_key=os.environ.get(env_var, ""))

    # -------------------------
    # Public API (xarray)
    # -------------------------
    def _query(
        self,
        polygon: Geom,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        variables: List[str],
        *,
        source: Sequence[FirmsSources] = "VIIRS_NOAA20_NRT", # ["VIIRS_NOAA21_SP"],
        crs: str = "EPSG:4326",
        return_by_source: bool = False,
    ) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        """
        Returns an xarray.Dataset of point detections inside polygon/date range.

        Output shape:
          dims: obs
          coords: latitude(obs), longitude(obs), time(obs) (if parseable)
          data_vars: remaining FIRMS columns + 'source' (if merged)
        """
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end must be >= start")
        
        buffered_polygon = buffer_polygon_meters(polygon, resolution_m=375, factor = 1)
        west, south, east, north = self._polygon_bounds_wsen(buffered_polygon)
        windows = self._chunk_date_range(start_ts, end_ts, max_days=5)

        session = self._session()

        out: Dict[str, xr.Dataset] = {}
        frames = []
        for w_start, w_end in windows:
            day_range = int((w_end - w_start).days) + 1
            df = self._area_csv_df(
                session=session,
                source=source,
                bbox_wsen=(west, south, east, north),
                day_range=day_range,
                date=w_start.strftime("%Y-%m-%d"),
            )
            frames.append(df)
        df_src = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df_src.empty:
            df_src = self._add_time_column_if_possible(df_src)
            return df_src

        df_src = self._dedupe_df(df_src)

        # Build time coordinate if possible
        df_src = self._add_time_column_if_possible(df_src)
        
        out = df_src.set_index(['time','latitude','longitude']).to_xarray().load()[variables]
        return out

    # -------------------------
    # HTTP session w/ retry
    # -------------------------
    def _session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    # -------------------------
    # FIRMS request -> DataFrame
    # -------------------------
    def _area_csv_df(
        self,
        session: requests.Session,
        source: FirmsSources,
        bbox_wsen: Tuple[float, float, float, float],
        day_range: int,
        date: str,
    ) -> pd.DataFrame:
        if not (1 <= day_range <= 5):
            raise ValueError("day_range must be 1..5 for FIRMS USFS Area API.")

        west, south, east, north = bbox_wsen
        area = f"{west},{south},{east},{north}"

        path = f"/usfs/api/area/csv/{self.map_key}/{source}/{area}/{day_range}/{date}"
        url = self.base_url.rstrip("/") + path

        r = session.get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"FIRMS request failed ({r.status_code}): {r.text[:300]}")

        df = pd.read_csv(StringIO(r.text))
        if not df.empty:
            if "longitude" not in df.columns or "latitude" not in df.columns:
                raise KeyError(f"Missing latitude/longitude in FIRMS CSV. Columns: {list(df.columns)[:30]}")
        return df

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _polygon_bounds_wsen(polygon: Geom) -> Tuple[float, float, float, float]:
        minx, miny, maxx, maxy = polygon.bounds
        return (float(minx), float(miny), float(maxx), float(maxy))

    @staticmethod
    def _chunk_date_range(
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        max_days: int = 5,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        windows = []
        cur = start_ts
        while cur <= end_ts:
            w_end = min(cur + pd.Timedelta(days=max_days - 1), end_ts)
            windows.append((cur, w_end))
            cur = w_end + pd.Timedelta(days=1)
        return windows

    @staticmethod
    def _dedupe_df(df: pd.DataFrame) -> pd.DataFrame:
        key_cols = [c for c in ["acq_date", "acq_time", "latitude", "longitude", "satellite"] if c in df.columns]
        if key_cols:
            return df.drop_duplicates(subset=key_cols).reset_index(drop=True)
        return df.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _add_time_column_if_possible(df: pd.DataFrame) -> pd.DataFrame:
        """
        FIRMS area CSV commonly provides:
          - acq_date (YYYY-MM-DD) or (YYYYMMDD)
          - acq_time (HHMM as string/int)
        We'll construct a datetime64[ns] column named 'time' if present.
        """
        if df.empty:
            df['time'] = []
            return df

        if "acq_date" in df.columns and "acq_time" in df.columns:
            # normalize types
            d = df["acq_date"].astype(str)
            t = df["acq_time"].astype(str).str.zfill(4)
            # Try common date formats
            # If acq_date is like '2025-08-01', concatenation will be '2025-08-011250' -> parse with %Y-%m-%d%H%M
            try:
                df["time"] = pd.to_datetime(d + t, format="%Y-%m-%d%H%M", errors="coerce")
            except Exception:
                df["time"] = pd.NaT

            # fallback if date is YYYYMMDD
            if df["time"].isna().all():
                df["time"] = pd.to_datetime(d + t, format="%Y%m%d%H%M", errors="coerce")
        else:
            df["time"] = pd.Timestamp("1999-01-01")
        return df

    @staticmethod
    def _df_to_xr(df: pd.DataFrame, add_source: Optional[str] = None) -> xr.Dataset:
        """
        Convert FIRMS detections DataFrame to xr.Dataset with dim 'obs'.
        """
        if df.empty:
            return FirmsClient._empty_dataset()

        df2 = df.copy()

        if add_source is not None:
            df2["source"] = add_source

        coords = {
            "latitude": ("time", df2["latitude"].to_numpy(dtype=float)),
            "longitude": ("time", df2["longitude"].to_numpy(dtype=float)),
        }

        if "time" in df2.columns and pd.api.types.is_datetime64_any_dtype(df2["time"]):
            coords["time"] = ("obs", df2["time"].to_numpy())

        # Put remaining columns as data_vars (excluding lat/lon, and time if used as coord)
        drop_cols = {"latitude", "longitude"}
        if "time" in coords:
            drop_cols.add("time")

        data_vars = {}
        for c in df2.columns:
            if c in drop_cols:
                continue
            data_vars[c] = ("obs", df2[c].to_numpy())

        return xr.Dataset(data_vars=data_vars, coords=coords)

    @staticmethod
    def _empty_dataset() -> xr.Dataset:
        return xr.Dataset(
            coords={
                "obs": np.array([], dtype=int),
                "latitude": ("obs", np.array([], dtype=float)),
                "longitude": ("obs", np.array([], dtype=float)),
            }
        )
    
if __name__ == "__main__":
    from shapely.geometry import box
    firms_key_path = os.path.join(CLIENTS_DIR,
                                  FIRMS_KEY_FNAME)

    set_env_var("FIRMS_MAP_KEY", firms_key_path)

    client = FirmsClient() #.from_env()

    poly = box(-119.05, 33.60, -117.50, 34.85)

    ds = client.query(
        polygon=poly,
        start="2025-08-01",
        end="2025-08-20",
        variables = ['frp'],
        source="VIIRS_NOAA20_NRT",
    )

    print(ds)
    print(ds.coords)
    print(ds.data_vars)