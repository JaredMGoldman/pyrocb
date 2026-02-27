# download subset of HRRR data based on 
#   - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
#   - fire location
#   - variables of interest
#       - hdw:              'vpd_2m', 'wind_speed'
#       - hwp: HWP=0.213*G^(1.5)*vpd^(0.73)(1-M)^(5.10)S
#           - https://doi.org/10.1175/WAF-D-24-0068.1
#           - G: max(3, 10-m wind gust potential)
#           - VPD: 2-m vapor pressure deficit
#               - calculate from relative humidity (RH) and temperature (TMP)
#               - SVP = 610.7*10^{7.5*TMP/(237.3+T)}/1000
#               - AVP = SVP * RH/100
#               - VPD = SVP(1 - RH/100) 
#           - M: soil moisture availability (MSTAV)
#           - S: snow water equivalent term (WEASD)
#       - hrrr_met: VPD, WIND

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Union, Literal, Tuple

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
import os

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep

from herbie import Herbie, FastHerbie
import uuid
import shutil
from datetime import datetime, timezone
import socket

def make_run_id() -> str:
    # Example: run_20260226T235901Z_5f2c9c3a0b8c4d8aa2a1a0f5b7b20d3e
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    u = uuid.uuid4().hex
    return f"{ts}_{u}"

def make_cache_dir(base: Path, prefix: str = "cache") -> Path:
    run_id = make_run_id()
    pid = os.getpid()
    d = base / f"{run_id}_pid{pid}"
    d.mkdir(parents=True, exist_ok=False)
    return d

Geom = Union[Polygon, MultiPolygon]
Freq = Literal["1H", "1h", "60min"]

RegionMode = Literal["auto", "force_hrrr", "force_ifs"]

drop_coords = ["heightAboveGround", "depthBelowLand", "valid_time"]

@dataclass
class HRRRClient:
    index_fps = []
    product: str = "sfc"
    fxx: int = 0
    model: str = "hrrr"
    freq: Freq = "6h"
    remove_grib: bool = True
    n_jobs: int = 4

    # NEW: IFS fallback config
    region_mode: RegionMode = "auto"
    ifs_model: str = "ifs"
    ifs_product: str = "oper"
    ifs_init_freq: str = "12H"  # IFS oper runs at 00z & 12z

    # NEW: CONUS bbox heuristic (lon/lat)
    conus_bbox: Tuple[float, float, float, float] = (-125.0, 24.0, -66.0, 50.0)

    def __init__(self, *args, **kwargs):
        self.save_dir = make_cache_dir(Path(f"{os.environ.get('HOME')}/data/herbie"), prefix="")
        # self.save_dir = f"{os.environ.get('HOME')}/data/herbie/.{uuid.uuid4().hex}"

    def _polygon_in_conus(self, polygon: Geom) -> bool:
        minx, miny, maxx, maxy = polygon.bounds
        west, south, east, north = self.conus_bbox
        return (minx >= west) and (maxx <= east) and (miny >= south) and (maxy <= north)

    def _choose_model_product_and_times(
        self, polygon: Geom, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> tuple[str, str, pd.DatetimeIndex]:
        """
        Returns (model, product, init_times).

        - HRRR: hourly times (as you currently do)
        - IFS oper: init times every 12h (00/12) :contentReference[oaicite:3]{index=3}
        """
        if self.region_mode == "force_hrrr":
            use_ifs = False
        elif self.region_mode == "force_ifs":
            use_ifs = True
        else:
            use_ifs = not self._polygon_in_conus(polygon)

        if not use_ifs:
            times = pd.date_range(start=start_ts, end=end_ts, freq=self.freq)
            return self.model, self.product, times

        # IFS fallback
        # Use 12-hourly init times; Herbie will fetch grib for each init+fxx
        # (Your fxx still applies.)
        times = pd.date_range(start=start_ts.floor("12H"), end=end_ts.ceil("12H"), freq=self.ifs_init_freq)
        return self.ifs_model, self.ifs_product, times
    def query(
        self,
        polygon: Geom,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        variables: Sequence[str],
        *,
        bbox_first: bool = True,
        time_dim: str = "time",
    ) -> xr.Dataset:
        try:
            out = self._query(  polygon = polygon,
                                start = start,
                                end = end,
                                variables = variables,
                                bbox_first = bbox_first,
                                time_dim = time_dim).load()
            self._remove_idx_files()
            return out
        except Exception as e:
            self._remove_idx_files()
            raise RuntimeError(f"[ERROR] HRRR failed with expection {e}")

    def _query(
        self,
        polygon: Geom,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        variables: Sequence[str],
        *,
        bbox_first: bool = True,
        time_dim: str = "time",
    ) -> xr.Dataset:
        """
        Parameters
        ----------
        polygon : shapely Polygon/MultiPolygon in lon/lat (EPSG:4326)
        start, end : timestamps (ideally UTC) inclusive
        variables : list of Herbie search strings OR bare GRIB tokens.
            Recommended: full regex snippets like [":TMP:2 m", ":RH:2 m", ":UGRD:10 m", ":VGRD:10 m"].
            These are combined with '|' into one regex for Herbie.xarray(). :contentReference[oaicite:4]{index=4}
        bbox_first : if True, do a fast bbox crop before polygon mask
        time_dim : output time dimension name (usually "time")

        Returns
        -------
        xr.Dataset subset to times and grid cells intersecting polygon.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        # HRRR is hourly; build list of initialization times (commonly UTC)
        model, product, times = self._choose_model_product_and_times(polygon, start_ts, end_ts)

        if len(times) == 0:
            raise ValueError("Empty time range after applying model-specific frequency.")

        search = self._build_search_string(variables)
        dses = []
        for time in times:
            H = Herbie(
                time,
                model=model,
                product=product,
                fxx=self.fxx,
                save_dir = self.save_dir,
            )
            print(f"search vars: {search}")
            dses.append(self.combine_dses(H.xarray(search, remove_grib=self.remove_grib), time_dim))
            self.index_fps.append(H.get_localIndexFilePath())
        return xr.merge(dses, join = 'outer')
    
    def _find_lat_lon_var(self, ds):
        lat_var = ""
        lon_var = ""
        if "latitude" in ds.dims:
            lat_var = "latitude"
        elif "y" in ds.dims:
            lat_var = "y"
        if "longitude" in ds.dims:
            lon_var = "longitude"
        elif "x" in ds.dims:
            lon_var = "x"
        return lat_var, lon_var
    
    def combine_dses(self, all_dses, time_dim):
        dses = []
        for ds in all_dses:
            ds = ds.expand_dims(time=[ds[time_dim].values]).drop_attrs()
            coords_to_rm = [coord for coord in drop_coords if coord in ds._coord_names]
            new_ds = ds.drop_vars(coords_to_rm)
            dses.append(new_ds)
            
        return xr.merge(dses, join = 'outer', compat = 'override')

    @staticmethod
    def _build_search_string(variables: Sequence[str]) -> str:
        """
        Herbie expects a regex-like searchString (wgrib2-style patterns).
        Example shown in docs: ":(?:TMP|RH):2 m" :contentReference[oaicite:7]{index=7}
        """
        if not variables:
            raise ValueError("variables must be a non-empty list of patterns.")

        # If user supplies bare tokens, encourage full patterns; still allow them.
        # Join with | into a non-capturing group
        parts = [v.strip() for v in variables if v and v.strip()]
        if not parts:
            raise ValueError("variables contained only empty strings.")
        if len(parts) == 1:
            return parts[0]
        return "(?:" + "|".join(parts) + ")"

    @staticmethod
    def _infer_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
        """
        HRRR xarray via cfgrib typically includes latitude/longitude coordinates.
        Names vary by engine/version; try common options.
        """
        # Common in cfgrib-backed Herbie datasets
        for lat_name in ("latitude", "lat", "gridlat", "y_lat"):
            if lat_name in ds:
                break
        else:
            raise KeyError("Could not find a latitude coordinate/variable in dataset.")

        for lon_name in ("longitude", "lon", "gridlon", "x_lon"):
            if lon_name in ds:
                break
        else:
            raise KeyError("Could not find a longitude coordinate/variable in dataset.")

        return lat_name, lon_name

    def _clip_to_polygon(self, ds: xr.Dataset, polygon: Geom, bbox_first: bool = True) -> xr.Dataset:
        lat_name, lon_name = self._infer_lat_lon(ds)

        # Wrap lon to [-180,180]
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})

        lat = ds[lat_name]
        lon = ds[lon_name]

        poly_prep = prep(polygon)
        minx, miny, maxx, maxy = polygon.bounds

        # -------------------------
        # Case 1: 2D lat/lon (HRRR typical)
        # -------------------------
        if lat.ndim == 2 and lon.ndim == 2:
            y_dim, x_dim = lat.dims

            if bbox_first:
                bbox_mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
                keep_y = bbox_mask.any(dim=x_dim)
                keep_x = bbox_mask.any(dim=y_dim)
                ds = ds.isel({y_dim: np.where(keep_y.values)[0], x_dim: np.where(keep_x.values)[0]})
                lat = ds[lat_name]
                lon = ds[lon_name]

            lonv = lon.values
            latv = lat.values
            mask = np.zeros(lonv.shape, dtype=bool)
            for j in range(lonv.shape[0]):
                mask[j, :] = [poly_prep.contains(Point(lonv[j, i], latv[j, i])) for i in range(lonv.shape[1])]

            return ds.where(xr.DataArray(mask, dims=(y_dim, x_dim)), drop=True)

        # -------------------------
        # Case 2: 1D lat/lon (common for IFS global grids)
        # -------------------------
        if lat.ndim == 1 and lon.ndim == 1:
            lat_dim = lat.dims[0]
            lon_dim = lon.dims[0]

            if bbox_first:
                # handle descending latitude arrays
                lat_asc = float(lat[0]) < float(lat[-1])
                lat_slice = slice(miny, maxy) if lat_asc else slice(maxy, miny)
                lon_slice = slice(minx, maxx) if float(lon[0]) < float(lon[-1]) else slice(maxx, minx)
                ds = ds.sel({lat_dim: lat_slice, lon_dim: lon_slice})
                lat = ds[lat_name]
                lon = ds[lon_name]

            # Build a 2D mask over the selected 1D grid
            lon2d, lat2d = np.meshgrid(lon.values, lat.values)
            mask = np.zeros(lon2d.shape, dtype=bool)
            for j in range(lon2d.shape[0]):
                mask[j, :] = [poly_prep.contains(Point(lon2d[j, i], lat2d[j, i])) for i in range(lon2d.shape[1])]

            mask_da = xr.DataArray(mask, dims=(lat_dim, lon_dim))
            return ds.where(mask_da, drop=True)

        raise ValueError(f"Unsupported lat/lon shapes: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")

    def _remove_idx_files(self):
        print(f"[INFO] removing {len(self.index_fps)} HRRR indices...")
        # dirnames = list(set([os.path.dirname(fname) for fname in self.index_fps]))
        # for fname in self.index_fps:
        #     os.remove(fname)
        # for dirname in dirnames:
        #     os.rmdir(dirname)
        shutil.rmtree(self.save_dir)
        print("[INFO] HRRR indices removed")

if __name__ == "__main__":
    from shapely.geometry import box

    la_poly = box(-119.05, 33.60, -117.50, 34.85)

    client = HRRRClient(product="sfc", fxx=0, n_jobs=6)

    # ds_us = client.query(
    #     polygon=la_poly,
    #     start="2025-07-01 00:00",
    #     end="2025-07-01 06:00",
    #     variables=[
    #         ":TMP:2 m",      # 2m temperature
    #         ":RH:2 m",       # 2m relative humidity
    #         ":(?:UGRD|VGRD):10 m",  # 10m wind components (regex)
    #     ],
    # )

    # print(ds_us)

    # Outside CONUS -> should auto-switch to IFS oper
    central_canada_poly = box(-105.5, 52.5, -104.0, 54.0)  # Mediterranean-ish
    vars = [
        ":tp:",
        ":u:1000",
        ":v:1000",
        ":r:",
        ":2t:",
        ":sd:",
        ":ssw:",
        ":2d:",
    ]
    ds_ocean = client.query(
        polygon=central_canada_poly,
        start="2025-07-01 00:00",
        end="2025-07-01 12:00",
        variables=vars,
    )
    print(ds_ocean)