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
from typing import Iterable, Sequence, Optional, Union, Literal

import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep

from herbie import Herbie, FastHerbie


Geom = Union[Polygon, MultiPolygon]
Freq = Literal["1H", "1h", "60min"]


@dataclass
class HRRRPolygonHerbie:
    """
    Query HRRR (surface product by default) via Herbie and return an xarray.Dataset
    clipped to grid cells that intersect a polygon.

    Typical usage: HRRR analyses (fxx=0) for each hour in a time range.
    """
    product: str = "sfc"        # HRRR surface fields product :contentReference[oaicite:3]{index=3}
    fxx: int = 0                # lead time in hours (0 = analysis)
    model: str = "hrrr"
    freq: Freq = "1H"
    remove_grib: bool = False   # keep GRIB file (useful if you re-open/cached)
    n_jobs: int = 4             # FastHerbie threads

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
        times = pd.date_range(start=start_ts, end=end_ts, freq=self.freq)

        if len(times) == 0:
            raise ValueError("Empty time range after applying freq.")

        # Combine requested variable patterns into one regex string for Herbie.xarray
        search = self._build_search_string(variables)
        # Use FastHerbie to manage many Herbie objects and open/concat efficiently :contentReference[oaicite:5]{index=5}
        FH = FastHerbie(
            times,
            model=self.model,
            product=self.product,
            fxx=[self.fxx]*len(times)
        )

        # Open only the GRIB messages matching `search` into xarray :contentReference[oaicite:6]{index=6}
        dses = FH.xarray(search, remove_grib=self.remove_grib)

        # Standardize time coordinate name if needed
        # import ipdb; ipdb.set_trace()        

        # Clip to polygon-intersecting grid cells
        dses = [self._clip_to_polygon(ds, polygon, bbox_first=bbox_first) for ds in dses]

        dses_stacked = [ds.stack(tcell=(time_dim, "y", "x")) for ds in dses]
        ds_stacked = xr.concat(dses_stacked, dim='tcell') 

        return ds_stacked

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
        lat = ds[lat_name]
        lon = ds[lon_name]

        # Identify spatial dims from lat/lon
        if lat.ndim != 2 or lon.ndim != 2:
            raise ValueError(
                "Expected 2D latitude/longitude grids. If you have 1D lat/lon, "
                "this clipper needs a small adjustment."
            )
        y_dim, x_dim = lat.dims

        poly = polygon
        if isinstance(poly, MultiPolygon):
            # union is OK, but prep() is faster for repeated contains checks
            poly = poly.union(poly)  # noop-ish; keeps type stable

        # Optional fast bbox crop first (reduces mask work)
        if bbox_first:
            minx, miny, maxx, maxy = polygon.bounds
            # mask bbox in lat/lon space
            bbox_mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
            # keep rows/cols that have any True
            keep_y = bbox_mask.any(dim=x_dim)
            keep_x = bbox_mask.any(dim=y_dim)
            ds = ds.isel({y_dim: np.where(keep_y.values)[0], x_dim: np.where(keep_x.values)[0]})
            lat = ds[lat_name]
            lon = ds[lon_name]

        # Polygon mask (intersecting cells via centroid-in-polygon test)
        # Note: If you need strict *cell intersects polygon* (not centroid),
        # you’d build cell polygons. Centroid mask is the usual fast approximation.
        poly_prep = prep(polygon)

        lonv = lon.values
        latv = lat.values
        mask = np.zeros(lonv.shape, dtype=bool)
        # vectorized-ish loop over rows to avoid huge Python object creation
        for j in range(lonv.shape[0]):
            mask[j, :] = [poly_prep.contains(Point(lonv[j, i], latv[j, i])) for i in range(lonv.shape[1])]

        # Apply mask: set outside to NaN, then drop fully-empty rows/cols
        ds = ds.where(xr.DataArray(mask, dims=(y_dim, x_dim)), drop=True)

        return ds


if __name__ == "__main__":
    from shapely.geometry import box

    la_poly = box(-119.05, 33.60, -117.50, 34.85)

    client = HRRRPolygonHerbie(product="sfc", fxx=0, n_jobs=6)

    ds = client.query(
        polygon=la_poly,
        start="2025-07-01 00:00",
        end="2025-07-01 06:00",
        variables=[
            ":TMP:2 m",      # 2m temperature
            ":RH:2 m",       # 2m relative humidity
            ":(?:UGRD|VGRD):10 m",  # 10m wind components (regex)
        ],
    )

    print(ds)