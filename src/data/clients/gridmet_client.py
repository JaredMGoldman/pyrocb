from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from data.clients.base_client import BaseClient


GeometryLike = Union[Polygon, MultiPolygon]


@dataclass
class GridMETClient(BaseClient):
    """
    Query NW Knowledge 'metdata/data/permanent/YYYY/' daily GridMET netCDFs
    and return a GeoDataFrame for all grid cells intersecting a polygon.

    Directory includes files like 'permanent_gridmet_YYYYMMDD.nc'.  :contentReference[oaicite:1]{index=1}
    """
    base_url: str = "https://www.northwestknowledge.net/metdata/data/permanent"
    cache_dir: Union[str, Path] = "nwk_cache"
    timeout_s: int = 120

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Public API
    # ----------------------------
    def _query(
        self,
        polygon: GeometryLike,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        *,
        year: Optional[int] = None,
        variables: Optional[Sequence[str]] = None,
        predicate: str = "intersects",
    ) -> xr.Dataset:
        """
        Returns GeoDataFrame with columns:
          - time (datetime64)
          - cell_id (int)
          - lon, lat
          - all requested variables (or all in file)
          - geometry (point at cell centroid)

        predicate: "intersects" or "within" for spatial filter against polygon.
        """
        poly = unary_union(polygon) if isinstance(polygon, MultiPolygon) else polygon
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        # Determine year(s). The directory is per-year.
        years = sorted({start_ts.year, end_ts.year}) if year is None else [year]

        # Download all required files across years
        local_files = []
        for y in years:
            local_files.extend(self._ensure_files_for_range(y, start_ts, end_ts))

        if not local_files:
            raise FileNotFoundError("No netCDF files found for the requested date range.")

        # Open dataset(s)
        ds = xr.open_mfdataset(
            [str(p) for p in local_files],
            combine="by_coords"
        )
        ds = ds.assign_coords(day=np.datetime64(f"{start_ts.year}-01-01") \
                              + (ds["day"].dt.dayofyear - 1).astype("timedelta64[D]"))

        # Keep variables if requested
        if variables is not None:
            missing = [v for v in variables if v not in ds.data_vars]
            if missing:
                raise KeyError(f"Requested variables not found in dataset: {missing}")
            ds = ds[list(variables)]

        # Subset in time
        if "day" not in ds.dims and "day" in ds.coords:
            # sometimes time is a coord not a dim
            ds = ds.swap_dims({list(ds.dims)[0]: "day"})
        
        ds = ds.sel(day=slice(start_ts, end_ts))

        # Detect lat/lon coords
        lat_name, lon_name = self._detect_lat_lon(ds)
        lat = ds[lat_name]
        lon = ds[lon_name]

        # Build cell centroids as GeoDataFrame, then spatially filter to polygon
        gdf_cells, cell_id_map = self._cells_gdf(lat, lon)

        poly_gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
        gdf_cells = gpd.sjoin(gdf_cells, poly_gdf, predicate=predicate, how="inner").drop(columns="index_right")

        keep_cell_ids = gdf_cells["cell_id"].to_numpy()

        # Convert ds to long table with a stacked cell dimension that matches our cell_id
        y_dim, x_dim = self._detect_xy_dims(ds, lat_name, lon_name)
        ds_stack = ds.stack(cell=(y_dim, x_dim))

        # Compute cell_id in the same order as stack
        # stack order is consistent with row-major indexing: cell_id = y_index * nx + x_index
        nx = ds.sizes[x_dim]
        # stacked retains coordinate values of y_dim/x_dim; we need positional indices
        # so we create integer indices from 0..ny-1 and 0..nx-1.
        # If dims are already integer-like 0..N-1, this is fine; else we map positions.
        y_pos = xr.DataArray(np.arange(ds.sizes[y_dim]), dims=(y_dim,), coords={y_dim: ds[y_dim]})
        x_pos = xr.DataArray(np.arange(ds.sizes[x_dim]), dims=(x_dim,), coords={x_dim: ds[x_dim]})
        cell_id_da = (y_pos * nx + x_pos).stack(cell=(y_dim, x_dim))

        # Filter cells
        mask = np.isin(cell_id_da.values, keep_cell_ids)
        return ds_stack.isel(cell=mask)

    # ----------------------------
    # Internals
    # ----------------------------
    def _year_url(self, year: int) -> str:
        return f"{self.base_url.rstrip('/')}/{year}/"

    def _list_remote_filenames(self, year: int) -> list[str]:
        url = self._year_url(year)
        r = requests.get(url, timeout=self.timeout_s)
        r.raise_for_status()

        # Directory listing contains filenames like permanent_gridmet_20250101.nc :contentReference[oaicite:2]{index=2}
        names = re.findall(r'href="([^"]+\.nc)"', r.text)
        return sorted(set(names))

    def _ensure_files_for_range(self, year: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[Path]:
        all_names = self._list_remote_filenames(year)

        # Daily pattern: permanent_gridmet_YYYYMMDD.nc (ignore DROUGHT files unless you want them)
        pat = re.compile(rf"^permanent_gridmet_{year}\d{{4}}\.nc$")
        daily = [n for n in all_names if pat.match(n)]

        # Build required date strings in this year
        y_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        y_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))

        needed = set()
        d = y_start
        while d <= y_end:
            needed.add(f"permanent_gridmet_{d.strftime('%Y%m%d')}.nc")
            d += pd.Timedelta(days=1)

        # Only download those that exist
        to_get = [n for n in daily if n in needed]

        local_paths: list[Path] = []
        for name in to_get:
            local = self.cache_dir / str(year) / name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or local.stat().st_size == 0:
                self._download(self._year_url(year) + name, local)
            local_paths.append(local)

        return local_paths

    def _download(self, url: str, out_path: Path) -> None:
        with requests.get(url, stream=True, timeout=self.timeout_s) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

    @staticmethod
    def _detect_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
        # Common names in NWK/GridMET-style files
        lat_candidates = ["lat", "latitude", "LAT", "y"]
        lon_candidates = ["lon", "longitude", "LON", "x"]

        lat_name = next((c for c in lat_candidates if c in ds.coords or c in ds.variables), None)
        lon_name = next((c for c in lon_candidates if c in ds.coords or c in ds.variables), None)

        if lat_name is None or lon_name is None:
            raise KeyError("Could not find lat/lon in dataset. Inspect ds.coords and ds.variables.")

        return lat_name, lon_name

    @staticmethod
    def _detect_xy_dims(ds: xr.Dataset, lat_name: str, lon_name: str) -> tuple[str, str]:
        # Prefer dims attached to lat/lon
        lat_dims = ds[lat_name].dims
        lon_dims = ds[lon_name].dims
        # If lat/lon are 2D, they share (y,x)
        if len(lat_dims) == 2 and lat_dims == lon_dims:
            return lat_dims[0], lat_dims[1]
        # If 1D lat/lon, dims might be (y) and (x)
        if len(lat_dims) == 1 and len(lon_dims) == 1:
            return lat_dims[0], lon_dims[0]
        # Fallback: pick two non-time dims
        non_time = [d for d in ds.dims if d != "time"]
        if len(non_time) < 2:
            raise ValueError("Dataset does not appear to have 2 spatial dims.")
        return non_time[0], non_time[1]

    @staticmethod
    def _cells_gdf(lat: xr.DataArray, lon: xr.DataArray) -> tuple[gpd.GeoDataFrame, np.ndarray]:
        """
        Returns:
          gdf_cells with columns: cell_id, lon, lat, geometry
          cell_id_map: flattened cell_id ordering for (y,x) row-major
        """
        # Make lon/lat 2D grids
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon.values, lat.values)
        else:
            lat2d = np.asarray(lat.values)
            lon2d = np.asarray(lon.values)

        ny, nx = lat2d.shape
        cell_id_map = np.arange(ny * nx, dtype=int)

        lon_flat = lon2d.reshape(-1)
        lat_flat = lat2d.reshape(-1)

        gdf = gpd.GeoDataFrame(
            {
                "cell_id": cell_id_map,
                "lon": lon_flat,
                "lat": lat_flat,
            },
            geometry=gpd.points_from_xy(lon_flat, lat_flat),
            crs="EPSG:4326",
        )
        return gdf, cell_id_map
    
if __name__ == "__main__":
    from shapely.geometry import box

    client = GridMETClient(cache_dir="nwk_gridmet_cache")

    # Polygon around Los Angeles (bbox) in lon/lat
    la_poly = box(-119.05, 33.60, -117.50, 34.85)

    ds = client.query(
        polygon=la_poly,
        start="2024-07-01",
        end="2024-07-07",
        predicate="intersects",
    )

    print(ds.data_vars)
    print(ds.dims)