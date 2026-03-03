from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
import requests
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import shutil
from rio_utils import download_file_safe, open_geotiff_safe

# Optional but recommended for HDF-EOS -> xarray
import rasterio
from utils import add_lonlat_coords, make_cache_dir, buffer_polygon_meters, \
                    CLIENTS_DIR, CACHE_BASE_DIR
import os


Geom = Union[Polygon, MultiPolygon]


@dataclass
class MODISClient:
    """
    Download MODIS tiled products from LAADS 'archive/allData' for a polygon+date range,
    based on MODIS sinusoidal tile IDs (hXXvYY), and return an xarray.Dataset clipped to polygon.

    Example archive root: https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD14A1/YEAR/DOY/
    Token must be provided as Authorization: Bearer <token>.  (store it in modis.key)
    """
    product: str = "MYD14A1"
    collection: str = "61"
    cache_files: bool = False
    cached_files = []
    key_file: Union[str, Path] = os.path.join(CLIENTS_DIR, "modis.key")
    timeout_s: int = 120

    base_url: str = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData"

    # MODIS Sinusoidal sphere radius used in MODLAND grid (common constant)
    R: float = 6371007.181

    def __init__(self):
        self.save_dir = make_cache_dir(Path(f"{CACHE_BASE_DIR}/modis"))
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._token = self._read_token(self.key_file)
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._token}"})
        
    # -------------------------
    # Public API
    # -------------------------
    def query(self,
        polygon: Geom,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        variables: Sequence[str] = None,
        prefer_subdatasets_regex: bool = True,
        drop_outside: bool = True,
        ) -> xr.Dataset:
        try:
            return self._query(
                polygon, start, end, variables,
                prefer_subdatasets_regex = prefer_subdatasets_regex, 
                drop_outside = drop_outside)
        except Exception as e:
            self._remove_cached_files()
            raise RuntimeError(f"[ERROR] MODIS failed: {e}")
        
    def _query(
        self,
        polygon: Geom,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        prefer_subdatasets_regex: bool = True,
        drop_outside: bool = True,
    ) -> xr.Dataset:
        """
        Download all tiles intersecting polygon for each day in [start,end] and return xarray.Dataset.

        variables:
          - If None: reads ALL subdatasets (often too big)
          - If list: selects subdatasets whose names match your strings.
            If prefer_subdatasets_regex=True, each entry is treated as a regex.

        drop_outside:
          - If True: mask outside polygon and drop fully-empty rows/cols.

        Returns:
          xr.Dataset with dims typically (time, y, x)
        """
        poly = buffer_polygon_meters(polygon, resolution_m=500, factor = 1)
        if isinstance(poly, MultiPolygon):
            poly = MultiPolygon(poly.geoms)  # keep; prepared uses it fine

        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        # compute intersecting tiles once for polygon
        tiles = self.tiles_for_polygon(poly)

        # build day list (daily directories in LAADS path)
        days = pd.date_range(start_ts, end_ts, freq="D")

        # download needed files
        local_files = []
        doy_buckets = {}
        fname_map = {}
        for d in days:
            year = d.year
            doy = int(d.strftime("%j"))
            bucket = int((doy - 1)/8)*8+1
            position = (doy - 1) % 8 + 1
            my_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=bucket + position - 2)
            if not bucket in doy_buckets.keys():
                fname = self._download_day_tiles(year, bucket, tiles)
                self.cached_files.extend(fname)
                local_files.extend(fname)
                doy_buckets[bucket] = [position]
                for fn in fname:
                    fname_map[fn] = {'positions' : [position], 'dates' : [my_date]}
            else:
                for fn in fname:
                    fname_map[fn]['positions'].append(position)
                    fname_map[fn]['dates'].append(my_date)

        if not local_files:
            raise FileNotFoundError("No matching files found/downloaded for requested range/tiles.")

        # Open files -> xarray list per time, then concat
        dsets = []
        for fp in local_files:
            positions = fname_map[fn]['positions']
            my_dates = fname_map[fn]['dates']
            try:
                ds = self._open_hdf_as_dataset(fp, variables=variables, regex=prefer_subdatasets_regex, days = positions)
            except:
                raise RuntimeError("unable to open MODIS data...")
            
            if my_dates is not None:
                ds = ds.expand_dims(time=my_dates)
            dsets.append(ds)

        ds_all = xr.concat(dsets, dim="time")

        # Clip/mask to polygon (works if ds has lon/lat grids or x/y in a projected CRS via rioxarray)
        if drop_outside:
            ds_all = self._mask_to_polygon(ds_all, poly)
        
        ds_all = add_lonlat_coords(ds_all).load()

        self._remove_cached_files()
        return ds_all

    # -------------------------
    # Tile selection
    # -------------------------
    def tiles_for_polygon(self, polygon: Geom) -> list[str]:
        """
        Return list of tile IDs like ['h08v05', ...] that intersect the polygon.

        Uses MODIS Sinusoidal tiling math; then refines by intersecting with tile corner polygon
        converted back to lon/lat.
        """
        minx, miny, maxx, maxy = polygon.bounds

        # sample corners (and a few midpoints) to bound in sinusoidal meters
        pts = [
            (minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy),
            ((minx+maxx)/2, miny), ((minx+maxx)/2, maxy),
            (minx, (miny+maxy)/2), (maxx, (miny+maxy)/2),
        ]
        xs, ys = zip(*(self._ll_to_sinu(lon, lat) for lon, lat in pts))
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        tile_size = (2 * np.pi * self.R) / 36.0  # ~1111950.52 m
        x0 = -np.pi * self.R                # left boundary
        y_top = (np.pi * self.R) / 2.0      # top boundary

        h0 = int(np.floor((x_min - x0) / tile_size))
        h1 = int(np.floor((x_max - x0) / tile_size))
        v0 = int(np.floor((y_top - y_max) / tile_size))
        v1 = int(np.floor((y_top - y_min) / tile_size))

        # clamp to valid tile index ranges
        h0, h1 = max(0, h0), min(35, h1)
        v0, v1 = max(0, v0), min(17, v1)

        # refine candidates by intersecting polygon with tile-corner polygon
        candidates = []
        for h in range(h0, h1 + 1):
            for v in range(v0, v1 + 1):
                tile_poly_ll = self._tile_polygon_lonlat(h, v, tile_size, x0, y_top)
                if tile_poly_ll.intersects(polygon):
                    candidates.append(f"h{h:02d}v{v:02d}")

        return sorted(set(candidates))

    def _ll_to_sinu(self, lon_deg: float, lat_deg: float) -> tuple[float, float]:
        """Lon/lat (deg) -> MODIS Sinusoidal meters (approx)."""
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        x = self.R * lon * np.cos(lat)
        y = self.R * lat
        return float(x), float(y)

    def _sinu_to_ll(self, x: float, y: float) -> tuple[float, float]:
        """MODIS Sinusoidal meters -> lon/lat degrees (approx inverse)."""
        lat = y / self.R
        # avoid div-by-zero near poles
        coslat = np.cos(lat)
        coslat = np.where(np.abs(coslat) < 1e-12, 1e-12, coslat)
        lon = x / (self.R * coslat)
        return float(np.rad2deg(lon)), float(np.rad2deg(lat))

    def _tile_polygon_lonlat(self, h: int, v: int, tile_size: float, x0: float, y_top: float) -> Polygon:
        """Approx tile footprint from tile corner points, converted to lon/lat polygon."""
        # tile bounds in sinusoidal meters
        x_left = x0 + h * tile_size
        x_right = x_left + tile_size
        y_upper = y_top - v * tile_size
        y_lower = y_upper - tile_size

        corners = [
            (x_left, y_upper),
            (x_right, y_upper),
            (x_right, y_lower),
            (x_left, y_lower),
            (x_left, y_upper),
        ]
        ll = [self._sinu_to_ll(x, y) for x, y in corners]
        return Polygon(ll)

    # -------------------------
    # Remote listing + download
    # -------------------------
    def _dir_url(self, year: int, doy: int) -> str:
        return f"{self.base_url.rstrip('/')}/{self.collection}/{self.product}/{year}/{doy:03d}/"

    def _list_dir(self, year: int, doy: int) -> list[str]:
        url = self._dir_url(year, doy)
        r = self._session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        # parse links to .hdf files
        return sorted(set(re.findall(r'href="([^"]+\.hdf)"', r.text)))

    def _download_day_tiles(self, year: int, doy: int, tiles: Sequence[str]) -> list[str]:
        names = self._list_dir(year, doy)
        if not names:
            return []

        out = []
        for tile in tiles:
            # match tile in filename
            matches = [n for n in names if f".{tile}." in n]
            for name in matches:
                local = self.save_dir / name.split(os.sep)[-1]
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists() or local.stat().st_size == 0:
                    self._download_file(name, local)
                out.append(str(local))
        return out

    def _download_file(self, filename: str, out_path: Path) -> None:
        url = filename
        download_file_safe(url, out_path, self._session)

    def _remove_cached_files(self):
        if not self.cache_files:
            print('cleaning up MODIS cache')
            [os.remove(fname) for fname in self.cached_files if os.path.exists(fname)]
            shutil.rmtree(self.save_dir)

    @staticmethod
    def _read_token(key_file: Union[str, Path]) -> str:
        token = Path(key_file).read_text().strip()
        if not token:
            raise ValueError(f"No token found in {key_file}")
        return token

    # -------------------------
    # HDF -> xarray
    # -------------------------
    def _open_hdf_as_dataset(
        self,
        hdf_path: str,
        *,
        variables: Optional[Sequence[str]],
        regex: bool,
        days: List[int]
    ) -> xr.Dataset:
        """
        Opens HDF subdatasets into a single xr.Dataset.

        'variables' matches subdataset names (e.g., 'FireMask', 'MaxFRP', etc. depending on product).
        """
        with rasterio.open(hdf_path) as src:
            subds = list(getattr(src, "subdatasets", []))

        if not subds:
            raise RuntimeError(
                f"No HDF subdatasets detected in {hdf_path}. "
                "This usually means your GDAL/rasterio build can't read HDF4/HDF-EOS."
            )

        # filter subdatasets
        if variables is None:
            chosen = subds
        else:
            chosen = []
            for s in subds:
                # subdataset strings often end with the SDS name
                for v in variables:
                    if (re.search(v, s) if regex else (v in s)):
                        chosen.append(s)
                        break
            chosen = sorted(set(chosen))
            if not chosen:
                raise KeyError(f"No subdatasets matched variables={variables}. Example subdataset: {subds[0]}")

        # open each subdataset with rioxarray
        dataarrays = []
        names = []
        
        for s in chosen:
            da =  open_geotiff_safe(s).sel(band=days) # rioxarray.open_rasterio(s, masked=True).sel(band=days)
            # name from last colon token
            nm = s.split(":")[-1]
            dataarrays.append(da)
            names.append(nm)

        ds = xr.merge([da.to_dataset(name=nm) for da, nm in zip(dataarrays, names)])
        return ds

    # -------------------------
    # Polygon mask (simple centroid-in-polygon mask using lon/lat grids if available)
    # -------------------------
    def _mask_to_polygon(self, ds: xr.Dataset, polygon: Geom) -> xr.Dataset:
        # Many MODIS land HDF subdatasets will carry an implicit sinusoidal CRS in rio attrs.
        # If ds has 2D lon/lat already, use them. Otherwise, rely on rioxarray clip if possible.
        try:            # Reproject polygon into raster CRS if needed
            if ds.rio.crs is not None and str(ds.rio.crs).upper() != "EPSG:4326":
                gdf = gdf.to_crs(ds.rio.crs)

            mask = geometry_mask(
                geometries=[geom.__geo_interface__ for geom in gdf.geometry],
                out_shape=(ds.sizes["y"], ds.sizes["x"]),
                transform=ds.rio.transform(),
                invert=True,         
                all_touched=True,
            )
            return ds.where(mask)
            # return ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
        except Exception:
            # fallback: do nothing (or you can implement a manual mask if you add lon/lat coords)
            return ds
        
if __name__ == "__main__":
    from shapely.geometry import box

    client = MODISClient()

    # Los Angeles-ish bounding polygon (lon/lat)
    poly = box(-119.05, 33.60, -117.50, 34.85)

    ds = client.query(
        polygon=poly,
        start="2025-07-01",
        end="2025-07-03",
        variables=None,  # change to the SDS names you want
    )

    print(ds)