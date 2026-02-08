# https://gis1.servirglobal.net/data/esi/4WK/2017/
from __future__ import annotations

from pathlib import Path
import requests
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Point
from datetime import date

class ESIClient:
    def __init__(self, year: str, month: int, day: int):
        self.base_url = "https://gis1.servirglobal.net/data/esi/4WK/"
        self.year = year
        self.fname = f"DFPPM_4WK_{year}{self.day_of_year(month, day)}.tif"

    def download_file(self, out_dir: Path, chunk_size: int = 1 << 20) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        with requests.get(f"{self.base_url}/{self.year}/{self.fname}", stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_dir / self.fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return out_dir / self.fname

    def day_of_year(self, month: int, day: int) -> int:
        d = date(self.year, month, day)
        doy = d.timetuple().tm_yday

        snapped = 1 + ((doy - 1) // 7) * 7
        return f"{snapped:03d}"

    def raster_to_points_gdf(self, tif_path: Path, band: int = 1, stride: int = 1) -> gpd.GeoDataFrame:
        """
        Convert raster pixels to point GeoDataFrame (centroid points).
        stride=1 keeps all pixels; stride=5 keeps every 5th pixel (much smaller).
        """
        with rasterio.open(tif_path) as src:
            arr = src.read(band)
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

            # Downsample by stride to keep size sane
            arr_s = arr[::stride, ::stride]
            rows, cols = np.where(~np.isnan(arr_s) if nodata is None else (arr_s != nodata))

            # Map back to full-res indices
            rows = rows * stride
            cols = cols * stride

            values = arr[rows, cols]

            # pixel center coords
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
            geom = [Point(x, y) for x, y in zip(xs, ys)]

        gdf = gpd.GeoDataFrame({"value": values}, geometry=geom, crs=crs)
        return gdf


    def raster_to_polygons_gdf(self, tif_path: Path, band: int = 1) -> gpd.GeoDataFrame:
        """
        Polygonize raster into polygons for each contiguous region of equal value.
        Warning: can still be large for continuous rasters.
        """
        with rasterio.open(tif_path) as src:
            arr = src.read(band)
            nodata = src.nodata
            mask = np.ones(arr.shape, dtype=bool) if nodata is None else (arr != nodata)

            geoms = []
            vals = []
            for geom, val in shapes(arr, mask=mask, transform=src.transform):
                geoms.append(shape(geom))
                vals.append(val)

            gdf = gpd.GeoDataFrame({"value": vals}, geometry=geoms, crs=src.crs)
        return gdf


if __name__ == "__main__":
    year = 2018
    month = 6
    day = 11

    EC = ESIClient(year, month, day)
    tif_path = EC.download_file(Path("data"))

    # Option A: points (use stride to reduce size)
    gdf_pts = EC.raster_to_points_gdf(tif_path)
    print("Points:", gdf_pts.shape, gdf_pts.crs)

    # Option B: polygons (can be huge if raster has many unique values)
    gdf_poly = EC.raster_to_polygons_gdf(tif_path)
    print("Polygons:", gdf_poly.shape, gdf_poly.crs)
