from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from utils.utils import EARTHACCESS_KEY_NAME, CLIENTS_DIR, CACHE_BASE_DIR, \
                    buffer_polygon_meters, make_cache_dir

import os
import numpy as np
import pandas as pd
import xarray as xr
import earthaccess as ea
from shapely.geometry import Polygon, MultiPolygon


EarthAccessSources = Literal[
    "VJ114IMG",
    "VJ114MOD",
    "VNP14IMG",
    "VNP14IMG",
]

Geom = Union[Polygon, MultiPolygon]

class EarthAccessClient:
    map_key_path: str  = os.path.join(CLIENTS_DIR,
                                  EARTHACCESS_KEY_NAME)
    
    timeout: int = 60
    max_retries: int = 5
    backoff_factor: float = 0.8
    map_key: str = ""
    if os.path.exists(map_key_path):
            with open(map_key_path, 'r') as f:
                map_key = f.read()
                os.environ['EARTHDATA_TOKEN'] = map_key
    else:
        raise ValueError(f"Save your Earthaccess Key key to {map_key_path}")
    
    def __init__(self):
        self.save_dir = make_cache_dir(Path(os.path.join(CACHE_BASE_DIR, "earthdata")))

    # -------------------------
    # Public API (xarray)
    # -------------------------
    def query(
        self,
        polygon: Geom,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        variables: List[str],
        *,
        source: Sequence[EarthAccessSources] = "VJ114IMG", 
        version: str = "002",
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
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("end must be >= start")
        
        buffered_polygon = buffer_polygon_meters(polygon, resolution_m=375, factor = 1)

        out: Dict[str, xr.Dataset] = {}
        ea.login(strategy ="netrc")
        granules = ea.search_data(
            short_name=source,
            polygon=buffered_polygon.exterior.coords[::-1],
            version = version,
            temporal=(start_ts, end_ts),
            count=-1
        )

        fnames = ea.download(granules, self.save_dir)
        for fname in fnames:
            ds = self.subset_ds(fname, buffered_polygon, variables, start, end)


        return out

    # -------------------------
    # Utilities
    # -------------------------
    def subset_ds(self, fname, polygon, variables, start, end):
        ds = xr.open_dataset(fname)
        import ipdb; ipdb.set_trace()

    @staticmethod
    def _polygon_bounds_wsen(polygon: Geom) -> Tuple[float, float, float, float]:
        minx, miny, maxx, maxy = polygon.bounds
        return (float(minx), float(miny), float(maxx), float(maxy))

    

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
    client = EarthAccessClient()

    poly = box(-119.05, 33.60, -117.50, 34.85)

    ds = client.query(
        polygon=poly,
        start="2025-01-07",
        end="2025-01-31",
        variables = ['frp'],
        source="VJ114IMG",
    )

    print(ds)
    print(ds.coords)
    print(ds.data_vars)