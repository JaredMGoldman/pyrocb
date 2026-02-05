from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict
import os
from io import StringIO, BytesIO
from utils import FIRMS_KEY_FNAME, CLIENTS_PATH, get_repo_root, set_env_var

import pandas as pd
import geopandas as gpd
import requests
from zipfile import ZipFile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FirmsSources = Literal["LANDSAT_NRT", # [US/Canada only] (LANDSAT Near Real-Time, Real-Time and Ultra Real-Time *)
    "MODIS_NRT", # (MODIS Near Real-Time, Real-Time and Ultra Real-Time *)
    "MODIS_SP", # (MODIS Standard Processing)
    "VIIRS_NOAA20_NRT", # (VIIRS NOAA-20 Near Real-Time, Real-Time and Ultra Real-Time *)
    "VIIRS_NOAA20_SP", # (VIIRS NOAA-20 Standard Processing)
    "VIIRS_NOAA21_NRT", # (VIIRS NOAA-21 Near Real-Time, Real-Time and Ultra Real-Time *)
    "VIIRS_SNPP_NRT", # (VIIRS Suomi-NPP Near Real-Time, Real-Time and Ultra Real-Time *)
    "VIIRS_SNPP_SP" # (VIIRS Suomi-NPP Standard Processing)
]

FirmsSensor = Literal[
    "c6.1", # (MODIS Near Real-Time, Real-Time and Ultra Real-Time *)
    "landsat", # (LANDSAT Near Real-Time, Real-Time and Ultra Real-Time *)
    "suomi-npp-viirs-c2", # (VIIRS Suomi-NPP Near Real-Time, Real-Time and Ultra Real-Time *)
    "noaa-20-viirs-c2",# (VIIRS NOAA-20 Near Real-Time, Real-Time and Ultra Real-Time *)
    "noaa-21-viirs-c2", # (VIIRS NOAA-21 Near Real-Time, Real-Time and Ultra Real-Time *)
]
FirmsRegion = Literal[
    "canada",
    "alaska",
    "usa_contiguous_and_hawaii"
]
FirmsSpans = Literal[
    "24h", "48h", "72h", "7d"
]

@dataclass(frozen=True)
class FirmsClient:
    map_key: str
    base_url: str = "https://firms.modaps.eosdis.nasa.gov"
    timeout: int = 60  # seconds

    def __post_init__(self):
        if not self.map_key:
            raise ValueError("FIRMS MAP_KEY is required.")

    @classmethod
    def from_env(cls) -> "FirmsClient":
        return cls(map_key=os.environ.get("FIRMS_MAP_KEY", ""))

    def _session(self) -> requests.Session:
        # Build a session with retry/backoff (pythonic & production-friendly)
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        return s
    
    def fire_detection(
            self,
            sensor: FirmsSensor,
            region: FirmsRegion,
            datespan: FirmsSpans,
            crs: str = "EPSG:4326"
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Fetch FIRMS detections in CSV for a bounding box.
        - sensor: (west, south, east, north)
        - region: alaska, canada, usa & hawaii
        - date: optional start date (YYYY-MM-DD). If None, returns most recent.
        """

        # URL template per FIRMS US/Canada docs
        # /usfs/api/kml_fire_footprints/?region=[REGION]&date_span=[DATE_SPAN]&sensor=[SENSOR]
        path = f"/usfs/api/kml_fire_footprints/?region={region}&date_span={datespan}&sensor={sensor}"
        url = self.base_url.rstrip("/") + path

        with self._session() as s:
            res = s.get(url, timeout=self.timeout)
            # FIRMS returns KML; errors can come back as HTML/text
            if res.status_code != 200:
                raise RuntimeError(f"FIRMS request failed ({res.status_code}): {res.text[:300]}")

        with ZipFile(BytesIO(res.content)) as z:
            # Find the first .kml inside the KMZ (often "doc.kml")
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("KMZ contains no .kml file.")
            kml_name = kml_names[0]

            kml_bytes = z.read(kml_name)

        # geopandas can read KML from a BytesIO in many environments
        # Explicit driver helps
        layers = gpd.list_layers(BytesIO(kml_bytes)).name.to_list()
        return {layer :  gpd.read_file(BytesIO(kml_bytes), driver="KML", layer=layer) for layer in layers}

    def area_csv(
        self,
        source: FirmsSources,
        bbox_wsen: Tuple[float, float, float, float],
        day_range: int = 1,
        date: Optional[str] = None,  # "YYYY-MM-DD"
        crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Fetch FIRMS detections in CSV for a bounding box.
        - bbox_wsen: (west, south, east, north)
        - day_range: 1..5 per docs
        - date: optional start date (YYYY-MM-DD). If None, returns most recent.
        """
        if not (1 <= day_range <= 5):
            raise ValueError("day_range must be 1..5 for FIRMS USFS Area API.")

        west, south, east, north = bbox_wsen
        area = f"{west},{south},{east},{north}"

        # URL template per FIRMS US/Canada docs
        # /usfs/api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE](/[DATE])
        path = f"/usfs/api/area/csv/{self.map_key}/{source}/{area}/{day_range}"
        if date:
            path += f"/{date}"

        url = self.base_url.rstrip("/") + path

        with self._session() as s:
            r = s.get(url, timeout=self.timeout)
            # FIRMS returns CSV; errors can come back as HTML/text
            if r.status_code != 200:
                raise RuntimeError(f"FIRMS request failed ({r.status_code}): {r.text[:300]}")
            df = pd.read_csv(StringIO(r.text))
            return gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df["longitude"], df["latitude"]), crs=crs)
        
        
if __name__ == "__main__":
    firms_key_path = os.path.join(get_repo_root(),
                                  CLIENTS_PATH,
                                  FIRMS_KEY_FNAME)

    set_env_var("FIRMS_MAP_KEY", firms_key_path)

    client = FirmsClient.from_env()

    df_area = client.area_csv(
        source="VIIRS_NOAA20_SP",
        bbox_wsen=(-85, -57, -32, 14),
        day_range=2,
        date="2025-08-01",
    )
    print(df_area.head())

    df_detection = client.fire_detection(
        sensor = "c6.1",
        region = "canada",
        datespan='24h'
    )
    [print(f"layer {k}:\n{v.head()}") for k, v in df_detection.items()]