from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import time
import zipfile

import numpy as np
import pandas as pd
import requests
import xarray as xr

# optional but recommended for GeoTIFF -> xarray with CRS/transform
import rioxarray  # noqa: F401  (needed for .rio accessor)


try:
    from shapely.geometry import Polygon, MultiPolygon
except Exception as e:
    raise ImportError("Install shapely: pip install shapely") from e

Geom = Union["Polygon", "MultiPolygon"]


@dataclass
class LANDFIREClient:
    """
    LANDFIRE Product Service (LFPS) client.

    This client follows the standard LFPS workflow:
      1) submit a job with products + AOI bbox + email
      2) poll job status until it succeeds
      3) download a zip containing GeoTIFF(s) (+ metadata)
      4) open GeoTIFF(s) as xarray and return as xr.Dataset

    Notes:
      - LFPS expects AOI in WGS84 bbox order: xmin, ymin, xmax, ymax
      - Email is required by LFPS v2 (per public client docs like rlandfire)
    """
    base_url: str = "https://lfps.usgs.gov"
    cache_dir: Union[str, Path] = "lfps_cache"
    timeout_s: int = 120
    poll_interval_s: float = 5.0
    max_wait_s: int = 60 * 30  # 30 minutes default

    # LFPS endpoints (per guide summary + common LFPS v2 usage patterns)
    submit_endpoint: str = "/api/job/submit"
    status_endpoint: str = "/api/job/status"
    download_endpoint: str = "/api/job/{job_id}/download"

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        # you can add retries/backoff here if you want:
        # from requests.adapters import HTTPAdapter
        # from urllib3.util.retry import Retry

    # --------------------------
    # Public API
    # --------------------------
    def query(
        self,
        polygon: Geom,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        variables: Sequence[str],
        *,
        email: str,
        projection_wkid: Optional[int] = None,
        resolution_m: Optional[int] = None,
        out_name: Optional[str] = None,
        keep_zip: bool = True,
    ) -> xr.Dataset:
        """
        Parameters
        ----------
        polygon : shapely (lon/lat EPSG:4326) Polygon/MultiPolygon
        start, end : accepted but only used for metadata / your own product selection logic
        variables : LFPS product codes (e.g., "240EVC", "220CC_22", "200EVT")
        email : required by LFPS v2
        projection_wkid : optional output projection WKID (e.g., 32611)
        resolution_m : optional output resolution (30..9999 meters)
        out_name : optional base name for cached downloads
        keep_zip : keep zip in cache (True) or delete after extraction (False)

        Returns
        -------
        xr.Dataset
            Dataset with one DataArray per product (variable).
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        if not variables:
            raise ValueError("variables must be a non-empty list of LFPS product codes.")

        bbox = self._polygon_to_wgs84_bbox(polygon)
        job = self._submit_job(
            products=list(variables),
            aoi_bbox=bbox,
            email=email,
            projection_wkid=projection_wkid,
            resolution_m=resolution_m,
        )

        job_id = job.get("job_id") or job.get("id") or job.get("jobId")
        if not job_id:
            raise RuntimeError(f"Could not find job_id in submit response: keys={list(job.keys())}")

        status = self._poll_job(job_id)

        # download
        zip_path = self._download_zip(status, out_name=out_name or f"lfps_{job_id}.zip")
        extract_dir = self.cache_dir / f"unzipped_{job_id}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        if not keep_zip:
            try:
                zip_path.unlink()
            except OSError:
                pass

        ds = self._tifs_to_xarray_dataset(extract_dir, requested_products=list(variables))

        # attach some lightweight metadata
        ds.attrs.update(
            {
                "lfps_job_id": str(job_id),
                "lfps_status": str(status.get("status", "")),
                "request_start": str(start_ts),
                "request_end": str(end_ts),
                "aoi_bbox_wgs84": ",".join(map(str, bbox)),
            }
        )
        return ds

    # --------------------------
    # Job lifecycle
    # --------------------------
    def _submit_job(
        self,
        *,
        products: List[str],
        aoi_bbox: Tuple[float, float, float, float],
        email: str,
        projection_wkid: Optional[int],
        resolution_m: Optional[int],
    ) -> Dict:
        """
        Submit LFPS job.

        Common request pattern (per public LFPS clients like rlandfire) is:
          - products (list)
          - aoi bbox (xmin,ymin,xmax,ymax) in WGS84
          - email required
          - optional projection/resolution
        """
        url = self.base_url.rstrip("/") + self.submit_endpoint

        payload: Dict = {
            "Layer_List": ";".join(products),
            "Area_of_Interest": " ".join(list(map(str, aoi_bbox))),
            "Email": email,
        }
        if projection_wkid is not None:
            payload["projection"] = int(projection_wkid)
        if resolution_m is not None:
            payload["resolution"] = int(resolution_m)
            
        r = self._session.post(url, json=payload, timeout=self.timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"LFPS submit failed ({r.status_code}): {r.text[:400]}")
        return r.json()

    def _poll_job(self, job_id: str) -> Dict:
        url = self.base_url.rstrip("/") + self.status_endpoint
        payload = {"JobId" : job_id}
        t0 = time.time()
        while True:
            r = self._session.post(url, json = payload, timeout=self.timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(f"LFPS status failed ({r.status_code}): {r.text[:400]}")
            status = r.json()

            # Try common fields
            state = (status.get("status") or status.get("state") or "").lower()

            if state in {"succeeded", "success", "complete", "completed"}:
                return status
            if state in {"failed", "error"}:
                raise RuntimeError(f"LFPS job failed: {json.dumps(status)[:800]}")

            if (time.time() - t0) > self.max_wait_s:
                raise TimeoutError(f"Timed out waiting for LFPS job {job_id}")

            time.sleep(self.poll_interval_s)

    def _download_zip(self, status, *, out_name: str) -> Path:
        url = status.get('outputFile')
        out_path = self.cache_dir / out_name

        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        with self._session.get(url, stream=True, timeout=self.timeout_s) as r:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return out_path

    # --------------------------
    # Geo helpers
    # --------------------------
    @staticmethod
    def _polygon_to_wgs84_bbox(polygon: Geom) -> Tuple[float, float, float, float]:
        """
        LFPS wants bbox in WGS84 order xmin, ymin, xmax, ymax.
        Assumes polygon coords are lon/lat already.
        """
        minx, miny, maxx, maxy = polygon.bounds
        return float(minx), float(miny), float(maxx), float(maxy)

    # --------------------------
    # Output assembly
    # --------------------------
    def _tifs_to_xarray_dataset(self, extracted_dir: Path, requested_products: List[str]) -> xr.Dataset:
        """
        Convert extracted GeoTIFFs into a single xr.Dataset.

        LFPS commonly returns one or more .tif files (sometimes multi-band) in the zip.
        We:
          - open each tif with rioxarray.open_rasterio
          - convert band->variable when possible
          - merge into dataset
        """
        tifs = sorted(extracted_dir.rglob("*.tif")) + sorted(extracted_dir.rglob("*.tiff"))
        if not tifs:
            raise FileNotFoundError(f"No GeoTIFFs found in {extracted_dir}")

        data_vars = {}
        for tif in tifs:
            da = rioxarray.open_rasterio(tif)  # dims: band, y, x
            band_names = da.long_name

            # best effort variable naming:
            # If multi-band, name bands as <stem>_b{n}; otherwise just <stem>.
            if "band" in da.dims and da.sizes.get("band", 1) > 1:
                for bi in range(da.sizes["band"]):
                    import ipdb; ipdb.set_trace()
                    data_vars[band_names[bi]] = da.isel(band=bi).drop_vars("band", errors="ignore")
            else:
                # single band
                if "band" in da.dims:
                    da = da.isel(band=0).drop_vars("band", errors="ignore")
                data_vars[da.BandName] = da

        ds = xr.Dataset(data_vars=data_vars)

        # If you requested product codes, try to map them:
        # LFPS often names tif files with product codes inside; if not, you’ll still
        # get tif-stem-based names. You can rename after the fact.
        return ds
    
if __name__ == "__main__":
    from shapely.geometry import box

    poly = box(-120.5, 35.0, -118.0, 37.0)  # lon/lat
    vars_ = ["ELEV2020", "ASP2020"]  # product codes

    client = LANDFIREClient(cache_dir="data/lfps")

    ds = client.query(
        polygon=poly,
        start="2022-01-01",
        end="2022-12-31",
        variables=vars_,
        email="jaredgoldman@ucla.edu",
        projection_wkid=4326,   # optional
        resolution_m=90,        # optional
    )

    print(ds)
    print(list(ds.data_vars))