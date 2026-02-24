from __future__ import annotations

from pathlib import Path
import os, time, random, threading
from collections import defaultdict
import requests
import xarray as xr
import rioxarray as rxr
import rasterio

def _validate_tif_readable(path: Path) -> None:
    # tiny window read: cheap but catches most truncated/corrupt tiffs
    with rasterio.open(path) as src:
        w = rasterio.windows.Window(0, 0, min(64, src.width), min(64, src.height))
        _ = src.read(1, window=w)

def validate_tif_download(
    url: str,
    out_path: Path,
    session: requests.Session,
    ):
    download_file_safe(url, out_path, session)
    try:
        _validate_tif_readable(path)
    except Exception:
        path.unlink(missing_ok=True)
        path = download_file_safe(url, out_path, session, tries=3)
        _validate_tif_readable(path)
    return path

def open_geotiff_safe(
    path: str,
    *,
    masked: bool = True,
    chunks: dict | None = None,
    squeeze_band: bool = True,
    load: bool = False,
) :
    """
    Download GeoTIFF safely then open with rioxarray.
    """

    da = rxr.open_rasterio(path, masked=masked, chunks=chunks)
    if load:
        da = da.load()  # or da.compute() if dask-backed
    return da

def open_netcdf_safe(
    url: str,
    out_path: Path,
    session: requests.Session,
    *,
    engine: str | None = None,
    load: bool = False,
    open_kwargs: dict | None = None,
) -> xr.Dataset:
    """
    Download NetCDF safely then open with xarray.
    """
    open_kwargs = open_kwargs or {}

    path = download_file_safe(url, out_path, session)

    try:
        ds = xr.open_dataset(path, engine=engine, **open_kwargs)
        if load:
            ds = ds.load()
        return ds
    except Exception:
        # If it can't open, assume corruption/truncation and retry once by deleting
        path.unlink(missing_ok=True)
        path = download_file_safe(url, out_path, session, tries=3)
        ds = xr.open_dataset(path, engine=engine, **open_kwargs)
        if load:
            ds = ds.load()
        return ds
    
def open_netcdf_safe_cached(
    url: str,
    out_path: Path,
    session: requests.Session,
    *,
    engine: str | None = None,
    load: bool = False,
    open_kwargs: dict | None = None,
) -> xr.Dataset:
    """
    Download NetCDF safely then open with xarray.
    """
    open_kwargs = open_kwargs or {}

    try:
        ds = xr.open_dataset(out_path, engine=engine, **open_kwargs)
        if load:
            ds = ds.load()
        return ds
    except Exception:
        # If it can't open, assume corruption/truncation and retry once by deleting
        path.unlink(missing_ok=True)
        path = download_file_safe(url, out_path, session, tries=3)
        ds = xr.open_dataset(path, engine=engine, **open_kwargs)
        if load:
            ds = ds.load()
        return ds

_path_locks = defaultdict(threading.Lock)

def download_file_safe(
    url: str,
    out_path: Path,
    session: requests.Session,
    *,
    tries: int = 5,
    chunk: int = 1 << 20,
) -> Path:
    """
    Atomic download to out_path:
      - writes to out_path.part
      - verifies Content-Length if provided
      - os.replace() to commit atomically
      - retries with exponential backoff
      - per-path thread lock prevents concurrent writes to same file
    """
    lock = _path_locks[str(out_path)]
    with lock:
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")

        for k in range(tries):
            try:
                with session.get(url, stream=True, timeout=180) as r:
                    r.raise_for_status()
                    expected = int(r.headers.get("Content-Length", "0") or 0)

                    with open(tmp, "wb") as f:
                        for b in r.iter_content(chunk_size=chunk):
                            if b:
                                f.write(b)

                if expected and tmp.stat().st_size != expected:
                    raise IOError(f"Truncated download: got {tmp.stat().st_size}, expected {expected}")

                os.replace(tmp, out_path)  # atomic commit
                return out_path

            except Exception:
                tmp.unlink(missing_ok=True)
                out_path.unlink(missing_ok=True)
                if k == tries - 1:
                    raise
                time.sleep((2**k) + random.random())

    return out_path