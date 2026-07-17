from utils.logging_utils import *
import numpy as np
from pathlib import Path
import requests
import shutil
from utils.io_utils import make_cache_dir
from utils.rio_utils import download_file_safe
from abc import ABC, abstractmethod


# log_client = contextvars.ContextVar("log_client", default="library")
# log_task = contextvars.ContextVar("log_task", default="-")
# log_run_id = contextvars.ContextVar("log_run_id", default="-")


class BaseClient(ABC):
    """
    Superclass for all data clients.
    Provides self.logger automatically.
    """

    def __init__(
        self,
        *,
        cache_dir: str = None,
    ) -> None:
        self.save_dir = make_cache_dir(Path(cache_dir))
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = make_client_logger(self.__class__.__name__)
        self._client_token = set_client(self.__class__.__name__)
        self.cached_files = []
        self._session = requests.Session()

        self.logger.info(f"{self.__class__.__name__} client initialized")
    
    @abstractmethod
    def _query(self, *args, **kwargs):
        pass

    def _query_worker(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        try:
            out = self._query(*args, **kwargs)
            self._remove_cached_files()
            return out
        except Exception as e:
            self._remove_cached_files()
            self.logger.error(f"[ERROR] {self.__class__.__name__} failed: {e}")
            raise RuntimeError(f"[ERROR] {self.__class__.__name__} failed: {e}")
        
    def parallel_query(self, *args, **kwargs):
        try:
            out = self._query_worker(*args, **kwargs)
            self._remove_cached_files()
            return out
        except Exception as e:
            self._remove_cached_files()
            self.logger.error(f"[ERROR] {self.__class__.__name__} failed: {e}")
            raise RuntimeError(f"[ERROR] {self.__class__.__name__} failed: {e}")
        
    def _download(self, url: str) -> Path:
        fname = url.split("/")[-1]
        out = self.save_dir / fname
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.exists() and out.stat().st_size > 0:
            return out

        return download_file_safe(url, out, self._session)

    def _remove_cached_files(self):
        self.logger.info(f'cleaning up {self.__class__.__name__} cache')
        shutil.rmtree(self.save_dir)

    def _subset_dataset(self, lat, lon, ds, pool_n=None):
        if isinstance(lat, list):
            lat_min, lat_max = lat
            lon_min, lon_max = lon
            
            # 1. Generate boolean array masks along your coordinates
            lat_mask = (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
            lon_mask = (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
            
            # 2. Subset using .where() and drop the outer non-matching grid cells
            target = ds.where(lat_mask & lon_mask, drop=True)
            
            if (pool_n is not None) and (pool_n > 1):
                # Dynamically resolve dimension strings ('latitude'/'longitude' or 'y'/'x')
                lat_dim = target.latitude.dims[0]
                lon_dim = target.longitude.dims[0]
                
                # Ensure the dataset isn't smaller than the coarsen pooling factor
                if target.sizes[lat_dim] >= pool_n and target.sizes[lon_dim] >= pool_n:
                    coarsen_dict = {lat_dim: pool_n, lon_dim: pool_n}
                    target = target.coarsen(coarsen_dict, boundary="trim").mean()
        else:
            # Point fallback
            target = ds.sel(latitude=lat, longitude=lon, method="nearest")
            
        return target

    def _make_n_len_str(self, value, n):
        return f"%0{n}d" % (int(value),)

    def __del__(self):
        try:
            log_client.reset(self._client_token)
        except:
            pass