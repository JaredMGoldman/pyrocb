from utils.logging_utils import *
from pathlib import Path
import requests
import shutil
from utils.utils import make_cache_dir
from utils.rio_utils import download_file_safe

# log_client = contextvars.ContextVar("log_client", default="library")
# log_task = contextvars.ContextVar("log_task", default="-")
# log_run_id = contextvars.ContextVar("log_run_id", default="-")


class BaseClient:
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

    def __del__(self):
        try:
            log_client.reset(self._client_token)
        except:
            pass