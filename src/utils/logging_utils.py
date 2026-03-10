from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import contextvars

_worker_log_queue = None

log_client = contextvars.ContextVar("log_client", default="-")
log_cp = contextvars.ContextVar("log_cp", default="-")

def set_client(name: str):
    return log_client.set(name)


def set_cp(cp_idx):
    return log_cp.set(str(cp_idx))

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "client=%(client)s | "
    "cp=%(cp_idx)s | "
    "pid=%(process)d | thread=%(threadName)s | "
    "func=%(funcName)s | "
    "%(message)s"
)

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:

        if not hasattr(record, "client"):
            record.client = log_client.get()

        if not hasattr(record, "cp_idx"):
            record.cp_idx = log_cp.get()

        return True

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:

        if not hasattr(record, "client"):
            record.client = log_client.get()

        if not hasattr(record, "cp_idx"):
            record.cp_idx = log_cp.get()

        return True

class SafeExtraFormatter(logging.Formatter):
    """
    Formatter that ensures custom fields always exist.
    """
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "client"):
            record.client = "-"
        return super().format(record)


class ClientLoggerAdapter(logging.LoggerAdapter):
    """
    Injects client-specific context into each record.
    """
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        kwargs["extra"] = {**self.extra, **extra}
        return msg, kwargs


@dataclass
class ClientLogger:
    """
    Thin wrapper around LoggerAdapter for nicer method names.
    """
    _logger: ClientLoggerAdapter

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)

def start_log_listener(log_file: Path, level: int = logging.DEBUG):
    """
    Start a queue listener that writes all logs to a single file.
    Returns (log_queue, listener).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    manager = mp.Manager()
    log_queue = manager.Queue()
    
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ContextFilter())

    ch.setLevel(level)
    ch.setFormatter(formatter)
    ch.addFilter(ContextFilter())

    listener = logging.handlers.QueueListener(
        log_queue,
        file_handler,
        respect_handler_level=True,
    )

    listener.start()

    return log_queue, listener

def configure_queue_logging(log_queue: mp.Queue, level: int = logging.DEBUG) -> None:
    """
    Configure the current process so all log records go to the queue.
    Call this once per worker process/thread entrypoint.
    """
    formatter = logging.Formatter(LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    qh = logging.handlers.QueueHandler(log_queue)
    ch = logging.StreamHandler()

    ch.setFormatter(formatter)
    ch.addFilter(ContextFilter())

    root.addHandler(ch)
    root.addHandler(qh)
    root.addFilter(ContextFilter)

def init_worker_logging(log_queue):
    global _worker_log_queue
    _worker_log_queue = log_queue
    formatter = logging.Formatter(LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    qh = logging.handlers.QueueHandler(_worker_log_queue)
    ch = logging.StreamHandler()

    ch.setFormatter(formatter)
    ch.addFilter(ContextFilter())

    root.addHandler(qh)
    root.addHandler(ch)

def make_client_logger(client_name: str) -> ClientLogger:
    """
    Create a logger with client context.
    """
    base = logging.getLogger(client_name)
    adapter = ClientLoggerAdapter(base, {"client": client_name})
    return ClientLogger(adapter)

def reset_tokens(cp_token = None, client_token = None):
    try:
        log_cp.reset(cp_token)
    except Exception:
        pass

    try:
        log_client.reset(client_token)
    except Exception:
        pass