# ThreeWToolkit/logging_config.py
from __future__ import annotations

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

_CONFIGURED: bool = False
_LOG_FILE: Path | None = None


def setup_default_logging(
    logs_dir: Path, run_id: str, level: int = logging.INFO
) -> Path:
    global _CONFIGURED, _LOG_FILE

    if _CONFIGURED and _LOG_FILE is not None:
        return _LOG_FILE

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"run_{run_id}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(level)

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    root.addHandler(sh)
    root.addHandler(fh)

    _CONFIGURED = True
    _LOG_FILE = log_file
    return log_file
