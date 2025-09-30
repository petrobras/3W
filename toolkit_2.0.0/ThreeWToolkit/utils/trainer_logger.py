import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional
from pathlib import Path


class TrainerLogger:
    """
    A utility class for logging training and optimization progress.
    Supports multiple file formats including JSON and Pickle.
    """

    # Supported file formats
    SUPPORTED_FORMATS = {"json", "pickle"}

    # Class-level logger to avoid duplicate handlers
    _logger: Optional[logging.Logger] = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        """Get or create the class logger with proper configuration."""
        if cls._logger is None:
            cls._logger = logging.getLogger("TrainerLogger")
            if not cls._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                cls._logger.addHandler(handler)
                cls._logger.setLevel(logging.INFO)
        return cls._logger

    @classmethod
    def log_optimization_progress(
        cls,
        progress_log: Dict[str, Any],
        log_dir: Union[str, Path] = "logs",
        file_format: str = "json",
    ) -> str:
        """
        Logs the optimization progress to a JSON or Pickle file.

        Args:
            progress_log (dict): Dictionary containing optimization results.
            log_dir (str): Directory where the log file will be saved.
            file_format (str): Format of the file: 'json' or 'pickle'.

        Returns:
            str: Path to the saved log file.
        """
        logger = cls._get_logger()

        if file_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"file_format must be one of {cls.SUPPORTED_FORMATS}")

        if not isinstance(progress_log, dict):
            raise TypeError("progress_log must be a dictionary")

        if not progress_log:
            raise ValueError("progress_log must be a non-empty dictionary")

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = ".json" if file_format == "json" else ".pkl"
        filename = f"log_{timestamp}{extension}"
        file_path = os.path.join(log_dir, filename)

        try:
            if file_format == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(progress_log, f, ensure_ascii=False)
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(progress_log, f)

            logger.info("Log saved at: %s", file_path)
        except Exception as e:
            logger.error("Failed to save optimization progress: %s", e)
            raise

        return file_path
