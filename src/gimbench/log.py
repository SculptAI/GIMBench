import logging

from logging.config import dictConfig
from pathlib import Path
from sys import stdout
from typing import Any


APP_NAME = "gimbench"

LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        },
        "no_datetime": {"format": "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": stdout,
            "formatter": "no_datetime",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": None,
            "maxBytes": 1024**2 * 10,
            "backupCount": 10,
            "formatter": "standard",
        },
    },
    "loggers": {
        APP_NAME: {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}


def get_logger(name: str, log_dir: str = ".", log_filename: str = "eval.log") -> logging.Logger:
    """returns the project logger, scoped to a child name if provided
    Args:
        name: will define a child logger
    """

    def _setup_logfile(log_dir: str = ".", log_filename: str = "eval.log") -> Path:
        """ensure the logger filepath is in place

        Returns: the logfile Path
        """
        logfile = Path(log_dir) / log_filename
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.touch(exist_ok=True)
        return logfile

    logfile = _setup_logfile(log_dir, log_filename)
    LOGGING_CONFIG["handlers"]["file"]["filename"] = str(logfile)

    dictConfig(LOGGING_CONFIG)

    return logging.getLogger("" if name.startswith(APP_NAME) else APP_NAME).getChild(name)
