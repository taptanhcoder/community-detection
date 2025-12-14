from __future__ import annotations

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("osnclusters")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
