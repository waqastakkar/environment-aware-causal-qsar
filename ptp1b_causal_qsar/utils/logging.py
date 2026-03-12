from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def configure_pipeline_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger("ptp1bqsar")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def dump_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
