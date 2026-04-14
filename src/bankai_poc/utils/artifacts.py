from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .io import dump_json, ensure_dir


def timestamp_ms() -> int:
    return int(time.time() * 1000)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("created_at_ms", timestamp_ms())
    dump_json(path, payload)


def results_dir(root: Path) -> Path:
    return ensure_dir(root / "results")

