"""Load config.toml and expose a stable hash of the resolved values."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(value) if isinstance(value, dict) else value

    @property
    def hash(self) -> str:
        blob = json.dumps(self, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def data_dir(self) -> Path:
        p = Path(self["paths"]["data_dir"]).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, "rb") as f:
        return Config(tomllib.load(f))
