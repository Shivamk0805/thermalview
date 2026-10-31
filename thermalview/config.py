"""Configuration loading. All paths are resolved relative to the project root."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"


class Config(dict):
    """A dict with helpers for paths. Sections are plain dicts."""

    root: Path

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else self.root / p

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for section, values in (overrides or {}).items():
        data.setdefault(section, {}).update(values)
    cfg = Config(data)
    cfg.root = path.resolve().parent
    if os.environ.get("THERMALVIEW_DB"):                 # e.g. a Docker volume
        cfg["paths"]["database"] = os.environ["THERMALVIEW_DB"]
    return cfg
