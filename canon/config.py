from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root: Path

    @property
    def data_dir(self) -> Path:
        override = os.getenv("CANON_DATA_DIR")
        return Path(override) if override else self.root / self.raw["paths"]["data_dir"]

    @property
    def reports_dir(self) -> Path:
        override = os.getenv("CANON_REPORTS_DIR")
        return Path(override) if override else self.root / self.raw["paths"]["reports_dir"]

    @property
    def fixtures_dir(self) -> Path:
        return self.data_dir / "fixtures"

    @property
    def contact_email(self) -> str:
        return self.raw["project"]["contact_email"]


def load_settings(path: str | None = None) -> Settings:
    settings_path = Path(path or os.getenv("CANON_SETTINGS", "conf/settings.toml"))
    if not settings_path.is_absolute():
        settings_path = Path.cwd() / settings_path
    with settings_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = settings_path.parent.parent
    return Settings(raw=raw, root=root)
