from __future__ import annotations

import os
from pathlib import Path


LOADED = False
ALLOWED_ENV_KEYS = {
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "COHERE_API_KEY",
    "CANON_DATA_DIR",
    "CANON_REPORTS_DIR",
    "CANON_SETTINGS",
}


def load_local_env(path: Path | None = None) -> None:
    global LOADED
    if LOADED:
        return
    LOADED = True
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ALLOWED_ENV_KEYS or key in os.environ:
            continue
        os.environ[key] = clean_value(value)


def clean_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped
