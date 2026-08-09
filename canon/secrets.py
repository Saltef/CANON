from __future__ import annotations

import os
import threading
from pathlib import Path


LOADED = False
LOAD_LOCK = threading.Lock()
ALLOWED_ENV_KEYS = {
    "OPENROUTER_API_KEY",
    "COHERE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "CANON_VECTORSTORE",
    "CANON_ALLOWED_ORIGINS",
    "CANON_API_KEY",
    "CANON_BASIC_AUTH_USER",
    "CANON_BASIC_AUTH_PASSWORD",
    "CANON_REQUIRE_AUTH",
    "CANON_MAX_CONCURRENCY",
    "CANON_MAX_QUEUE_DEPTH",
    "OPENALEX_API_KEY",
    "CANON_DATA_DIR",
    "CANON_REPORTS_DIR",
    "CANON_SETTINGS",
}


def load_local_env(path: Path | None = None) -> None:
    global LOADED
    with LOAD_LOCK:
        if LOADED:
            return
        env_path = path or Path.cwd() / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key not in ALLOWED_ENV_KEYS or key in os.environ:
                    continue
                os.environ[key] = clean_value(value)
        LOADED = True


def clean_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped
