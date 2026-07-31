from __future__ import annotations

import atexit
import json
import urllib.request
from typing import Any


_HTTPX_CLIENT: Any | None = None


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST JSON through httpx pooling when available, with stdlib fallback."""
    try:
        client = pooled_httpx_client()
    except ImportError:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers or {},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    response = client.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("JSON response must be an object.")
    return result


def pooled_httpx_client() -> Any:
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is None:
        import httpx

        _HTTPX_CLIENT = httpx.Client(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _HTTPX_CLIENT


def close_pooled_httpx_client() -> None:
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is not None:
        _HTTPX_CLIENT.close()
        _HTTPX_CLIENT = None


atexit.register(close_pooled_httpx_client)
