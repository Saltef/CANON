from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GrobidResult:
    ok: bool
    status_code: int | None
    tei_xml: str | None
    error: str | None


def grobid_health(grobid_url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(f"{grobid_url.rstrip('/')}/api/isalive", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def parse_pdf_with_grobid(pdf_path: Path, grobid_url: str, timeout: int = 60) -> GrobidResult:
    boundary = "----canon-grobid-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="input"; filename="paper.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + pdf_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        f"{grobid_url.rstrip('/')}/api/processFulltextDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return GrobidResult(
                ok=response.status == 200,
                status_code=response.status,
                tei_xml=response.read().decode("utf-8"),
                error=None,
            )
    except urllib.error.HTTPError as error:
        return GrobidResult(ok=False, status_code=error.code, tei_xml=None, error=str(error))
    except (urllib.error.URLError, TimeoutError) as error:
        return GrobidResult(ok=False, status_code=None, tei_xml=None, error=str(error))
