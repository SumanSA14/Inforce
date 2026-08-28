"""Cached HTTP fetching.

Ground rule: every fetch is cached to disk permanently. No eval run, demo or
test may ever depend on a live request to rbi.org.in.
"""
from __future__ import annotations

import gzip
import hashlib
import pathlib
import time

import requests

from . import config


class FetchError(RuntimeError):
    pass


class InvalidPayloadError(FetchError):
    """The server answered 200 but the body is not the resource we asked for.

    rbidocs.rbi.org.in returns an HTML interstitial with a 200 status when it
    dislikes the User-Agent. Trusting the status code silently stores garbage,
    so payloads are validated by content, never by status or length.
    """


PDF_MAGIC = b"%PDF-"


def user_agent_for(url: str) -> str:
    return config.DOCS_USER_AGENT if config.DOCS_HOST in url else config.USER_AGENT


def _read_cache(path: pathlib.Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _write_cache(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        # Notification pages are ~110 kB of boilerplate around ~3 kB of content.
        # Gzipping keeps the full-corpus cache around 55 MB instead of ~390 MB.
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def fetch_text(
    url: str,
    cache_path: pathlib.Path,
    *,
    refresh: bool = False,
    timeout: int = 120,
    retries: int = 3,
    backoff: float = 2.0,
) -> tuple[str, bool]:
    """Return (text, from_cache).

    Writes UTF-8 to `cache_path` on a successful fetch. Re-reads from cache
    unless `refresh` is set.
    """
    if cache_path.exists() and not refresh:
        return _read_cache(cache_path), True

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": config.USER_AGENT},
                timeout=timeout,
            )
            resp.raise_for_status()
            # RBI pages do not always declare a charset; apparent_encoding
            # reads the actual bytes rather than trusting the header.
            resp.encoding = resp.apparent_encoding or "utf-8"
            text = resp.text
            _write_cache(cache_path, text)
            return text, False
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)

    # Keep the underlying cause in the message: it is persisted to the DB as a
    # string, so losing it means a failed document cannot be diagnosed later.
    raise FetchError(
        f"failed to fetch {url} after {retries} attempts — "
        f"{type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def fetch_bytes(
    url: str,
    cache_path: pathlib.Path,
    *,
    refresh: bool = False,
    timeout: int = 180,
    retries: int = 3,
    backoff: float = 2.0,
    expect_pdf: bool = True,
    referer: str | None = None,
) -> tuple[bytes, bool]:
    """Binary variant for PDFs. Returns (payload, from_cache).

    Validates the magic bytes: a 200 status is not evidence that the body is a
    PDF on this host.
    """
    if cache_path.exists() and not refresh:
        payload = cache_path.read_bytes()
        if expect_pdf and not payload.startswith(PDF_MAGIC):
            # A previously-poisoned cache entry. Drop it and re-fetch rather
            # than serving known-bad bytes forever.
            cache_path.unlink(missing_ok=True)
        else:
            return payload, True

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent": user_agent_for(url),
                    "Accept": "application/pdf,*/*;q=0.8",
                    "Accept-Language": "en-GB,en;q=0.9",
                    "Referer": referer or config.MASTER_DIRECTIONS_URL,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            if expect_pdf and not resp.content.startswith(PDF_MAGIC):
                head = resp.content[:80].decode("utf-8", "replace").replace("\n", " ")
                raise InvalidPayloadError(
                    f"expected PDF, got {resp.headers.get('Content-Type')!r} "
                    f"({len(resp.content)} bytes): {head!r}"
                )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(resp.content)
            return resp.content, False
        except InvalidPayloadError:
            # Deterministic rejection — retrying identical headers cannot help.
            raise
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)

    # Keep the underlying cause in the message: it is persisted to the DB as a
    # string, so losing it means a failed document cannot be diagnosed later.
    raise FetchError(
        f"failed to fetch {url} after {retries} attempts — "
        f"{type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def content_sha1(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha1(payload).hexdigest()
