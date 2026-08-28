"""M2 — crawl loops.

Both crawls are resumable and idempotent: work is selected by what is missing
from the `document` table, every response is cached to disk permanently, and
re-running only touches what failed. A crawl killed halfway costs nothing.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

from . import config, documents, fetch, store


@dataclass
class CrawlStats:
    attempted: int = 0
    from_network: int = 0
    from_cache: int = 0
    ok: int = 0
    thin: int = 0
    from_pdf: int = 0
    errors: int = 0
    error_samples: list[str] = field(default_factory=list)

    def note_error(self, label: str, exc: Exception) -> None:
        self.errors += 1
        if len(self.error_samples) < 8:
            self.error_samples.append(f"{label}: {type(exc).__name__}: {exc}"[:160])


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def crawl_notifications(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    delay: float = config.CRAWL_DELAY_SECONDS,
    refresh: bool = False,
    progress_every: int = 50,
    log=print,
) -> CrawlStats:
    """Fetch the withdrawn circulars that carry a stable RBI document ID."""
    stats = CrawlStats()
    doc_ids = store.pending_notification_ids(conn, limit)
    log(f"pending notifications: {len(doc_ids):,}")

    for i, doc_id in enumerate(doc_ids, 1):
        url = config.NOTIFICATION_URL.format(id=doc_id)
        cache_path = config.NOTIFICATION_CACHE_DIR / f"{doc_id}.html.gz"
        stats.attempted += 1
        try:
            html, from_cache = fetch.fetch_text(url, cache_path, refresh=refresh)
            stats.from_cache += from_cache
            stats.from_network += not from_cache

            doc = documents.parse_notification(html, doc_id, url)
            body, body_source = doc.body_text, "html"
            pdf_hit = False

            # Master-Direction-style pages carry only a title plus a PDF link;
            # the circular text is entirely inside the PDF. Fall back to it
            # rather than recording a near-empty document.
            if len(body) < config.MIN_BODY_CHARS and doc.pdf_url:
                pdf_cache = config.NOTIFICATION_PDF_CACHE_DIR / f"{doc_id}.pdf"
                payload, pdf_cached = fetch.fetch_bytes(
                    doc.pdf_url, pdf_cache, refresh=refresh
                )
                pdf_hit = not pdf_cached
                extracted = documents.pdf_to_text(payload)
                if documents.looks_like_block_page(extracted):
                    raise fetch.InvalidPayloadError(
                        "PDF text is the CDN interstitial, not the document"
                    )
                if len(extracted) > len(body):
                    body, body_source = extracted, "pdf"
                    cache_path = pdf_cache
                    stats.from_pdf += 1

            thin = len(body) < config.MIN_BODY_CHARS
            stats.thin += thin
            stats.ok += not thin

            store.upsert_document(
                conn,
                doc_key=f"notif:{doc_id}",
                source_kind="notification",
                status="withdrawn",
                rbi_doc_id=doc_id,
                url=url,
                title=doc.title,
                rbi_ref=doc.rbi_ref,
                dept_ref=doc.dept_ref,
                doc_date=_iso(doc.doc_date),
                pdf_url=doc.pdf_url,
                body_text=body,
                body_len=len(body),
                body_source=body_source,
                content_sha1=fetch.content_sha1(body),
                cache_path=str(cache_path),
                fetch_status="thin" if thin else "ok",
                fetch_error=None,
            )

            if (not from_cache or pdf_hit) and delay:
                time.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - recorded, crawl continues
            stats.note_error(f"Id={doc_id}", exc)
            store.upsert_document(
                conn,
                doc_key=f"notif:{doc_id}",
                source_kind="notification",
                status="withdrawn",
                rbi_doc_id=doc_id,
                url=url,
                fetch_status="error",
                fetch_error=str(exc)[:400],
            )

        if i % progress_every == 0:
            conn.commit()
            log(
                f"  {i:>5,}/{len(doc_ids):,}  ok={stats.ok:,} thin={stats.thin} "
                f"err={stats.errors} net={stats.from_network:,}"
            )

    conn.commit()
    return stats


def crawl_master_directions(
    conn: sqlite3.Connection,
    *,
    download_pdfs: bool = True,
    limit: int | None = None,
    delay: float = config.CRAWL_DELAY_SECONDS,
    refresh: bool = False,
    progress_every: int = 25,
    log=print,
) -> tuple[CrawlStats, documents.MDIndexReport]:
    """Parse the Master Directions index and optionally fetch each PDF."""
    stats = CrawlStats()
    html, from_cache = fetch.fetch_text(
        config.MASTER_DIRECTIONS_URL, config.MASTER_DIRECTIONS_CACHE, refresh=refresh
    )
    log(f"index: {len(html):,} chars ({'cache' if from_cache else 'network'})")

    mds, report = documents.parse_master_directions(html)
    log(f"index rows={report.rows_seen:,}  documents={len(mds):,}  "
        f"categories={len(report.categories)}  date-rows={report.date_rows}")

    if limit:
        mds = mds[:limit]

    for i, md in enumerate(mds, 1):
        doc_key = f"md:{fetch.content_sha1(md.pdf_url)[:16]}"
        stats.attempted += 1
        base = dict(
            doc_key=doc_key,
            source_kind="master_direction",
            status="in_force",
            url=md.pdf_url,
            title=md.title,
            doc_date=_iso(md.doc_date),
            category=md.category,
            updated_as_on=_iso(md.updated_as_on),
            pdf_url=md.pdf_url,
            pdf_size_kb=md.size_kb,
        )

        if not download_pdfs:
            store.upsert_document(conn, **base, fetch_status="meta")
            continue

        try:
            cache_path = config.MD_PDF_CACHE_DIR / f"{doc_key.split(':')[1]}.pdf"
            payload, from_cache = fetch.fetch_bytes(md.pdf_url, cache_path, refresh=refresh)
            stats.from_cache += from_cache
            stats.from_network += not from_cache

            text = documents.pdf_to_text(payload)
            if documents.looks_like_block_page(text):
                raise fetch.InvalidPayloadError(
                    "PDF text is the CDN interstitial, not the document"
                )
            thin = len(text) < config.MIN_BODY_CHARS
            stats.thin += thin
            stats.ok += not thin

            store.upsert_document(
                conn,
                **base,
                body_text=text,
                body_len=len(text),
                body_source="pdf",
                content_sha1=fetch.content_sha1(payload),
                cache_path=str(cache_path),
                fetch_status="thin" if thin else "ok",
                fetch_error=None,
            )
            if not from_cache and delay:
                time.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - recorded, crawl continues
            stats.note_error(md.title[:60], exc)
            store.upsert_document(
                conn, **base, fetch_status="error", fetch_error=str(exc)[:400]
            )

        if i % progress_every == 0:
            conn.commit()
            log(f"  {i:>4,}/{len(mds):,}  ok={stats.ok} thin={stats.thin} err={stats.errors}")

    conn.commit()
    return stats, report
