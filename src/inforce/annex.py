"""M1 — parse the RBI withdrawn-circulars annex into labelled records.

The annex page carries three independent tables, each a distinct withdrawal
batch with its own column layout. Layout is detected from the header row
rather than assumed by position, so a reordering upstream fails loudly
instead of silently mislabelling 9,000 documents.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from bs4 import BeautifulSoup

from . import config

_WS = re.compile(r"\s+")
_ID_IN_HREF = re.compile(r"[?&]Id=(\d+)", re.IGNORECASE)
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y")


def norm(text: str) -> str:
    """Collapse whitespace and strip. RBI markup contains runs of spaces and
    newlines inside cells ('August    21, 2019')."""
    return _WS.sub(" ", text).strip()


def parse_date(raw: str) -> date | None:
    """Parse an RBI date cell, tolerating the malformations present in the
    real data: trailing periods ('October 20, 2011.') and missing space after
    the comma ('August 07,1989')."""
    s = norm(raw).rstrip(".")
    s = re.sub(r",(?=\S)", ", ", s)
    s = s.replace(" ", " ")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def content_key(
    batch: str,
    serial_no: int | None,
    circular_number: str | None,
    department: str | None,
    subject: str,
    circular_date_raw: str,
) -> str:
    """Hash of a row's identifying content. Deliberately excludes position, so
    the key survives RBI inserting or removing unrelated rows."""
    parts = [
        batch,
        str(serial_no or ""),
        circular_number or "",
        department or "",
        subject,
        norm(circular_date_raw),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnnexRow:
    batch: str
    source_table_index: int
    serial_no: int | None
    circular_number: str | None
    subject: str
    department: str | None
    circular_date_raw: str
    circular_date: date | None
    rbi_doc_id: int | None
    source_url: str | None
    deferred_repeal: bool
    withdrawn_on: str | None
    # Nth occurrence of otherwise-identical content. The legacy table genuinely
    # repeats (date, subject, department) tuples; without this they collide on
    # one primary key and rows are silently lost.
    dup_index: int = 0

    @property
    def content_key(self) -> str:
        return content_key(
            self.batch,
            self.serial_no,
            self.circular_number,
            self.department,
            self.subject,
            self.circular_date_raw,
        )

    @property
    def natural_key(self) -> str:
        return f"{self.content_key}#{self.dup_index}"


@dataclass
class ParseReport:
    tables_seen: int = 0
    rows_by_batch: dict[str, int] = field(default_factory=dict)
    linked_by_batch: dict[str, int] = field(default_factory=dict)
    skipped_rows: int = 0
    unparsed_dates: list[str] = field(default_factory=list)
    deferred_repeal_rows: int = 0
    duplicate_content_rows: int = 0
    unknown_withdrawal_date: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_by_batch.values())

    @property
    def total_linked(self) -> int:
        return sum(self.linked_by_batch.values())


def identify_batch(headers: list[str]) -> str | None:
    """Identify which withdrawal batch a table is, from its header signature.

    Position is never used. The two 4-column tables differ only in their third
    header ("Circular Name/Title" vs "Subject"), so `must_not_have` is what
    keeps them apart.
    """
    blob = " | ".join(h.lower() for h in headers)
    if not blob.strip():
        return None
    for key, meta in config.BATCHES.items():
        if all(term in blob for term in meta["must_have"]) and not any(
            term in blob for term in meta["must_not_have"]
        ):
            return key
    return None


def _extract_link(tr) -> tuple[int | None, str | None]:
    anchor = tr.find("a", href=_ID_IN_HREF)
    if anchor is None:
        return None, None
    href = anchor.get("href", "")
    match = _ID_IN_HREF.search(href)
    if match is None:
        return None, None
    return int(match.group(1)), href


def _strip_deferred_marker(value: str) -> tuple[str, bool]:
    """Rows carrying a TRAILING '@' are repealed on a deferred date, per the
    table footnote: '@-circulars shall be repealed with effect from
    January 01, 2026'.

    In the live data the marker is appended to the title text, e.g.
    '...Regional Rural Banks, 2022@'. Only a trailing '@' counts — a stray '@'
    mid-string is content, not a marker.
    """
    stripped = value.rstrip()
    if stripped.endswith("@"):
        return norm(stripped[:-1]), True
    return value, False


def parse_annex(html: str) -> tuple[list[AnnexRow], ParseReport]:
    soup = BeautifulSoup(html, "lxml")
    report = ParseReport()
    rows: list[AnnexRow] = []

    tables = soup.find_all("table", class_="tablebg")
    report.tables_seen = len(tables)
    if len(tables) != len(config.BATCHES):
        report.warnings.append(
            f"expected {len(config.BATCHES)} tables, found {len(tables)} — "
            "RBI may have restructured the annex; verify before trusting counts"
        )

    seen_batches: set[str] = set()
    content_seen: Counter[str] = Counter()

    for index, table in enumerate(tables):
        headers = [norm(th.get_text()) for th in table.find_all("th")]
        batch_key = identify_batch(headers)
        if batch_key is None:
            report.warnings.append(
                f"table[{index}] headers not recognised ({headers!r}); skipped"
            )
            continue
        if batch_key in seen_batches:
            report.warnings.append(
                f"table[{index}] matched batch '{batch_key}', already seen; skipped"
            )
            continue
        seen_batches.add(batch_key)

        meta = config.BATCHES[batch_key]
        if index != meta["expected_index"]:
            report.warnings.append(
                f"batch '{batch_key}' found at table index {index}, expected "
                f"{meta['expected_index']} — identified by header signature so labels "
                "are still correct, but the page structure has changed"
            )

        layout = meta["layout"]
        expected_cells = config.CELLS_PER_LAYOUT[layout]
        report.rows_by_batch.setdefault(batch_key, 0)
        report.linked_by_batch.setdefault(batch_key, 0)

        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) != expected_cells:
                # header rows (0 td) and the footnote row (1 td)
                if cells:
                    report.skipped_rows += 1
                continue

            values = [norm(td.get_text()) for td in cells]
            doc_id, href = _extract_link(tr)

            if layout == "legacy3":
                date_raw, subject, department = values
                serial_no = None
                circular_number = None
                subject, deferred = _strip_deferred_marker(subject)
            else:
                serial_raw, circular_number, subject, date_raw = values
                department = None
                # The marker appears on the title in the live data; check the
                # other identifier cells too rather than assume one placement.
                subject, d0 = _strip_deferred_marker(subject)
                serial_raw, d1 = _strip_deferred_marker(serial_raw)
                circular_number, d2 = _strip_deferred_marker(circular_number)
                deferred = d0 or d1 or d2
                digits = re.sub(r"\D", "", serial_raw)
                serial_no = int(digits) if digits else None
                circular_number = circular_number or None

            if not subject:
                report.skipped_rows += 1
                continue

            parsed = parse_date(date_raw)
            if parsed is None:
                report.unparsed_dates.append(date_raw)
            if deferred:
                report.deferred_repeal_rows += 1

            withdrawn_on = (
                config.DEFERRED_REPEAL_DATE if deferred else meta["withdrawn_on"]
            )

            # RBI appends newly-withdrawn circulars to the same annex table over
            # time, so batch membership does NOT imply a single withdrawal date:
            # the DOR table already contains circulars issued months after its
            # own consolidation event. A circular cannot be withdrawn on or
            # before the day it was issued, so where the batch date is
            # impossible, record the date as unknown rather than assert a wrong
            # one. M6's as-of-date logic must treat NULL as "withdrawn, date
            # unknown" and not as "still in force".
            if (
                withdrawn_on
                and parsed is not None
                and parsed.isoformat() >= withdrawn_on
            ):
                withdrawn_on = None
                report.unknown_withdrawal_date += 1

            ckey = content_key(
                batch_key, serial_no, circular_number, department or None,
                subject, date_raw,
            )
            dup_index = content_seen[ckey]
            content_seen[ckey] += 1
            if dup_index:
                report.duplicate_content_rows += 1

            rows.append(
                AnnexRow(
                    batch=batch_key,
                    source_table_index=index,
                    serial_no=serial_no,
                    circular_number=circular_number,
                    subject=subject,
                    department=department or None,
                    circular_date_raw=date_raw,
                    circular_date=parsed,
                    rbi_doc_id=doc_id,
                    source_url=href,
                    deferred_repeal=deferred,
                    withdrawn_on=withdrawn_on,
                    dup_index=dup_index,
                )
            )
            report.rows_by_batch[batch_key] += 1
            if doc_id is not None:
                report.linked_by_batch[batch_key] += 1

        # Row-count drift is only meaningful against the whole live page;
        # parsing a fragment (as tests do) would otherwise warn spuriously.
        expected = meta.get("expected_rows")
        actual = report.rows_by_batch[batch_key]
        if expected and actual != expected and len(tables) == len(config.BATCHES):
            report.warnings.append(
                f"batch '{batch_key}': parsed {actual} rows, expected ~{expected} "
                "— RBI updates this list over time, so small drift is normal; "
                "large drift means the layout changed"
            )

    return rows, report
