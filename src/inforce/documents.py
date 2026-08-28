"""M2 — parse RBI notification pages and the Master Directions index.

Two very different sources:

* **Notification pages** (`NotificationUser.aspx?Id=N`) carry the full circular
  text inline in HTML, inside `#pnlDetails`. No PDF parsing needed — a useful
  finding, since it removes the whole table-extraction problem for the
  withdrawn corpus.

* **The Master Directions index** is a flat, stateful table: a category heading
  row, then a date row, then one or more document rows belonging to that
  category and date. Category and date must be carried down while walking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from bs4 import BeautifulSoup

from .annex import norm, parse_date

# The document PDF lives under /rdocs/notification/PDFs/. Every page also links
# /rdocs/content/pdfs/ boilerplate (accessibility notice, Utkarsh brochure) —
# matching on '.pdf' alone picks up the wrong file.
_DOC_PDF = re.compile(r"rbidocs\.rbi\.org\.in/rdocs/notification/PDFs/.+?\.PDF", re.I)
_ANY_RBIDOCS_PDF = re.compile(r"rbidocs\.rbi\.org\.in/.+?\.PDF", re.I)
_LEADING_LABEL = re.compile(r"^\s*Notifications?\s*\(\s*[\d.,]+\s*[kKmM][bB]\s*\)\s*")
_RBI_REF = re.compile(r"\bRBI/\d{4}-\d{2,4}/\d+\b")
_DEPT_REF = re.compile(r"\b[A-Z]{2,8}[A-Za-z0-9.&()\-]*\.[A-Za-z0-9.&()\-]*/[A-Za-z0-9.\-/()]+")
_UPDATED_AS_ON = re.compile(r"Updated\s+as\s+on\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4})", re.I)
_SIZE = re.compile(r"([\d,.]+)\s*(kb|mb)", re.I)
_INLINE_DATE = re.compile(r"\b([A-Z][a-z]+\s+\d{1,2},\s*\d{4})\b")


@dataclass
class NotificationDoc:
    rbi_doc_id: int
    url: str
    title: str | None
    rbi_ref: str | None
    dept_ref: str | None
    doc_date: date | None
    body_text: str
    pdf_url: str | None

    @property
    def body_len(self) -> int:
        return len(self.body_text)


@dataclass
class MasterDirection:
    title: str
    pdf_url: str
    category: str | None
    doc_date: date | None
    updated_as_on: date | None
    size_kb: int | None


@dataclass
class MDIndexReport:
    rows_seen: int = 0
    categories: list[str] = field(default_factory=list)
    date_rows: int = 0
    unclassified_single_cells: list[str] = field(default_factory=list)
    rows_without_pdf: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_notification(html: str, rbi_doc_id: int, url: str) -> NotificationDoc:
    soup = BeautifulSoup(html, "lxml")

    container = soup.select_one("#pnlDetails") or soup.select_one(".tablecontent2")
    raw = norm(container.get_text(" ")) if container else ""
    # Pages prefix the content with a PDF-size label, e.g. 'Notifications (892 kb)'.
    body = _LEADING_LABEL.sub("", raw).strip()

    rbi_ref_m = _RBI_REF.search(body)
    rbi_ref = rbi_ref_m.group(0) if rbi_ref_m else None

    # The departmental reference sits immediately after the RBI reference when
    # both are present; searching only the head avoids matching URLs in the body.
    head = body[:600]
    dept_ref = None
    for cand in _DEPT_REF.findall(head):
        if rbi_ref and cand in rbi_ref:
            continue
        dept_ref = cand.rstrip(".,")
        break

    title = None
    if rbi_ref_m:
        candidate = body[: rbi_ref_m.start()].strip(" -–—")
        if 5 < len(candidate) < 400:
            title = candidate
    if title is None:
        title = body[:200].strip() or None

    doc_date = None
    for m in _INLINE_DATE.finditer(head):
        doc_date = parse_date(m.group(1))
        if doc_date:
            break

    pdf_url = None
    for anchor in soup.find_all("a", href=True):
        if _DOC_PDF.search(anchor["href"]):
            pdf_url = anchor["href"]
            break

    return NotificationDoc(
        rbi_doc_id=rbi_doc_id,
        url=url,
        title=title,
        rbi_ref=rbi_ref,
        dept_ref=dept_ref,
        doc_date=doc_date,
        body_text=body,
        pdf_url=pdf_url,
    )


def _looks_like_date(text: str) -> date | None:
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d,%Y", "%B %d,%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_master_directions(html: str) -> tuple[list[MasterDirection], MDIndexReport]:
    """Walk the flat index, carrying category and date down to document rows."""
    soup = BeautifulSoup(html, "lxml")
    report = MDIndexReport()
    out: list[MasterDirection] = []

    tables = soup.find_all("table", class_="tablebg")
    if not tables:
        report.warnings.append("no table.tablebg found — index layout changed")
        return out, report

    current_category: str | None = None
    current_date: date | None = None

    for tr in tables[0].find_all("tr"):
        cells = tr.find_all("td")
        report.rows_seen += 1

        if len(cells) == 1:
            text = norm(cells[0].get_text())
            if not text:
                continue
            as_date = _looks_like_date(text)
            if as_date is not None:
                current_date = as_date
                report.date_rows += 1
            elif len(text) < 120:
                current_category = text
                current_date = None  # a new category restarts the date run
                report.categories.append(text)
            else:
                report.unclassified_single_cells.append(text[:80])
            continue

        anchor = tr.find("a", href=_ANY_RBIDOCS_PDF)
        if anchor is None:
            report.rows_without_pdf += 1
            continue

        title = norm(cells[0].get_text())
        if not title:
            report.rows_without_pdf += 1
            continue

        size_kb = None
        size_m = _SIZE.search(norm(tr.get_text()))
        if size_m:
            value = float(size_m.group(1).replace(",", ""))
            size_kb = int(value * 1024) if size_m.group(2).lower() == "mb" else int(value)

        upd = _UPDATED_AS_ON.search(title)
        updated = parse_date(upd.group(1)) if upd else None

        out.append(
            MasterDirection(
                title=title,
                pdf_url=anchor["href"],
                category=current_category,
                doc_date=current_date,
                updated_as_on=updated,
                size_kb=size_kb,
            )
        )

    if not out:
        report.warnings.append("index parsed but yielded zero documents")
    return out, report


# Second line of defence. Even a structurally valid PDF can contain the CDN's
# interstitial text, and a length threshold will never catch it — the block page
# is ~315 characters, comfortably above any sane minimum.
_BLOCK_MARKERS = (
    "please enable javascript to view the page content",
    "your support id is",
    "this question is for testing whether",
)


def looks_like_block_page(text: str) -> bool:
    head = text[:1200].lower()
    return any(marker in head for marker in _BLOCK_MARKERS)


def pdf_to_text(payload: bytes) -> str:
    """Extract text from a PDF. PyMuPDF is used for speed and because Master
    Directions are text PDFs, not scans — no OCR path is needed."""
    try:
        import pymupdf
    except ImportError:  # PyMuPDF < 1.24 only exposes the legacy name
        import fitz as pymupdf

    with pymupdf.open(stream=payload, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)
