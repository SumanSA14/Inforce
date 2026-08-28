"""Paths, URLs and batch definitions for InForce."""
from __future__ import annotations

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "inforce.db"

USER_AGENT = "InForce/0.1 (regulatory-retrieval research; contact via repo)"

# rbidocs.rbi.org.in (the document CDN) rejects unrecognised User-Agent strings:
# it answers HTTP 200 with a 315-byte HTML interstitial instead of the file, so
# the failure is invisible unless the payload is validated. A standard browser
# UA is served the real PDF on the first request — there is no JS challenge to
# solve. We identify ourselves honestly to www.rbi.org.in, which accepts it, and
# fall back to a conventional UA only for the CDN that does not.
DOCS_HOST = "rbidocs.rbi.org.in"
DOCS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

# The Annex referenced by both RBI/2025-26/100 (28 Nov 2025) and the
# 31 Jul 2026 supervisory repeal. One static HTML page, ~3.6 MB.
ANNEX_URL = "https://www.rbi.org.in/scripts/NotificationUserWithdrawnCircular.aspx"
ANNEX_CACHE = RAW_DIR / "withdrawn_annex.html"

# M2 sources.
NOTIFICATION_URL = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={id}&Mode=0"
NOTIFICATION_CACHE_DIR = RAW_DIR / "notifications"
# Some notification pages carry only a title and a PDF link — the circular text
# lives entirely in the PDF. Those need a binary fallback.
NOTIFICATION_PDF_CACHE_DIR = RAW_DIR / "notification_pdfs"

MASTER_DIRECTIONS_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx"
MASTER_DIRECTIONS_CACHE = RAW_DIR / "master_directions_index.html"
MD_PDF_CACHE_DIR = RAW_DIR / "md_pdfs"

# Politeness. rbi.org.in is not hostile, but 3,516 notification pages plus 410
# PDFs is a real load — go slowly and cache permanently so it is paid once.
CRAWL_DELAY_SECONDS = 1.0

# A notification page whose extracted body is shorter than this almost
# certainly failed to render content rather than genuinely being that short.
MIN_BODY_CHARS = 200

# The page carries three separate <table class="tablebg"> elements. Each is a
# distinct withdrawal batch with its own column layout — this is why the annex
# does NOT need a Wayback diff to recover batch membership.
#
# Batches are identified by their HEADER SIGNATURE, never by DOM position. If
# RBI reorders or inserts a table, position-based identification would stamp
# ~9,500 rows with the wrong withdrawal date and do it silently. `must_have`
# terms must all appear somewhere in the header row; `must_not_have` terms
# disambiguate the two 4-column tables, whose headers differ only in the third
# column ("Circular Name/Title" vs "Subject").
BATCHES = {
    "legacy": {
        "layout": "legacy3",
        "label": "Legacy withdrawn circulars (pre-consolidation)",
        # No single withdrawal event — these accrued individually over time.
        "withdrawn_on": None,
        "expected_rows": 714,
        "expected_index": 0,
        "must_have": ("date", "subject", "department"),
        "must_not_have": (),
    },
    "dor_consolidation": {
        "layout": "numbered4",
        "label": "Dept. of Regulation consolidation (RBI/2025-26/100)",
        "withdrawn_on": "2025-11-28",
        "expected_rows": 9462,
        "expected_index": 1,
        "must_have": ("circular number", "circular name", "date"),
        "must_not_have": ("department",),
    },
    "supervisory_consolidation": {
        "layout": "numbered4",
        "label": "Supervisory instructions consolidation",
        "withdrawn_on": "2026-07-31",
        "expected_rows": 628,
        "expected_index": 2,
        "must_have": ("circular number", "subject", "date"),
        "must_not_have": ("department", "circular name"),
    },
}

CELLS_PER_LAYOUT = {"legacy3": 3, "numbered4": 4}

# Rows flagged with '@' carry a deferred repeal date, per the table footnote:
# "@-circulars shall be repealed with effect from January 01, 2026"
DEFERRED_REPEAL_DATE = "2026-01-01"
