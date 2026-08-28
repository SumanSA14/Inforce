"""M6 bi-temporal tests.

The rule these all defend: `valid_to IS NULL` is ambiguous between "still in
force" and "withdrawn on an unpublished date". Collapsing the two would
silently resurrect dead regulation — the exact failure this project measures.
"""
from __future__ import annotations

import pytest

from inforce.temporal import (IN_FORCE, NOT_YET, UNKNOWN, WITHDRAWN, migrate,
                              snapshot, validity_at)
from inforce import store


def test_live_master_direction_is_in_force_now():
    assert validity_at("2025-11-28", None, "in_force", True, "2026-08-08") == IN_FORCE


def test_withdrawn_document_is_in_force_before_its_withdrawal():
    """The whole point of valid time: repealed rules still governed past conduct."""
    assert validity_at("2016-10-06", "2025-11-28", "withdrawn", True, "2022-03-14") == IN_FORCE


def test_withdrawn_document_is_withdrawn_after_its_withdrawal():
    assert validity_at("2016-10-06", "2025-11-28", "withdrawn", True, "2026-08-08") == WITHDRAWN


def test_withdrawal_date_boundary_is_inclusive():
    assert validity_at("2016-10-06", "2025-11-28", "withdrawn", True, "2025-11-28") == WITHDRAWN
    assert validity_at("2016-10-06", "2025-11-28", "withdrawn", True, "2025-11-27") == IN_FORCE


def test_document_not_yet_issued_is_not_in_force():
    assert validity_at("2025-11-28", None, "in_force", True, "2020-01-01") == NOT_YET


def test_withdrawn_with_unknown_date_is_unknown_never_in_force():
    """The 17 rows RBI appended without publishing a date. Rounding these to
    'in force' resurrects dead law; rounding to 'withdrawn' invents a date."""
    assert validity_at("2026-02-13", None, "withdrawn", False, "2026-08-08") == UNKNOWN
    assert validity_at("2026-02-13", None, "withdrawn", False, "2026-03-01") == UNKNOWN


def test_uncertain_flag_dominates_a_present_date():
    assert validity_at("2016-01-01", None, "withdrawn", False, "2026-08-08") == UNKNOWN


def test_unknown_is_not_silently_counted_as_either(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    migrate(conn)
    now = "2026-08-08T00:00:00+00:00"
    rows = [
        # (key, status, valid_from, valid_to, certain)
        ("md:live", "in_force", "2025-11-28", None, 1),
        ("notif:dead", "withdrawn", "2016-10-06", "2025-11-28", 1),
        ("notif:murky", "withdrawn", "2026-02-13", None, 0),
    ]
    for key, status, vf, vt, certain in rows:
        conn.execute(
            """INSERT INTO document (doc_key, source_kind, status, url, fetch_status,
                                     fetched_at, valid_from, valid_to, validity_certain)
               VALUES (?,?,?,?,'ok',?,?,?,?)""",
            (key, "notification", status, "http://x", now, vf, vt, certain),
        )
    conn.commit()

    snap = snapshot(conn, "2026-08-08")
    assert snap.in_force == 1
    assert snap.withdrawn == 1
    assert snap.unknown == 1
    assert snap.total == 3, "every document must land in exactly one bucket"


def test_historical_snapshot_reverses_a_document(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    migrate(conn)
    conn.execute(
        """INSERT INTO document (doc_key, source_kind, status, url, fetch_status,
                                 fetched_at, valid_from, valid_to, validity_certain)
           VALUES ('notif:x','notification','withdrawn','http://x','ok','t',
                   '2016-10-06','2025-11-28',1)"""
    )
    conn.commit()
    assert snapshot(conn, "2022-06-01").in_force == 1   # governed conduct then
    assert snapshot(conn, "2026-06-01").withdrawn == 1  # not now


def test_hit_reports_validity_on_the_queried_date_not_today(tmp_path):
    """Regression guard for a demo bug.

    The UI badge originally showed `status` (present tense), so a historical
    query for 2022 labelled the Master Circulars that genuinely governed then as
    "REPEALED — 5 of 5 results are dead law". Correct behaviour displayed as
    failure, which would undercut the entire premise in front of an audience.
    """
    from inforce.retrieve import Hit

    hit = Hit(
        rank=1, score=0.8, chunk_id=1, doc_key="notif:x",
        text="...", title="Master Circular - Management of Advances",
        status="withdrawn", source_kind="notification",
        doc_date="2015-07-01", withdrawn_on="2025-11-28",
        rbi_ref=None, url=None,
        valid_from="2015-07-01", valid_to="2025-11-28",
        validity_certain=True, known_from="2026-08-07T00:00:00+00:00",
    )
    assert hit.is_withdrawn is True, "present-tense status is unchanged"
    assert hit.validity_on("2022-03-14") == IN_FORCE, "it governed conduct in 2022"
    assert hit.validity_on("2026-08-08") == WITHDRAWN
    assert hit.validity_on("2014-01-01") == NOT_YET


def test_migrate_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    first = migrate(conn)
    second = migrate(conn)
    assert set(first) == {"valid_from", "valid_to", "validity_certain", "known_from"}
    assert second == [], "re-running must add nothing"
