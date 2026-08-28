"""Unit tests for the M1 annex parser.

Fixtures mirror the real markup: runs of whitespace inside cells, en-dashes,
the rupee sign, malformed dates, and the '@' deferred-repeal marker.
"""
from __future__ import annotations

from datetime import date

from inforce.annex import parse_annex, parse_date

LEGACY_TABLE = """
<table class="tablebg">
  <tr><th>Date</th><th>Subject</th><th>Department</th></tr>
  <tr>
    <td>December 24,    2019</td>
    <td>Introduction of a new type of semi-closed PPI &ndash; PPIs upto &#8377;10,000/-</td>
    <td>Department of Payment and Settlement Systems</td>
  </tr>
  <tr>
    <td>August 21, 2019</td>
    <td><a href="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11667&amp;Mode=0">RTGS &ndash; Increase in operating hours</a></td>
    <td>Department of Payment and Settlement Systems</td>
  </tr>
  <tr><td>October 20, 2011.</td><td>Trailing period date</td><td>Department of Regulation</td></tr>
  <tr><td colspan="3">a footnote row with one cell</td></tr>
</table>
"""

NUMBERED_TABLE = """
<table class="tablebg">
  <tr><th>S No.</th><th>Circular Number</th><th>Circular Name/Title</th><th>Date</th></tr>
  <tr>
    <td>1.</td>
    <td>DOR.CRE.REC.402/07-01-001/2025-26</td>
    <td><a href="/Scripts/NotificationUser.aspx?Id=13297&amp;Mode=0">Commercial Banks &ndash; Credit Facilities</a></td>
    <td>February 13, 2026</td>
  </tr>
  <tr>
    <td>2.</td>
    <td>DBOD.No.IBS.BC.84/23.09.001/2000-01</td>
    <td>Remittance of Surplus to Head Office, 2022@</td>
    <td>August 07,1989</td>
  </tr>
</table>
"""

DUPLICATE_LEGACY = """
<table class="tablebg">
  <tr><th>Date</th><th>Subject</th><th>Department</th></tr>
  <tr><td>July 1, 2013</td><td>Interest Rates on Deposits</td><td>Department of Regulation</td></tr>
  <tr><td>July 1, 2013</td><td>Interest Rates on Deposits</td><td>Department of Regulation</td></tr>
  <tr><td>July 1, 2013</td><td>Interest Rates on Deposits</td><td>Department of Regulation</td></tr>
</table>
"""

SUPERVISORY_TABLE = """
<table class="tablebg">
  <tr><th>Sr. No.</th><th>Circular Number</th><th>Subject</th><th>Date</th></tr>
  <tr>
    <td>1.</td>
    <td>DoS.CO.PPG.SEC.1/11.01.005/2026-27</td>
    <td>Fair Practices Code for Lenders</td>
    <td>May 21, 2026</td>
  </tr>
</table>
"""

FULL = LEGACY_TABLE + NUMBERED_TABLE + SUPERVISORY_TABLE


def test_parse_date_handles_real_malformations():
    assert parse_date("December 24, 2019") == date(2019, 12, 24)
    assert parse_date("October   20, 2011.") == date(2011, 10, 20)
    assert parse_date("August 07,1989") == date(1989, 8, 7)
    assert parse_date("February 3, 1981") == date(1981, 2, 3)
    assert parse_date("not a date") is None


def test_three_batches_are_separated():
    rows, report = parse_annex(FULL)
    assert report.tables_seen == 3
    assert report.rows_by_batch["legacy"] == 3
    assert report.rows_by_batch["dor_consolidation"] == 2
    assert report.rows_by_batch["supervisory_consolidation"] == 1
    assert len(rows) == 6


def test_legacy_layout_fields():
    rows, _ = parse_annex(LEGACY_TABLE)
    first = rows[0]
    assert first.batch == "legacy"
    assert first.department == "Department of Payment and Settlement Systems"
    assert first.circular_number is None
    assert first.serial_no is None
    assert first.circular_date == date(2019, 12, 24)
    # whitespace collapsed, entities decoded
    assert "₹10,000" in first.subject or "₹10,000" in first.subject
    assert "  " not in first.subject


def test_link_extraction():
    rows, report = parse_annex(LEGACY_TABLE)
    linked = [r for r in rows if r.rbi_doc_id is not None]
    assert len(linked) == 1
    assert linked[0].rbi_doc_id == 11667
    assert report.linked_by_batch["legacy"] == 1


def test_numbered_layout_and_deferred_marker():
    rows, report = parse_annex(NUMBERED_TABLE)
    assert rows[0].serial_no == 1
    assert rows[0].circular_number == "DOR.CRE.REC.402/07-01-001/2025-26"
    assert rows[0].rbi_doc_id == 13297
    assert rows[0].deferred_repeal is False
    # This fixture row is copied from real data and is dated February 13, 2026 —
    # after its own batch's consolidation event — so its withdrawal date is
    # genuinely unknown rather than 2025-11-28.
    assert rows[0].withdrawn_on is None

    # The live data puts the '@' marker at the end of the TITLE, not the serial.
    assert rows[1].serial_no == 2
    assert rows[1].deferred_repeal is True
    assert rows[1].withdrawn_on == "2026-01-01"
    assert rows[1].subject == "Remittance of Surplus to Head Office, 2022"
    assert "@" not in rows[1].subject
    assert report.deferred_repeal_rows == 1


IMPOSSIBLE_DATE_TABLE = """
<table class="tablebg">
  <tr><th>S No.</th><th>Circular Number</th><th>Circular Name/Title</th><th>Date</th></tr>
  <tr><td>1.</td><td>DoR.X/2025-26</td><td>Issued after the batch withdrawal date</td>
      <td>February 13, 2026</td></tr>
  <tr><td>2.</td><td>DoR.Y/2015-16</td><td>Issued long before it</td>
      <td>July 1, 2015</td></tr>
</table>
"""


def test_impossible_withdrawal_date_becomes_unknown():
    """RBI appends to the annex over time, so batch membership does not imply a
    single withdrawal date. A circular dated after its batch's event cannot have
    been withdrawn by it — record unknown rather than assert a wrong date."""
    rows, report = parse_annex(IMPOSSIBLE_DATE_TABLE)
    after, before = rows[0], rows[1]
    assert after.circular_date.year == 2026
    assert after.withdrawn_on is None, "impossible date must not be asserted"
    assert before.withdrawn_on == "2025-11-28", "plausible dates are kept"
    assert report.unknown_withdrawal_date == 1


def test_unknown_withdrawal_date_still_means_withdrawn():
    """NULL means 'withdrawn, date unknown' — never 'still in force'."""
    rows, _ = parse_annex(IMPOSSIBLE_DATE_TABLE)
    assert rows[0].batch == "dor_consolidation"
    assert rows[0].withdrawn_on is None


def test_trailing_at_only_mid_string_at_is_content():
    from inforce.annex import _strip_deferred_marker

    assert _strip_deferred_marker("Foo 2022@") == ("Foo 2022", True)
    assert _strip_deferred_marker("mail@rbi.org.in") == ("mail@rbi.org.in", False)


def test_identical_rows_are_kept_not_collapsed():
    """The legacy table genuinely repeats (date, subject, department) tuples.
    Collapsing them onto one primary key would silently lose rows."""
    rows, report = parse_annex(DUPLICATE_LEGACY)
    assert len(rows) == 3
    assert [r.dup_index for r in rows] == [0, 1, 2]
    assert len({r.natural_key for r in rows}) == 3
    assert len({r.content_key for r in rows}) == 1
    assert report.duplicate_content_rows == 2


def test_footnote_and_header_rows_are_skipped_not_stored():
    rows, report = parse_annex(LEGACY_TABLE)
    assert len(rows) == 3
    assert report.skipped_rows == 1  # the single-cell footnote


def test_batch_identified_by_header_not_position():
    """The supervisory table sits at DOM index 0 here. Identifying batches by
    position would label it 'legacy' and stamp the wrong withdrawal date."""
    rows, _ = parse_annex(SUPERVISORY_TABLE)
    assert rows[0].batch == "supervisory_consolidation"
    assert rows[0].withdrawn_on == "2026-07-31"
    assert rows[0].circular_number == "DoS.CO.PPG.SEC.1/11.01.005/2026-27"
    assert rows[0].serial_no == 1


def test_reordered_tables_still_labelled_correctly():
    """Regression guard: if RBI reorders the annex, every row must keep its
    correct batch and withdrawal date."""
    shuffled = SUPERVISORY_TABLE + LEGACY_TABLE + NUMBERED_TABLE
    rows, report = parse_annex(shuffled)
    by_batch = {r.batch for r in rows}
    assert by_batch == {"legacy", "dor_consolidation", "supervisory_consolidation"}

    dor = [r for r in rows if r.batch == "dor_consolidation"]
    # None is valid here: a row dated after its batch's event has an unknown
    # withdrawal date. What must never happen is a row landing in the wrong batch.
    assert all(r.withdrawn_on in ("2025-11-28", "2026-01-01", None) for r in dor)
    sup = [r for r in rows if r.batch == "supervisory_consolidation"]
    assert all(r.withdrawn_on == "2026-07-31" for r in sup)
    assert any("page structure has changed" in w for w in report.warnings)


def test_duplicate_batch_tables_are_skipped_not_double_counted():
    rows, report = parse_annex(SUPERVISORY_TABLE + SUPERVISORY_TABLE)
    assert len(rows) == 1
    assert any("already seen" in w for w in report.warnings)


def test_unrecognised_headers_warn_rather_than_mislabel():
    bad = '<table class="tablebg"><tr><th>Foo</th><th>Bar</th></tr>' \
          "<tr><td>x</td><td>y</td></tr></table>"
    rows, report = parse_annex(bad)
    assert rows == []
    assert any("not recognised" in w for w in report.warnings)


def test_natural_key_is_stable_and_unique():
    rows, _ = parse_annex(FULL)
    keys = [r.natural_key for r in rows]
    assert len(keys) == len(set(keys))
    again, _ = parse_annex(FULL)
    assert [r.natural_key for r in again] == keys
