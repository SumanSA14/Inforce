"""Unit tests for the M2 document parsers."""
from __future__ import annotations

from datetime import date

from inforce.documents import parse_master_directions, parse_notification

NOTIFICATION = """
<html><body>
<div id="pnlDetails">
  <table class="tablebg"><tr><td>
    Notifications (892 kb)
    Reserve Bank of India (Commercial Banks &ndash; Credit Facilities) Amendment Directions, 2026
    RBI/2025-26/211
    DOR.CRE.REC.402/07-01-001/2025-26
    February 13, 2026
    All Commercial Banks. Madam / Dear Sir, In exercise of the powers conferred by
    Section 35A of the Banking Regulation Act, 1949 the Reserve Bank hereby directs
    that the following amendments shall apply with immediate effect.
  </td></tr></table>
</div>
<a href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT2197C8F6.PDF">PDF</a>
<a href="https://rbidocs.rbi.org.in/rdocs/content/pdfs/Utkarsh202910042026.pdf">Utkarsh</a>
<a href="https://rbidocs.rbi.org.in/rdocs/content/pdfs/Accessibility20012026.pdf">Access</a>
</body></html>
"""

PDF_ONLY_NOTIFICATION = """
<html><body>
<div id="pnlDetails"><table class="tablebg"><tr><td>
  Notifications (2,048 kb)
  Reserve Bank of India (Urban Co-operative Banks &ndash; Branch Authorisation) Directions, 2025
</td></tr></table></div>
<a href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/271MDB844E1A6.PDF">PDF</a>
</body></html>
"""

MD_INDEX = """
<html><body><table class="tablebg">
  <tr><td>Banker and Debt Manager to Government</td></tr>
  <tr><td>Jul 03, 2018</td></tr>
  <tr><td>Master Directions on Relief/Savings Bonds</td><td>256 kb</td>
      <td><a href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/61MD0825.PDF">x</a></td></tr>
  <tr><td>Commercial Banks</td></tr>
  <tr><td>Nov 28, 2025</td></tr>
  <tr><td>RBI (Commercial Banks - Internal Ombudsman) Directions, 2026 (Updated as on June 24, 2026)</td>
      <td><a href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/380MD1401.PDF">2.8 MB</a></td></tr>
  <tr><td>RBI (Commercial Banks - Capital Adequacy) Directions, 2026</td>
      <td><a href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/381MD1401.PDF">293 kb</a></td></tr>
</table></body></html>
"""


def test_notification_strips_leading_size_label():
    doc = parse_notification(NOTIFICATION, 13297, "http://x")
    assert not doc.body_text.startswith("Notifications")
    assert doc.body_text.startswith("Reserve Bank of India")


def test_notification_extracts_both_reference_numbers():
    doc = parse_notification(NOTIFICATION, 13297, "http://x")
    assert doc.rbi_ref == "RBI/2025-26/211"
    assert doc.dept_ref == "DOR.CRE.REC.402/07-01-001/2025-26"
    assert doc.doc_date == date(2026, 2, 13)


def test_notification_title_is_text_before_the_rbi_reference():
    doc = parse_notification(NOTIFICATION, 13297, "http://x")
    assert doc.title.endswith("Amendment Directions, 2026")
    assert "RBI/2025-26/211" not in doc.title


def test_notification_picks_document_pdf_not_site_boilerplate():
    """Every page also links the Utkarsh brochure and accessibility notice
    under /rdocs/content/pdfs/. Matching on '.pdf' alone grabs the wrong file."""
    doc = parse_notification(NOTIFICATION, 13297, "http://x")
    assert doc.pdf_url.endswith("NT2197C8F6.PDF")
    assert "Utkarsh" not in doc.pdf_url
    assert "Accessibility" not in doc.pdf_url


def test_pdf_only_page_yields_thin_body_but_keeps_pdf_url():
    """These pages carry a title and nothing else — the crawler must be able to
    detect the shortfall and fall back to the PDF."""
    doc = parse_notification(PDF_ONLY_NOTIFICATION, 13035, "http://x")
    assert doc.body_len < 200
    assert doc.pdf_url is not None
    assert doc.title.startswith("Reserve Bank of India")


def test_md_index_carries_category_and_date_down_to_rows():
    mds, report = parse_master_directions(MD_INDEX)
    assert len(mds) == 3
    assert mds[0].category == "Banker and Debt Manager to Government"
    assert mds[0].doc_date == date(2018, 7, 3)
    # both rows under 'Commercial Banks / Nov 28, 2025' inherit the same state
    assert mds[1].category == "Commercial Banks"
    assert mds[2].category == "Commercial Banks"
    assert mds[1].doc_date == mds[2].doc_date == date(2025, 11, 28)
    assert report.date_rows == 2


def test_md_index_extracts_updated_as_on_and_sizes():
    mds, _ = parse_master_directions(MD_INDEX)
    assert mds[1].updated_as_on == date(2026, 6, 24)
    assert mds[2].updated_as_on is None
    assert mds[0].size_kb == 256
    assert mds[1].size_kb == int(2.8 * 1024)


def test_new_category_resets_the_date_run():
    """A date belongs to the category above it; leaking it across a category
    boundary would silently date-stamp documents wrongly."""
    mds, _ = parse_master_directions(MD_INDEX)
    assert mds[0].doc_date != mds[1].doc_date


def test_missing_table_warns_rather_than_returning_silently_empty():
    mds, report = parse_master_directions("<html><body><p>nothing</p></body></html>")
    assert mds == []
    assert any("no table.tablebg" in w for w in report.warnings)
