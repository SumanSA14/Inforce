"""Guards against the failure that silently poisoned 50 documents:
rbidocs.rbi.org.in answers HTTP 200 with a 315-byte HTML interstitial when it
dislikes the User-Agent. Status codes and length thresholds both miss it.
"""
from __future__ import annotations

import pytest

from inforce import config, fetch
from inforce.documents import looks_like_block_page

INTERSTITIAL = (
    "Please enable JavaScript to view the page content.\n"
    "Your support ID is: 10988882067393225720. This question is for\n"
    "testing whether you are a human visitor and to prevent automated spam."
)

REAL_CIRCULAR = (
    "Reserve Bank of India (Commercial Banks - Treatment of Wilful Defaulters "
    "and Large Defaulters) Directions, 2025. In exercise of the powers conferred "
    "by Section 35A of the Banking Regulation Act, 1949, the Reserve Bank hereby "
    "issues the following Directions. JavaScript is not mentioned anywhere here."
)


def test_block_page_is_detected():
    assert looks_like_block_page(INTERSTITIAL)


# Measured on the live CDN: every blocked response extracted to 314-315 chars,
# identical across all 50 documents fetched before the bug was caught. The
# fixture above is a paraphrase, so assert against the observed value.
OBSERVED_BLOCK_PAGE_CHARS = 315


def test_block_page_length_is_above_any_sane_minimum():
    """This is why a length threshold could never have caught it, and why the
    detector matches on content instead."""
    assert OBSERVED_BLOCK_PAGE_CHARS > config.MIN_BODY_CHARS


def test_real_circular_is_not_flagged():
    assert not looks_like_block_page(REAL_CIRCULAR)


def test_docs_cdn_gets_a_conventional_user_agent():
    pdf = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/61MD.PDF"
    assert fetch.user_agent_for(pdf) == config.DOCS_USER_AGENT


def test_main_site_keeps_the_honest_user_agent():
    page = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=1&Mode=0"
    assert fetch.user_agent_for(page) == config.USER_AGENT
    assert "InForce" in fetch.user_agent_for(page)


def test_non_pdf_payload_in_cache_is_discarded_not_served(tmp_path, monkeypatch):
    """A cache entry poisoned by an earlier run must not be trusted forever."""
    poisoned = tmp_path / "bad.pdf"
    poisoned.write_bytes(b"<!DOCTYPE html><html>interstitial</html>")

    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        raise RuntimeError("network disabled in test")

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    with pytest.raises(fetch.FetchError):
        fetch.fetch_bytes("http://x/doc.pdf", poisoned, retries=1, backoff=0)

    # It must have discarded the bad cache and attempted a real fetch.
    assert calls, "poisoned cache was served instead of re-fetched"
    assert not poisoned.exists()


def test_valid_pdf_in_cache_is_served_without_network(tmp_path, monkeypatch):
    good = tmp_path / "good.pdf"
    good.write_bytes(b"%PDF-1.5\nreal content")

    def fake_get(url, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("network hit despite a valid cache entry")

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    payload, from_cache = fetch.fetch_bytes("http://x/doc.pdf", good)
    assert from_cache is True
    assert payload.startswith(b"%PDF-")
