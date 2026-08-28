"""Entity detection tests.

Detection is deterministic keyword matching rather than an LLM call, so its
errors are inspectable. These tests pin the ambiguous cases, which are the ones
that actually matter: several category names are substrings of each other.
"""
from __future__ import annotations

import pytest

from inforce.entities import categories, detect, same_subject, subject_of


@pytest.mark.parametrize("question,expected", [
    ("What minimum CRAR must a small finance bank maintain?", "Small Finance Banks"),
    ("What ratio applies to an SFB?", "Small Finance Banks"),
    ("What Pillar 1 CRAR applies to a commercial bank?", "Commercial Banks"),
    ("What KYC rules govern an urban co-operative bank?", "Urban Co-operative Banks"),
    ("What applies to a UCB?", "Urban Co-operative Banks"),
    ("What KYC requirements apply to an NBFC?", "Non-Banking Financial Companies"),
    ("What rules govern a payments bank?", "Payments Banks"),
    ("How must an ARC update its records?", "Asset Reconstruction Companies"),
    ("What fee cap applies to a credit information company?", "Credit Information Companies"),
    ("What target applies to an all India financial institution?",
     "All India Financial Institutions"),
    ("What applies to a local area bank?", "Local Area Banks"),
])
def test_detects_named_entities(question, expected):
    assert detect(question) == expected


def test_regional_rural_beats_rural_cooperative():
    """'Regional Rural Banks' and 'Rural Co-operative Banks' both contain
    'rural'. Matching the wrong one scopes retrieval to the wrong corpus."""
    assert detect("What applies to a regional rural bank?") == "Regional Rural Banks"
    assert detect("What applies to an RRB?") == "Regional Rural Banks"


def test_rural_cooperative_is_not_regional_rural():
    assert detect("What applies to a rural co-operative bank?") == "Rural Co-operative Banks"


def test_urban_and_rural_cooperatives_are_distinguished():
    assert detect("rules for urban co-operative banks") == "Urban Co-operative Banks"
    assert detect("rules for rural co-operative banks") == "Rural Co-operative Banks"


def test_no_entity_named_returns_none():
    """A question with no entity must NOT be scoped — returning a guess here
    would filter out the correct answer."""
    assert detect("Which banknote denominations must be machine-processed?") is None
    assert detect("What timeframe applies to resolving a PPI complaint?") is None


def test_empty_input_is_safe():
    assert detect("") is None
    assert detect(None) is None


def test_all_patterns_have_a_canonical_category():
    assert len(categories()) == len(set(categories()))


def test_entity_mask_keeps_department_documents(tmp_path, monkeypatch):
    """RBI's index mixes entity types with functional departments. Scoping a
    question to 'Commercial Banks' must not exclude a Financial Inclusion
    document — only sibling *entity* documents are genuine competitors.

    Before this fix, a priority-sector question mentioning "commercial bank"
    filtered out the department document that actually answered it.
    """
    import numpy as np
    from inforce import retrieve, store

    conn = store.connect(tmp_path / "e.db")
    store.init_schema(conn)
    rows = [
        ("md:cb", "Commercial Banks"),                       # target entity
        ("md:sfb", "Small Finance Banks"),                   # rival entity -> drop
        ("md:psl", "Financial Inclusion and Development"),   # department -> keep
        ("notif:x", None),                                   # unknown -> keep
    ]
    for i, (key, cat) in enumerate(rows, start=1):
        conn.execute(
            """INSERT INTO document (doc_key, source_kind, status, url, category,
                                     fetch_status, fetched_at)
               VALUES (?,?,'in_force','http://x',?,'ok','t')""",
            (key, "master_direction", cat))
        conn.execute(
            """INSERT INTO chunk (chunk_id, doc_key, ordinal, text, char_start,
                                  char_end, embedding)
               VALUES (?,?,0,'t',0,1,?)""", (i, key, b"\x00" * 4))
    conn.commit()

    retrieve._ENTITY_CACHE.clear()
    monkeypatch.setattr(retrieve, "load_index",
                        lambda: (np.zeros((4, 1), dtype=np.float32),
                                 np.array([1, 2, 3, 4], dtype=np.int64)))

    mask = retrieve.entity_mask(conn, "Commercial Banks")
    assert mask.tolist() == [True, False, True, True], (
        "target kept, rival entity dropped, department and unknown kept")
    retrieve._ENTITY_CACHE.clear()


# --- subject families -------------------------------------------------------
#
# These 15 cases are the complete set of category mismatches from the
# time_aware+inferred+hybrid+doc+expand+llm run. Each was adjudicated
# individually and then adversarially reviewed; the reviewer upheld all 15.
# Pinning them here stops the entity/subject split from silently drifting,
# because the whole point of the split is that it changes what a headline
# number means.
ADJUDICATED = [
    # (expected title, returned title, same_subject)
    ("RBI (Regional Rural Banks - Credit Cards and Debit Cards) Directions",
     "RBI (Urban Co-operative Banks - Credit Cards and Debit Cards) Directions", True),
    ("RBI (Regional Rural Banks - Miscellaneous) Directions",
     "RBI (Rural Co-operative Banks - Miscellaneous) Directions", True),
    ("RBI (Local Area Banks - Classification, Valuation and Operation of "
     "Investment Portfolio) Directions",
     "RBI (Small Finance Banks - Classification, Valuation and Operation of "
     "Investment Portfolio) Directions", True),
    ("RBI (Commercial Banks - Know Your Customer) Directions",
     "RBI (Regional Rural Banks - Know Your Customer) Directions", True),
    ("RBI (Commercial Banks - Know Your Customer) Directions",
     "RBI (Non-Banking Financial Companies - Know Your Customer) Directions", True),
    ("RBI (Rural Co-operative Banks - Responsible Business Conduct) Directions",
     "RBI (Commercial Banks - Responsible Business Conduct) Directions", True),
    ("RBI (Rural Co-operative Banks - Credit Information Reporting) Directions",
     "RBI (Commercial Banks - Credit Information Reporting) Directions", True),
    ("RBI (Non-Banking Financial Companies - Securitisation of Standard Assets) "
     "Directions",
     "RBI (Small Finance Banks - Securitisation Transactions) Directions", True),
    # subject misses filed under an entity label before the split
    ("RBI (All India Financial Institutions - Transfer and Distribution of "
     "Credit Risk) Directions",
     "RBI (Small Finance Banks - Treatment of Wilful Defaulters) Directions", False),
    ("RBI (All India Financial Institutions - Transfer and Distribution of "
     "Credit Risk) Directions",
     "RBI (Housing Finance Companies) Directions, 2025", False),
    ("Master Direction - Foreign Investment in India",
     "RBI (Non-Banking Financial Companies - Undertaking of Activities) Directions",
     False),
    ("RBI (Payments Banks - Classification, Valuation and Operation of "
     "Investment Portfolio) Directions",
     "RBI (Urban Co-operative Banks - Financial Statements) Directions", False),
    ("Master Directions on Prepaid Payment Instruments (PPIs)",
     "RBI (Non-Bank Prepaid Payment Instruments Issuers) Directions", False),
    ("RBI (Non-Banking Financial Companies - Securitisation of Standard Assets) "
     "Directions",
     "RBI (Commercial Banks - Prudential Norms on Capital Adequacy)", False),
]


@pytest.mark.parametrize("expected,actual,same", ADJUDICATED)
def test_subject_split_matches_adjudication(expected, actual, same):
    assert same_subject(expected, actual) is same


def test_master_directions_have_no_entity_sibling():
    """No entity split in the title means an entity mismatch is impossible, so
    any miss against it must be attributed to subject, not entity."""
    assert subject_of("Master Direction - Foreign Investment in India") is None
    assert subject_of("Master Directions on Prepaid Payment Instruments") is None


def test_subject_extracted_from_entity_split_title():
    assert subject_of(
        "RBI (Commercial Banks - Know Your Customer) Directions") == "know your customer"


def test_same_subject_is_symmetric():
    a = "RBI (Payments Banks - Know Your Customer) Directions"
    b = "RBI (Commercial Banks - Know Your Customer) Directions"
    assert same_subject(a, b) == same_subject(b, a)


def test_same_subject_handles_missing_titles():
    assert same_subject(None, "RBI (X - Y) Directions") is False
    assert same_subject("RBI (X - Y) Directions", None) is False
