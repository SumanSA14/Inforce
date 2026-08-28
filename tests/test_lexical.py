"""M9 — BM25 and reciprocal rank fusion tests."""
from __future__ import annotations

import pytest

from inforce import lexical, store


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    store.init_schema(c)
    rows = [
        (1, "md:a", 0, "A bank shall maintain a minimum CRAR of 15 per cent on an ongoing basis."),
        (2, "md:b", 0, "Know your customer periodic updation shall follow a risk based approach."),
        (3, "md:c", 0, "Circular DOR.CRE.REC.402/07-01-001/2025-26 dated February 13, 2026."),
        (4, "md:d", 0, "Walk-in customers where the amount exceeds ₹50,000 require identification."),
    ]
    for cid, key, ordinal, text in rows:
        c.execute(
            """INSERT INTO chunk (chunk_id, doc_key, ordinal, text, char_start, char_end)
               VALUES (?,?,?,?,0,?)""", (cid, key, ordinal, text, len(text)))
    c.commit()
    lexical.build(c, log=lambda *_: None)
    return c


def test_exact_acronym_is_found(conn):
    """Acronyms like CRAR are where dense retrieval is weakest and BM25 shines."""
    assert lexical.search(conn, "What CRAR must a bank keep?")[0] == 1


def test_rupee_threshold_is_found(conn):
    assert 4 in lexical.search(conn, "identification above 50,000 rupees")


def test_circular_reference_is_found(conn):
    hits = lexical.search(conn, "DOR.CRE.REC.402/07-01-001/2025-26")
    assert 3 in hits


def test_query_punctuation_does_not_break_match_syntax(conn):
    """Raw user text is not valid FTS5 MATCH syntax — quotes, hyphens and
    parentheses are operators. Passing it through unescaped raises
    OperationalError, so the query must be reduced to quoted terms."""
    for q in ['what is the "minimum" CRAR (per cent)?',
              "a bank's ratio -- really?",
              "NEAR( capital )",
              "AND OR NOT"]:
        lexical.search(conn, q)  # must not raise


def test_stopword_only_query_returns_nothing(conn):
    assert lexical.search(conn, "what is the of and to") == []
    assert lexical.to_match_query("what is the of and to") == ""


def test_match_query_quotes_every_term(conn):
    q = lexical.to_match_query("minimum CRAR 15 per-cent")
    assert '"minimum"' in q and '"CRAR"' in q
    assert " OR " in q


def test_rrf_rewards_agreement_between_rankers():
    """An item ranked mid-table by both should beat one ranked first by only
    one — that is the whole point of fusing."""
    dense = [10, 20, 30]
    sparse = [40, 20, 50]
    fused = dict(lexical.reciprocal_rank_fusion([dense, sparse]))
    assert fused[20] > fused[10]
    assert fused[20] > fused[40]


def test_rrf_is_rank_based_not_score_based():
    """BM25 scores are negative in SQLite and cosine is 0..1; fusing on raw
    scores would silently weight one ranker to nothing."""
    fused = lexical.reciprocal_rank_fusion([[1, 2], [2, 1]])
    assert len(fused) == 2
    assert abs(fused[0][1] - fused[1][1]) < 1e-9, "symmetric input, equal scores"


def test_rrf_limit_truncates(conn):
    fused = lexical.reciprocal_rank_fusion([[1, 2, 3, 4]], limit=2)
    assert len(fused) == 2


def test_index_reports_built(conn):
    assert lexical.is_built(conn) is True


def test_hybrid_without_an_fts_index_raises_rather_than_degrading(tmp_path, monkeypatch):
    """Regression guard. Hybrid with no BM25 index initially fused dense against
    an empty ranking, producing numbers *identical to the dense baseline* while
    reporting success — an improvement that silently wasn't one."""
    import numpy as np
    from inforce import index as idx, retrieve

    c = store.connect(tmp_path / "n.db")
    store.init_schema(c)
    assert lexical.is_built(c) is False

    monkeypatch.setattr(retrieve, "load_index",
                        lambda: (np.zeros((3, 4), dtype=np.float32),
                                 np.array([1, 2, 3], dtype=np.int64)))
    monkeypatch.setattr(retrieve, "check_fresh", lambda conn: None)
    monkeypatch.setattr(retrieve.embedding, "embed_query",
                        lambda q: np.zeros(4, dtype=np.float32))

    with pytest.raises(retrieve.IndexNotBuilt, match="BM25"):
        retrieve.search(c, "capital adequacy", k=3, retrieval="hybrid")
