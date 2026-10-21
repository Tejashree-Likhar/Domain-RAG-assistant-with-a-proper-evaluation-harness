import json, pathlib, re
import numpy as np
import pytest
from ragkit.corpus import parse_sections, clean_body
from ragkit.chunking import chunk_corpus, chunk_section, Section
from ragkit.bm25 import BM25, tokenize
from ragkit.generate import parse_citations, SYSTEM_PROMPT, build_user_prompt
from eval.metrics import retrieval_summary, key_fact_recall, load_questions, norm
from eval.refusal_eval import calibrate_threshold
from eval.calibrate_judge import agreement

RST = """.. _x:

=====
Title
=====

Intro text that is long enough to keep around here.

Sub heading
-----------

.. currentmodule:: sklearn.foo

See :class:`~sklearn.foo.Bar` and ``code`` for details on this subsection of text.

.. note:: remember this
"""


def test_rst_headings_and_cleaning():
    secs = parse_sections(RST, "d")
    assert [s.headings for s in secs] == [["Title"], ["Title", "Sub heading"]]
    body = secs[1].text
    assert "Bar" in body and ":class:" not in body and "currentmodule" not in body and "Note: remember this" in body


def test_chunking_respects_size_and_overlap():
    para = lambda i: " ".join(f"w{i}_{j}" for j in range(40))
    sec = Section("d", "T", ["T"], "\n\n".join(para(i) for i in range(10)))
    chunks = chunk_section(sec, max_words=100, overlap_words=45)
    assert all(len(c.split()) <= 100 for c in chunks) and len(chunks) > 3
    assert any(p in chunks[1] for p in [para(1)])             # trailing paragraph carried into next chunk


def test_giant_paragraph_is_windowed():
    sec = Section("d", "T", ["T"], " ".join(f"w{i}" for i in range(1000)))
    chunks = chunk_section(sec, 200, 30)
    assert max(len(c.split()) for c in chunks) <= 200 and len(chunks) >= 5


def test_bm25_prefers_matching_doc_and_splits_identifiers():
    bm = BM25(["kmeans n_clusters inertia", "decision tree depth", "svm kernel"])
    assert bm.scores("number of clusters").argmax() == 0
    assert "clusters" in tokenize("n_clusters")


def test_retrieval_metrics():
    s = retrieval_summary([1, 2, None, 6, 11], n_boot=50)
    assert s["recall@1"] == 0.2 and s["recall@5"] == 0.4 and s["recall@10"] == 0.6
    assert abs(s["mrr@10"] - (1 + .5 + 0 + 1 / 6 + 0) / 5) < 1e-9


def test_key_fact_and_citation_parsing():
    assert key_fact_recall("It uses k-means++ init", ["K-Means++", "eps"]) == 0.5
    assert parse_citations("A [1] B [3][9] C [2]", 3) == ([1, 2, 3], [9])


def test_threshold_calibration_separates_clean_case():
    t, bal = calibrate_threshold([.9, .8, .85, .2, .1], [True, True, True, False, False])
    assert bal == 1.0 and .2 < t < .8


def test_judge_agreement_math():
    h = [{"faithfulness": 2, "relevance": 2}, {"faithfulness": 0, "relevance": 1}, {"faithfulness": 1, "relevance": 0}]
    assert agreement(h, h)["faithfulness"]["weighted_kappa"] == 1.0


def test_no_eval_question_leaks_into_prompts_or_source():
    """Trap: leaking eval questions into prompts. Scan the prompt template and all library source."""
    qs = [norm(q["question"]) for q in load_questions()]
    haystacks = [norm(SYSTEM_PROMPT)] + [norm(p.read_text()) for p in pathlib.Path("src/ragkit").glob("*.py")]
    for q in qs:
        assert not any(q in h for h in haystacks), f"eval question leaked: {q}"
    # key facts / evidence must not appear in the system prompt either
    ev = [norm(e) for q in load_questions() for e in q["evidence"]]
    assert not any(e in norm(SYSTEM_PROMPT) for e in ev)


def test_question_set_shape():
    qs = load_questions()
    assert len({q["id"] for q in qs}) == len(qs)
    ans = [q for q in qs if q["type"] == "answerable"]; un = [q for q in qs if q["type"] == "unanswerable"]
    assert len(qs) >= 50 and len(un) >= 10
    assert {"dev", "test"} == {q["split"] for q in qs}
    assert all(q["evidence"] and q["doc"] for q in ans)
