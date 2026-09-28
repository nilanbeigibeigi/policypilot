"""Offline tests (TF-IDF search, no LLM) — run with: pytest -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import rag
import eval as ev


@pytest.fixture(scope="module")
def index():
    return rag.Index.build("tfidf")


def test_chunks_have_citations():
    chunks = rag.load_chunks()
    assert len(chunks) >= 20
    assert all(c["title"] and c["section"] and c["text"] for c in chunks)


def test_finds_sick_days(index):
    top = index.search("How many sick days do I get?", k=1)[0]
    assert top["section"] == "Sick Days"


def test_prompt_numbers_sources(index):
    p = rag.build_prompt("q", index.search("refund over 100 dollars", k=2))
    assert "[1]" in p and "[2]" in p and "QUESTION: q" in p


def test_retrieval_hit_rate(index):
    summary, _ = ev.run(index=index, llm="none", k=3, verbose=False)
    assert summary["retrieval_hit_rate"] >= 90


def test_generate_rejects_unknown_llm(index):
    with pytest.raises(ValueError):
        rag.generate("q", index.search("q"), llm="gpt")
