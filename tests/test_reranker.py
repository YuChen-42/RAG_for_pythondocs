from src.reranker import Reranker


def test_reranker():
    r = Reranker()
    query = "how to use asyncio"
    docs = ["asyncio is a library", "json is for parsing", "asyncio tutorial"]
    results = r.rerank(query, docs, top_k=2)
    assert len(results) == 2
    assert "asyncio" in results[0]["text"].lower()
