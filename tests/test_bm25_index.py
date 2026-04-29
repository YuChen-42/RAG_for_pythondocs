import os
from src.bm25_index import BM25Index


def test_bm25_index():
    idx = BM25Index()
    texts = ["hello world", "hello python", "python asyncio"]
    idx.build(texts)
    results = idx.query("python", top_k=2)
    assert len(results) == 2
    assert any("python" in r["text"] for r in results)
    idx.save("test_bm25.pkl")
    assert os.path.exists("test_bm25.pkl")
    idx2 = BM25Index.load("test_bm25.pkl")
    results2 = idx2.query("python", top_k=2)
    assert len(results2) == 2
