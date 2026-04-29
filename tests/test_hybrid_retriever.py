from src.hybrid_retriever import HybridRetriever, rrf_fusion


def test_rrf_fusion():
    vector_results = [
        {"id": "a", "text": "A"},
        {"id": "b", "text": "B"},
    ]
    bm25_results = [
        {"id": "b", "text": "B"},
        {"id": "c", "text": "C"},
    ]
    fused = rrf_fusion(vector_results, bm25_results, k=60, top_k=3)
    assert len(fused) == 3
    ids = [r["id"] for r in fused]
    assert "b" in ids


def test_hybrid_retriever_query():
    from src.hybrid_retriever import HybridRetriever
    assert hasattr(HybridRetriever, "retrieve")
