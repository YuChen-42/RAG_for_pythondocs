from src.rag_engine import RAGEngine


def test_rag_engine_init():
    engine = RAGEngine()
    assert engine.retriever is not None
