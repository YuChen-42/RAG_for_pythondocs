import numpy as np
from src.vector_store import VectorStore


def test_vector_store():
    vs = VectorStore(collection_name="test_collection")
    texts = ["hello world", "foo bar"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadatas = [{"source": "a.html"}, {"source": "b.html"}]
    vs.add(texts, embeddings, metadatas)
    results = vs.query(np.array([1.0, 0.0]), top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "a.html"
