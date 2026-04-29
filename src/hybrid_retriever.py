from src import config
from src.vector_store import VectorStore
from src.bm25_index import BM25Index
from src.embedding import EmbeddingModel


def rrf_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60, top_k: int = 20) -> list[dict]:
    scores = {}
    for rank, item in enumerate(vector_results):
        key = item.get("id", item.get("text"))
        scores[key] = {"score": 1.0 / (k + rank + 1), "item": item}
    for rank, item in enumerate(bm25_results):
        key = item.get("id", item.get("text"))
        if key in scores:
            scores[key]["score"] += 1.0 / (k + rank + 1)
        else:
            scores[key] = {"score": 1.0 / (k + rank + 1), "item": item}
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["item"] for r in sorted_results[:top_k]]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        if top_k is None:
            top_k = config.TOP_K_FINAL

        query_embedding = self.embedding_model.encode([query])[0]
        vector_results = self.vector_store.query(query_embedding, top_k=config.TOP_K_VECTOR)
        vector_results = [{"id": r["id"], "text": r["text"], "metadata": r["metadata"]} for r in vector_results]

        bm25_results = self.bm25_index.query(query, top_k=config.TOP_K_BM25)
        bm25_results = [{"id": f"bm25_{r['index']}", "text": r["text"], "metadata": {}} for r in bm25_results]

        fused = rrf_fusion(vector_results, bm25_results, k=config.RRF_K, top_k=config.TOP_K_RRF)
        return fused[:top_k]
