from sentence_transformers import CrossEncoder
from src import config


class Reranker:
    def __init__(self, model_path: str = None):
        self.model = CrossEncoder(model_path or config.RERANKER_MODEL_PATH)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = sorted(
            [{"text": doc, "score": float(score)} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True
        )
        return scored_docs[:top_k]
