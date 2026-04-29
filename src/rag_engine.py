import os
from src import config
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.bm25_index import BM25Index
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.query_rewriter import QueryRewriter
from src.generator import Generator


class RAGEngine:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        if os.path.exists(config.BM25_INDEX_PATH):
            self.bm25_index = BM25Index.load(config.BM25_INDEX_PATH)
        else:
            self.bm25_index = BM25Index()
        self.retriever = HybridRetriever(self.vector_store, self.bm25_index, self.embedding_model)
        self.reranker = Reranker() if config.USE_RERANK else None
        self.query_rewriter = QueryRewriter()
        self.generator = Generator()

    def query(self, user_query: str, stream: bool = False, use_query_rewrite: bool = None, use_rerank: bool = None) -> dict:
        if use_query_rewrite is None:
            use_query_rewrite = config.USE_QUERY_REWRITE
        if use_rerank is None:
            use_rerank = config.USE_RERANK
        rewritten_queries = self.query_rewriter.rewrite(user_query, use_query_rewrite=use_query_rewrite)

        all_results = []
        for q in rewritten_queries:
            results = self.retriever.retrieve(q)
            all_results.extend(results)

        seen = set()
        unique_results = []
        for r in all_results:
            key = r.get("text", "")
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        if use_rerank and self.reranker:
            docs = [r["text"] for r in unique_results]
            reranked = self.reranker.rerank(user_query, docs, top_k=config.TOP_K_FINAL)
            final_chunks = [{"text": r["text"], "metadata": {}} for r in reranked]
            text_to_meta = {r["text"]: r.get("metadata", {}) for r in unique_results}
            for c in final_chunks:
                c["metadata"] = text_to_meta.get(c["text"], {})
        else:
            final_chunks = unique_results[:config.TOP_K_FINAL]

        sources = []
        for i, c in enumerate(final_chunks, start=1):
            sources.append({
                "index": i,
                "source": c.get("metadata", {}).get("source", ""),
                "text": c.get("text", "")[:300]
            })

        if stream:
            return {
                "stream": self.generator.generate(user_query, final_chunks, stream=True),
                "sources": sources
            }
        answer = self.generator.generate(user_query, final_chunks)
        return {
            "answer": answer,
            "sources": sources
        }
