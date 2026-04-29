import chromadb
from chromadb.config import Settings
from src import config


class VectorStore:
    def __init__(self, collection_name: str = "python_docs"):
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, texts: list[str], embeddings: list, metadatas: list[dict], ids: list[str] = None):
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding, top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        return output

    def count(self) -> int:
        return self.collection.count()
