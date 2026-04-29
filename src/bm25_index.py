import pickle
import re
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.corpus = []
        self.index = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, texts: list[str]):
        self.corpus = texts
        tokenized = [self.tokenize(t) for t in texts]
        self.index = BM25Okapi(tokenized)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        tokenized_query = self.tokenize(query_text)
        scores = self.index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"index": i, "text": self.corpus[i], "score": float(scores[i])} for i in top_indices]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "index": self.index}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.corpus = data["corpus"]
        obj.index = data["index"]
        return obj
