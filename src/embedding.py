import numpy as np
from sentence_transformers import SentenceTransformer
from src import config


class EmbeddingModel:
    def __init__(self, model_path: str = None):
        self.model = SentenceTransformer(model_path or config.EMBEDDING_MODEL_PATH)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
