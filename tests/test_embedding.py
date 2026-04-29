import numpy as np
from src.embedding import EmbeddingModel


def test_embedding_model():
    model = EmbeddingModel()
    texts = ["hello world", "test sentence"]
    embeddings = model.encode(texts)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
    assert isinstance(embeddings, np.ndarray)
