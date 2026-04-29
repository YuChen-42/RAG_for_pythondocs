from src.llm_client import LLMClient


def test_llm_client_init():
    client = LLMClient()
    assert client.model is not None
