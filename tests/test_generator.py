from src.generator import Generator


def test_generator_build_prompt():
    g = Generator()
    chunks = [
        {"text": "asyncio is a library", "metadata": {"source": "asyncio.html"}},
    ]
    prompt = g.build_prompt("what is asyncio?", chunks)
    assert "asyncio is a library" in prompt[1]["content"]
    assert "asyncio.html" in prompt[1]["content"]
    assert "what is asyncio?" in prompt[1]["content"]
