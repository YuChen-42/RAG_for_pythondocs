from src.chunker import split_text_into_chunks


def test_chunking():
    text = "A" * 300 + "\n\n" + "B" * 300 + "\n\n" + "C" * 300
    chunks = split_text_into_chunks(text, "test.html", "Test", chunk_size=512, overlap=100)
    assert len(chunks) > 0
    assert all("text" in c and "metadata" in c for c in chunks)
    assert chunks[0]["metadata"]["source"] == "test.html"
