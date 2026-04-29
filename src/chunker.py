def split_text_into_chunks(text: str, source: str, title: str, chunk_size: int = 512, overlap: int = 100) -> list:
    separators = ["\n\n", "\n", " ", ""]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in separators:
                idx = text.rfind(sep, start + chunk_size - overlap, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {"source": source, "title": title}
            })
        start = end - overlap if end < len(text) else end
    return chunks
