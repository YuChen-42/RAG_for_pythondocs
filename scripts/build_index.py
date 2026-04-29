import json
import os
from tqdm import tqdm

from src import config
from src.document_parser import parse_html_document
from src.chunker import split_text_into_chunks
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.bm25_index import BM25Index


def main():
    os.makedirs("data", exist_ok=True)

    print("[1/4] 解析文档...")
    all_chunks = []
    for doc_name in tqdm(config.SELECTED_DOCS):
        path = os.path.join("documents", doc_name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        parsed = parse_html_document(html, doc_name)
        chunks = split_text_into_chunks(
            parsed["text"], parsed["source"], parsed["title"],
            chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)

    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"共生成 {len(all_chunks)} 个 chunks")

    print("[2/4] 编码向量...")
    model = EmbeddingModel()
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts)

    print("[3/4] 存入 ChromaDB...")
    vs = VectorStore()
    try:
        vs.collection.delete(where={})
    except Exception:
        pass
    vs.add(
        texts=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]
    )
    print(f"向量库共 {vs.count()} 条记录")

    print("[4/4] 构建 BM25 索引...")
    bm25 = BM25Index()
    bm25.build(texts)
    bm25.save(config.BM25_INDEX_PATH)
    print(f"BM25 索引已保存到 {config.BM25_INDEX_PATH}")

    print("索引构建完成！")


if __name__ == "__main__":
    main()
