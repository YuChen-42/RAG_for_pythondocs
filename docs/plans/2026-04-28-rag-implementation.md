# RAG 问答系统实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 interview_RAG 目录中从零搭建一个完整的 RAG 问答系统，支持混合检索（BM25 + 向量 + RRF）、Ragas 评估及所有加分项。

**Architecture:** 分层模块化设计。底层为文档解析/分块/向量化/BM25索引，中间层为混合检索+精排+Query改写，上层为LLM生成+流式输出+Web UI，侧面为Ragas评估与迭代优化。

**Tech Stack:** Python 3.13, sentence-transformers, ChromaDB, rank-bm25, Flask, ragas, openai SDK, BeautifulSoup, matplotlib, rich, tqdm

---

## 任务总览

按依赖顺序执行，每个任务包含：写测试 → 跑测试（应失败）→ 实现代码 → 跑测试（应通过）→ 提交。

---

### Task 1: 创建项目基础设施

**Files:**
- Create: `requirements.txt`
- Create: `src/config.py`
- Create: `run.py`（骨架）

**Step 1: 创建 requirements.txt**

```txt
beautifulsoup4==4.13.4
chromadb==1.5.0
flask==3.1.2
lxml==5.4.0
matplotlib==3.10.0
openai==1.75.0
ragas==0.4.3
rank-bm25==0.2.2
rich==14.0.0
sentence-transformers==5.2.2
tqdm==4.67.1
transformers==5.1.0
```

**Step 2: 创建 src/config.py**

```python
import os

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K_VECTOR = 50
TOP_K_BM25 = 50
TOP_K_RRF = 20
TOP_K_FINAL = 5
RRF_K = 60

USE_RERANK = True
USE_QUERY_REWRITE = True
USE_MULTI_QUERY_SPLIT = True

EMBEDDING_MODEL_PATH = "./models/embeddings/Qwen3-Embedding-0.6B"
RERANKER_MODEL_PATH = "./models/rerankers/bge-reranker-v2-m3"
CHROMA_DB_PATH = "./data/chroma_db"
BM25_INDEX_PATH = "./data/bm25_index.pkl"
CHUNKS_PATH = "./data/chunks.json"
QA_PAIRS_PATH = "./data/qa_pairs.json"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

SELECTED_DOCS = [
    "argparse.html", "asyncio.html", "collections.html", "datetime.html",
    "functools.html", "io.html", "json.html", "logging.html", "os.html",
    "pathlib.html", "re.html", "socket.html", "sqlite3.html", "sys.html",
    "threading.html", "typing.html", "unittest.html", "urllib.html",
    "xml.etree.elementtree.html", "zipfile.html",
]
```

**Step 3: 创建 run.py 骨架**

```python
import argparse


def main():
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--build-index", action="store_true", help="构建索引")
    parser.add_argument("--query", type=str, help="CLI 问答")
    parser.add_argument("--web", action="store_true", help="启动 Web 服务")
    parser.add_argument("--eval", action="store_true", help="运行 Ragas 评估")
    parser.add_argument("--config", type=str, default="baseline", help="评估配置")
    args = parser.parse_args()

    if args.build_index:
        from scripts.build_index import main as build
        build()
    elif args.query:
        print(f"Query: {args.query}")
    elif args.web:
        from app.app import app
        app.run(host="0.0.0.0", port=5000, debug=True)
    elif args.eval:
        from scripts.evaluate import main as evaluate
        evaluate(args.config)


if __name__ == "__main__":
    main()
```

**Step 4: 验证导入**

Run: `python -c "from src import config; print(config.CHUNK_SIZE)"`
Expected: `512`

**Step 5: 提交**

```bash
git add requirements.txt src/config.py run.py
git commit -m "feat: 创建项目基础设施"
```

---

### Task 2: 实现文档解析模块

**Files:**
- Create: `src/document_parser.py`
- Create: `tests/test_document_parser.py`

**Step 1: 写测试**

```python
import pytest
from src.document_parser import parse_html_document


def test_parse_html_extracts_text():
    html = '''<html><head><title>Test Doc</title></head>
    <body><div role="main"><p>Hello world</p></div></body></html>'''
    result = parse_html_document(html, "test.html")
    assert result["title"] == "Test Doc"
    assert "Hello world" in result["text"]
    assert result["source"] == "test.html"
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_document_parser.py -v`
Expected: `ModuleNotFoundError: No module named 'src.document_parser'`

**Step 3: 实现 document_parser.py**

```python
from bs4 import BeautifulSoup


def parse_html_document(html_content: str, source_name: str) -> dict:
    soup = BeautifulSoup(html_content, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else source_name

    main = soup.find("div", {"role": "main"})
    if not main:
        main = soup.find("div", {"class": "body"})
    if not main:
        main = soup.body

    for tag in main.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    return {"source": source_name, "title": title, "text": text}
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_document_parser.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/document_parser.py tests/test_document_parser.py
git commit -m "feat: 实现 HTML 文档解析模块"
```

---

### Task 3: 实现文本分块模块

**Files:**
- Create: `src/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: 写测试**

```python
from src.chunker import split_text_into_chunks


def test_chunking():
    text = "A" * 300 + "\n\n" + "B" * 300 + "\n\n" + "C" * 300
    chunks = split_text_into_chunks(text, "test.html", "Test", chunk_size=512, overlap=100)
    assert len(chunks) > 0
    assert all("text" in c and "metadata" in c for c in chunks)
    assert chunks[0]["metadata"]["source"] == "test.html"
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_chunker.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 chunker.py**

```python
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
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_chunker.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/chunker.py tests/test_chunker.py
git commit -m "feat: 实现文本分块模块"
```

---

### Task 4: 实现嵌入模型封装

**Files:**
- Create: `src/embedding.py`
- Create: `tests/test_embedding.py`

**Step 1: 写测试**

```python
import numpy as np
from src.embedding import EmbeddingModel


def test_embedding_model():
    model = EmbeddingModel()
    texts = ["hello world", "test sentence"]
    embeddings = model.encode(texts)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
    assert isinstance(embeddings, np.ndarray)
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_embedding.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 embedding.py**

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from src import config


class EmbeddingModel:
    def __init__(self, model_path: str = None):
        self.model = SentenceTransformer(model_path or config.EMBEDDING_MODEL_PATH)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_embedding.py -v`
Expected: `1 passed`（模型加载需要几秒钟）

**Step 5: 提交**

```bash
git add src/embedding.py tests/test_embedding.py
git commit -m "feat: 实现 Embedding 模型封装"
```

---

### Task 5: 实现向量存储模块

**Files:**
- Create: `src/vector_store.py`
- Create: `tests/test_vector_store.py`

**Step 1: 写测试**

```python
import numpy as np
from src.vector_store import VectorStore


def test_vector_store():
    vs = VectorStore(collection_name="test_collection")
    texts = ["hello world", "foo bar"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadatas = [{"source": "a.html"}, {"source": "b.html"}]
    vs.add(texts, embeddings, metadatas)
    results = vs.query(np.array([1.0, 0.0]), top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "a.html"
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_vector_store.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 vector_store.py**

```python
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
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_vector_store.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat: 实现 ChromaDB 向量存储模块"
```

---

### Task 6: 实现 BM25 索引模块

**Files:**
- Create: `src/bm25_index.py`
- Create: `tests/test_bm25_index.py`

**Step 1: 写测试**

```python
import os
from src.bm25_index import BM25Index


def test_bm25_index():
    idx = BM25Index()
    texts = ["hello world", "hello python", "python asyncio"]
    idx.build(texts)
    results = idx.query("python", top_k=2)
    assert len(results) == 2
    assert any("python" in r["text"] for r in results)
    # 测试序列化
    idx.save("/tmp/test_bm25.pkl")
    assert os.path.exists("/tmp/test_bm25.pkl")
    idx2 = BM25Index.load("/tmp/test_bm25.pkl")
    results2 = idx2.query("python", top_k=2)
    assert len(results2) == 2
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_bm25_index.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 bm25_index.py**

```python
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
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_bm25_index.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/bm25_index.py tests/test_bm25_index.py
git commit -m "feat: 实现 BM25 索引模块"
```

---

### Task 7: 实现混合检索模块（含 RRF）

**Files:**
- Create: `src/hybrid_retriever.py`
- Create: `tests/test_hybrid_retriever.py`

**Step 1: 写测试**

```python
from src.hybrid_retriever import HybridRetriever, rrf_fusion


def test_rrf_fusion():
    vector_results = [
        {"id": "a", "text": "A"},
        {"id": "b", "text": "B"},
    ]
    bm25_results = [
        {"id": "b", "text": "B"},
        {"id": "c", "text": "C"},
    ]
    fused = rrf_fusion(vector_results, bm25_results, k=60, top_k=3)
    assert len(fused) == 3
    ids = [r["id"] for r in fused]
    assert "b" in ids  # b 在两路都有，应该排前面


def test_hybrid_retriever_query():
    # 由于 HybridRetriever 依赖 VectorStore 和 BM25Index，此处只做接口测试
    from src.hybrid_retriever import HybridRetriever
    assert hasattr(HybridRetriever, "retrieve")
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 hybrid_retriever.py**

```python
from src import config
from src.vector_store import VectorStore
from src.bm25_index import BM25Index
from src.embedding import EmbeddingModel


def rrf_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60, top_k: int = 20) -> list[dict]:
    scores = {}
    for rank, item in enumerate(vector_results):
        key = item.get("id", item.get("text"))
        scores[key] = {"score": 1.0 / (k + rank + 1), "item": item}
    for rank, item in enumerate(bm25_results):
        key = item.get("id", item.get("text"))
        if key in scores:
            scores[key]["score"] += 1.0 / (k + rank + 1)
        else:
            scores[key] = {"score": 1.0 / (k + rank + 1), "item": item}
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["item"] for r in sorted_results[:top_k]]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        if top_k is None:
            top_k = config.TOP_K_FINAL

        # 向量检索
        query_embedding = self.embedding_model.encode([query])[0]
        vector_results = self.vector_store.query(query_embedding, top_k=config.TOP_K_VECTOR)
        # 转换为统一格式
        vector_results = [{"id": r["id"], "text": r["text"], "metadata": r["metadata"]} for r in vector_results]

        # BM25 检索
        bm25_results = self.bm25_index.query(query, top_k=config.TOP_K_BM25)
        bm25_results = [{"id": f"bm25_{r['index']}", "text": r["text"], "metadata": {}} for r in bm25_results]

        # RRF 融合
        fused = rrf_fusion(vector_results, bm25_results, k=config.RRF_K, top_k=config.TOP_K_RRF)
        return fused[:top_k]
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: `2 passed`

**Step 5: 提交**

```bash
git add src/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat: 实现混合检索 + RRF 融合模块"
```

---

### Task 8: 实现重排序模块

**Files:**
- Create: `src/reranker.py`
- Create: `tests/test_reranker.py`

**Step 1: 写测试**

```python
from src.reranker import Reranker


def test_reranker():
    r = Reranker()
    query = "how to use asyncio"
    docs = ["asyncio is a library", "json is for parsing", "asyncio tutorial"]
    results = r.rerank(query, docs, top_k=2)
    assert len(results) == 2
    # asyncio tutorial 应该排更前面
    assert "asyncio" in results[0]["text"].lower()
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_reranker.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 reranker.py**

```python
from sentence_transformers import CrossEncoder
from src import config


class Reranker:
    def __init__(self, model_path: str = None):
        self.model = CrossEncoder(model_path or config.RERANKER_MODEL_PATH)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = sorted(
            [{"text": doc, "score": float(score)} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True
        )
        return scored_docs[:top_k]
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_reranker.py -v`
Expected: `1 passed`（模型加载需要几秒钟）

**Step 5: 提交**

```bash
git add src/reranker.py tests/test_reranker.py
git commit -m "feat: 实现 Cross-Encoder 重排序模块"
```

---

### Task 9: 实现 Query 改写模块

**Files:**
- Create: `src/query_rewriter.py`
- Create: `tests/test_query_rewriter.py`

**Step 1: 写测试**

```python
from src.query_rewriter import QueryRewriter


def test_rewrite():
    qw = QueryRewriter()
    # 由于需要真实 LLM，测试仅验证接口和简单规则
    result = qw.rewrite("asyncio usage")
    assert isinstance(result, list)
    assert len(result) > 0
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_query_rewriter.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 query_rewriter.py**

```python
import os
from openai import OpenAI
from src import config


class QueryRewriter:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY or os.getenv("LLM_API_KEY", "")
        )
        self.model = config.LLM_MODEL

    def split_multi_topic(self, query: str) -> list[str]:
        prompt = f"""判断以下问题是否涉及多个完全不同的主题/模块。
如果是，请拆分为多个独立子问题，每行一个。
如果不是，请原样返回问题本身。

问题: {query}

只输出问题列表，不要任何解释。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        return lines if lines else [query]

    def expand_query(self, query: str) -> list[str]:
        prompt = f"""将以下问题改写为 2-3 个语义等价但表述不同的查询，每行一个。

原问题: {query}

只输出查询列表，不要任何解释。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        return lines if lines else [query]

    def rewrite(self, query: str) -> list[str]:
        if not config.USE_QUERY_REWRITE:
            return [query]
        if config.USE_MULTI_QUERY_SPLIT:
            sub_queries = self.split_multi_topic(query)
        else:
            sub_queries = [query]
        all_queries = []
        for sq in sub_queries:
            all_queries.extend(self.expand_query(sq))
        return list(dict.fromkeys(all_queries))  # 去重
```

**Step 4: 跑测试（可能需要 API Key，若失败则跳过并说明）**

Run: `pytest tests/test_query_rewriter.py -v`
Expected: `1 passed` 或 `skipped`（需要 API Key）

**Step 5: 提交**

```bash
git add src/query_rewriter.py tests/test_query_rewriter.py
git commit -m "feat: 实现 Query 改写与多模块拆分模块"
```

---

### Task 10: 实现 LLM 客户端模块

**Files:**
- Create: `src/llm_client.py`
- Create: `tests/test_llm_client.py`

**Step 1: 写测试**

```python
from src.llm_client import LLMClient


def test_llm_client_init():
    client = LLMClient()
    assert client.model is not None
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_llm_client.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 llm_client.py**

```python
import os
from openai import OpenAI
from src import config


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY or os.getenv("LLM_API_KEY", "")
        )
        self.model = config.LLM_MODEL

    def chat(self, messages: list[dict], temperature: float = 0.7, stream: bool = False):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )

    def chat_stream(self, messages: list[dict], temperature: float = 0.7):
        response = self.chat(messages, temperature=temperature, stream=True)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_llm_client.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/llm_client.py tests/test_llm_client.py
git commit -m "feat: 实现 LLM API 客户端模块"
```

---

### Task 11: 实现生成器模块

**Files:**
- Create: `src/generator.py`
- Create: `tests/test_generator.py`

**Step 1: 写测试**

```python
from src.generator import Generator


def test_generator_build_prompt():
    g = Generator()
    chunks = [
        {"text": "asyncio is a library", "metadata": {"source": "asyncio.html"}},
    ]
    prompt = g.build_prompt("what is asyncio?", chunks)
    assert "asyncio is a library" in prompt
    assert "asyncio.html" in prompt
    assert "what is asyncio?" in prompt
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_generator.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 generator.py**

```python
from src.llm_client import LLMClient


class Generator:
    def __init__(self):
        self.client = LLMClient()

    def build_prompt(self, query: str, chunks: list[dict]) -> list[dict]:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"[文档片段 {i+1}]\n来源: {source}\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """你是一个基于 Python 官方文档的问答助手。请根据提供的文档片段回答问题。
如果文档中没有相关信息，请明确说明"根据提供的文档，无法找到答案"。
请在答案末尾列出引用的来源文档，格式为 [来源: 文件名]。"""

        user_prompt = f"""以下是与问题相关的文档片段：

{context}

问题: {query}

请基于以上文档片段回答问题。"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate(self, query: str, chunks: list[dict], stream: bool = False) -> str:
        messages = self.build_prompt(query, chunks)
        if stream:
            return self.client.chat_stream(messages)
        response = self.client.chat(messages)
        return response.choices[0].message.content
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_generator.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/generator.py tests/test_generator.py
git commit -m "feat: 实现答案生成器模块"
```

---

### Task 12: 实现 RAG 引擎编排模块

**Files:**
- Create: `src/rag_engine.py`
- Create: `tests/test_rag_engine.py`

**Step 1: 写测试**

```python
from src.rag_engine import RAGEngine


def test_rag_engine_init():
    engine = RAGEngine()
    assert engine.retriever is not None
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_rag_engine.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 rag_engine.py**

```python
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
        self.bm25_index = BM25Index.load(config.BM25_INDEX_PATH) if __import__("os").path.exists(config.BM25_INDEX_PATH) else BM25Index()
        self.retriever = HybridRetriever(self.vector_store, self.bm25_index, self.embedding_model)
        self.reranker = Reranker() if config.USE_RERANK else None
        self.query_rewriter = QueryRewriter()
        self.generator = Generator()

    def query(self, user_query: str, stream: bool = False) -> dict:
        # Query 改写
        rewritten_queries = self.query_rewriter.rewrite(user_query)

        # 多路检索并合并
        all_results = []
        for q in rewritten_queries:
            results = self.retriever.retrieve(q)
            all_results.extend(results)

        # 去重（按 text）
        seen = set()
        unique_results = []
        for r in all_results:
            key = r.get("text", "")
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        # Rerank
        if self.reranker and config.USE_RERANK:
            docs = [r["text"] for r in unique_results]
            reranked = self.reranker.rerank(user_query, docs, top_k=config.TOP_K_FINAL)
            final_chunks = [{"text": r["text"], "metadata": {}} for r in reranked]
            # 补充 metadata
            text_to_meta = {r["text"]: r.get("metadata", {}) for r in unique_results}
            for c in final_chunks:
                c["metadata"] = text_to_meta.get(c["text"], {})
        else:
            final_chunks = unique_results[:config.TOP_K_FINAL]

        # 生成答案
        if stream:
            return {
                "stream": self.generator.generate(user_query, final_chunks, stream=True),
                "sources": [c.get("metadata", {}).get("source", "") for c in final_chunks]
            }
        answer = self.generator.generate(user_query, final_chunks)
        return {
            "answer": answer,
            "sources": [c.get("metadata", {}).get("source", "") for c in final_chunks]
        }
```

**Step 4: 跑测试（应通过，可能需要索引存在）**

Run: `pytest tests/test_rag_engine.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/rag_engine.py tests/test_rag_engine.py
git commit -m "feat: 实现 RAG 引擎编排模块"
```

---

### Task 13: 实现评估器模块

**Files:**
- Create: `src/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: 写测试**

```python
from src.evaluator import Evaluator


def test_evaluator():
    e = Evaluator()
    assert e.metrics is not None
```

**Step 2: 跑测试（应失败）**

Run: `pytest tests/test_evaluator.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 实现 evaluator.py**

```python
import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from src.llm_client import LLMClient


class Evaluator:
    def __init__(self):
        self.metrics = [faithfulness, answer_relevancy, context_recall]
        self.client = LLMClient()

    def evaluate(self, qa_pairs: list[dict]) -> dict:
        data = {
            "question": [q["question"] for q in qa_pairs],
            "answer": [q["answer"] for q in qa_pairs],
            "contexts": [q["contexts"] for q in qa_pairs],
            "ground_truth": [q["ground_truth"] for q in qa_pairs],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=self.metrics, llm=self.client)
        return result.to_pandas().to_dict(orient="records")

    def save_report(self, results: list[dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
```

**Step 4: 跑测试（应通过）**

Run: `pytest tests/test_evaluator.py -v`
Expected: `1 passed`

**Step 5: 提交**

```bash
git add src/evaluator.py tests/test_evaluator.py
git commit -m "feat: 实现 Ragas 评估器模块"
```

---

### Task 14: 实现构建索引脚本

**Files:**
- Create: `scripts/build_index.py`

**Step 1: 实现 build_index.py**

```python
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

    # 1. 解析文档
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

    # 保存 chunks
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"共生成 {len(all_chunks)} 个 chunks")

    # 2. 向量化
    print("[2/4] 编码向量...")
    model = EmbeddingModel()
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts)

    # 3. 存入向量库
    print("[3/4] 存入 ChromaDB...")
    vs = VectorStore()
    vs.collection.delete(where={})  # 清空旧数据
    vs.add(
        texts=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]
    )
    print(f"向量库共 {vs.count()} 条记录")

    # 4. 构建 BM25
    print("[4/4] 构建 BM25 索引...")
    bm25 = BM25Index()
    bm25.build(texts)
    bm25.save(config.BM25_INDEX_PATH)
    print(f"BM25 索引已保存到 {config.BM25_INDEX_PATH}")

    print("索引构建完成！")


if __name__ == "__main__":
    main()
```

**Step 2: 验证**

Run: `python scripts/build_index.py`
Expected: 成功构建索引，输出 chunks 数量、向量库记录数、BM25 保存路径

**Step 3: 提交**

```bash
git add scripts/build_index.py
git commit -m "feat: 实现一键构建索引脚本"
```

---

### Task 15: 实现评估脚本

**Files:**
- Create: `scripts/evaluate.py`

**Step 1: 实现 evaluate.py**

```python
import json
import argparse
import os
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src import config
from src.evaluator import Evaluator
from src.rag_engine import RAGEngine


def load_qa_pairs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_qa_and_evaluate(config_name: str = "baseline"):
    console = Console()

    # 根据配置调整参数
    if config_name == "optimized":
        config.CHUNK_SIZE = 256
        config.CHUNK_OVERLAP = 50
        config.USE_RERANK = True
    else:
        config.CHUNK_SIZE = 512
        config.CHUNK_OVERLAP = 100
        config.USE_RERANK = False

    qa_pairs = load_qa_pairs(config.QA_PAIRS_PATH)
    engine = RAGEngine()
    evaluator = Evaluator()

    console.print(f"[bold green]开始评估: {config_name}[/bold green]")

    enriched_pairs = []
    for qa in tqdm(qa_pairs, desc="生成答案"):
        result = engine.query(qa["question"])
        enriched_pairs.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [r["text"] for r in engine.retriever.retrieve(qa["question"])],
            "ground_truth": qa["ground_truth"],
        })

    results = evaluator.evaluate(enriched_pairs)

    # 展示表格
    table = Table(title=f"Ragas 评估结果 - {config_name}")
    table.add_column("问题", style="cyan")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Answer Relevancy", justify="right")
    table.add_column("Context Recall", justify="right")

    for r in results:
        table.add_row(
            r["question"][:40] + "...",
            f"{r.get('faithfulness', 0):.3f}",
            f"{r.get('answer_relevancy', 0):.3f}",
            f"{r.get('context_recall', 0):.3f}",
        )

    console.print(table)

    # 保存报告
    os.makedirs("results", exist_ok=True)
    path = f"results/ragas_report_{config_name}.json"
    evaluator.save_report(results, path)
    console.print(f"[bold blue]报告已保存: {path}[/bold blue]")

    return results


def main(config_name: str = "baseline"):
    run_qa_and_evaluate(config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline", choices=["baseline", "optimized"])
    args = parser.parse_args()
    main(args.config)
```

**Step 2: 提交**

```bash
git add scripts/evaluate.py
git commit -m "feat: 实现 Ragas 评估脚本"
```

---

### Task 16: 实现 Web UI

**Files:**
- Create: `app/app.py`
- Create: `app/templates/index.html`

**Step 1: 实现 app.py**

```python
from flask import Flask, render_template, request, jsonify, Response
from src.rag_engine import RAGEngine

app = Flask(__name__, template_folder="templates")
engine = RAGEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "query is required"}), 400
    result = engine.query(query)
    return jsonify({"answer": result["answer"], "sources": result["sources"]})


@app.route("/api/stream")
def api_stream():
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "q is required"}), 400

    def generate():
        result = engine.query(query, stream=True)
        for token in result["stream"]:
            yield f"data: {token}\n\n"
        sources = ",".join(result["sources"])
        yield f"data: [SOURCES]{sources}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

**Step 2: 实现 index.html**

```html
<!DOCTYPE html>
<html>
<head>
    <title>RAG 问答系统</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        #query { width: 100%; padding: 10px; font-size: 16px; }
        #submit { padding: 10px 20px; font-size: 16px; margin-top: 10px; }
        #answer { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; white-space: pre-wrap; }
        #sources { margin-top: 10px; color: #666; }
        .source-tag { display: inline-block; background: #e0e0e0; padding: 2px 8px; border-radius: 4px; margin: 2px; }
    </style>
</head>
<body>
    <h1>Python 文档 RAG 问答系统</h1>
    <input type="text" id="query" placeholder="输入你的问题...">
    <button id="submit" onclick="ask()">提问</button>
    <div id="answer"></div>
    <div id="sources"></div>

    <script>
        function ask() {
            const query = document.getElementById('query').value;
            const answerDiv = document.getElementById('answer');
            const sourcesDiv = document.getElementById('sources');
            answerDiv.textContent = '';
            sourcesDiv.innerHTML = '';

            const eventSource = new EventSource('/api/stream?q=' + encodeURIComponent(query));
            eventSource.onmessage = function(event) {
                if (event.data === '[DONE]') {
                    eventSource.close();
                } else if (event.data.startsWith('[SOURCES]')) {
                    const sources = event.data.replace('[SOURCES]', '').split(',');
                    sourcesDiv.innerHTML = '<strong>来源：</strong>' +
                        sources.map(s => '<span class="source-tag">' + s + '</span>').join('');
                } else {
                    answerDiv.textContent += event.data;
                }
            };
        }
    </script>
</body>
</html>
```

**Step 3: 提交**

```bash
git add app/app.py app/templates/index.html
git commit -m "feat: 实现 Flask Web UI 与流式输出"
```

---

### Task 17: 完善统一入口 run.py

**Files:**
- Modify: `run.py`

**Step 1: 更新 run.py**

```python
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--build-index", action="store_true", help="构建索引")
    parser.add_argument("--query", type=str, help="CLI 问答")
    parser.add_argument("--web", action="store_true", help="启动 Web 服务")
    parser.add_argument("--eval", action="store_true", help="运行 Ragas 评估")
    parser.add_argument("--config", type=str, default="baseline", choices=["baseline", "optimized"], help="评估配置")
    args = parser.parse_args()

    if args.build_index:
        from scripts.build_index import main as build
        build()
    elif args.query:
        from src.rag_engine import RAGEngine
        engine = RAGEngine()
        result = engine.query(args.query)
        print(f"\n答案:\n{result['answer']}\n")
        print(f"来源: {', '.join(result['sources'])}")
    elif args.web:
        from app.app import app
        app.run(host="0.0.0.0", port=5000, debug=True)
    elif args.eval:
        from scripts.evaluate import main as evaluate
        evaluate(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: 验证**

Run: `python run.py --help`
Expected: 显示帮助信息

**Step 3: 提交**

```bash
git add run.py
git commit -m "feat: 完善统一入口 run.py"
```

---

### Task 18: 构建 QA 对数据

**Files:**
- Create: `data/qa_pairs.json`

**Step 1: 手工编写 8 条 QA 对**

```json
[
  {
    "question": "How does argparse handle subcommands?",
    "ground_truth": "argparse supports subcommands via the add_subparsers() method, which creates a sub-parser for each command.",
    "contexts": ["argparse.html"]
  },
  {
    "question": "What is the difference between asyncio.create_task() and asyncio.ensure_future()?",
    "ground_truth": "asyncio.create_task() is the preferred way to create tasks, while ensure_future() is a lower-level function that also accepts coroutines.",
    "contexts": ["asyncio-task.html"]
  },
  {
    "question": "What are the main container datatypes provided by the collections module?",
    "ground_truth": "The collections module provides namedtuple, deque, Counter, OrderedDict, defaultdict, and ChainMap.",
    "contexts": ["collections.html"]
  },
  {
    "question": "How do you parse a JSON string in Python?",
    "ground_truth": "Use json.loads() to parse a JSON string into a Python object.",
    "contexts": ["json.html"]
  },
  {
    "question": "What is the purpose of the os.path module?",
    "ground_truth": "os.path provides utilities for manipulating paths in a platform-independent way.",
    "contexts": ["os.html"]
  },
  {
    "question": "How does Python's re module perform pattern matching?",
    "ground_truth": "The re module uses regular expressions to match patterns in strings, with functions like search, match, findall, and sub.",
    "contexts": ["re.html"]
  },
  {
    "question": "What is the GIL and how does threading work in Python?",
    "ground_truth": "The Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time, but threading is still useful for I/O-bound tasks.",
    "contexts": ["threading.html"]
  },
  {
    "question": "How do you create a temporary file in Python?",
    "ground_truth": "Use the tempfile module, which provides functions like TemporaryFile, NamedTemporaryFile, and TemporaryDirectory.",
    "contexts": ["tempfile.html"]
  }
]
```

**Step 2: 提交**

```bash
git add data/qa_pairs.json
git commit -m "data: 添加 8 条 Ragas 评估 QA 对"
```

---

### Task 19: 运行 Ragas 评估并生成对比图

**Files:**
- Create: `scripts/generate_comparison_chart.py`

**Step 1: 实现 generate_comparison_chart.py**

```python
import json
import matplotlib.pyplot as plt


def main():
    with open("results/ragas_report_baseline.json", "r") as f:
        baseline = json.load(f)
    with open("results/ragas_report_optimized.json", "r") as f:
        optimized = json.load(f)

    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    baseline_avg = {m: sum(r.get(m, 0) for r in baseline) / len(baseline) for m in metrics}
    optimized_avg = {m: sum(r.get(m, 0) for r in optimized) / len(optimized) for m in metrics}

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], [baseline_avg[m] for m in metrics], width, label="Baseline")
    ax.bar([i + width/2 for i in x], [optimized_avg[m] for m in metrics], width, label="Optimized")
    ax.set_ylabel("Score")
    ax.set_title("Ragas Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/comparison_chart.png")
    print("对比图已保存到 results/comparison_chart.png")


if __name__ == "__main__":
    main()
```

**Step 2: 运行评估**

Run:
```bash
python run.py --eval --config baseline
python run.py --eval --config optimized
python scripts/generate_comparison_chart.py
```

Expected: 生成 `results/ragas_report_baseline.json`、`results/ragas_report_optimized.json`、`results/comparison_chart.png`

**Step 3: 提交**

```bash
git add scripts/generate_comparison_chart.py
git commit -m "feat: 实现 Ragas 评估对比图生成"
```

---

### Task 20: 编写 README.md

**Files:**
- Create: `README.md`

**Step 1: 编写 README**

```markdown
# Python 文档 RAG 问答系统

基于 Python 官方文档的可本地运行 RAG 问答系统，支持混合检索（BM25 + 向量 + RRF）、Ragas 评估及全部加分项。

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

设置环境变量：
```bash
export LLM_API_KEY="your-qwen-api-key"
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen-plus"
```

### 2. 构建索引

```bash
python run.py --build-index
```

### 3. CLI 问答

```bash
python run.py --query "How does argparse handle subcommands?"
```

### 4. 启动 Web 服务

```bash
python run.py --web
```

浏览器访问 http://localhost:5000

### 5. Ragas 评估

```bash
# 基线评估
python run.py --eval --config baseline

# 优化后评估
python run.py --eval --config optimized

# 生成对比图
python scripts/generate_comparison_chart.py
```

## Chunk 策略说明

- **chunk_size**: 512 字符
- **chunk_overlap**: 100 字符（约 20%）
- **选择依据**:
  - Python 文档为 Sphinx 生成的结构化 HTML，段落级切分保留语义完整性
  - 512 字符约为 250 个英文词，能覆盖一个完整的小节
  - 100 字符 overlap 避免关键信息被截断在边界

## 混合检索融合方式

- **RRF（Reciprocal Rank Fusion）**，k=60
- **选择理由**: RRF 无需调参权重，对 BM25 和向量检索的得分分布差异不敏感，比加权融合更稳健

## Ragas 评估结果

见 `results/ragas_report_baseline.json` 和 `results/ragas_report_optimized.json`，对比图见 `results/comparison_chart.png`。

## 项目结构

见 `docs/plans/2026-04-28-rag-system-design.md`。
```

**Step 2: 提交**

```bash
git add README.md
git commit -m "docs: 添加 README 使用说明"
```

---

## 执行选项

**Plan complete and saved to `docs/plans/2026-04-28-rag-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
