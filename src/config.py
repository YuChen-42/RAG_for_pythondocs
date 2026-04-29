import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

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

EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embeddings", "Qwen3-Embedding-0.6B")
RERANKER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rerankers", "bge-reranker-v2-m3")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
BM25_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "bm25_index.pkl")
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "chunks.json")
QA_PAIRS_PATH = os.path.join(PROJECT_ROOT, "data", "qa_pairs.json")

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
