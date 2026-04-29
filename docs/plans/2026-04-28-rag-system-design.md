# RAG 问答系统设计文档

## Context

本项目目标是在 `interview_RAG` 目录中从零搭建一个可本地运行的 RAG 问答系统，以 Python 官方文档（322 个扁平化 HTML 文件）作为语料库，支持混合检索（BM25 + 向量 + RRF）、Ragas 评估，并实现全部加分项。

## 关键资源

- **本地嵌入模型**：`models/embeddings/Qwen3-Embedding-0.6B/`
- **本地重排序模型**：`models/rerankers/bge-reranker-v2-m3/`
- **语料库**：`documents/` 目录下 322 个 Python 3.14.4 官方 HTML 文档
- **已安装依赖**：`chromadb`, `Flask`, `ragas`, `rank-bm25`, `sentence-transformers`, `torch`, `transformers`
- **LLM API**：通义千问（qwen），兼容 OpenAI 格式

## 设计决策

### Chunk 策略
- **方法**：递归字符文本分割（RecursiveCharacterTextSplitter）
- **chunk_size**：512 字符
- **chunk_overlap**：100 字符（约 20%）
- **选择依据**：
  - Python 文档为 Sphinx 生成的结构化 HTML，段落级切分可保留语义完整性
  - 512 字符约为 150–200 个中文词或 250 个英文词，能覆盖一个完整的小节（如一个函数说明）
  - 100 字符 overlap 避免关键信息被截断在边界，同时控制冗余

### 融合方式
- **方式**：RRF（Reciprocal Rank Fusion），参数 `k=60`
- **选择依据**：
  - RRF 无需调参权重，对两路检索的得分分布差异不敏感，更稳健
  - 相比加权融合，RRF 不需要人工校准 BM25 和向量检索的得分尺度

### 模型与库选型
- **Embedding**：本地 `Qwen3-Embedding-0.6B`
- **Reranker**：本地 `bge-reranker-v2-m3`
- **向量库**：ChromaDB
- **BM25**：`rank_bm25`
- **LLM**：通义千问（qwen）兼容 OpenAI 格式
- **Web UI**：Flask

## 架构设计

### 文件结构
```
interview_RAG/
├── run.py                      # 统一入口：--build-index / --web / --eval / --query
├── requirements.txt
├── README.md
├── src/
│   ├── config.py               # 全局配置（chunk参数、模型路径、API配置、功能开关）
│   ├── document_parser.py      # 解析 documents/*.html，提取正文和标题
│   ├── chunker.py              # 递归字符分块，记录来源元数据
│   ├── embedding.py            # 本地 Qwen3-Embedding 封装
│   ├── vector_store.py         # ChromaDB 向量存储
│   ├── bm25_index.py           # BM25 索引构建与查询
│   ├── hybrid_retriever.py     # 混合检索：向量 + BM25 + RRF 融合
│   ├── reranker.py             # 本地 bge-reranker-v2-m3 精排
│   ├── query_rewriter.py       # Query 改写/扩展 + 多模块拆分
│   ├── llm_client.py           # 通义千问 API 客户端（支持流式）
│   ├── generator.py            # Prompt 组装 + 答案生成 + 来源标注
│   ├── rag_engine.py           # 编排完整 RAG 流水线
│   └── evaluator.py            # Ragas 评估封装
├── app/
│   ├── app.py                  # Flask Web 服务
│   └── templates/index.html    # 单页 UI
├── scripts/
│   ├── build_index.py          # 一键构建向量和 BM25 索引
│   └── evaluate.py             # 运行 Ragas 评估并输出报告
├── data/
│   ├── selected_docs.json      # 选中的 20–25 个文档列表
│   ├── chunks.json             # 分块结果
│   ├── qa_pairs.json           # 手工构建的 8 条 QA 对
│   ├── chroma_db/              # ChromaDB 持久化数据
│   └── bm25_index.pkl          # 序列化 BM25 索引
└── results/
    ├── ragas_report_baseline.json   # 基线评估报告
    ├── ragas_report_optimized.json  # 优化后评估报告
    └── comparison_chart.png         # Ragas 指标对比图
```

### 核心数据流
```
用户 Query
    ↓
LLM 判断多模块拆分（可选）
    ↓
Query 改写（2-3个等价表述）
    ↓
  ├─→ 向量检索（ChromaDB, top 50） ──┐
  └─→ BM25 检索（top 50） ────────────┤
                                      ↓
                                  RRF 融合（k=60, top 20）
                                      ↓
                              Rerank 精排（可选开关, top 5）
                                      ↓
                              Top-K chunks + 来源元数据
                                      ↓
                              LLM 生成答案（流式）
                                      ↓
                            答案 + [来源: xxx.html]
```

## 模块详细设计

### 1. 文档解析与分块
- **document_parser.py**：
  - 用 `BeautifulSoup` 过滤 `mobile-nav`、`sphinxsidebar`、`<script>`、`<style>`，只保留 `<div role="main">` 内的正文和各级标题
  - 从 `<title>` 提取文档标题，从 `<h1>` / `<h2>` 提取章节标题
  - 从 `config.SELECTED_DOCS` 中读取 20-25 个文档进行解析
- **chunker.py**：
  - `RecursiveCharacterTextSplitter`，分隔符优先级：`\n\n` → `\n` → ` ` → `''`
  - `chunk_size=512`, `chunk_overlap=100`
  - 每个 chunk 的元数据：`{source: 'argparse.html', title: 'argparse — Parser...', section: 'Basic Example'}`

### 2. 索引构建
- **embedding.py**：`SentenceTransformer` 加载本地 `Qwen3-Embedding-0.6B`，`batch_size=32`
- **vector_store.py**：ChromaDB `Client` + collection `python_docs`，持久化到 `./data/chroma_db`
- **bm25_index.py**：`rank_bm25.BM25Okapi`，分词用 `re.findall(r'\w+', text.lower())`，序列化到 `./data/bm25_index.pkl`
- **scripts/build_index.py**：一键完成解析 → 分块 → 向量索引 → BM25 索引，全程 `tqdm` 进度条可视化

### 3. 混合检索
- **hybrid_retriever.py**：
  - 向量检索：ChromaDB `query` 接口，取 top 50
  - BM25 检索：`bm25.get_scores(tokenized_query)`，取 top 50
  - RRF 融合：`score = 1/(60+rank_vec) + 1/(60+rank_bm25)`，去重后取 top 20
- **reranker.py**：本地 `bge-reranker-v2-m3` Cross-Encoder 精排，输入 query + chunk 对，输出 relevance score，取 top 5
- **开关控制**：`config.USE_RERANK = True/False`，支持对比实验

### 4. Query 改写
- **query_rewriter.py**：
  - **多模块拆分**：LLM 判断 Query 是否涉及多个独立主题，若是则拆分为子问题（如 "argparse and asyncio" → ["argparse usage", "asyncio usage"]）
  - **Query 改写**：每个子问题用 LLM 扩展为 2-3 个等价表述
  - 多路并行检索后合并去重
- **开关控制**：`config.USE_QUERY_REWRITE = True`, `config.USE_MULTI_QUERY_SPLIT = True`

### 5. LLM 生成与流式输出
- **llm_client.py**：
  - `openai` SDK，配置从环境变量读取：`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`
  - 支持 `stream=True/False`
- **generator.py**：
  - System Prompt + Top-K chunks（含来源） + User Query
  - 要求模型在答案末尾列出 `[来源: 文件名]`
  - 非流式返回字符串，流式返回 `Iterator[str]`

### 6. Web UI
- **app/app.py**：
  - `GET /`：渲染首页
  - `POST /api/query`：非流式 JSON 返回
  - `GET /api/stream?q=xxx`：SSE 流式返回
  - 构建索引进度通过 SSE 推送百分比
- **app/templates/index.html**：单页应用，输入框 + 流式答案显示 + 引用来源列表

### 7. Ragas 评估与迭代
- **evaluator.py**：封装 `ragas` 评估，`faithfulness` / `answer_relevancy` / `context_recall`
- **data/qa_pairs.json**：8 条手工 QA 对，覆盖不同模块
- **scripts/evaluate.py**：
  - CLI `tqdm` 实时显示每个 QA 对的评估进度
  - 支持 `--config baseline`（chunk=512, rerank=off）和 `--config optimized`（chunk=256, rerank=on）两组实验
  - 输出 `rich` 表格展示每个 QA 对的指标 + 平均值
  - 用 `matplotlib` 生成柱状图对比两组指标，保存到 `results/comparison_chart.png`

### 8. 配置管理
```python
# config.py 关键参数
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
SELECTED_DOCS = [...]  # 20-25 个文档列表
```

### 9. 运行入口
```bash
python run.py --build-index    # 解析 → 分块 → 向量索引 → BM25 索引（带 tqdm 进度）
python run.py --query "xxx"    # CLI 问答（流式输出）
python run.py --web            # 启动 Flask 服务
python run.py --eval           # 运行 Ragas 基线评估
python run.py --eval --config optimized  # 运行优化后评估
```

## 验证方案

1. **索引构建验证**：运行 `python run.py --build-index`，确认 `data/chunks.json` 非空，ChromaDB collection 中文档数 > 0，`bm25_index.pkl` 可加载
2. **检索验证**：运行 `python run.py --query "What is asyncio?"`，确认返回结果含来源文件名和段落
3. **生成验证**：运行 `python run.py --query "What is asyncio?"`，确认答案末尾有 `[来源: xxx.html]`
4. **Rerank 开关验证**：分别设置 `USE_RERANK=True/False`，对比检索到的 top 5 结果差异
5. **Ragas 验证**：运行 `python run.py --eval`，确认输出 faithfulness、answer_relevancy、context_recall 三个指标
6. **评估迭代验证**：运行 baseline 和 optimized 两组实验，确认 `results/comparison_chart.png` 生成且指标有变化
7. **Web UI 验证**：运行 `python run.py --web`，在浏览器中访问 `http://localhost:5000`，输入问题，验证流式答案和来源标注正常显示

## 风险与注意事项

- **环境变量**：LLM API Key 必须通过环境变量传入，代码中不得硬编码
- **本地模型路径**：`embedding.py` 和 `reranker.py` 中模型路径使用相对路径 `./models/...`，确保跨平台兼容
- **文档解析**：Python HTML 文档为 Sphinx 生成，需精确过滤 `mobile-nav`、`sphinxsidebar` 等非内容元素
- **文档选择**：从 322 个文件中筛选 20–25 个核心模块，确保覆盖常用标准库
- **Ragas 版本**：当前安装 `ragas 0.4.3`，API 可能与最新文档有差异，需根据实际版本调整调用方式
- **多模块拆分**：LLM 判断可能产生误判，需在 Prompt 中明确 "只有涉及完全不同主题时才拆分"
