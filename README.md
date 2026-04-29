# Python 文档 RAG 问答系统

基于 Python 官方文档的可本地运行 RAG（Retrieval-Augmented Generation）问答系统，支持混合检索、Query 改写、重排序、流式 Web UI、Ragas 评估及消融实验。

## 核心特性

- **混合检索**：向量语义检索 + BM25 关键词检索双路召回，通过 RRF 融合
- **Query 改写**：LLM 驱动的多主题拆分与查询扩展，提升检索覆盖率
- **重排序**：基于 Cross-Encoder（`bge-reranker-v2-m3`）的精排优化
- **流式 Web UI**：基于 Flask + Server-Sent Events 的实时答案流式输出
- **Ragas 评估**：集成 `faithfulness`、`answer_relevancy`、`context_recall` 指标
- **消融实验**：支持一键对比 Query Rewrite、Rerank 及联合策略的效果

## 技术栈

- **语言**：Python 3
- **Web 框架**：Flask
- **向量数据库**：ChromaDB
- **Embedding**：`sentence-transformers`（Qwen3-Embedding-0.6B）
- **稀疏检索**：`rank-bm25`
- **重排序**：`sentence-transformers` CrossEncoder（bge-reranker-v2-m3）
- **LLM 客户端**：`openai` SDK（兼容 DashScope / 通义千问）
- **评估**：Ragas

---

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
# 编辑 .env，将 your_qwen_api_key_here 替换为实际的 API Key
```

### 2. 构建索引

```bash
python run.py --build-index
```

该命令会依次执行：
1. 解析 `documents/` 目录下的 HTML 文档
2. 将文档切分为 chunks
3. 使用 Embedding 模型编码并写入 ChromaDB
4. 构建 BM25 索引并序列化到本地

### 3. CLI 问答

```bash
python run.py --query "How does argparse handle subcommands?"
```

如需禁用 Query 改写以对比效果：

```bash
python run.py --query "你的问题" --no-query-rewrite
```

### 4. 启动 Web 服务

```bash
python run.py --web
# 或指定端口
python run.py --web --port 8080
```

浏览器访问 http://localhost:5000，支持流式输出与 Query 改写开关。

### 5. Ragas 评估

```bash
# 基线评估（关闭 Query Rewrite 与 Rerank）
python run.py --eval --config baseline

# 优化后评估（开启全部优化策略）
python run.py --eval --config optimized

# 生成对比图
python scripts/generate_comparison_chart.py
```

### 6. 消融实验

```bash
python run.py --ablation
```

将自动运行三组实验：仅 Query Rewrite、仅 Rerank、联合策略，并输出对比结果。

---

## 系统架构

```
User Query
    |
    v
+------------------+
| Query Rewriter   |  LLM 多主题拆分 + 查询扩展
+------------------+
    |
    v
+------------------+     +------------------+
| Vector Retrieval |     | BM25 Retrieval   |
| (ChromaDB)       |     | (Okapi BM25)     |
| Top-50           |     | Top-50           |
+------------------+     +------------------+
    |                           |
    +-----------+---------------+
                |
                v
    +----------------------+
    | RRF Fusion (k=60)    |
    | Top-20               |
    +----------------------+
                |
                v
    +----------------------+
    | Reranker (optional)  |
    | Cross-Encoder        |
    | Top-5                |
    +----------------------+
                |
                v
    +----------------------+
    | Prompt Builder       |
    | Generator (LLM)      |
    +----------------------+
                |
                v
        Answer + Sources
```

---

## 文档解析与分块

### 支持格式

系统支持解析 **Sphinx 生成的 HTML 文档**（如 Python 官方文档）。解析器（`src/document_parser.py`）会提取 `<title>` 作为标题，并定位 `<div role="main">` 或 `<div class="body">` 提取正文，同时过滤掉 `script`、`style`、`nav`、`header`、`footer` 等无关标签。

### Chunk 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `CHUNK_SIZE` | 512（字符） | 每个 chunk 的最大字符数 |
| `CHUNK_OVERLAP` | 100（字符） | 相邻 chunk 之间的重叠字符数，约 20% |

### 选择依据

- **格式特点**：Python 官方文档为 Sphinx 生成的结构化 HTML，段落级切分能够保留语义完整性。
- **Size 选择**：512 字符约为 250 个英文词，通常能覆盖一个完整的小节或一个独立的概念说明，既不会因过小而破坏上下文，也不会因过大而稀释检索精度。
- **Overlap 选择**：100 字符的 overlap 约等于一个短句的长度，可有效避免关键信息被截断在 chunk 边界，同时保证冗余度可控。

切分策略采用**递归字符切分**（`src/chunker.py`），优先按 `\n\n`（段落）、`\n`（换行）、` `（空格）寻找断点，尽量在语义边界处分割。

---

## 向量化与存储

### Embedding 模型

使用本地部署的 **`Qwen3-Embedding-0.6B`** 模型，通过 `sentence-transformers` 加载。该模型专为文本嵌入优化，在中文和英文场景下均有良好表现。

```python
from src.embedding import EmbeddingModel
model = EmbeddingModel()
embeddings = model.encode(texts)
```

### 向量库

向量数据持久化存储在 **ChromaDB**（`data/chroma_db`）中，采用 `PersistentClient` 模式，支持本地离线查询。集合（Collection）名称为 `python_docs`，存储内容包含：

- `documents`：原始 chunk 文本
- `embeddings`：对应的向量表示
- `metadatas`：来源文档名、标题等元信息

### 稀疏索引

除向量库外，系统同时使用 **BM25Okapi**（`rank-bm25`）构建稀疏索引，用于关键词匹配。索引构建后会通过 `pickle` 序列化到 `data/bm25_index.pkl`，启动时直接加载，无需重复构建。

---

## 混合检索（BM25 + 向量）

### 双路召回

系统同时运行两路检索，互为补充：

1. **向量语义检索**：将查询文本编码为向量，在 ChromaDB 中执行近似最近邻搜索，取 **Top-50**。擅长捕捉语义相关性，对同义词、表述变体鲁棒。
2. **BM25 关键词检索**：基于 Okapi BM25 对查询中的关键词进行精确匹配，取 **Top-50**。擅长命中包含精确术语的文档片段。

### 融合方式：RRF

两路结果通过 **Reciprocal Rank Fusion（RRF）** 合并，参数 `k=60`，最终取 **Top-20**。

RRF 公式：
```
score(d) = Σ 1 / (k + rank_i(d))
```

其中 `rank_i(d)` 为文档 `d` 在第 `i` 路结果中的排名。

**选择理由**：RRF 无需人工调参融合权重，对 BM25 和向量检索的得分分布差异不敏感，比简单的加权求和更稳健，且实现简洁。

### 可选增强：重排序

在 RRF 融合后，可启用 **Cross-Encoder 重排序**（`bge-reranker-v2-m3`）对 Top-20 结果进行精排，取最终 **Top-5** 送入生成阶段。重排序能显著提升结果的相关性，但会带来额外的推理开销。

该功能可通过 `src/config.py` 中的 `USE_RERANK` 开关控制，也支持在 `RAGEngine.query()` 调用时动态传入。

---

## 检索与生成

### 完整链路

```
User Query
    → Query Rewriter（LLM 多主题拆分 / 查询扩展）
    → 对每个子查询执行混合检索（Vector + BM25 → RRF）
    → 合并去重后的结果
    → 可选 Rerank
    → Prompt 组装（包含检索到的 Chunk 与引用来源）
    → LLM 生成答案（通义千问 qwen-plus，支持流式 / 非流式）
    → 输出答案 + 引用来源列表
```

### Query 改写

`src/query_rewriter.py` 实现了两层改写策略：

1. **多主题拆分**：LLM 判断查询是否涉及多个不同主题，若是则拆分为多个独立子问题。
2. **查询扩展**：对每个子问题生成 2-3 个语义等价但表述不同的查询变体。

通过多个查询变体分别检索，可以显著提高召回率，覆盖文档中不同表述的相关内容。

**运行时开关**：Query 改写可通过 `--no-query-rewrite` CLI 参数临时关闭，或在 `src/config.py` 中修改 `USE_QUERY_REWRITE` / `USE_MULTI_QUERY_SPLIT` 进行全局配置。

### 引用标注

答案输出时会附带 `sources` 列表，每个来源包含：

- `index`：引用序号
- `source`：来源文档文件名
- `text`：被引用的文档片段前 300 字符

在 Web UI 中，来源以卡片形式展示；CLI 模式下直接打印来源列表。

### 生成 Prompt 示例

```
[文档片段 1]
来源: argparse.html
...（chunk 内容）...

---

[文档片段 2]
来源: argparse.html
...（chunk 内容）...

问题: How does argparse handle subcommands?

请基于以上文档片段，用清晰的分点/分段格式回答问题。
```

---

## 评估与实验

### Ragas 评估

系统内置 Ragas 评估流程，覆盖以下指标：

- **Faithfulness**：答案是否忠实于检索到的上下文
- **Answer Relevancy**：答案与问题的相关程度
- **Context Recall**：检索到的上下文是否包含回答问题所需的全部信息

评估数据集位于 `data/qa_pairs.json`，结果输出到 `results/ragas_report_baseline.json` 和 `results/ragas_report_optimized.json`。

### 消融实验

运行 `python run.py --ablation` 会自动执行以下三组实验并输出对比：

| 实验组 | Query Rewrite | Rerank |
|--------|--------------|--------|
| 仅 Query Rewrite | 开 | 关 |
| 仅 Rerank | 关 | 开 |
| 联合策略 | 开 | 开 |

通过对比可量化每项优化策略对最终答案质量的独立贡献与协同效应。

---

## 项目结构

```
interview_RAG/
├── app/                      # Flask Web 应用
│   ├── app.py                # Web 服务入口（SSE 流式 + JSON API）
│   └── templates/
├── data/                     # 持久化数据
│   ├── chroma_db/            # ChromaDB 向量库
│   ├── bm25_index.pkl        # BM25 索引
│   ├── chunks.json           # 文档 chunks
│   └── qa_pairs.json         # 评估问答对
├── documents/                # 原始语料（Python 官方 HTML 文档）
├── models/                   # 本地模型
│   ├── embeddings/Qwen3-Embedding-0.6B
│   └── rerankers/bge-reranker-v2-m3
├── scripts/                  # 工具脚本
│   ├── build_index.py        # 构建索引
│   ├── evaluate.py           # Ragas 评估
│   ├── generate_comparison_chart.py
│   └── run_ablation_tests.py # 消融实验
├── src/                      # 核心源码
│   ├── config.py             # 全局配置
│   ├── document_parser.py    # HTML 文档解析
│   ├── chunker.py            # 文本切分
│   ├── embedding.py          # Embedding 编码
│   ├── vector_store.py       # ChromaDB 封装
│   ├── bm25_index.py         # BM25 索引
│   ├── hybrid_retriever.py   # 混合检索 + RRF
│   ├── reranker.py           # Cross-Encoder 重排
│   ├── query_rewriter.py     # Query 改写
│   ├── generator.py          # Prompt 组装与答案生成
│   ├── llm_client.py         # LLM 客户端封装
│   ├── rag_engine.py         # RAG 全流程编排
│   └── evaluator.py          # Ragas 评估适配器
├── tests/                    # 单元测试
├── docs/plans/               # 设计文档
├── run.py                    # 统一 CLI 入口
├── requirements.txt
└── .env.example
```

---

## 配置说明

### 环境变量（`.env`）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | 通义千问 API Key | 必填 |
| `LLM_BASE_URL` | OpenAI 兼容接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | 模型名称 | `qwen-plus` |

### 可调参数（`src/config.py`）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CHUNK_SIZE` | Chunk 大小（字符） | 512 |
| `CHUNK_OVERLAP` | Chunk 重叠（字符） | 100 |
| `TOP_K_VECTOR` | 向量检索返回数量 | 50 |
| `TOP_K_BM25` | BM25 检索返回数量 | 50 |
| `TOP_K_RRF` | RRF 融合后数量 | 20 |
| `TOP_K_FINAL` | 最终送入生成的数量 | 5 |
| `RRF_K` | RRF 参数 | 60 |
| `USE_RERANK` | 是否启用重排序 | `True` |
| `USE_QUERY_REWRITE` | 是否启用 Query 改写 | `True` |
| `USE_MULTI_QUERY_SPLIT` | 是否启用多主题拆分 | `True` |

---

## 详细设计文档

完整的系统设计、模块接口与数据流说明见 `docs/plans/2026-04-28-rag-system-design.md`。
