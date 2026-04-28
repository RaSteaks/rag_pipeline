# RAG Pipeline

本地 RAG（检索增强生成）知识库系统，与 OpenClaw memory_search 互补。

## 架构

```
GPU :8080  → Embedding 模型 (llama.cpp, 向量化)
CPU :8081  → Reranker 模型 (llama.cpp, 精排)
CPU :8900  → FastAPI (ChromaDB + BM25 + RRF + Reranker + Diversity Filter)
```

> **显存 ≤ 8GB 时**：Embedding 模型部署在 GPU，Reranker 模型必须部署在 CPU（`-ngl 0`），
> 否则会 OOM。显存充裕时可同时部署在 GPU。

### 检索流程

```
用户查询
    ↓
向量检索 Top-30 + BM25 检索 Top-30
    ↓
RRF 倒数秩融合 → 去重截断 Top-20
    ↓
Reranker 精排 (CPU, /v1/rerank)
    ↓ Reranker 不可用时降级为 RRF
文档多样性过滤 (max_chunks_per_doc)
    ↓
返回 Top-5/8
```

### 增量更新

- 文件 SHA256 hash 变更 → 删除旧 chunks → 重建该文件 chunks
- 文件删除 → 同步删除 ChromaDB 和 BM25 中对应条目
- watchdog 实时监听知识源目录变更

## 快速开始

### 前置条件

- Python 3.10+ 和 uv
- llama.cpp (已编译，包含 embedding 和 reranker 支持)
- Embedding 模型 (gguf 格式)
- Reranker 模型 (gguf 格式)

### 安装

```bash
cd /path/to/rag_pipeline
uv venv
uv pip install -e .
```

### 启动服务

**1. Embedding 服务** 

```powershell
llama-server.exe -m /path/to/embedding-model.gguf `
  --host 127.0.0.1 --port 8080 `
  --embedding --pooling last -ngl 32 -t 8
```

**2. Reranker 服务** (必须带 `--rerank` 参数)

```powershell
llama-server.exe -m /path/to/reranker-model.gguf `
  --host 127.0.0.1 --port 8081 `
  --pooling rank --rerank `
  -ngl 0 -t 8
```

**3. RAG API 服务**

```bash
python rag_service.py
```

首次启动会自动全量同步所有知识源文件。

### 使用 CLI

```bash
# 搜索
python cli.py search "查询内容"

# 指定返回数量和 debug 模式
python cli.py search "查询内容" --top-k 5 --debug

# 增量同步（检测文件变更）
python cli.py sync

# 全量重建
python cli.py sync --rebuild

# 按知识源重建
python cli.py sync --source <source_name>

# 查看统计
python cli.py stats
```

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/search` | 搜索知识库，参数：`query`, `top_k`, `debug` |
| POST | `/sync` | 同步索引，参数：`source_name`, `rebuild` |
| POST | `/reload-config` | 重载配置文件 |
| GET | `/stats` | 索引统计信息 |
| GET | `/health` | 健康检查 |

## 配置

编辑 `config.yaml`（参考 `config.example.yaml`）。

| 配置项 | 说明 | 改后需重建？ |
|--------|------|-------------|
| `knowledge_sources` | 知识源目录、权重、文件类型 | ✅ 是 |
| `chunking` | 分块大小、重叠、最小 chunk | ✅ 是 |
| `exclude` | 排除的目录和文件模式 | ✅ 是 |
| `retrieval` | Top-K 值、RRF 参数 | ❌ 重启即可 |
| `reranker` | 端点、超时、候选数 | ❌ 重启即可 |
| `embedding` | 端点、模型名、批大小 | ❌ 重启即可 |

### 知识源权重说明

- `weight: 1.15` → 高优先级知识源，检索时 RR 分数乘以权重
- `weight: 0.9` → 低优先级知识源（如日常笔记），降低噪音干扰

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.yaml` | 生产配置 |
| `config.example.yaml` | 配置模板 |
| `config.py` | Pydantic 配置校验，所有默认值为空，强制从 YAML 读取 |
| `parsers.py` | 文档解析器，支持 MD/PDF/DOCX/HTML/IPynb/TXT/PY/JSON/YAML |
| `chunker.py` | 语义分块器，按 markdown 标题递归切分，含截断保护 |
| `vector_store.py` | ChromaDB 存储 + Embedding 向量化，分批容错（单条失败时跳过） |
| `reranker.py` | Reranker 客户端，调用 llama.cpp `/v1/rerank` 端点，超时降级到 RRF |
| `retriever.py` | 混合检索：向量+BM25→RRF 融合→Reranker 精排→多样性过滤 |
| `image_describer.py` | PDF 页面渲染 + 视觉模型描述（InternVL2.5-4B），按需调用，非常驻 |
| `ingest.py` | 增量索引（SHA256+manifest+watchdog）+ 全量重建 + 按源重建 + 图片描述 |
| `rag_service.py` | FastAPI 主服务，启动时检测 Embedding/Reranker 可用性 |
| `cli.py` | CLI 工具（search/sync/stats），Windows UTF-8 兼容 |
| `start.bat` | Windows 启动脚本（检查 Embedding → 启动 RAG API） |
| `start-reranker.ps1` | Reranker 启动脚本（CPU, --rerank --pooling rank -ngl 0） |

## 降级机制

- **Embedding 不可用** → 向量检索失败，RAG API 启动时警告，搜索返回空结果
- **Reranker 不可用/超时** → 自动降级为 RRF 融合结果，日志输出 WARN
- **ChromaDB 损坏** → 需手动 `python cli.py sync --rebuild` 全量重建

## 技术栈

| 层级 | 组件 | 说明 |
|------|------|------|
| **推理引擎** | llama.cpp | Embedding 和 Reranker 模型服务 |
| **向量数据库** | ChromaDB | 嵌入式持久化存储，HNSW 索引，cosine 距离 |
| **关键词检索** | jieba + rank_bm25 | 中文分词 + BM25Okapi 算法 |
| **融合排序** | RRF (Reciprocal Rank Fusion) | k=60，向量与 BM25 倒数秩融合 |
| **精排模型** | Reranker (llama.cpp /v1/rerank) | Cross-Encoder 精排，CPU 推理 |
| **文档解析** | PyMuPDF / python-docx / BeautifulSoup | PDF / DOCX / HTML 多格式解析 |
| **分块策略** | Markdown 标题递归切分 | 按标题层级分割，超大段落按空行再分 |
| **增量更新** | SHA256 + manifest + watchdog | 文件哈希比对，watchdog 实时监听，防抖 2-10s |
| **API 框架** | FastAPI + uvicorn | REST API，异步处理 |
| **配置管理** | Pydantic + YAML | 类型校验，热重载，不含硬编码 |
| **CLI** | argparse | search / sync / stats 子命令 |
| **Python** | 3.10+ | uv 管理虚拟环境和依赖 |