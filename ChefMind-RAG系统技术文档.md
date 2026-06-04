# ChefMind RAG 系统技术文档

> **项目名称**：ChefMind（私厨大脑）
> **项目定位**：基于 LangChain 的智能菜谱 RAG 助手，解析中文菜谱 Markdown 文件，索引至向量数据库，通过自然语言问答提供带引用的烹饪指导。
> **架构**：FastAPI 后端 + Celery 异步任务 + Next.js 前端

---

## 目录

- [1. 技术栈总览](#1-技术栈总览)
- [2. RAG 流水线架构](#2-rag-流水线架构)
- [3. 各阶段详解](#3-各阶段详解)
  - [阶段 0：文档解析](#阶段-0文档解析离线)
  - [阶段 1：领域分块](#阶段-1领域分块chunking)
  - [阶段 2：向量嵌入](#阶段-2向量嵌入embedding)
  - [阶段 3：向量存储](#阶段-3向量存储)
  - [阶段 4：查询分类与路由](#阶段-4查询分类与路由)
  - [阶段 5：查询改写](#阶段-5查询改写)
  - [阶段 6：混合检索](#阶段-6混合检索)
  - [阶段 7：重排序](#阶段-7重排序reranking)
  - [阶段 8：答案生成](#阶段-8答案生成)
- [4. 支撑能力](#4-支撑能力)
- [5. 数据模型](#5-数据模型)
- [6. API 接口](#6-api-接口)
- [7. 配置与部署](#7-配置与部署)
- [8. 流程总览图](#8-流程总览图)

---

## 1. 技术栈总览

| 组件 | 技术选型 |
|---|---|
| LLM（主） | SiliconFlow（Qwen/Qwen3-8B）、DeepSeek-V3、Qwen-Max、OpenAI GPT-4o-mini |
| 嵌入模型 | BAAI/bge-large-zh-v1.5（1024 维，sentence-transformers） |
| 重排序模型 | BAAI/bge-reranker-large（CrossEncoder） |
| 向量数据库 | ChromaDB（本地）/ Milvus 2.5（生产） |
| 关系型数据库 | SQLite（本地）/ PostgreSQL 16（生产） |
| RAG 框架 | LangChain 0.3.13（langchain-core、langchain-openai、langchain-community） |
| 后端框架 | FastAPI 0.115.6 + uvicorn |
| 异步任务 | Celery 5.4.0 + Redis |
| 前端 | Next.js 14 + Tailwind CSS |
| 文档处理 | PyMuPDF、Unstructured |
| 日志监控 | structlog、prometheus-client |

---

## 2. RAG 流水线架构

流水线由 `backend/app/services/rag_pipeline.py` 中的 `RAGPipeline.run()` 方法编排，共 **8 个阶段**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        离线索引流程                                  │
│  Markdown 文件 → 文档解析 → 领域分块 → 向量嵌入 → 向量存储          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        在线查询流程                                  │
│  用户提问 → 查询分类 → 查询改写 → 混合检索 → 重排序 → 答案生成      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 各阶段详解

### 阶段 0：文档解析（离线）

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/recipe_parser.py` |
| **类** | `RecipeParser` |
| **技术** | Python 正则表达式、Markdown 解析 |
| **输入** | `data/cook/dishes/` 下 323 个中文菜谱 `.md` 文件 |
| **输出** | 结构化 `ParsedRecipe` 对象 |

**实现效果**：

- 解析 Markdown 菜谱文件，提取以下字段：
  - 标题（title）
  - 描述（description）
  - 食材清单（ingredients）—— 含用量
  - 烹饪步骤（steps）
  - 小贴士（tips）
  - 图片（images）
  - 难度（difficulty）—— 星级评分
  - 烹饪时间（cooking_time）
  - 标签（tags）
- **自动标签生成**：根据食材关键词（海鲜、禽类等）和烹饪方式（蒸、炒等）自动打标签
- 支持 `parse_directory()` 批量解析整个目录树

**菜谱分类**（11 类）：水产、早餐、调味品、甜品、饮品、肉菜、半成品、汤品、主食、模板、素菜

---

### 阶段 1：领域分块（Chunking）

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/recipe_chunker.py` |
| **类** | `RecipeChunker` |
| **技术** | 自定义分块策略、正则匹配、Token 计数截断 |
| **输入** | `ParsedRecipe` 对象 |
| **输出** | `RecipeChunk` 对象列表 |

**实现效果**：

- 按 **4 种类型** 分块：
  | 分块类型 | 说明 |
  |---|---|
  | `ingredient` | 食材信息 |
  | `step` | 烹饪步骤 |
  | `tip` | 小贴士 |
  | `image_desc` | 图片描述 |

- **自包含设计**：每个步骤块自动附加完整食材清单，确保独立检索时信息完整
- **关联食材提取**：通过正则匹配提取步骤中涉及的具体食材
- **短步骤合并**：连续短步骤合并，避免碎片化
- **自动截断**：超长内容截断至 512 tokens，标记 `truncated` 字段
- 每个块前缀包含菜谱名称和元数据头（难度、烹饪时间）

---

### 阶段 2：向量嵌入（Embedding）

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/embedding_service.py` |
| **类** | `EmbeddingService` |
| **技术** | `sentence-transformers`、`BAAI/bge-large-zh-v1.5` |
| **输入** | 文本字符串 |
| **输出** | 1024 维归一化向量 |

**实现效果**：

- 使用 BGE-large-zh-v1.5 模型生成中文文本向量
- **查询与文档差异化处理**：
  - 查询向量添加指令前缀：`"为这段文本生成向量表示："`
  - 文档向量 **不加** 前缀（遵循 BGE 模型规范）
- 批量处理：`batch_size=32`
- 向量归一化：启用 `normalize_embeddings=True`
- **降级策略**：模型不可用时回退到 `MockEmbeddingService`（确定性随机向量）

---

### 阶段 3：向量存储

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/core/milvus.py` |
| **技术** | ChromaDB（本地）/ Milvus 2.5（生产） |
| **输入** | 向量 + 元数据 + 文档内容 |
| **输出** | 持久化向量索引 |

**实现效果**：

- ChromaDB `PersistentClient` 持久化存储，数据目录：`./chroma_data/`
- Collection 名称：`recipes`
- 存储内容：ids、embeddings、documents（文本内容）、metadata
- **批量插入**：每批 500 条
- **相似度检索**：余弦相似度 ANN 搜索
- 函数命名兼容 Milvus 接口（`connect_milvus()`、`create_recipe_collection()`），便于生产环境切换

---

### 阶段 4：查询分类与路由

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/query_classifier.py` |
| **类** | `QueryClassifier` |
| **技术** | 正则规则匹配（主）、LLM 零样本分类（辅） |
| **输入** | 用户查询文本 |
| **输出** | 查询类型 + `RouteConfig`（权重 + 过滤器） |

**实现效果**：

- **7 种查询类型**：

  | 查询类型 | 说明 | 语义权重 | BM25 权重 | 过滤分块类型 |
  |---|---|---|---|---|
  | `ingredient_search` | 食材搜索 | 0.4 | 0.6 | ingredient |
  | `step_lookup` | 步骤查找 | 0.5 | 0.5 | step, ingredient |
  | `substitution` | 替代建议 | 0.6 | 0.4 | tip, ingredient |
  | `tip` | 小贴士 | 0.5 | 0.5 | tip |
  | `image_request` | 图片请求 | 0.7 | 0.3 | image_desc |
  | `generate_recipe` | 生成菜谱 | 0.6 | 0.4 | 全部 |
  | `general` | 通用查询 | 0.5 | 0.5 | 全部 |

- 主分类使用 **正则表达式**（零 LLM 调用开销）
- 可选 LLM 零样本分类作为增强
- 每种类型的 `RouteConfig` 决定后续检索的权重分配和分块过滤

---

### 阶段 5：查询改写

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/query_rewriter.py` |
| **类** | `QueryRewriter` |
| **技术** | LLM（Qwen3-8B）、对话历史、正则回退 |
| **输入** | 原始查询 + 对话历史 |
| **输出** | 改写后的查询列表 |

**实现效果**：

- **三种改写策略**：

  | 策略 | 说明 | 示例 |
  |---|---|---|
  | 指代消解 | 解析对话中的代词 | "它怎么做" → "清蒸鲈鱼怎么做" |
  | 意图拆解 | 拆分复合查询 | "红烧肉和糖醋排骨怎么做" → 两个子查询 |
  | 多视角生成 | 生成同义/改写变体 | 提升召回率的多种表达方式 |

- 每种策略均有 **LLM 实现 + 正则规则回退**
- 支持 **HyDE**（Hypothetical Document Embeddings）：生成假设性答案文档用于检索
- 对话历史通过 `ConversationMemory` 注入

---

### 阶段 6：混合检索

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/hybrid_retriever.py` |
| **类** | `HybridRetriever` |
| **技术** | ChromaDB 向量检索 + `jieba` 分词 + `BM25Okapi` + RRF 融合 |
| **输入** | 改写后的查询 + RouteConfig |
| **输出** | 候选文档列表（top-30） |

**实现效果**：

- **双路检索**：
  - **语义检索**：查询向量化 → ChromaDB 余弦相似度搜索 → top-30
  - **BM25 关键词检索**：jieba 中文分词 → 构建 BM25Okapi 索引 → top-30

- **RRF 融合**（Reciprocal Rank Fusion）：
  ```
  score = weight / (k + rank)
  ```
  - `k = 60`（RRF 常数）
  - 权重由阶段 4 的 `RouteConfig` 动态决定

- 降级策略：无向量数据库时回退到 `MockHybridRetriever`

---

### 阶段 7：重排序（Reranking）

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/reranker_service.py` |
| **类** | `RerankerService` |
| **技术** | `sentence-transformers.CrossEncoder`、`BAAI/bge-reranker-large` |
| **输入** | 查询 + 候选文档列表 |
| **输出** | 精排后的 top-5 文档 |

**实现效果**：

- 使用 CrossEncoder 对查询-文档对进行精细打分
- 文本截断至 1024 字符后评分
- 按分数降序排列，返回 **top-5**
- **相关性阈值**：默认 `0.15`
  - 最高分 < 阈值 → 判定"无相关内容" → 触发 **拒答机制**
- 降级策略：模型不可用时回退到 `MockRerankerService`

---

### 阶段 8：答案生成

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/answer_generator.py` |
| **类** | `AnswerGenerator` |
| **技术** | LangChain `ChatPromptTemplate` + LLM 链、SSE 流式输出 |
| **输入** | 查询 + 重排后的文档 + 对话历史 |
| **输出** | 答案文本 + 引用列表 |

**实现效果**：

- **System Prompt 规则**：
  - 必须使用 `[N]` 格式引用来源
  - 禁止编造信息
  - 保留食材精确用量
  - 信息不足时必须拒答

- **上下文构建**：将检索到的块格式化为带编号的引用，包含菜谱标题、来源文件、分块类型、步骤编号

- **幻觉检测**（后处理）：
  - 检查答案中所有 `[N]` 引用编号是否存在于引用映射中
  - 移除虚假引用编号
  - 添加幻觉告警

- **引用构建**：提取所有有效 `[N]` 引用，映射为结构化 `Citation` 对象

- **流式输出**：支持 SSE 逐 token 推送，最终事件携带引用和调试信息

- 降级策略：无 LLM 时回退到 `_generate_simple_answer()`（规则化格式输出）

---

## 4. 支撑能力

### 4.1 LLM 工厂

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/llm_factory.py` |
| **技术** | `langchain_openai.ChatOpenAI` |

- 支持 4 个 LLM 提供商，统一使用 OpenAI 兼容接口
- **自动降级链**：SiliconFlow → DeepSeek → Qwen → OpenAI
- 每个提供商配置独立的 `base_url` 和 API Key

### 4.2 对话记忆

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/conversation_memory.py` |
| **技术** | `deque(maxlen=10)`、正则实体抽取 |

- 每会话存储最近 **5 轮对话**（10 条消息）
- **实体追踪**：正则提取用户消息中的菜谱名称
- 提供 LangChain 兼容的 `ConversationBufferWindowMemory` 接口
- 历史注入查询改写和答案生成阶段

### 4.3 缓存服务

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/cache_service.py` |
| **技术** | 内存 TTL 缓存 |

- 默认 TTL：**1 小时**
- 缓存键：MD5(查询 + 上下文)
- 缓存内容：查询改写结果、检索结果

### 4.4 容错机制

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/fault_tolerance.py` |
| **技术** | 熔断器、指数退避重试 |

- **熔断器**：3 次失败 → 断开，60 秒后恢复
- **重试**：指数退避，最多 3 次，超时 2 秒
- **降级策略**：每个流水线阶段均有静态降级方案

### 4.5 多模态服务

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/multimodal_service.py` |
| **技术** | DeepSeek-OCR（视觉 LLM）、base64 编码 |

- 处理流程：图片上传 → 保存（本地/MinIO）→ base64 编码 → 视觉 LLM 生成描述 → 索引为 `image_desc` 分块
- 视觉模型：DeepSeek-OCR（通过 SiliconFlow 调用）

### 4.6 日志与监控

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/logging_service.py` |
| **技术** | `structlog`、`prometheus-client` |

- 结构化日志：记录完整查询链路
- **MetricsCollector** 指标：
  - 查询计数
  - 拒答率
  - 引用覆盖率
  - 延迟分位数

### 4.7 评估服务

| 项目 | 内容 |
|---|---|
| **文件** | `backend/app/services/evaluation_service.py` |
| **技术** | Recall@5、引用准确率、忠实度、相关性 |

- **评估指标**：
  - Recall@5：前 5 个结果的召回率
  - Citation Accuracy：引用准确率
  - Faithfulness：忠实度
  - Answer Relevance：答案相关性
  - Refuse Accuracy：拒答准确率
- 内置 **6 个测试用例**，覆盖不同查询类型

---

## 5. 数据模型

**文件**：`backend/app/models/recipe.py`

| 模型 | 说明 |
|---|---|
| `Recipe` | 菜谱元数据（标题、食材 JSON、步骤 JSON、小贴士、难度、标签、图片） |
| `RecipeChunk` | 分块跟踪（分块类型、内容、步骤号、元数据） |
| `Document` | 上传文档跟踪（状态：pending → parsing → chunking → indexing → done / error） |
| `Conversation` | 聊天历史（会话 ID、角色、内容、引用） |
| `UserFeedback` | 用户评分（1-5 分）+ 评论 |

---

## 6. API 接口

### 6.1 聊天接口

**文件**：`backend/app/api/endpoints/chat.py`

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | 标准 RAG 查询，返回答案 + 引用 + 调试信息 |
| `POST` | `/api/chat/stream` | SSE 流式响应，逐 token 推送 + 最终元数据事件 |
| `POST` | `/api/new_session` | 清除会话记忆 |
| `GET` | `/api/session/{session_id}` | 获取会话摘要 |

### 6.2 菜谱接口

**文件**：`backend/app/api/endpoints/recipes.py`

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/recipes` | 菜谱列表（搜索/过滤/分页） |
| `GET` | `/api/recipes/{recipe_id}` | 菜谱详情 |
| `POST` | `/api/recipes/upload` | 图片上传 + 多模态处理 |

### 6.3 评估接口

**文件**：`backend/app/api/endpoints/evaluation.py`

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/evaluation/summary` | 评估汇总 |
| `GET` | `/api/evaluation/test-cases` | 测试用例列表 |
| `POST` | `/api/evaluation/run` | 运行评估 |

---

## 7. 配置与部署

### 7.1 两种部署模式

| 模式 | 配置文件 | 向量数据库 | 关系型数据库 | 存储 |
|---|---|---|---|---|
| **Lite（本地开发）** | `.env.lite` | ChromaDB | SQLite | 本地文件 |
| **Full（生产）** | `.env.example` | Milvus 2.5 | PostgreSQL 16 | MinIO |

### 7.2 关键配置参数

| 参数 | 值 | 说明 |
|---|---|---|
| `RETRIEVAL_TOP_K` | 30 | 检索阶段返回的候选数量 |
| `RERANK_TOP_K` | 5 | 重排序后保留的文档数量 |
| `RELEVANCE_THRESHOLD` | 0.15 | 相关性阈值，低于此值拒答 |
| `CHUNK_MAX_TOKENS` | 512 | 分块最大 token 数 |

### 7.3 Docker Compose 服务

**文件**：`docker-compose.yml`

| 服务 | 说明 |
|---|---|
| etcd | Milvus 依赖 |
| MinIO | 对象存储（生产模式） |
| Milvus | 向量数据库（生产模式） |
| PostgreSQL 16 | 关系型数据库 |
| Redis 7 | Celery 消息队列 |
| FastAPI API | 后端服务 |
| Celery Worker | 异步任务 |
| Next.js | 前端服务 |

---

## 8. 流程总览图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          离线索引流程                                    │
│                                                                         │
│   data/cook/dishes/*.md                                                 │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  RecipeParser │───▶│ RecipeChunker│───▶│EmbeddingSvc  │              │
│   │  正则+MD解析  │    │ 4类领域分块  │    │ BGE-large-zh │              │
│   └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                   │                     │
│                                                   ▼                     │
│                                          ┌──────────────┐              │
│                                          │  ChromaDB    │              │
│                                          │  向量持久化   │              │
│                                          └──────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          在线查询流程                                    │
│                                                                         │
│   用户提问                                                              │
│      │                                                                  │
│      ▼                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │QueryClassifier│───▶│QueryRewriter │───▶│HybridRetriever│             │
│   │ 7类+路由权重  │    │ 消解/拆解/改写│    │向量+BM25+RRF │              │
│   └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                   │                     │
│                                                   ▼                     │
│   ┌──────────────┐    ┌──────────────┐                           │
│   │AnswerGenerator│◀───│ RerankerSvc  │◀──────────────────────────┘      │
│   │ LLM+反幻觉   │    │CrossEncoder  │              │
│   └──────┬───────┘    └──────────────┘              │
│          │                                                                  │
│          ▼                                                                  │
│   答案 + 引用 + 调试信息                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

> **文档生成日期**：2026-06-04
> **项目版本**：PRD v3.2
