# ChefMind 私厨大脑 – 完整产品需求文档（PRD） v3.2

---


## 1. 产品概述

**ChefMind** 是一款面向个人与家庭烹饪场景的智能食谱助手，支持文字、PDF、图片等多模态食谱的摄入与解析，提供自然语言问答、食材检索、步骤指引、图片展示等功能。系统基于先进 RAG 技术，结合查询重构、混合检索与重排序，保障精准度；内置完善的防幻觉与拒答机制，并支持**多文档引用溯源**，确保每个回答都有据可查，做到“知之为知之，不知为不知”。

**核心价值**：
- 多源食谱一键导入，自动结构化索引
- 深度理解用户口语化、模糊、复合型查询
- 生成回答强制引用原文，精确到文档名、菜谱名和步骤编号
- 无匹配时直接拒答，杜绝编造
- 支持多模态（图片描述、步骤图返回）
- 从检索到生成全链路可观测、可评估

---

## 2. 技术选型与模型选择

| 类别               | 技术/模型                                                      | 用途与说明                                                                 |
|-------------------|---------------------------------------------------------------|----------------------------------------------------------------------------|
| **大语言模型 (LLM)** | DeepSeek-V3 (API) 或 Qwen-Max                                  | 最终答案生成、查询重构、HyDE（可选）                                        |
| **轻量分类/路由模型** | Qwen-Turbo (API)                                               | 查询重构、意图分类，低延迟场景                                               |
| **嵌入模型**        | `BAAI/bge-large-zh-v1.5` (本地部署)                           | 将文本片段向量化，维度 1024，支持中文语义匹配                               |
| **重排序模型**      | `BAAI/bge-reranker-large` (本地部署)                          | 对检索候选进行精细排序，提升相关性                                           |
| **多模态模型**      | GPT-4o-mini (API) 或 Qwen-VL (本地)                            | 图片描述生成、手写食谱 OCR+理解                                              |
| **RAG 编排框架**    | **LangChain** (Python)                                         | 基于 LCEL 构建可组合的 RAG 管道，利用 RouterChain、Memory、HyDE 等模块，并提供丰富的检索器集成与回调机制 |
| **向量数据库**      | Milvus 2.5 (单机)                                              | 存储向量索引与标量字段，支持 BM25 混合检索与元数据过滤                       |
| **关键词索引**      | Milvus 内置 BM25                                              | 针对食材名、菜名等进行精确关键词召回                                         |
| **关系数据库**      | PostgreSQL 16                                                  | 存储菜谱基础信息、对话日志、用户反馈、引用数据                                 |
| **对象存储**        | MinIO (兼容 S3)                                               | 存储原始文档、菜谱图片、步骤图                                               |
| **消息队列**        | Redis + Celery                                                | 异步处理文档解析、图片描述生成等耗时任务                                     |
| **后端框架**        | FastAPI (Python 3.11+)                                        | 高性能异步 API，支持流式响应（SSE）                                          |
| **前端**            | Next.js 14 + Tailwind CSS                                     | 流式对话 UI、步骤卡片、图片查看器、引用悬浮卡片                                |
| **监控与日志**      | Prometheus + Grafana，结构化日志 (structlog)                   | 全链路指标监控、错误追踪                                                     |
| **部署**            | Docker Compose                                                | 一键启动本地开发环境，所有组件容器化                                         |

---

## 3. 系统架构设计

### 3.1 架构示意图（文字描述）
```
┌─────────────┐       ┌──────────────────────────────────────────────┐
│   用户浏览器   │ ←SSE→  │                FastAPI 网关                    │
└─────────────┘       │   /chat   /recipes/upload   /eval/feedback     │
                      └──┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┘
                         │   │   │   │   │   │   │   │   │   │
              ┌──────────┘   │   │   │   │   │   │   │   │   └──────────┐
              │  查询重构服务  │   │   │   │   │   │   │   │              │
              │  (LLM Call)   │   │   │   │   │   │   │   │      ┌───────┴───────┐
              └──────────────┘   │   │   │   │   │   │   │      │  异步任务队列   │
                                 │   │   │   │   │   │   │      │ (Celery Worker) │
              ┌──────────────────┘   │   │   │   │   │   │      └───┬───────┬─────┘
              │  路由分类服务 (LLM)   │   │   │   │   │   │          │       │
              └──────────────────────┘   │   │   │   │   │  ┌───────┘       └────────┐
                                         │   │   │   │   │  │  文档解析 (Unstructured)  │
              ┌──────────────────────────┘   │   │   │   │  └──────────────────────────┘
              │  混合检索协调器 (LangChain)   │   │   │   │
              │  ┌──────────┐ ┌───────────┐   │   │   │   │  ┌──────────────────────────┐
              │  │ 向量检索  │ │ BM25 检索  │   │   │   │   │  │  多模态处理 (GPT-4o-mini) │
              │  │(Milvus)  │ │ (Milvus)  │   │   │   │   │  └──────────────────────────┘
              │  └──────────┘ └───────────┘   │   │   │   │
              └──────────────────────────────┘   │   │   │
                                 │              │   │   │
              ┌──────────────────┘              │   │   │
              │ 重排序服务 (BGE-Reranker)        │   │   │
              └─────────────────────────────────┘   │   │
                                                     │   │
              ┌──────────────────────────────────────┘   │
              │  生成服务 (LLM) + 引用构建 + 防幻觉校验    │
              └──────────────────────────────────────────┘
```

### 3.2 组件交互说明（LangChain 集成）
- **API 网关**：接收所有请求，进行基本鉴权（MVP 可无），路由到不同模块。
- **查询重构服务**：基于 LangChain 的 `ChatPromptTemplate` + LLM 调用链，结合对话记忆，将模糊问题改写为精准查询。
- **路由分类服务**：使用 LangChain 的 `RouterChain` 或自定义 LCEL 分支，根据问题类型决定后续检索策略。
- **混合检索协调器**：通过 LangChain 的自定义检索器（`BaseRetriever`）封装 Milvus 的向量检索与 BM25 检索，利用 LCEL 组合并执行 RRF 融合。
- **重排序服务**：作为 LangChain 检索管道中的 `ContextualCompressionRetriever` 或独立后处理节点，对候选集打分排序。
- **生成与引用构建服务**：通过 LangChain 的 `ChatPromptTemplate` + LLM 生成带引用编号的回答，经后处理构建结构化引用数组，并结合 `OutputParser` 进行防幻觉校验。
- **异步任务队列**：Celery 处理文档上传后的解析、分块、嵌入、图片描述生成等任务，任务中可复用 LangChain 的文档加载器和文本分割器。
- **存储层**：Milvus（向量+关键词索引）、PostgreSQL（菜谱元数据、对话日志、用户反馈、引用数据）、MinIO（文件/图片）。

---

## 4. RAG 链路详细设计

### 4.1 文档处理
**支持格式**：
- 纯文本（.txt）
- Markdown（.md）
- PDF（含文字型与扫描型）
- 图片（.jpg/.png，手写菜谱、成品图）

**处理流水线**：
1. **上传**：通过 FastAPI 接口接收，原文件存入 MinIO。
2. **格式识别**：根据 MIME 类型选择解析器。
3. **解析**：
   - 文本/Markdown：直接读取。
   - PDF：使用 LangChain 集成的 `UnstructuredLoader` 或 `PyMuPDFLoader` 提取文本与表格，保留阅读顺序。
   - 图片：调用多模态模型（GPT-4o-mini 或 Qwen-VL）生成结构化的食谱描述（菜名、食材、步骤、小贴士）；同时可提取 OCR 文本作为辅助。
4. **结构化处理**：将解析结果标准化为 JSON，包含 `title`, `ingredients[]`, `steps[]`, `tips`, `raw_text`, `images[]`, `source_file` 等字段。
5. **异步状态反馈**：前端可轮询或通过 WebSocket 获取处理进度（`PENDING→PARSING→CHUNKING→INDEXING→DONE`）。

### 4.2 Chunk 切分策略及优化
基于食谱结构设计**领域专用分块器**，继承 LangChain 的 `BaseDocumentTransformer` 或自定义文本分割器。

**分块类型与策略**：

| 块类型 (`chunk_type`) | 分割依据                                   | 优化细节                                                                                                                                                   |
|----------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ingredient`         | 按食材项（行）分割，每项一个 chunk         | 每条前自动拼接菜名与类别（如“《麻婆豆腐》食材：豆腐 200g”），嵌入时保留上下文；元数据包含 `ingredient_list` 数组，供关键词精确匹配。                         |
| `step`               | 以步骤序号（`^\d+[.、)]`）为边界分割       | 每个步骤单独成块，但文本内容自动拼接“菜谱名《xxx》步骤 N：”前缀，并附加上一步与下一步的摘要作为增强上下文（`prev_step_summary`, `next_step_summary`）。元数据包含 `step_number`。 |
| `tip`                | 以小标题“小贴士/注意/提示”等分割          | 独立索引，关联所属菜谱 ID。                                                                                                                             |
| `image_desc`         | 每张图片生成一段描述文字                   | 描述由多模态模型生成，文本后附加原图 MinIO 路径，元数据含 `image_path` 和 `image_type`（成品图/步骤图）。                                                  |

**通用元数据字段**（每个 chunk 均携带）：
- `recipe_id`：所属菜谱唯一 ID
- `recipe_title`：菜谱名称
- `chunk_type`：上述类型（`ingredient` / `step` / `tip` / `image_desc`）
- `chunk_id`：全局唯一的 chunk 标识（如 `rec_042_step_3`），用于后续引用映射
- `step_number`：步骤序号（仅 step 类型）
- `cooking_time`：烹饪时长（如“30分钟”）
- `difficulty`：难度（简单/中等/困难）
- `tags`：标签数组（如“素食”、“快手菜”）
- `document_source`：来源文件名（如“川菜经典菜谱.pdf”）

**优化点**：
- 防止长步骤截断：如果某步骤过长，允许其独立成块，但设置最大长度 512 tokens 并自动截断，同时标记 `truncated: true`。
- 食材与步骤的隐含关联：在每个步骤 chunk 末尾追加“用到的食材：豆腐、牛肉末”，增强跨类型检索能力。
- 块大小控制：通过 `BAAI/bge-large-zh-v1.5` 的最佳输入长度（512 tokens）进行调整，重叠量设为 0（因为已经基于结构分割）。

### 4.3 Embedding 向量化
- **模型**：`BAAI/bge-large-zh-v1.5`，通过 `sentence-transformers` 或 HuggingFace TEI 部署为本地服务。
- **指令前缀**：嵌入文本时使用 BGE 模型的推荐前缀 `"为这段文本生成向量表示："`，以提升检索效果。
- **批量处理**：上传菜谱后，异步任务收集所有 chunk，以 batch_size=32 调用嵌入接口。
- **存储**：使用 LangChain 的 `Milvus` 向量存储接口，将向量与元数据一起 upsert 到 Milvus 集合中。
- **向量维度**：1024，索引类型使用 IVF_FLAT 或 HNSW 以平衡速度与精度。

### 4.4 检索策略及优化
检索管道串联了查询重构、路由分类、混合检索与融合，是系统最核心的优化环节。整体通过 LangChain 的 LCEL 管道串联。

#### 4.4.1 查询重构 (Query Rewriting)
在收到用户原始问题后，先通过 LLM 进行多策略改写，构建 LangChain 的 `RunnableBranch` 或自定义链：

- **指代消解与上下文补全**：输入当前问题 + 最近 5 轮对话摘要（从 `ConversationBufferWindowMemory` 获取），输出独立完整的查询。
- **复合意图分解**：对于明显的并列问题，分解为 1~3 个子查询。
- **多视角生成**：对同一个问题生成 2~3 种不同说法。
- **HyDE（可选优化）**：使用 LangChain 的 `HypotheticalDocumentEmbedder`，让 LLM 生成假设的菜谱摘要文本，使用该摘要进行向量检索。

**执行策略**：
- 所有改写查询并行执行（利用 `asyncio.gather` 或 LangChain 的 `RunnableParallel`），最多保留 3 路召回。
- 重构结果短期缓存（Redis，TTL 1h），降低重复调用成本。

#### 4.4.2 查询分类与路由
对主查询（重构后的第一候选）进行分类，确定问题类型。使用 LangChain 的 `RouterChain` 或自定义 LCEL 分支：

**类型枚举**：
- `ingredient_search`：基于冰箱食材的匹配
- `step_lookup`：询问具体步骤
- `substitution`：食材替换建议
- `tip`：烹饪技巧
- `image_request`：要求看图
- `generate_recipe`：直接生成菜谱（检索作为参考）

**实现**：用 Qwen-Turbo 零样本分类，prompt 输出严格 JSON。路由根据 `type` 跳转到不同的检索权重配置和提示词模板。

#### 4.4.3 混合检索与融合
- **向量检索**：通过 LangChain 的 `Milvus` 检索器，用重构后的每个查询分别进行 ANN 检索，返回 Top-30 候选。
- **关键词检索**：使用 Milvus 内置 BM25，LangChain 中可通过自定义检索器或直接调用 Milvus API 实现，返回 Top-30。
- **融合**：采用倒数排名融合（RRF），在自定义 `Retriever` 中实现，公式为 `score(d) = Σ 1/(k + rank_i(d))`，k=60。候选集大小设为 60。
- **动态权重**：根据路由类型对不同检索来源的结果进行加权微调：

| 问题类型           | 语义权重 | BM25 权重 | 说明                           |
|--------------------|----------|-----------|--------------------------------|
| `ingredient_search`| 0.4      | 0.6       | 食材名必须精确命中             |
| `step_lookup`      | 0.7      | 0.3       | 步骤描述依赖语义               |
| `tip`              | 0.8      | 0.2       | 技巧多为长句，语义为主         |
| `substitution`     | 0.5      | 0.5       | 替换需兼顾语义与关键词         |

- **过滤**：融合时可根据元数据提前过滤（如只取 `chunk_type` 为 `step` 和 `ingredient`），但 MVP 可先不做。

#### 4.4.4 优化措施
- **查询缓存**：高频问题的检索结果在 Redis 缓存 1 小时。
- **并行执行**：重构、分类、多路检索均通过 LangChain 的 `RunnableParallel` 或 `asyncio` 并发处理。
- **失败降级**：重构失败直接使用原始查询；某一路检索超时则忽略该路结果。

### 4.5 Rerank 重排序
- **模型**：`BAAI/bge-reranker-large`，输入查询 + 候选文本，输出相关性分数。
- **集成方式**：作为 LangChain 检索链中的 `ContextualCompressionRetriever`，将 Reranker 封装为 `BaseDocumentCompressor`。
- **输入**：用户原始查询（非重构后） + RRF 融合后的 Top-60 候选 chunk 文本。
- **输出**：按分数降序排列，截取 Top-5。
- **阈值截断**：若最大分数 < `RELEVANCE_THRESHOLD` (默认 0.35)，则判定为无相关信息，触发拒答。
- **性能优化**：重排序模型本地部署，支持 batch 推理；对超长候选文本进行截断（保留前 512 字符）。

### 4.6 回答生成
- **提示词设计**（LangChain `ChatPromptTemplate` 系统消息，含引用要求）：
```
你是一个严谨的食谱问答助手。你只能依据下方提供的“参考食谱片段”来回答问题。

## 回答要求
1. **必须标注引用**：每个基于片段的内容，请在句末标注引用来源，格式为 `[来源编号]`。
2. **引用编号对应**：来源编号必须与下方片段前的编号严格对应。
3. **不得编造**：如果片段信息不足，请直接回复“根据现有食谱库，无法回答此问题”，绝对禁止编造任何步骤、食材或技巧。
4. **格式示例**：
   用户问：麻婆豆腐怎么做？
   回答：
   麻婆豆腐是一道经典川菜。首先将豆腐切块焯水[1]。然后热锅倒油，放入牛肉末煸炒[2]。最后加入豆瓣酱和豆腐煮至入味[3]。

> 📎 引用来源：
> [1] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤1
> [2] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤2
> [3] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤3
```
- **上下文构建**：将重排序 Top-5 的文本按以下格式拼接：
```
[1] 来源《麻婆豆腐》(川菜经典菜谱.pdf) 步骤1：
将豆腐切成2cm见方的小块，放入加了盐的沸水中焯烫1分钟...

[2] 来源《麻婆豆腐》(川菜经典菜谱.pdf) 步骤2：
锅中倒油烧至六成热，放入牛肉末煸炒至变色...

[3] 来源《家常豆腐》(每日一菜.md) 步骤1：
将豆腐切块，用厨房纸吸干水分...
```
- **生成参数**：temperature=0.1，max_tokens=800，使用 LangChain 的 `ChatDeepSeek` 或 `ChatTongyi` 组件，开启流式输出（通过 FastAPI SSE 透传）。
- **引用强制**：结合 `OutputParser` 对 LLM 输出进行后处理，若发现提及的菜谱名未在上下文中出现，则追加不确定性声明或转拒答。

### 4.7 多文档引用溯源（完整方案）

#### 4.7.1 引用数据结构
在回答中返回结构化引用，前端可渲染为可点击链接或展开卡片。

```python
from pydantic import BaseModel

class Citation(BaseModel):
    citation_id: str          # 唯一引用ID，如 "ref_001"
    recipe_title: str         # 菜谱名称
    chunk_type: str           # 块类型：step / ingredient / tip / image_desc
    step_number: int | None   # 步骤序号（仅 step 类型）
    excerpt: str              # 引用的原文摘录（前100字符）
    document_source: str      # 文档来源文件名
```

**引用数据示例**：
```json
{
  "citation_id": "ref_001",
  "recipe_title": "麻婆豆腐",
  "chunk_type": "step",
  "step_number": 3,
  "excerpt": "锅中倒油，烧至六成热，放入牛肉末煸炒至变色...",
  "document_source": "川菜经典菜谱.pdf"
}
```

#### 4.7.2 检索阶段保留引用元数据
检索时，通过 LangChain `Document` 对象携带完整元数据：

```python
from langchain.schema import Document

doc = Document(
    page_content="锅中倒油，烧至六成热，放入牛肉末煸炒至变色...",
    metadata={
        "recipe_id": "rec_042",
        "recipe_title": "麻婆豆腐",
        "chunk_type": "step",
        "step_number": 3,
        "document_source": "川菜经典菜谱.pdf",
        "chunk_id": "rec_042_step_3"
    }
)
```

#### 4.7.3 LLM 输出示例（带引用编号）
经过提示词约束后，LLM 输出带引用编号的回答：

```
麻婆豆腐是一道经典川菜，麻辣鲜香。

首先将豆腐切成2cm见方的小块，放入加盐的沸水中焯烫1分钟，去除豆腥味[1]。
然后热锅倒油烧至六成热，放入牛肉末100g煸炒至变色，加入豆瓣酱炒出红油[2]。
最后加入焯好的豆腐和适量清水，小火煮3分钟使豆腐入味，撒上花椒粉即可[3]。

> 📎 引用来源：
> [1] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤1
> [2] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤2
> [3] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤3
```

#### 4.7.4 跨文档引用示例
当回答涉及多个文档时，引用会自动体现不同来源：

```
豆腐是一种百搭食材，有多种做法：

麻婆豆腐：豆腐切块焯水后与牛肉末、豆瓣酱同煮，麻辣鲜香[1]。
家常豆腐：豆腐切片煎至两面金黄，再与青椒、木耳翻炒[2]。
皮蛋豆腐：嫩豆腐切块，淋上调料汁，放上皮蛋碎，是凉菜[3]。

> 📎 引用来源：
> [1] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤摘要
> [2] 《家常豆腐》(家常菜100道.pdf) - 步骤1
> [3] 《皮蛋豆腐》(夏日凉菜集.md) - 做法概述
```

#### 4.7.5 后处理：构建引用映射与输出格式化
在 LangChain 生成链末尾添加后处理函数，提取引用并构建映射。

```python
import re

def build_citation_map(retrieved_docs: list[Document]) -> dict:
    """为检索到的文档构建引用映射"""
    citation_map = {}
    for idx, doc in enumerate(retrieved_docs, start=1):
        citation_map[str(idx)] = {
            "recipe_title": doc.metadata.get("recipe_title", "未知菜谱"),
            "chunk_type": doc.metadata.get("chunk_type", "unknown"),
            "step_number": doc.metadata.get("step_number", None),
            "excerpt": doc.page_content[:100],
            "document_source": doc.metadata.get("document_source", "未知来源")
        }
    return citation_map

def format_final_response(llm_output: str, citation_map: dict) -> dict:
    """将 LLM 输出与引用映射合并为结构化响应"""
    cited_ids = set(re.findall(r'\[(\d+)\]', llm_output))
    
    citations = []
    for cid in sorted(cited_ids, key=int):
        if cid in citation_map:
            citations.append({
                "citation_id": f"ref_{cid}",
                "recipe_title": citation_map[cid]["recipe_title"],
                "document_source": citation_map[cid]["document_source"],
                "chunk_type": citation_map[cid]["chunk_type"],
                "step_number": citation_map[cid]["step_number"],
                "excerpt": citation_map[cid]["excerpt"]
            })
    
    return {
        "answer": llm_output,
        "citations": citations,
        "has_related_content": len(citations) > 0
    }
```

#### 4.7.6 LangChain 完整生成链代码

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

def build_rag_chain_with_citations(retriever, llm):
    """构建带引用的 RAG 链"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # 含引用要求的系统提示词
        ("human", human_prompt)
    ])
    
    def format_docs(docs: list[Document]) -> tuple[str, dict]:
        """格式化文档并构建引用映射"""
        formatted = []
        citation_map = {}
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata
            source = f"[{idx}] 来源《{meta.get('recipe_title', '未知')}》" \
                     f"({meta.get('document_source', '未知来源')})"
            if meta.get('step_number'):
                source += f" 步骤{meta['step_number']}"
            source += f"：\n{doc.page_content}"
            formatted.append(source)
            citation_map[str(idx)] = meta
        return "\n\n".join(formatted), citation_map
    
    def generate_with_citations(inputs: dict) -> dict:
        docs_str, citation_map = format_docs(inputs["docs"])
        response = llm.invoke(
            prompt.format(
                query=inputs["query"],
                chat_history=inputs.get("chat_history", ""),
                context=docs_str
            )
        )
        return format_final_response(response.content, citation_map)
    
    chain = (
        RunnablePassthrough.assign(docs=itemgetter("query") | retriever)
        | RunnableLambda(generate_with_citations)
    )
    
    return chain
```

#### 4.7.7 前端渲染方案
前端根据 `citations` 数组进行优雅展示：

**方式一：悬浮卡片（推荐 MVP）**
- 鼠标悬停在回答文本中的 `[1]` 上时，弹出小卡片显示：“《麻婆豆腐》步骤3 · 川菜经典菜谱.pdf”，并可预览原文摘录前 100 字符。

**方式二：侧栏引用面板**
- 回答右侧展示引用面板，列出所有引用来源，点击可展开查看原文摘录。

**方式三：底部引用区（推荐 MVP）**
- 回答文本底部自动追加“📎 参考来源”区域，列出所有被引用的文档及对应片段。

**MVP 推荐组合**：方式一（悬浮卡片）+ 方式三（底部引用区），实现简单且用户体验好。

### 4.8 幻觉处理（多层防护）
1. **检索置信度拦截**：重排序最高分数 < 0.35 时，不进入生成阶段，直接返回拒答。
2. **系统提示词强约束**：明确禁止使用内部知识，反复强调“仅基于片段”。
3. **输出后校验**：
   - 在 LangChain 生成链末尾添加自定义函数，正则提取回答中出现的菜谱名称，与当前上下文中的 `recipe_title` 对比。
   - 若出现非上下文的菜谱名，则替换回答为：“抱歉，我无法基于当前知识确认该信息，请重新提问。”
4. **引用校验**：若 LLM 输出中包含 `[引用编号]` 但该编号在 `citation_map` 中不存在，则判定为幻觉引用，返回拒答。
5. **用户反馈闭环**：
   - 每个回答附带“赞/踩”按钮，存入 PostgreSQL。
   - 踩数据定期分析，用于调整阈值、优化分块策略。
6. **定期回归测试**：建立包含故意诱导幻觉的测试集，通过 RAGAS 评估框架验证系统稳定性。

---

## 5. 对话记忆设计
- **短期记忆**：使用 LangChain 的 `ConversationBufferWindowMemory`，保留最近 5 轮对话，存储于内存或 Redis（用于多进程）。
- **实体追踪**：从历史中提取最近一次提及的 `recipe_title` 作为 `current_recipe`，在查询重构时注入。
- **记忆集成**：通过 LangChain 的 `RunnableWithMessageHistory` 自动管理对话历史。
- **记忆清理**：前端可调用 `/new_session` 接口，触发记忆清除。

---

## 6. 多模态处理流程
1. **图片上传**：通过 `/recipes/upload` 接收，存至 MinIO。
2. **异步任务**：Celery Worker 执行：
   - 调用多模态模型生成图片描述（prompt：“请详细描述这张图片中的菜品，包括菜名、外观、食材、可能的口味”）。
   - 生成结构化 JSON：`{"title": "...", "ingredients": [...], "description": "..."}`。
   - 将描述文本作为 `image_desc` 类型的 chunk，通过 LangChain 的 `Document` 对象封装，进行嵌入和索引，元数据包含 `image_path`、`document_source`。
3. **图片检索与返回**：当用户请求图片时，检索 `image_desc` 块，将原图 URL 返回，并附引用来源。
4. **多模态模型降级**：若 API 调用失败，回退为 OCR 文本（PaddleOCR），仅索引纯文本。

---

## 7. 工程化细节

### 7.1 日志与监控
- **结构化日志**：使用 `structlog`，通过 LangChain 的 `CallbackHandler` 记录每次查询的完整链路数据。每条日志包含：`request_id`, `timestamp`, `original_query`, `rewritten_queries`, `route_type`, `retrieval_top_scores`, `rerank_max_score`, `generated_answer`, `citations_count`, `user_feedback`。
- **指标**：通过 Prometheus 暴露：
  - `rag_query_duration_seconds`（直方图）
  - `rag_retrieval_recall`（基于测试集）
  - `rag_refuse_rate`（拒答比例）
  - `rag_citation_coverage`（回答中有引用的比例）
  - `llm_call_count_total`
- **Grafana 面板**：展示延迟分布、拒答趋势、引用覆盖率、错误率。

### 7.2 缓存策略
- **查询重构结果缓存**：Redis，key = `query_rewrite:{hash(原始问题+记忆摘要)}`，TTL 1h。
- **检索结果缓存**：Redis，key = `retrieval:{hash(重构查询+分类)}`，TTL 1h。
- **嵌入缓存**：LangChain 的 `CacheBackedEmbeddings` 可用于缓存嵌入结果，避免重复调用。

### 7.3 容错与降级
- **重构超时**（2s）→ 跳过重构，使用原始查询。
- **重排序服务不可用** → 直接使用融合后的 Top-5 顺序。
- **LLM 不可用** → 返回固定错误提示：“助手暂时无法响应，请稍后再试。”
- **Milvus 连接失败** → 尝试重试 3 次，之后抛异常，API 返回 503。
- **引用构建失败** → 降级返回纯文本回答，不含结构化引用。

### 7.4 API 响应结构
```python
# /chat 接口返回结构
{
    "request_id": "uuid",
    "answer": "麻婆豆腐是一道经典川菜...\n\n> 📎 引用来源：\n...",
    "citations": [
        {
            "citation_id": "ref_1",
            "recipe_title": "麻婆豆腐",
            "document_source": "川菜经典菜谱.pdf",
            "chunk_type": "step",
            "step_number": 1,
            "excerpt": "将豆腐切成2cm见方的小块..."
        }
    ],
    "has_related_content": true,
    "refused": false
}
```

### 7.5 部署
```yaml
# docker-compose.yml 核心服务
services:
  api:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
  redis:
    image: redis:7
  milvus:
    image: milvusdb/milvus:v2.5.0
  minio:
    image: minio/minio
  worker:
    build: ./worker
  postgres:
    image: postgres:16
```

---

## 8. 评估方案
- **测试集**：自建 50 条中文菜谱问题，覆盖所有路由类型、模糊查询、跨文档查询、无答案查询。
- **自动化指标**：
  - **Recall@5**：期望菜谱 chunk 在检索 Top-5 中的命中率。
  - **RAGAS**：Faithfulness（忠实度）、Answer Relevance（答案相关性）、Context Recall（上下文召回率）。
  - **拒答准确率**：对于无答案问题，系统是否拒答。
  - **引用准确率**：回答中引用的来源编号是否全部真实存在于检索结果中。
- **人工评估**：邀请 3 人，按 1-5 分对准确性、步骤完整性、引用准确性、图片相关性打分。

---

## 9. 开发路线图

| 阶段 | 内容                                         | 工期 |
|------|----------------------------------------------|------|
| M1   | 基础 RAG + 自定义分块 + 防幻觉提示词          | 3 天 |
| M2   | 混合检索 + 重排序 + 检索阈值拒答              | 2 天 |
| M3   | 查询重构 + 路由分类 + 对话记忆                | 2 天 |
| M4   | 多文档引用溯源（元数据、提示词、后处理）       | 2 天 |
| M5   | 多模态摄入与展示                             | 2 天 |
| M6   | 工程化加固（日志、监控、缓存、容错）          | 2 天 |
| M7   | 评估、测试与演示（录制）                      | 2 天 |

---

> 此 PRD v3.2 以 LangChain 为核心编排框架，完整涵盖了从文档处理、自定义分块、混合检索、重排序，到多文档引用溯源、生成防幻觉的 RAG 全链路。每个回答均附带结构化引用数据，精确追溯到文档名、菜谱名和步骤编号，是 ChefMind 开发的最终权威指南。如需任何模块的示例代码或细化 Prompt 模板，可随时补充。