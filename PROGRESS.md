# ChefMind 开发进度

## 已完成的RAG链路模块

### ✅ 4.1 文档处理
- **Markdown解析器** (`backend/app/services/recipe_parser.py`)
  - 支持解析标准格式的Markdown菜谱文件
  - 提取标题、食材、步骤、小贴士、图片等结构化信息
  - 批量解析目录功能
  - 已测试：成功解析323个菜谱

- **结构化数据模型** (`backend/app/schemas/parsed_recipe.py`)
  - ParsedRecipe: 解析后的菜谱结构
  - RecipeChunk: 分块后的数据单元

### ✅ 4.2 Chunk切分策略
- **领域专用分块器** (`backend/app/services/recipe_chunker.py`)
  - 支持4种块类型：ingredient/step/tip/image_desc
  - 自动拼接菜名前缀
  - 步骤块附加相关食材
  - 支持上下文摘要（prev/next step）
  - 自动截断（最大512 tokens）
  - 已测试：成功分块清蒸鲈鱼（14个块）

### ✅ 4.3 Embedding向量化
- **嵌入服务** (`backend/app/services/embedding_service.py`)
  - 集成BAAI/bge-large-zh-v1.5模型
  - 支持指令前缀
  - 批量嵌入（batch_size=32）
  - 向量维度：1024
  - 已测试：成功生成1024维向量

### ✅ 4.4.1 查询重构
- **查询重构服务** (`backend/app/services/query_rewriter.py`)
  - 指代消解与上下文补全
  - 复合意图分解
  - 多视角生成
  - HyDE（可选）
  - 已测试：成功分解"红烧肉和糖醋排骨怎么做"

### ✅ 4.4.2 查询分类与路由
- **查询分类器** (`backend/app/services/query_classifier.py`)
  - 支持6种查询类型
  - 动态权重配置
  - chunk类型过滤
  - 已测试：成功分类各种查询

### ✅ 4.4.3 混合检索协调器
- **混合检索器** (`backend/app/services/hybrid_retriever.py`)
  - 向量检索（Milvus ANN）
  - BM25关键词检索
  - RRF融合（k=60）
  - 动态权重调整
  - Mock模式用于测试

### ✅ 4.5 Rerank重排序
- **重排序服务** (`backend/app/services/reranker_service.py`)
  - 集成BAAI/bge-reranker-large模型
  - 阈值截断（0.35）
  - 文本截断（512字符）
  - 已测试：成功重排序结果

### ✅ 4.6 回答生成
- **回答生成服务** (`backend/app/services/answer_generator.py`)
  - 强制引用提示词
  - 引用校验（防幻觉）
  - 结构化引用构建
  - 拒答机制
  - 已测试：成功生成带引用的回答

### ✅ 完整RAG管道
- **RAG管道** (`backend/app/services/rag_pipeline.py`)
  - 整合所有模块
  - 支持流式输出
  - 错误处理和降级

### ✅ API端点
- **聊天API** (`backend/app/api/endpoints/chat.py`)
  - POST /api/chat - 标准聊天
  - POST /api/chat/stream - 流式聊天
  - 符合PRD 7.4节响应结构

---

## 待完成模块

### ⏳ 4.7 多文档引用溯源
- 引用数据结构完善
- 跨文档引用支持
- 前端渲染方案

### ⏳ 4.8 幻觉处理
- 检索置信度拦截完善
- 输出后校验增强
- 用户反馈闭环

### ⏳ 4.9 对话记忆
- ConversationBufferWindowMemory
- 实体追踪
- 会话管理

### ⏳ 5.0 文档上传
- 文件上传接口
- 异步处理任务
- 进度反馈

### ⏳ 6.0 前端开发
- Next.js应用
- 流式对话UI
- 引用展示组件

---

## 如何运行

### 1. 安装依赖
```bash
cd backend
pip install -r requirements-lite.txt
```

### 2. 配置环境变量
```bash
copy ..\.env.lite ..\.env
# 编辑 .env 填入 API 密钥
```

### 3. 启动服务
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问API文档
http://localhost:8000/docs

---

## 测试命令

```bash
# 测试解析器
python -c "from app.services.recipe_parser import recipe_parser; print(recipe_parser.parse_file('../data/cook/dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md'))"

# 测试分块器
python -c "from app.services.recipe_parser import recipe_parser; from app.services.recipe_chunker import recipe_chunker; r = recipe_parser.parse_file('../data/cook/dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md'); print(len(recipe_chunker.chunk_recipe(r)))"

# 测试嵌入服务
python -c "from app.services.embedding_service import get_embedding_service; s = get_embedding_service(); print(len(s.embed_query('test')))"
```
