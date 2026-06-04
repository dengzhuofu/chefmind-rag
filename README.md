# ChefMind 私厨大脑

基于 LangChain 的智能食谱 RAG 助手，支持多模态食谱摄入、自然语言问答、食材检索、步骤指引等功能。

## 项目结构

```
rag-langchain/
├── backend/                    # FastAPI 后端
│   ├── app/
│   │   ├── api/               # API 路由
│   │   │   └── endpoints/     # 具体端点
│   │   ├── core/              # 核心配置
│   │   │   ├── config.py      # 应用配置
│   │   │   ├── database.py    # 数据库配置
│   │   │   └── milvus.py      # Milvus配置
│   │   ├── models/            # 数据库模型
│   │   ├── schemas/           # Pydantic 模式
│   │   └── services/          # 业务逻辑
│   ├── Dockerfile
│   ├── requirements.txt       # 完整依赖
│   └── requirements-lite.txt  # 简化依赖(本地开发)
├── worker/                     # Celery 异步任务
├── frontend/                   # Next.js 前端
├── docker-compose.yml          # Docker 编排配置
├── .env.example               # 环境变量示例(完整版)
├── .env.lite                  # 环境变量示例(简化版)
├── LOCAL_SETUP.md             # 本地开发环境安装指南
└── README.md
```

## 环境要求

### 已检查
- ✅ Python 3.13.13 (满足 3.11+ 要求)
- ✅ pip 26.1.1

## 快速开始 (本地开发)

### 1. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 安装简化版依赖
pip install -r backend/requirements-lite.txt
```

### 2. 配置环境变量

```bash
# 复制简化版配置
copy .env.lite .env

# 编辑 .env 文件，填入你的 API 密钥
# DEEPSEEK_API_KEY=your_key_here
# DASHSCOPE_API_KEY=your_key_here
```

### 3. 运行后端

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问服务

- **API 文档**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 本地开发方案

本项目支持两种开发模式：

### 简化模式 (推荐入门)
- **数据库**: SQLite (无需安装)
- **向量数据库**: Milvus Lite (嵌入式，无需Docker)
- **文件存储**: 本地文件系统
- **配置文件**: `.env.lite`

### 完整模式
- **数据库**: PostgreSQL 16
- **向量数据库**: Milvus 2.5 Server
- **文件存储**: MinIO
- **消息队列**: Redis + Celery
- **配置文件**: `.env.example`

## 核心技术栈

| 组件 | 技术 |
|------|------|
| LLM | DeepSeek-V3 / Qwen-Max |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 |
| 重排序模型 | BAAI/bge-reranker-large |
| RAG 框架 | LangChain |
| 向量数据库 | Milvus Lite / Milvus 2.5 |
| 关系数据库 | SQLite / PostgreSQL 16 |
| 后端框架 | FastAPI |
| 前端框架 | Next.js 14 |

## API 密钥获取

### DeepSeek API (推荐)
- 注册: https://platform.deepseek.com/
- 获取 API Key

### 通义千问 API
- 注册: https://dashscope.aliyun.com/
- 获取 API Key

### OpenAI API (用于多模态)
- 注册: https://platform.openai.com/
- 获取 API Key

## 开发路线图

- [ ] M1: 基础 RAG + 自定义分块 + 防幻觉提示词
- [ ] M2: 混合检索 + 重排序 + 检索阈值拒答
- [ ] M3: 查询重构 + 路由分类 + 对话记忆
- [ ] M4: 多文档引用溯源
- [ ] M5: 多模态摄入与展示
- [ ] M6: 工程化加固
- [ ] M7: 评估、测试与演示

## License

MIT
