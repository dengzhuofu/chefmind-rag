# 本地开发环境安装指南

## 方案A: 简化开发 (推荐先跑通RAG)

使用 SQLite + Milvus Lite + 本地文件存储，无需安装额外服务。

### 1. 创建虚拟环境

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r backend/requirements-lite.txt
```

### 3. 配置环境变量

```bash
copy .env.lite .env
# 编辑 .env 填入 API 密钥
```

### 4. 运行

```bash
cd backend
uvicorn app.main:app --reload
```

---

## 方案B: 安装完整服务

### 1. 安装 PostgreSQL

**Windows 安装:**
- 下载: https://www.postgresql.org/download/windows/
- 安装时记住设置的密码

**创建数据库:**
```sql
CREATE DATABASE chefmind;
CREATE USER chefmind WITH PASSWORD 'chefmind123';
GRANT ALL PRIVILEGES ON DATABASE chefmind TO chefmind;
```

### 2. 安装 Redis

**Windows 安装:**
- 下载 Redis for Windows: https://github.com/tporadowski/redis/releases
- 或使用 WSL2: `sudo apt install redis-server`

**启动:**
```bash
redis-server
```

### 3. Milvus Lite (推荐)

Milvus Lite 是嵌入式版本，无需独立服务：

```bash
pip install milvus[model]
```

### 4. 配置环境变量

```bash
copy .env.example .env
# 编辑 .env 配置数据库连接
```

---

## API 密钥获取

### DeepSeek API
- 注册: https://platform.deepseek.com/
- 获取 API Key

### 通义千问 API
- 注册: https://dashscope.aliyun.com/
- 获取 API Key

### OpenAI API (用于多模态)
- 注册: https://platform.openai.com/
- 获取 API Key
