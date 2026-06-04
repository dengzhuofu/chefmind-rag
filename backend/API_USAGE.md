# ChefMind API 使用指南

## 基础信息

- **基础URL**: `http://localhost:8000`
- **API文档**: `http://localhost:8000/docs`
- **版本**: 1.0.0

---

## 聊天API

### 1. 标准聊天

**POST** `/api/chat`

```json
{
  "query": "清蒸鲈鱼怎么做？",
  "session_id": "user_123"
}
```

**响应**:
```json
{
  "request_id": "uuid",
  "answer": "清蒸鲈鱼的做法是...",
  "citations": [
    {
      "citation_id": "ref_1",
      "recipe_title": "清蒸鲈鱼",
      "chunk_type": "step",
      "step_number": 1,
      "excerpt": "将鲈鱼清洗干净...",
      "document_source": "清蒸鲈鱼.md"
    }
  ],
  "has_related_content": true,
  "refused": false
}
```

### 2. 流式聊天

**POST** `/api/chat/stream`

```json
{
  "query": "宫保鸡丁需要哪些食材？",
  "session_id": "user_123"
}
```

**响应** (SSE):
```
data: {"text": "宫保鸡丁的主要食材包括"}
data: {"text": "鸡肉、花生、干辣椒等"}
data: [DONE]
```

### 3. 创建新会话

**POST** `/api/new_session`

```json
{
  "session_id": "user_123"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "Session user_123 cleared"
}
```

### 4. 获取会话信息

**GET** `/api/session/{session_id}`

**响应**:
```json
{
  "session_id": "user_123",
  "message_count": 4,
  "current_recipe": "清蒸鲈鱼",
  "last_message": {
    "role": "assistant",
    "content": "清蒸鲈鱼的做法是...",
    "timestamp": "2026-06-03T09:41:25.463723"
  }
}
```

---

## 菜谱API

### 1. 上传菜谱图片

**POST** `/api/recipes/upload`

**请求** (multipart/form-data):
- `file`: 图片文件（JPEG、PNG、WebP、GIF）
- `recipe_id`: 菜谱ID（可选）
- `recipe_title`: 菜谱名称（可选）

**响应**:
```json
{
  "status": "success",
  "image_path": "./uploads/general/abc123.jpg",
  "recipe_id": "recipe_001",
  "recipe_title": "红烧肉",
  "chunk_content": "菜品图片描述：红烧肉。外观：色泽红亮...",
  "message": "Image uploaded and processed successfully"
}
```

### 2. 获取菜谱图片

**GET** `/api/recipes/{recipe_id}/images`

**响应**:
```json
{
  "images": [
    {
      "filename": "abc123.jpg",
      "path": "./uploads/recipe_001/abc123.jpg",
      "url": "/uploads/abc123.jpg"
    }
  ]
}
```

---

## 评估API

### 1. 获取评估摘要

**GET** `/api/evaluation/summary`

**响应**:
```json
{
  "total_tests": 6,
  "passed": 5,
  "failed": 1,
  "pass_rate": 0.83,
  "average_metrics": {
    "recall_at_5": 0.85,
    "citation_accuracy": 0.95,
    "faithfulness": 0.80,
    "answer_relevance": 0.75,
    "refuse_accuracy": 0.90
  }
}
```

### 2. 获取测试用例

**GET** `/api/evaluation/test-cases`

**响应**:
```json
{
  "test_cases": [
    {
      "id": "TC001",
      "query": "清蒸鲈鱼怎么做？",
      "expected_recipes": ["清蒸鲈鱼"],
      "route_type": "step_lookup",
      "difficulty": "easy",
      "description": "基础步骤查询"
    }
  ]
}
```

### 3. 运行评估

**POST** `/api/evaluation/run`

**响应**:
```json
{
  "status": "success",
  "message": "Evaluation completed",
  "summary": {...}
}
```

---

## 健康检查

**GET** `/health`

**响应**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "milvus_connected": true,
  "database_connected": true
}
```

---

## 错误处理

所有API在出错时返回：
```json
{
  "detail": "错误信息"
}
```

常见HTTP状态码：
- `200`: 成功
- `400`: 请求参数错误
- `500`: 服务器内部错误
- `503`: 服务不可用（如数据库连接失败）

---

## 使用示例

### Python示例

```python
import requests

# 聊天
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "query": "红烧肉怎么做？",
        "session_id": "user_001"
    }
)
print(response.json())

# 上传图片
with open("dish.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/recipes/upload",
        files={"file": f},
        data={"recipe_title": "红烧肉"}
    )
print(response.json())

# 创建新会话
response = requests.post(
    "http://localhost:8000/api/new_session",
    json={"session_id": "user_001"}
)
print(response.json())
```

### cURL示例

```bash
# 聊天
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "清蒸鲈鱼怎么做？", "session_id": "user_001"}'

# 上传图片
curl -X POST http://localhost:8000/api/recipes/upload \
  -F "file=@dish.jpg" \
  -F "recipe_title=红烧肉"

# 创建新会话
curl -X POST http://localhost:8000/api/new_session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user_001"}'
```

---

## 启动服务

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看完整的API文档。
