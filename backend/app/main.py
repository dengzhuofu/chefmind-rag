"""ChefMind 主应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import sys

from app.core.config import settings, ensure_directories
from app.core.database import init_db, close_db
from app.core.milvus import connect_milvus, create_recipe_collection
from app.core.redis import init_redis, close_redis

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Starting ChefMind application...")

    # 确保目录存在
    ensure_directories()

    # 初始化数据库
    await init_db()
    logger.info("Database initialized")

    # 连接Milvus并创建集合
    try:
        connect_milvus()
        create_recipe_collection()
        logger.info("Milvus connected and collection ready")
    except Exception as e:
        logger.warning(f"Milvus connection failed: {e}. Vector search will be unavailable.")

    # 连接Redis
    try:
        await init_redis()
        # 初始化Redis版对话记忆
        from app.services.conversation_memory import _init_conversation_memory
        await _init_conversation_memory()
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching will be unavailable.")

    yield

    # 关闭时执行
    await close_redis()
    await close_db()
    logger.info("Application shutdown complete")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description="智能食谱RAG助手 - 支持多模态食谱摄入、自然语言问答、食材检索、步骤指引",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 基础路由 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to ChefMind - 私厨大脑",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    redis_ok = False
    try:
        from app.core.redis import get_redis
        redis = await get_redis()
        await redis.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "version": "1.0.0",
        "milvus_connected": True,  # TODO: 实际检查
        "database_connected": True,  # TODO: 实际检查
        "redis_connected": redis_ok,
    }


# ==================== 注册路由 ====================

from app.api.endpoints import chat, recipes, evaluation
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(recipes.router, prefix="/api", tags=["recipes"])
app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])

# 挂载静态文件目录，使上传的图片可通过URL访问
import os
upload_dir = settings.UPLOAD_DIR if hasattr(settings, 'UPLOAD_DIR') else "./uploads"
os.makedirs(upload_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")

# TODO: 注册其他API路由
# from app.api.endpoints import documents
# app.include_router(documents.router, prefix="/api/documents", tags=["documents"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
