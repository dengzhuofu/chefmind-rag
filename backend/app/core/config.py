"""应用配置模块"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """应用设置"""

    # 应用配置
    APP_NAME: str = "ChefMind"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # 数据库配置
    DATABASE_URL: str = "sqlite+aiosqlite:///./chefmind.db"

    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"

    # Milvus配置
    MILVUS_URI: str = "./milvus_data/chefmind.db"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "recipes"
    USE_MILVUS_LITE: bool = True  # 使用Milvus Lite

    # 存储配置
    STORAGE_PATH: str = "./storage"
    UPLOAD_PATH: str = "./storage/uploads"
    IMAGE_PATH: str = "./storage/images"
    UPLOAD_DIR: str = "./uploads"

    # LLM API配置
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com"
    DASHSCOPE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # SiliconFlow API配置
    SILICONFLOW_API_KEY: Optional[str] = None
    SILICONFLOW_API_BASE: str = "https://api.siliconflow.cn/v1"
    SILICONFLOW_TEXT_MODEL: str = "Qwen/Qwen3-8B"
    SILICONFLOW_VISION_MODEL: str = "deepseek-ai/DeepSeek-OCR"

    # 模型配置
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_MODEL_DIMENSION: int = 1024
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-large"

    # RAG配置
    RETRIEVAL_TOP_K: int = 30
    RERANK_TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.15
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP: int = 0

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 创建全局设置实例
settings = Settings()


def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        settings.STORAGE_PATH,
        settings.UPLOAD_PATH,
        settings.IMAGE_PATH,
        settings.UPLOAD_DIR,
        "./milvus_data",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
