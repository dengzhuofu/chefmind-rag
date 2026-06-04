"""
Redis 客户端模块
提供异步 Redis 连接管理
"""

import logging
import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)

# 全局 Redis 客户端实例
_redis_client: aioredis.Redis = None


async def get_redis() -> aioredis.Redis:
    """获取 Redis 客户端实例"""
    global _redis_client
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_redis() first.")
    return _redis_client


async def init_redis():
    """初始化 Redis 连接"""
    global _redis_client
    _redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    # 验证连接
    await _redis_client.ping()
    logger.info(f"Redis connected: {settings.REDIS_URL}")


async def close_redis():
    """关闭 Redis 连接"""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")
