"""
缓存服务
实现PRD 7.2节定义的缓存策略
使用 Redis 作为后端存储
"""

import logging
import hashlib
import json
from typing import Any, Optional, Dict
from app.core.redis import get_redis

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis 缓存实现

    PRD规范：
    - 查询重构结果缓存：key = query_rewrite:{hash(原始问题+记忆摘要)}，TTL 1h
    - 检索结果缓存：key = retrieval:{hash(重构查询+分类)}，TTL 1h
    - 嵌入缓存：LangChain的CacheBackedEmbeddings
    """

    def __init__(self, default_ttl: int = 3600):
        """
        初始化缓存

        Args:
            default_ttl: 默认TTL（秒）
        """
        self.default_ttl = default_ttl

    def _generate_key(self, prefix: str, *args) -> str:
        """
        生成缓存键

        Args:
            prefix: 键前缀
            *args: 用于生成hash的参数

        Returns:
            缓存键
        """
        content = ":".join(str(arg) for arg in args)
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{hash_value}"

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在返回None
        """
        try:
            redis = await get_redis()
            raw = await redis.get(key)
            if raw is None:
                return None
            logger.debug(f"Cache hit: {key}")
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: TTL（秒）
        """
        try:
            redis = await get_redis()
            ttl = ttl or self.default_ttl
            await redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
            logger.debug(f"Cache set: {key}, ttl={ttl}s")
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")

    async def delete(self, key: str):
        """删除缓存"""
        try:
            redis = await get_redis()
            await redis.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete failed for key {key}: {e}")

    async def clear(self):
        """清除所有缓存（慎用）"""
        try:
            redis = await get_redis()
            await redis.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        try:
            redis = await get_redis()
            info = await redis.info("keyspace")
            db_info = info.get("db0", {})
            return {
                "size": db_info.get("keys", 0),
                "expires": db_info.get("expires", 0),
            }
        except Exception as e:
            logger.warning(f"Redis stats failed: {e}")
            return {"size": 0, "expires": 0}


class QueryRewriteCache:
    """查询重构缓存"""

    def __init__(self, cache: RedisCache):
        self.cache = cache

    async def get(self, query: str, memory_summary: str = "") -> Optional[list]:
        """获取缓存的查询重构结果"""
        key = self.cache._generate_key("query_rewrite", query, memory_summary)
        return await self.cache.get(key)

    async def set(self, query: str, results: list, memory_summary: str = ""):
        """缓存查询重构结果"""
        key = self.cache._generate_key("query_rewrite", query, memory_summary)
        await self.cache.set(key, results)


class RetrievalCache:
    """检索结果缓存"""

    def __init__(self, cache: RedisCache):
        self.cache = cache

    async def get(self, query: str, query_type: str) -> Optional[list]:
        """获取缓存的检索结果"""
        key = self.cache._generate_key("retrieval", query, query_type)
        return await self.cache.get(key)

    async def set(self, query: str, query_type: str, results: list):
        """缓存检索结果"""
        key = self.cache._generate_key("retrieval", query, query_type)
        await self.cache.set(key, results)


# 全局实例
memory_cache = RedisCache(default_ttl=3600)
query_rewrite_cache = QueryRewriteCache(memory_cache)
retrieval_cache = RetrievalCache(memory_cache)
