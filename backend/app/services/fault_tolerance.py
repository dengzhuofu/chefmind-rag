"""
容错与降级服务
实现PRD 7.3节定义的容错策略
"""

import logging
import asyncio
from typing import Any, Callable, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    熔断器

    PRD规范：
    - Milvus连接失败 → 尝试重试3次，之后抛异常，API返回503
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: int = 60,
        name: str = "default"
    ):
        """
        初始化熔断器

        Args:
            failure_threshold: 失败阈值
            reset_timeout: 重置超时（秒）
            name: 熔断器名称
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).seconds
                if elapsed >= self.reset_timeout:
                    self.state = "half-open"
                    return True
            return False

        # half-open state
        return True

    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")

    def reset(self):
        """重置熔断器"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None


class RetryHandler:
    """
    重试处理器

    PRD规范：
    - 重构超时（2s）→ 跳过重构，使用原始查询
    - 重排序服务不可用 → 直接使用融合后的Top-5顺序
    - LLM不可用 → 返回固定错误提示
    - 引用构建失败 → 降级返回纯文本回答
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 2.0,
        backoff_factor: float = 0.5
    ):
        """
        初始化重试处理器

        Args:
            max_retries: 最大重试次数
            timeout: 超时时间（秒）
            backoff_factor: 退避因子
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        执行函数，支持重试和降级

        Args:
            func: 要执行的函数
            *args: 位置参数
            fallback: 降级函数
            **kwargs: 关键字参数

        Returns:
            执行结果
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # 使用超时控制
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.timeout
                )
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            # 退避等待
            if attempt < self.max_retries - 1:
                wait_time = self.backoff_factor * (2 ** attempt)
                await asyncio.sleep(wait_time)

        # 所有重试都失败，使用降级方案
        if fallback:
            logger.info(f"Using fallback after {self.max_retries} failed attempts")
            return fallback(*args, **kwargs)

        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")


class FallbackStrategies:
    """
    降级策略

    PRD规范：
    - 重构超时 → 跳过重构，使用原始查询
    - 重排序服务不可用 → 直接使用融合后的Top-5顺序
    - LLM不可用 → 返回固定错误提示
    - 引用构建失败 → 降级返回纯文本回答
    """

    @staticmethod
    def fallback_query_rewrite(query: str, *args, **kwargs):
        """查询重构降级：返回原始查询"""
        logger.info("Fallback: using original query")
        from app.services.query_rewriter import RewrittenQuery
        return [RewrittenQuery(
            original=query,
            rewritten=query,
            strategy="fallback",
            confidence=0.5
        )]

    @staticmethod
    def fallback_rerank(query: str, results: list, *args, **kwargs):
        """重排序降级：直接返回Top-5"""
        logger.info("Fallback: using original ranking")
        return results[:5], len(results[:5]) > 0

    @staticmethod
    def fallback_llm_generation(*args, **kwargs):
        """LLM生成降级：返回固定提示"""
        logger.info("Fallback: using default error message")
        return "助手暂时无法响应，请稍后再试。"

    @staticmethod
    def fallback_citation_build(*args, **kwargs):
        """引用构建降级：返回空引用"""
        logger.info("Fallback: using empty citations")
        return []


# 全局实例
circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60, name="main")
retry_handler = RetryHandler(max_retries=3, timeout=2.0)
