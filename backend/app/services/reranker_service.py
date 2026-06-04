"""
重排序服务
实现PRD 4.5节定义的Rerank重排序
"""

import logging
from typing import List, Tuple, Optional

from app.core.config import settings
from app.services.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)

# 模型导入（可选）
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    logger.warning("CrossEncoder not installed. Reranker service unavailable.")


class RerankerService:
    """
    重排序服务

    实现PRD 4.5节规范：
    - 模型：BAAI/bge-reranker-large
    - 输入：用户原始查询 + RRF融合后的Top-60候选chunk文本
    - 输出：按分数降序排列，截取Top-5
    - 阈值截断：若最大分数 < RELEVANCE_THRESHOLD (默认0.35)，判定为无相关信息
    - 性能优化：对超长候选文本进行截断（保留前512字符）
    """

    # 文本截断长度（新格式chunk包含完整食材列表，需要更大空间）
    MAX_TEXT_LENGTH = 1024

    def __init__(self, model_name: Optional[str] = None):
        """
        初始化重排序服务

        Args:
            model_name: 模型名称
        """
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.model = None
        self._initialized = False

    def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return

        if not HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers is required for CrossEncoder. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading reranker model: {self.model_name}")
        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            self._initialized = True
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = settings.RERANK_TOP_K,
        threshold: float = settings.RELEVANCE_THRESHOLD
    ) -> Tuple[List[RetrievalResult], bool]:
        """
        重排序检索结果

        Args:
            query: 原始查询（非重构后）
            results: 检索结果列表
            top_k: 返回结果数
            threshold: 相关性阈值

        Returns:
            (重排序后的结果列表, 是否有相关内容)
        """
        if not results:
            return [], False

        self.initialize()

        # 准备输入对
        pairs = []
        for result in results:
            # 截断过长文本
            truncated_content = self._truncate_text(result.content)
            pairs.append((query, truncated_content))

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 将分数添加到结果
        scored_results = []
        for result, score in zip(results, scores):
            result.score = float(score)
            scored_results.append(result)

        # 按分数降序排序
        scored_results.sort(key=lambda x: x.score, reverse=True)

        # 检查最大分数是否超过阈值
        max_score = scored_results[0].score if scored_results else 0
        has_related_content = max_score >= threshold

        if not has_related_content:
            logger.info(f"No related content found. Max score: {max_score} < threshold: {threshold}")

        # 返回Top-K
        return scored_results[:top_k], has_related_content

    def _truncate_text(self, text: str) -> str:
        """
        截断文本到最大长度

        PRD规定：保留前512字符

        Args:
            text: 原始文本

        Returns:
            截断后的文本
        """
        if len(text) <= self.MAX_TEXT_LENGTH:
            return text

        return text[:self.MAX_TEXT_LENGTH] + "..."

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return HAS_CROSS_ENCODER


class MockRerankerService(RerankerService):
    """
    模拟重排序服务（用于测试）
    """

    def initialize(self):
        """跳过模型加载"""
        self._initialized = True
        logger.info("Using mock reranker service")

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
        threshold: float = 0.35
    ) -> Tuple[List[RetrievalResult], bool]:
        """返回模拟排序结果"""
        # 使用原始分数（如果有）或随机分数
        import random
        for i, result in enumerate(results):
            if result.score <= 0:
                result.score = 1.0 - (i * 0.1)  # 递减分数

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 模拟阈值检查
        max_score = results[0].score if results else 0
        has_related_content = max_score >= threshold

        return results[:top_k], has_related_content

    def is_available(self) -> bool:
        """模拟服务始终可用"""
        return True


def get_reranker_service() -> RerankerService:
    """
    获取重排序服务实例

    Returns:
        重排序服务实例
    """
    service = RerankerService()
    if not service.is_available():
        logger.warning("Real reranker service unavailable, using mock service")
        return MockRerankerService()
    return service
