"""
嵌入服务
实现PRD 4.3节定义的Embedding向量化
"""

import logging
from typing import List, Optional
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# 模型导入（可选）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Embedding service unavailable.")


class EmbeddingService:
    """
    嵌入服务

    实现PRD 4.3节规范：
    - 模型：BAAI/bge-large-zh-v1.5
    - 指令前缀："为这段文本生成向量表示："
    - 批量处理：batch_size=32
    - 向量维度：1024
    """

    # BGE模型推荐的指令前缀
    INSTRUCTION_PREFIX = "为这段文本生成向量表示："

    # 默认批量大小
    DEFAULT_BATCH_SIZE = 32

    def __init__(self, model_name: Optional[str] = None):
        """
        初始化嵌入服务

        Args:
            model_name: 模型名称，默认使用配置中的模型
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model = None
        self._initialized = False

    def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return

        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info(f"Embedding model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本

        Args:
            text: 查询文本

        Returns:
            向量列表
        """
        self.initialize()

        # 添加指令前缀（BGE模型推荐）
        prefixed_text = f"{self.INSTRUCTION_PREFIX}{text}"

        # 生成嵌入
        embedding = self.model.encode(prefixed_text, normalize_embeddings=True)

        return embedding.tolist()

    def embed_documents(
        self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[List[float]]:
        """
        批量嵌入文档文本

        Args:
            texts: 文本列表
            batch_size: 批量大小

        Returns:
            向量列表
        """
        self.initialize()

        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} documents with batch_size={batch_size}")

        # 批量处理
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # BGE模型文档嵌入不需要指令前缀
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=True if len(batch) > 10 else False
            )
            all_embeddings.extend(batch_embeddings.tolist())

        logger.info(f"Embedding complete. Shape: {len(all_embeddings)} x {len(all_embeddings[0])}")
        return all_embeddings

    def embed_recipe_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        嵌入菜谱分块

        Args:
            chunks: 分块列表，每个分块包含content字段

        Returns:
            添加了embedding字段的分块列表
        """
        if not chunks:
            return []

        # 提取文本内容
        texts = [chunk["content"] for chunk in chunks]

        # 批量嵌入
        embeddings = self.embed_documents(texts)

        # 将嵌入添加到分块
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks

    def get_dimension(self) -> int:
        """获取向量维度"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return settings.EMBEDDING_MODEL_DIMENSION

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return HAS_SENTENCE_TRANSFORMERS


class MockEmbeddingService(EmbeddingService):
    """
    模拟嵌入服务（用于测试）

    当真实模型不可用时，生成随机向量
    """

    def initialize(self):
        """跳过模型加载"""
        self._initialized = True
        logger.info("Using mock embedding service")

    def embed_query(self, text: str) -> List[float]:
        """生成随机向量"""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(settings.EMBEDDING_MODEL_DIMENSION).tolist()

    def embed_documents(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """生成随机向量"""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            embeddings.append(np.random.randn(settings.EMBEDDING_MODEL_DIMENSION).tolist())
        return embeddings

    def is_available(self) -> bool:
        """模拟服务始终可用"""
        return True


def get_embedding_service() -> EmbeddingService:
    """
    获取嵌入服务实例

    如果真实模型不可用，返回模拟服务
    """
    service = EmbeddingService()
    if not service.is_available():
        logger.warning("Real embedding service unavailable, using mock service")
        return MockEmbeddingService()
    return service
