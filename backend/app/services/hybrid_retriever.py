"""
混合检索协调器
实现PRD 4.4.3节定义的混合检索与融合策略
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from app.core.config import settings
from app.services.embedding_service import get_embedding_service
from app.services.query_classifier import QueryType, RouteConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # "vector", "bm25", or "hybrid"


class HybridRetriever:
    """
    混合检索协调器

    实现PRD 4.4.3节规范：
    - 向量检索：通过Milvus进行ANN检索，返回Top-30
    - 关键词检索：使用Milvus内置BM25，返回Top-30
    - 融合：采用倒数排名融合（RRF），k=60
    - 动态权重：根据路由类型调整权重
    """

    # RRF参数（PRD规定k=60）
    RRF_K = 60

    # 默认返回结果数
    DEFAULT_TOP_K = settings.RERANK_TOP_K

    def __init__(self, milvus_client=None):
        """
        初始化混合检索器

        Args:
            milvus_client: Milvus客户端实例
        """
        self.milvus_client = milvus_client
        self.embedding_service = get_embedding_service()

    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        混合检索

        Args:
            query: 查询文本
            query_type: 查询类型
            top_k: 返回结果数
            filters: 元数据过滤条件

        Returns:
            检索结果列表
        """
        # 获取权重配置
        weights = RouteConfig.get_weights(query_type)
        chunk_type_filter = RouteConfig.get_chunk_type_filter(query_type)

        # 合并过滤条件
        if chunk_type_filter and filters is None:
            filters = {"chunk_type": {"$in": chunk_type_filter}}
        elif chunk_type_filter and filters:
            filters["chunk_type"] = {"$in": chunk_type_filter}

        # 并行执行向量检索和BM25检索
        vector_results = await self._vector_search(query, top_k=30, filters=filters)
        bm25_results = await self._bm25_search(query, top_k=30, filters=filters)

        # RRF融合
        fused_results = self._rrf_fusion(
            vector_results,
            bm25_results,
            semantic_weight=weights["semantic"],
            bm25_weight=weights["bm25"]
        )

        # 返回Top-K
        return fused_results[:top_k]

    async def _vector_search(
        self,
        query: str,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            filters: 过滤条件

        Returns:
            检索结果列表
        """
        if not self.milvus_client:
            logger.warning("Milvus client not available, returning empty results")
            return []

        try:
            # 生成查询向量
            query_embedding = self.embedding_service.embed_query(query)

            # 检查是否是ChromaDB客户端
            if hasattr(self.milvus_client, 'get_collection'):
                # ChromaDB检索
                from app.core.milvus import search_vectors
                results = search_vectors(query_embedding, top_k=top_k)

                # 转换结果格式
                retrieval_results = []
                if results and len(results) > 0:
                    for hit in results[0]:
                        result = RetrievalResult(
                            chunk_id=hit.get("id", ""),
                            content=hit.get("entity", {}).get("content", ""),
                            score=1.0 - hit.get("distance", 0),  # ChromaDB返回距离，转换为相似度
                            metadata={
                                "recipe_id": hit.get("entity", {}).get("recipe_id"),
                                "recipe_title": hit.get("entity", {}).get("recipe_title"),
                                "chunk_type": hit.get("entity", {}).get("chunk_type"),
                                "step_number": hit.get("entity", {}).get("step_number"),
                                "document_source": hit.get("entity", {}).get("document_source"),
                                "image_path": hit.get("entity", {}).get("image_path"),
                            },
                            source="vector"
                        )
                        retrieval_results.append(result)

                return retrieval_results
            else:
                # Milvus检索
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": 128}
                }

                results = self.milvus_client.search(
                    collection_name=settings.MILVUS_COLLECTION_NAME,
                    data=[query_embedding],
                    limit=top_k,
                    output_fields=[
                        "chunk_id", "content", "recipe_id", "recipe_title",
                        "chunk_type", "step_number", "document_source",
                        "cooking_time", "difficulty", "tags", "image_path"
                    ],
                    search_params=search_params,
                    filter=self._build_filter_expr(filters)
                )

                # 转换结果格式
                retrieval_results = []
                if results and len(results) > 0:
                    for hit in results[0]:
                        result = RetrievalResult(
                            chunk_id=hit.entity.get("chunk_id", ""),
                            content=hit.entity.get("content", ""),
                            score=hit.distance,
                            metadata={
                                "recipe_id": hit.entity.get("recipe_id"),
                                "recipe_title": hit.entity.get("recipe_title"),
                                "chunk_type": hit.entity.get("chunk_type"),
                                "step_number": hit.entity.get("step_number"),
                                "document_source": hit.entity.get("document_source"),
                                "cooking_time": hit.entity.get("cooking_time"),
                                "difficulty": hit.entity.get("difficulty"),
                                "tags": hit.entity.get("tags"),
                                "image_path": hit.entity.get("image_path"),
                            },
                            source="vector"
                        )
                        retrieval_results.append(result)

                return retrieval_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _bm25_search(
        self,
        query: str,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        BM25关键词检索（使用rank_bm25本地实现）

        Args:
            query: 查询文本
            top_k: 返回结果数
            filters: 过滤条件

        Returns:
            检索结果列表
        """
        if not self.milvus_client:
            logger.warning("Milvus client not available, returning empty results")
            return []

        try:
            from rank_bm25 import BM25Okapi
            import jieba

            # 检查是否是ChromaDB客户端
            if hasattr(self.milvus_client, 'get_collection'):
                from app.core.milvus import get_chroma_client
                client = get_chroma_client()
                collection = client.get_collection(settings.MILVUS_COLLECTION_NAME)

                # 获取所有文档（ChromaDB不支持原生BM25，需要本地计算）
                all_docs = collection.get(
                    include=["documents", "metadatas"]
                )

                if not all_docs or not all_docs["ids"]:
                    return []

                # 分词处理
                doc_texts = all_docs["documents"]
                tokenized_docs = [list(jieba.cut(doc)) for doc in doc_texts]

                # 构建BM25索引
                bm25 = BM25Okapi(tokenized_docs)

                # 查询分词
                query_tokens = list(jieba.cut(query))
                scores = bm25.get_scores(query_tokens)

                # 获取top-k索引
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

                # 构建结果
                retrieval_results = []
                for idx in top_indices:
                    if scores[idx] <= 0:
                        continue

                    meta = all_docs["metadatas"][idx]
                    result = RetrievalResult(
                        chunk_id=all_docs["ids"][idx],
                        content=doc_texts[idx],
                        score=float(scores[idx]),
                        metadata={
                            "recipe_id": meta.get("recipe_id", ""),
                            "recipe_title": meta.get("recipe_title", ""),
                            "chunk_type": meta.get("chunk_type", ""),
                            "step_number": meta.get("step_number"),
                            "document_source": meta.get("document_source", ""),
                            "image_path": meta.get("image_path"),
                        },
                        source="bm25"
                    )
                    retrieval_results.append(result)

                logger.info(f"BM25 search returned {len(retrieval_results)} results for query: {query[:50]}")
                return retrieval_results
            else:
                # Milvus原生BM25（备用）
                search_params = {
                    "metric_type": "BM25",
                    "params": {"k1": 1.2, "b": 0.75}
                }
                results = self.milvus_client.search(
                    collection_name=settings.MILVUS_COLLECTION_NAME,
                    data=[query],
                    limit=top_k,
                    output_fields=[
                        "chunk_id", "content", "recipe_id", "recipe_title",
                        "chunk_type", "step_number", "document_source",
                        "image_path",
                    ],
                    search_params=search_params,
                    filter=self._build_filter_expr(filters)
                )
                retrieval_results = []
                if results and len(results) > 0:
                    for hit in results[0]:
                        result = RetrievalResult(
                            chunk_id=hit.entity.get("chunk_id", ""),
                            content=hit.entity.get("content", ""),
                            score=hit.distance,
                            metadata={
                                "recipe_id": hit.entity.get("recipe_id"),
                                "recipe_title": hit.entity.get("recipe_title"),
                                "chunk_type": hit.entity.get("chunk_type"),
                                "step_number": hit.entity.get("step_number"),
                                "document_source": hit.entity.get("document_source"),
                                "image_path": hit.entity.get("image_path"),
                            },
                            source="bm25"
                        )
                        retrieval_results.append(result)
                return retrieval_results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[RetrievalResult]:
        """
        倒数排名融合（RRF）

        PRD规范：
        - score(d) = Σ 1/(k + rank_i(d))
        - k=60

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            semantic_weight: 语义权重
            bm25_weight: BM25权重

        Returns:
            融合后的结果列表
        """
        # 构建chunk_id到结果的映射
        chunk_scores = {}
        chunk_data = {}

        # 处理向量检索结果
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.chunk_id
            rrf_score = semantic_weight / (self.RRF_K + rank)

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0
                chunk_data[chunk_id] = result

            chunk_scores[chunk_id] += rrf_score

        # 处理BM25检索结果
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.chunk_id
            rrf_score = bm25_weight / (self.RRF_K + rank)

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0
                chunk_data[chunk_id] = result

            chunk_scores[chunk_id] += rrf_score

        # 按分数排序
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 构建结果列表
        results = []
        for chunk_id, score in sorted_chunks:
            result = chunk_data[chunk_id]
            result.score = score
            result.source = "hybrid"
            results.append(result)

        return results

    def _build_filter_expr(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        构建Milvus过滤表达式

        Args:
            filters: 过滤条件字典

        Returns:
            过滤表达式字符串
        """
        if not filters:
            return None

        expressions = []

        for key, value in filters.items():
            if isinstance(value, dict):
                if "$in" in value:
                    # IN 表达式
                    values_str = ", ".join([f'"{v}"' for v in value["$in"]])
                    expressions.append(f'{key} in [{values_str}]')
                elif "$eq" in value:
                    expressions.append(f'{key} == "{value["$eq"]}"')
                elif "$gte" in value:
                    expressions.append(f'{key} >= {value["$gte"]}')
                elif "$lte" in value:
                    expressions.append(f'{key} <= {value["$lte"]}')
            else:
                # 简单相等
                if isinstance(value, str):
                    expressions.append(f'{key} == "{value}"')
                else:
                    expressions.append(f'{key} == {value}')

        return " and ".join(expressions) if expressions else None


class MockHybridRetriever(HybridRetriever):
    """
    模拟混合检索器（用于测试）

    当Milvus不可用时，返回模拟结果
    """

    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """返回模拟结果"""
        logger.info(f"Mock retrieve for: {query}")

        # 返回示例结果
        mock_results = [
            RetrievalResult(
                chunk_id="mock_step_1",
                content=f"《示例菜谱》步骤1：准备食材，与查询'{query}'相关",
                score=0.95,
                metadata={
                    "recipe_id": "mock_001",
                    "recipe_title": "示例菜谱",
                    "chunk_type": "step",
                    "step_number": 1,
                    "document_source": "示例.md",
                },
                source="mock"
            ),
            RetrievalResult(
                chunk_id="mock_ingredient",
                content="《示例菜谱》食材：主料、辅料、调料",
                score=0.85,
                metadata={
                    "recipe_id": "mock_001",
                    "recipe_title": "示例菜谱",
                    "chunk_type": "ingredient",
                    "document_source": "示例.md",
                },
                source="mock"
            ),
        ]

        return mock_results[:top_k]


def get_hybrid_retriever(milvus_client=None) -> HybridRetriever:
    """
    获取混合检索器实例

    Args:
        milvus_client: Milvus客户端

    Returns:
        检索器实例
    """
    if milvus_client:
        return HybridRetriever(milvus_client)
    else:
        # 尝试使用ChromaDB
        try:
            from app.core.milvus import get_chroma_client
            chroma_client = get_chroma_client()
            if chroma_client:
                logger.info("Using ChromaDB for retrieval")
                return HybridRetriever(chroma_client)
        except Exception as e:
            logger.warning(f"ChromaDB not available: {e}")

        logger.warning("No vector database available, using mock retriever")
        return MockHybridRetriever()
