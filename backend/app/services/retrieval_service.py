"""
检索服务
整合查询重构、分类和混合检索
"""

import logging
from typing import List, Dict, Any, Optional

from app.services.query_rewriter import QueryRewriter, query_rewriter
from app.services.query_classifier import QueryClassifier, query_classifier, QueryType
from app.services.hybrid_retriever import HybridRetriever, RetrievalResult, get_hybrid_retriever

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    检索服务

    整合PRD 4.4节定义的完整检索管道：
    1. 查询重构
    2. 查询分类
    3. 混合检索
    4. 结果融合
    """

    def __init__(
        self,
        query_rewriter: Optional[QueryRewriter] = None,
        query_classifier: Optional[QueryClassifier] = None,
        hybrid_retriever: Optional[HybridRetriever] = None
    ):
        """
        初始化检索服务

        Args:
            query_rewriter: 查询重构器
            query_classifier: 查询分类器
            hybrid_retriever: 混合检索器
        """
        self.query_rewriter = query_rewriter or query_rewriter
        self.query_classifier = query_classifier or query_classifier
        self.hybrid_retriever = hybrid_retriever or get_hybrid_retriever()

    async def retrieve(
        self,
        query: str,
        chat_history: Optional[str] = None,
        top_k: int = 5,
        use_rewrite: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整检索流程

        Args:
            query: 用户查询
            chat_history: 对话历史
            top_k: 返回结果数
            use_rewrite: 是否使用查询重构

        Returns:
            检索结果和元数据
        """
        # 1. 查询分类
        classification = await self.query_classifier.classify_and_get_config(query)
        query_type = classification["type"]

        logger.info(f"Query classified as: {query_type}")

        # 2. 查询重构（可选）
        queries = [query]
        if use_rewrite:
            rewritten = await self.query_rewriter.rewrite_query(query, chat_history)
            # 去重并保留原始查询
            seen = {query}
            for rq in rewritten:
                if rq.rewritten not in seen:
                    queries.append(rq.rewritten)
                    seen.add(rq.rewritten)

        logger.info(f"Executing {len(queries)} queries")

        # 3. 并行执行检索
        all_results = []
        for q in queries:
            results = await self.hybrid_retriever.retrieve(
                query=q,
                query_type=query_type,
                top_k=top_k * 2,  # 获取更多结果用于去重
                filters=classification.get("chunk_type_filter")
            )
            all_results.extend(results)

        # 4. 去重并排序
        unique_results = self._deduplicate_results(all_results)

        # 5. 返回Top-K
        final_results = unique_results[:top_k]

        return {
            "query": query,
            "query_type": query_type,
            "rewritten_queries": queries[1:] if len(queries) > 1 else [],
            "results": final_results,
            "total_candidates": len(all_results),
            "unique_results": len(unique_results),
        }

    def _deduplicate_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        去重结果

        Args:
            results: 原始结果列表

        Returns:
            去重后的结果列表
        """
        seen = set()
        unique_results = []

        for result in results:
            if result.chunk_id not in seen:
                seen.add(result.chunk_id)
                unique_results.append(result)

        # 按分数排序
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results

    async def retrieve_for_recipe_generation(
        self,
        query: str,
        reference_count: int = 10
    ) -> List[RetrievalResult]:
        """
        为菜谱生成检索参考材料

        Args:
            query: 查询
            reference_count: 参考材料数量

        Returns:
            检索结果列表
        """
        # 使用通用类型检索，获取更多结果
        results = await self.hybrid_retriever.retrieve(
            query=query,
            query_type=QueryType.GENERATE_RECIPE,
            top_k=reference_count
        )

        return results


# 创建全局实例
retrieval_service = RetrievalService()
