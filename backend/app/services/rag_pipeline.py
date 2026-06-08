"""
RAG管道
整合PRD 4.1-4.8节定义的完整RAG链路
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, Tuple

from app.services.query_rewriter import QueryRewriter, query_rewriter
from app.services.query_classifier import QueryClassifier, query_classifier
from app.services.hybrid_retriever import HybridRetriever, RetrievalResult, get_hybrid_retriever
from app.services.reranker_service import RerankerService, get_reranker_service
from app.services.answer_generator import AnswerGenerator, AnswerResult, answer_generator
from app.services.conversation_memory import ConversationMemory, conversation_memory
from app.services.conversation_service import conversation_service
from app.services.cache_service import query_rewrite_cache, retrieval_cache

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    完整RAG管道

    实现PRD定义的完整RAG链路：
    1. 查询重构 (4.4.1)
    2. 查询分类 (4.4.2)
    3. 混合检索 (4.4.3)
    4. 重排序 (4.5)
    5. 回答生成 (4.6)
    6. 引用构建 (4.7)
    7. 幻觉处理 (4.8)
    8. 对话记忆 (5.0)
    """

    def __init__(
        self,
        query_rewriter: Optional[QueryRewriter] = None,
        query_classifier: Optional[QueryClassifier] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[RerankerService] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        memory: Optional[ConversationMemory] = None
    ):
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.query_classifier = query_classifier or QueryClassifier()
        self.hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
        self.reranker = reranker or get_reranker_service()

        # 创建LLM并传入AnswerGenerator
        if answer_generator:
            self.answer_generator = answer_generator
        else:
            from app.services.llm_factory import get_llm_for_answer
            llm = get_llm_for_answer()
            self.answer_generator = AnswerGenerator(llm=llm)
            if llm:
                logger.info("AnswerGenerator initialized with LLM")
            else:
                logger.warning("No LLM available, using simple answer generation")

        self.memory = memory or conversation_memory

    async def run(
        self,
        query: str,
        session_id: str = "default",
        request_id: Optional[str] = None
    ) -> Tuple[AnswerResult, Dict[str, Any]]:
        """
        执行完整RAG流程

        Args:
            query: 用户查询
            session_id: 会话ID
            request_id: 请求ID

        Returns:
            (回答结果, 管道调试信息)
        """
        total_start = time.time()

        # 生成请求ID
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(f"[{request_id}] Starting RAG pipeline for query: {query}")

        # 调试信息收集
        debug_info = {
            "query_type": "",
            "query_weights": {},
            "rewritten_queries": [],
            "retrieval_count": 0,
            "retrieval_results": [],
            "rerank_results": [],
            "has_related_content": True,
            "generation_time_ms": 0,
            "total_time_ms": 0,
        }

        try:
            # Step 0: 获取对话历史
            chat_history = await self.memory.get_history_as_string(session_id)
            current_recipe = await self.memory.get_current_recipe(session_id)
            if current_recipe:
                logger.info(f"[{request_id}] Current recipe context: {current_recipe}")

            # 记录用户消息
            await self.memory.add_message(session_id, "user", query)
            await conversation_service.save_message(session_id, "user", query)

            # Step 1: 查询分类
            logger.info(f"[{request_id}] Step 1: Query classification")
            classification = await self.query_classifier.classify_and_get_config(query)
            query_type = classification["type"]
            debug_info["query_type"] = query_type.value if hasattr(query_type, 'value') else str(query_type)
            debug_info["query_weights"] = classification.get("weights", {})
            logger.info(f"[{request_id}] Query type: {query_type}")

            # Step 2: 查询重构（注入对话历史和当前菜谱上下文）
            logger.info(f"[{request_id}] Step 2: Query rewriting")
            cached_rewrite = await query_rewrite_cache.get(query, chat_history)
            if cached_rewrite is not None:
                queries = cached_rewrite
                logger.info(f"[{request_id}] Query rewrite cache hit")
            else:
                rewritten_queries = await self.query_rewriter.rewrite_query(query, chat_history)
                queries = [query] + [rq.rewritten for rq in rewritten_queries if rq.rewritten != query]
                await query_rewrite_cache.set(query, queries, chat_history)
            debug_info["rewritten_queries"] = queries[1:] if len(queries) > 1 else []
            logger.info(f"[{request_id}] Generated {len(queries)} queries")

            # Step 3: 混合检索
            logger.info(f"[{request_id}] Step 3: Hybrid retrieval")
            all_results = []
            query_type_str = query_type.value if hasattr(query_type, 'value') else str(query_type)
            for q in queries:
                cached_results = await retrieval_cache.get(q, query_type_str)
                if cached_results is not None:
                    results = [RetrievalResult(**r) for r in cached_results]
                    logger.info(f"[{request_id}] Retrieval cache hit for: {q[:50]}")
                else:
                    results = await self.hybrid_retriever.retrieve(
                        query=q,
                        query_type=query_type,
                        top_k=30
                    )
                    await retrieval_cache.set(
                        q, query_type_str,
                        [{"chunk_id": r.chunk_id, "content": r.content, "score": r.score, "metadata": r.metadata, "source": r.source} for r in results]
                    )
                all_results.extend(results)
            logger.info(f"[{request_id}] Retrieved {len(all_results)} candidates")

            # 去重
            seen = set()
            unique_results = []
            for result in all_results:
                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    unique_results.append(result)

            debug_info["retrieval_count"] = len(unique_results)
            debug_info["retrieval_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "source": r.source,
                    "recipe_title": r.metadata.get("recipe_title", ""),
                    "chunk_type": r.metadata.get("chunk_type", ""),
                    "step_number": r.metadata.get("step_number"),
                    "content_preview": r.content[:200],
                }
                for r in unique_results[:30]  # 最多记录30条
            ]

            # Step 4: 重排序
            logger.info(f"[{request_id}] Step 4: Reranking")
            reranked_results, has_related_content = await self.reranker.rerank(
                query=query,
                results=unique_results,
                top_k=5
            )
            debug_info["has_related_content"] = has_related_content
            debug_info["rerank_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "source": r.source,
                    "recipe_title": r.metadata.get("recipe_title", ""),
                    "chunk_type": r.metadata.get("chunk_type", ""),
                    "step_number": r.metadata.get("step_number"),
                    "content_preview": r.content[:200],
                }
                for r in reranked_results
            ]
            logger.info(f"[{request_id}] Reranked to {len(reranked_results)} results, has_content: {has_related_content}")

            # Step 5: 回答生成
            logger.info(f"[{request_id}] Step 5: Answer generation")
            gen_start = time.time()
            answer_result = await self.answer_generator.generate(
                query=query,
                retrieval_results=reranked_results,
                has_related_content=has_related_content,
                chat_history=chat_history,
                request_id=request_id
            )
            debug_info["generation_time_ms"] = int((time.time() - gen_start) * 1000)

            # 记录助手消息
            await self.memory.add_message(session_id, "assistant", answer_result.answer)
            await conversation_service.save_message(
                session_id, "assistant", answer_result.answer,
                citations=[{"citation_id": c.citation_id, "recipe_title": c.recipe_title, "excerpt": c.excerpt, "image_path": c.image_path} for c in answer_result.citations]
            )

            debug_info["total_time_ms"] = int((time.time() - total_start) * 1000)
            logger.info(f"[{request_id}] RAG pipeline completed. Refused: {answer_result.refused}")
            return answer_result, debug_info

        except Exception as e:
            logger.error(f"[{request_id}] RAG pipeline failed: {e}", exc_info=True)
            debug_info["total_time_ms"] = int((time.time() - total_start) * 1000)
            return AnswerResult(
                answer="抱歉，处理您的问题时出现错误，请稍后重试。",
                citations=[],
                has_related_content=False,
                refused=True,
                request_id=request_id
            ), debug_info

    async def run_with_streaming(
        self,
        query: str,
        session_id: str = "default",
        request_id: Optional[str] = None
    ):
        """
        执行RAG流程（流式输出）

        Args:
            query: 用户查询
            session_id: 会话ID
            request_id: 请求ID

        Yields:
            流式输出的文本片段（str），最后一个 yield 是包含 citations 和 debug_info 的 dict
        """
        # 生成请求ID
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(f"[{request_id}] Starting streaming RAG pipeline")

        total_start = time.time()
        debug_info = {
            "query_type": "",
            "query_weights": {},
            "rewritten_queries": [],
            "retrieval_count": 0,
            "retrieval_results": [],
            "rerank_results": [],
            "has_related_content": True,
            "generation_time_ms": 0,
            "total_time_ms": 0,
        }

        try:
            # 获取对话历史
            chat_history = await self.memory.get_history_as_string(session_id)

            # 记录用户消息
            await self.memory.add_message(session_id, "user", query)
            await conversation_service.save_message(session_id, "user", query)

            # Step 1: 查询分类
            classification = await self.query_classifier.classify_and_get_config(query)
            query_type = classification["type"]
            debug_info["query_type"] = query_type.value if hasattr(query_type, 'value') else str(query_type)
            debug_info["query_weights"] = classification.get("weights", {})

            # Step 2: 查询重构
            cached_rewrite = await query_rewrite_cache.get(query, chat_history)
            if cached_rewrite is not None:
                queries = cached_rewrite
                logger.info(f"[{request_id}] Query rewrite cache hit")
            else:
                rewritten_queries = await self.query_rewriter.rewrite_query(query, chat_history)
                queries = [query] + [rq.rewritten for rq in rewritten_queries if rq.rewritten != query]
                await query_rewrite_cache.set(query, queries, chat_history)
            debug_info["rewritten_queries"] = queries[1:] if len(queries) > 1 else []

            # Step 3: 混合检索
            all_results = []
            query_type_str = query_type.value if hasattr(query_type, 'value') else str(query_type)
            for q in queries:
                cached_results = await retrieval_cache.get(q, query_type_str)
                if cached_results is not None:
                    results = [RetrievalResult(**r) for r in cached_results]
                    logger.info(f"[{request_id}] Retrieval cache hit for: {q[:50]}")
                else:
                    results = await self.hybrid_retriever.retrieve(
                        query=q,
                        query_type=query_type,
                        top_k=30
                    )
                    await retrieval_cache.set(
                        q, query_type_str,
                        [{"chunk_id": r.chunk_id, "content": r.content, "score": r.score, "metadata": r.metadata, "source": r.source} for r in results]
                    )
                all_results.extend(results)

            seen = set()
            unique_results = []
            for result in all_results:
                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    unique_results.append(result)

            debug_info["retrieval_count"] = len(unique_results)
            debug_info["retrieval_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "source": r.source,
                    "recipe_title": r.metadata.get("recipe_title", ""),
                    "chunk_type": r.metadata.get("chunk_type", ""),
                    "step_number": r.metadata.get("step_number"),
                    "content_preview": r.content[:200],
                }
                for r in unique_results[:30]
            ]

            # Step 4: 重排序
            reranked_results, has_related_content = await self.reranker.rerank(
                query=query,
                results=unique_results,
                top_k=5
            )
            debug_info["has_related_content"] = has_related_content
            debug_info["rerank_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "source": r.source,
                    "recipe_title": r.metadata.get("recipe_title", ""),
                    "chunk_type": r.metadata.get("chunk_type", ""),
                    "step_number": r.metadata.get("step_number"),
                    "content_preview": r.content[:200],
                }
                for r in reranked_results
            ]

            # 流式生成回答
            if not has_related_content:
                answer = "根据现有食谱库，无法回答此问题。请尝试换个问题或提供更多细节。"
                await self.memory.add_message(session_id, "assistant", answer)
                await conversation_service.save_message(session_id, "assistant", answer)
                yield answer
                debug_info["total_time_ms"] = int((time.time() - total_start) * 1000)
                yield {"type": "metadata", "citations": [], "pipeline_debug": debug_info}
                return

            # 构建上下文
            context, citation_map = self.answer_generator._build_context(reranked_results)

            # 使用LLM流式生成
            gen_start = time.time()
            full_answer = ""
            if self.answer_generator.llm:
                async for chunk in self.answer_generator._generate_streaming(query, context, chat_history):
                    full_answer += chunk
                    yield chunk
            else:
                # 无LLM时返回简单回答
                full_answer = self.answer_generator._generate_simple_answer(query, reranked_results)
                yield full_answer

            debug_info["generation_time_ms"] = int((time.time() - gen_start) * 1000)

            # 构建引用
            citations = self.answer_generator._build_citations(full_answer, citation_map)
            citations_data = [
                {
                    "citation_id": c.citation_id,
                    "recipe_title": c.recipe_title,
                    "chunk_type": c.chunk_type,
                    "step_number": c.step_number,
                    "excerpt": c.excerpt,
                    "document_source": c.document_source,
                    "image_path": c.image_path,
                }
                for c in citations
            ]

            # 记录助手消息
            await self.memory.add_message(session_id, "assistant", full_answer)
            await conversation_service.save_message(
                session_id, "assistant", full_answer,
                citations=citations_data
            )

            debug_info["total_time_ms"] = int((time.time() - total_start) * 1000)

            # 最后 yield 元数据（citations + debug_info）
            yield {"type": "metadata", "citations": citations_data, "pipeline_debug": debug_info}

        except Exception as e:
            logger.error(f"[{request_id}] Streaming RAG failed: {e}")
            yield "抱歉，处理您的问题时出现错误，请稍后重试。"

    async def clear_session(self, session_id: str):
        """
        清除会话记忆

        Args:
            session_id: 会话ID
        """
        await self.memory.clear_session(session_id)
        logger.info(f"Cleared session: {session_id}")

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话摘要

        Args:
            session_id: 会话ID

        Returns:
            会话摘要
        """
        return await self.memory.get_memory_summary(session_id)


# 创建全局实例
rag_pipeline = RAGPipeline()
