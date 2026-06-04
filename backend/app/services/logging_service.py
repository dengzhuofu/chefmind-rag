"""
日志与监控服务
实现PRD 7.1节定义的结构化日志
"""

import logging
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar
from dataclasses import dataclass, asdict

# 请求上下文
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
request_start_time_var: ContextVar[datetime] = ContextVar('request_start_time', default=None)


@dataclass
class RAGLogEntry:
    """RAG查询日志条目"""
    request_id: str
    timestamp: str
    original_query: str
    rewritten_queries: list
    route_type: str
    retrieval_top_scores: list
    rerank_max_score: float
    generated_answer: str
    citations_count: int
    user_feedback: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


class StructuredLogger:
    """
    结构化日志记录器

    PRD规范：
    - 使用structlog
    - 通过LangChain的CallbackHandler记录每次查询的完整链路数据
    - 每条日志包含：request_id, timestamp, original_query, rewritten_queries,
      route_type, retrieval_top_scores, rerank_max_score, generated_answer,
      citations_count, user_feedback
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_structlog()

    def _setup_structlog(self):
        """设置structlog"""
        try:
            import structlog
            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer()
                ],
                logger_factory=structlog.PrintLoggerFactory(),
                wrapper_class=structlog.BoundLogger,
                cache_logger_on_first_use=True,
            )
            self.use_structlog = True
        except ImportError:
            self.use_structlog = False
            self.logger.warning("structlog not installed, using standard logging")

    def log_rag_query(self, entry: RAGLogEntry):
        """
        记录RAG查询日志

        Args:
            entry: 日志条目
        """
        log_data = asdict(entry)

        if self.use_structlog:
            import structlog
            logger = structlog.get_logger()
            logger.info("rag_query", **log_data)
        else:
            self.logger.info(f"RAG Query: {json.dumps(log_data, ensure_ascii=False)}")

    def log_retrieval(
        self,
        request_id: str,
        query: str,
        results_count: int,
        top_scores: list
    ):
        """记录检索日志"""
        self.logger.info(
            f"Retrieval: request_id={request_id}, query={query}, "
            f"results={results_count}, top_scores={top_scores}"
        )

    def log_reranking(
        self,
        request_id: str,
        input_count: int,
        output_count: int,
        max_score: float
    ):
        """记录重排序日志"""
        self.logger.info(
            f"Reranking: request_id={request_id}, input={input_count}, "
            f"output={output_count}, max_score={max_score}"
        )

    def log_generation(
        self,
        request_id: str,
        citations_count: int,
        answer_length: int
    ):
        """记录生成日志"""
        self.logger.info(
            f"Generation: request_id={request_id}, "
            f"citations={citations_count}, answer_length={answer_length}"
        )

    def log_feedback(
        self,
        request_id: str,
        rating: int,
        comment: Optional[str] = None
    ):
        """记录用户反馈"""
        self.logger.info(
            f"Feedback: request_id={request_id}, rating={rating}, comment={comment}"
        )

    def log_error(
        self,
        request_id: str,
        error: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录错误"""
        self.logger.error(
            f"Error: request_id={request_id}, error={error}, context={context}"
        )


class MetricsCollector:
    """
    指标收集器

    PRD规范：
    - rag_query_duration_seconds（直方图）
    - rag_retrieval_recall（基于测试集）
    - rag_refuse_rate（拒答比例）
    - rag_citation_coverage（回答中有引用的比例）
    - llm_call_count_total
    """

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "refused_queries": 0,
            "queries_with_citations": 0,
            "total_duration_ms": 0,
            "llm_calls": 0,
        }
        self.query_durations = []

    def record_query(
        self,
        duration_ms: float,
        refused: bool,
        citations_count: int
    ):
        """记录查询指标"""
        self.metrics["total_queries"] += 1
        self.metrics["total_duration_ms"] += duration_ms
        self.query_durations.append(duration_ms)

        if refused:
            self.metrics["refused_queries"] += 1

        if citations_count > 0:
            self.metrics["queries_with_citations"] += 1

    def record_llm_call(self):
        """记录LLM调用"""
        self.metrics["llm_calls"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标摘要"""
        total = self.metrics["total_queries"]
        if total == 0:
            return {
                "total_queries": 0,
                "refuse_rate": 0,
                "citation_coverage": 0,
                "avg_duration_ms": 0,
                "p95_duration_ms": 0,
                "llm_calls": self.metrics["llm_calls"],
            }

        return {
            "total_queries": total,
            "refuse_rate": self.metrics["refused_queries"] / total,
            "citation_coverage": self.metrics["queries_with_citations"] / total,
            "avg_duration_ms": self.metrics["total_duration_ms"] / total,
            "p95_duration_ms": self._percentile(self.query_durations, 95),
            "llm_calls": self.metrics["llm_calls"],
        }

    def _percentile(self, data: list, percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# 全局实例
structured_logger = StructuredLogger("chefmind")
metrics_collector = MetricsCollector()
