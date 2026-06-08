"""
评估服务
集成RAGAS框架实现PRD第8节定义的评估方案

RAGAS指标：
- Faithfulness: 回答是否忠实于检索到的上下文
- AnswerRelevancy: 回答是否与问题相关
- LLMContextRecall: 检索的上下文是否覆盖了标准答案
- ContextPrecision: 检索的上下文中相关内容的精确度

自定义指标：
- Recall@K: 期望菜谱在检索Top-K中的命中率
- CitationAccuracy: 引用来源编号是否真实存在
- RefuseAccuracy: 无答案问题是否正确拒答
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# 评估数据集路径
DATASET_PATH = Path(__file__).parent.parent.parent / "evaluation_dataset.json"


@dataclass
class TestCase:
    """测试用例"""
    id: str
    query: str
    expected_recipes: List[str]
    route_type: str
    difficulty: str
    ground_truth: str
    should_refuse: bool
    description: str


@dataclass
class EvaluationResult:
    """单个测试用例的评估结果"""
    test_case_id: str
    query: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: List[str]
    retrieved_recipes: List[str]
    expected_recipes: List[str]
    metrics: Dict[str, float]
    passed: bool
    errors: List[str]


class EvaluationService:
    """
    评估服务

    PRD规范（第8节）：
    - 测试集：自建50条中文菜谱问题，覆盖所有路由类型
    - 自动化指标：
      - Recall@5：期望菜谱chunk在检索Top-5中的命中率
      - RAGAS：Faithfulness、Answer Relevance、Context Recall
      - 拒答准确率：对于无答案问题，系统是否拒答
      - 引用准确率：回答中引用的来源编号是否全部真实存在
    """

    def __init__(self):
        """初始化评估服务"""
        self.test_cases: List[TestCase] = []
        self.results: List[EvaluationResult] = []
        self.ragas_scores: Optional[Dict[str, float]] = None
        self._evaluator_llm = None
        self._load_test_cases()

    def _load_test_cases(self):
        """从JSON文件加载测试用例"""
        if not DATASET_PATH.exists():
            logger.warning(f"评估数据集不存在: {DATASET_PATH}")
            return

        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                tc = TestCase(
                    id=item["id"],
                    query=item["query"],
                    expected_recipes=item.get("expected_recipes", []),
                    route_type=item.get("route_type", "unknown"),
                    difficulty=item.get("difficulty", "medium"),
                    ground_truth=item.get("ground_truth", ""),
                    should_refuse=item.get("should_refuse", False),
                    description=item.get("description", ""),
                )
                self.test_cases.append(tc)

            logger.info(f"加载了 {len(self.test_cases)} 条评估测试用例")
        except Exception as e:
            logger.error(f"加载评估数据集失败: {e}")

    def _get_evaluator_llm(self):
        """获取RAGAS评估用的LLM（复用项目现有LLM配置）"""
        if self._evaluator_llm is None:
            try:
                from ragas.llms import LangchainLLMWrapper
                from app.services.llm_factory import get_llm_for_answer

                llm = get_llm_for_answer()
                if llm:
                    self._evaluator_llm = LangchainLLMWrapper(llm)
                    logger.info("RAGAS评估LLM初始化成功")
                else:
                    logger.warning("无法创建RAGAS评估LLM")
            except ImportError as e:
                logger.error(f"RAGAS依赖未安装: {e}")
        return self._evaluator_llm

    def _get_evaluator_embeddings(self):
        """获取RAGAS评估用的Embeddings（复用项目的BGE模型）"""
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_core.embeddings import Embeddings
            from app.services.embedding_service import get_embedding_service

            class SentenceTransformerEmbeddings(Embeddings):
                """将项目的EmbeddingService包装为LangChain Embeddings接口"""
                def __init__(self, service):
                    self._service = service

                def embed_documents(self, texts):
                    return self._service.embed_documents(texts)

                def embed_query(self, text):
                    return self._service.embed_query(text)

            svc = get_embedding_service()
            if svc:
                svc.initialize()  # 延迟加载模型
                if svc.model:
                    return LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(svc))
        except Exception as e:
            logger.warning(f"无法创建RAGAS评估Embeddings: {e}")
        return None

    # ==================== 自定义指标 ====================

    def calculate_recall_at_k(
        self,
        retrieved_recipes: List[str],
        expected_recipes: List[str],
        k: int = 5
    ) -> float:
        """
        计算Recall@K

        Args:
            retrieved_recipes: 检索到的菜谱列表
            expected_recipes: 期望的菜谱列表
            k: Top-K

        Returns:
            Recall@K值
        """
        if not expected_recipes:
            return 1.0  # 无期望菜谱时，认为召回率为100%

        retrieved_set = set(retrieved_recipes[:k])
        expected_set = set(expected_recipes)

        hits = len(retrieved_set & expected_set)
        return hits / len(expected_set)

    def calculate_citation_accuracy(
        self,
        answer: str,
        citation_map: Dict[str, Any]
    ) -> float:
        """
        计算引用准确率

        Args:
            answer: 生成的回答
            citation_map: 引用映射

        Returns:
            引用准确率
        """
        cited_ids = set(re.findall(r'\[(\d+)\]', answer))

        if not cited_ids:
            return 1.0  # 无引用时，认为准确率为100%

        valid_citations = sum(1 for cid in cited_ids if cid in citation_map)
        return valid_citations / len(cited_ids)

    def calculate_refuse_accuracy(
        self,
        test_case: TestCase,
        refused: bool
    ) -> float:
        """
        计算拒答准确率

        Args:
            test_case: 测试用例
            refused: 系统是否拒答

        Returns:
            1.0 表示正确，0.0 表示错误
        """
        if test_case.should_refuse:
            return 1.0 if refused else 0.0
        else:
            return 0.0 if refused else 1.0

    # ==================== RAGAS评估 ====================

    def run_ragas_evaluation(self, eval_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        执行RAGAS评估

        Args:
            eval_data: 评估数据列表，每项包含:
                - user_input: 用户问题
                - response: 生成的回答
                - retrieved_contexts: 检索到的上下文列表
                - reference: 标准答案（ground_truth）

        Returns:
            RAGAS指标分数字典
        """
        try:
            from ragas import evaluate as ragas_evaluate
            from datasets import Dataset
            from ragas.metrics import faithfulness, answer_relevancy, ContextRecall, ContextPrecision
        except ImportError:
            logger.error("RAGAS未安装，请运行: pip install ragas datasets")
            return {}

        llm = self._get_evaluator_llm()
        if not llm:
            logger.warning("无可用的评估LLM，跳过RAGAS评估")
            return {}

        embeddings = self._get_evaluator_embeddings()

        # 过滤掉ground_truth为空的用例（RAGAS需要reference字段）
        valid_data = [d for d in eval_data if d.get("reference", "").strip()]
        if not valid_data:
            logger.warning("没有有效的评估数据（ground_truth全为空）")
            return {}

        try:
            # RAGAS 0.2.x 使用 HuggingFace Dataset，列名为:
            # question, answer, contexts, ground_truth
            dataset_dict = {
                "question": [d["user_input"] for d in valid_data],
                "answer": [d["response"] for d in valid_data],
                "contexts": [d["retrieved_contexts"] for d in valid_data],
                "ground_truth": [d["reference"] for d in valid_data],
            }
            dataset = Dataset.from_dict(dataset_dict)

            ctx_recall = ContextRecall()
            ctx_precision = ContextPrecision()
            metrics = [faithfulness, answer_relevancy, ctx_recall, ctx_precision]

            # 直接在需要 embeddings 的指标上设置，避免内部 embedding_factory() 调用
            if embeddings:
                from ragas.metrics.base import MetricWithEmbeddings
                for m in metrics:
                    if isinstance(m, MetricWithEmbeddings):
                        m.embeddings = embeddings

            result = ragas_evaluate(dataset=dataset, metrics=metrics, llm=llm)

            # 提取平均分数（ragas 0.2.x EvaluationResult 使用 _repr_dict 存储均值）
            scores = {}
            repr_dict = getattr(result, "_repr_dict", {})
            for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
                if metric_name in repr_dict:
                    scores[metric_name] = float(repr_dict[metric_name])

            # 将 nan 值转为 None（JSON 不支持 NaN）
            import math
            scores = {k: (None if math.isnan(v) else v) for k, v in scores.items()}

            logger.info(f"RAGAS评估完成: {scores}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS评估执行失败: {e}", exc_info=True)
            return {}

    # ==================== 单条评估 ====================

    def evaluate_single(
        self,
        test_case: TestCase,
        generated_answer: str,
        retrieved_contexts: List[str],
        retrieved_recipes: List[str],
        citation_map: Optional[Dict[str, Any]] = None,
        refused: bool = False
    ) -> EvaluationResult:
        """
        评估单个测试用例（自定义指标）

        Args:
            test_case: 测试用例
            generated_answer: 生成的回答
            retrieved_contexts: 检索到的上下文文本列表
            retrieved_recipes: 检索到的菜谱名列表
            citation_map: 引用映射
            refused: 是否拒答

        Returns:
            评估结果
        """
        errors = []
        metrics = {}

        # Recall@5
        recall = self.calculate_recall_at_k(retrieved_recipes, test_case.expected_recipes, k=5)
        metrics["recall_at_5"] = recall

        # 引用准确率
        if citation_map:
            citation_accuracy = self.calculate_citation_accuracy(generated_answer, citation_map)
            metrics["citation_accuracy"] = citation_accuracy
        else:
            metrics["citation_accuracy"] = 1.0

        # 拒答准确率
        refuse_accuracy = self.calculate_refuse_accuracy(test_case, refused)
        metrics["refuse_accuracy"] = refuse_accuracy
        if refuse_accuracy < 1.0:
            if test_case.should_refuse and not refused:
                errors.append("Expected refusal but got answer")
            elif not test_case.should_refuse and refused:
                errors.append("Expected answer but got refusal")

        # 判断是否通过自定义指标
        passed = all([
            metrics["recall_at_5"] >= 0.5,
            metrics["citation_accuracy"] >= 0.8,
            metrics["refuse_accuracy"] >= 0.8,
        ])

        return EvaluationResult(
            test_case_id=test_case.id,
            query=test_case.query,
            generated_answer=generated_answer,
            ground_truth=test_case.ground_truth,
            retrieved_contexts=retrieved_contexts,
            retrieved_recipes=retrieved_recipes,
            expected_recipes=test_case.expected_recipes,
            metrics=metrics,
            passed=passed,
            errors=errors,
        )

    # ==================== 端到端评估 ====================

    async def run_full_evaluation(self, rag_pipeline) -> Dict[str, Any]:
        """
        运行完整评估流程

        1. 对每个测试用例调用RAG管道
        2. 收集结果
        3. 运行RAGAS指标（Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision）
        4. 运行自定义指标（Recall@K, CitationAccuracy, RefuseAccuracy）
        5. 汇总输出

        Args:
            rag_pipeline: RAG管道实例

        Returns:
            完整评估结果
        """
        if not self.test_cases:
            return {"error": "没有加载测试用例"}

        logger.info(f"开始完整评估，共 {len(self.test_cases)} 条测试用例")
        self.results = []

        # 收集RAGAS评估数据
        ragas_eval_data = []

        for i, tc in enumerate(self.test_cases):
            logger.info(f"[{i+1}/{len(self.test_cases)}] 评估: {tc.id} - {tc.query}")

            try:
                # 调用RAG管道
                answer_result, debug_info = await rag_pipeline.run(tc.query)

                # 提取检索到的上下文（rerank后的top结果）
                retrieved_contexts = [
                    r.get("content_preview", "")
                    for r in debug_info.get("rerank_results", [])
                ]

                # 提取检索到的菜谱名
                retrieved_recipes = list(set([
                    r.get("recipe_title", "")
                    for r in debug_info.get("rerank_results", [])
                    if r.get("recipe_title")
                ]))

                # 构建引用映射
                citation_map = {}
                for j, c in enumerate(answer_result.citations):
                    citation_map[str(j + 1)] = {
                        "citation_id": c.citation_id,
                        "recipe_title": c.recipe_title,
                        "chunk_type": c.chunk_type,
                    }

                # 自定义指标评估
                result = self.evaluate_single(
                    test_case=tc,
                    generated_answer=answer_result.answer,
                    retrieved_contexts=retrieved_contexts,
                    retrieved_recipes=retrieved_recipes,
                    citation_map=citation_map,
                    refused=answer_result.refused,
                )
                self.results.append(result)

                # 收集RAGAS数据
                if tc.ground_truth.strip():
                    ragas_eval_data.append({
                        "user_input": tc.query,
                        "response": answer_result.answer,
                        "retrieved_contexts": retrieved_contexts if retrieved_contexts else [""],
                        "reference": tc.ground_truth,
                    })

            except Exception as e:
                logger.error(f"评估 {tc.id} 失败: {e}")
                self.results.append(EvaluationResult(
                    test_case_id=tc.id,
                    query=tc.query,
                    generated_answer="",
                    ground_truth=tc.ground_truth,
                    retrieved_contexts=[],
                    retrieved_recipes=[],
                    expected_recipes=tc.expected_recipes,
                    metrics={"recall_at_5": 0, "citation_accuracy": 0, "refuse_accuracy": 0},
                    passed=False,
                    errors=[str(e)],
                ))

        # 运行RAGAS评估
        logger.info("开始RAGAS评估...")
        self.ragas_scores = self.run_ragas_evaluation(ragas_eval_data)

        # 构建汇总
        summary = self.get_evaluation_summary()
        logger.info(f"评估完成: {summary.get('pass_rate', 0):.1%} 通过率")
        return summary

    # ==================== 结果汇总 ====================

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        获取评估摘要

        Returns:
            包含自定义指标和RAGAS指标的完整评估摘要
        """
        if not self.results:
            return {"message": "No evaluation results available"}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        # 自定义指标平均值
        custom_metric_names = ["recall_at_5", "citation_accuracy", "refuse_accuracy"]
        avg_custom = {}
        for name in custom_metric_names:
            values = [r.metrics.get(name, 0) for r in self.results]
            avg_custom[name] = sum(values) / len(values) if values else 0

        # 按路由类型分组统计
        type_stats = {}
        for r in self.results:
            # 找到对应的test_case获取route_type
            tc = next((t for t in self.test_cases if t.id == r.test_case_id), None)
            route_type = tc.route_type if tc else "unknown"
            if route_type not in type_stats:
                type_stats[route_type] = {"total": 0, "passed": 0, "metrics": {}}
            type_stats[route_type]["total"] += 1
            if r.passed:
                type_stats[route_type]["passed"] += 1

        # 按难度分组统计
        difficulty_stats = {}
        for r in self.results:
            tc = next((t for t in self.test_cases if t.id == r.test_case_id), None)
            difficulty = tc.difficulty if tc else "unknown"
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"total": 0, "passed": 0}
            difficulty_stats[difficulty]["total"] += 1
            if r.passed:
                difficulty_stats[difficulty]["passed"] += 1

        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "custom_metrics": avg_custom,
            "ragas_metrics": self.ragas_scores or {},
            "by_route_type": type_stats,
            "by_difficulty": difficulty_stats,
            "test_results": [
                {
                    "id": r.test_case_id,
                    "query": r.query,
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "errors": r.errors,
                }
                for r in self.results
            ],
        }

        return summary

    def export_results(self, file_path: str):
        """
        导出评估结果到JSON文件

        Args:
            file_path: 输出文件路径
        """
        summary = self.get_evaluation_summary()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已导出到: {file_path}")


# 全局实例
evaluation_service = EvaluationService()
