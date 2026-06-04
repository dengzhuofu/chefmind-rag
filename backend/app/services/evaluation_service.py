"""
评估服务
实现PRD第8节定义的评估方案
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """测试用例"""
    id: str
    query: str
    expected_answer: str
    expected_recipes: List[str]
    route_type: str
    difficulty: str  # easy, medium, hard
    description: str


@dataclass
class EvaluationResult:
    """评估结果"""
    test_case_id: str
    query: str
    generated_answer: str
    expected_answer: str
    retrieved_recipes: List[str]
    expected_recipes: List[str]
    metrics: Dict[str, float]
    passed: bool
    errors: List[str]


class EvaluationService:
    """
    评估服务

    PRD规范：
    - 测试集：自建50条中文菜谱问题，覆盖所有路由类型、模糊查询、跨文档查询、无答案查询
    - 自动化指标：
      - Recall@5：期望菜谱chunk在检索Top-5中的命中率
      - RAGAS：Faithfulness、Answer Relevance、Context Recall
      - 拒答准确率：对于无答案问题，系统是否拒答
      - 引用准确率：回答中引用的来源编号是否全部真实存在于检索结果中
    - 人工评估：邀请3人，按1-5分对准确性、步骤完整性、引用准确性、图片相关性打分
    """

    def __init__(self):
        """初始化评估服务"""
        self.test_cases: List[TestCase] = []
        self.results: List[EvaluationResult] = []
        self._load_test_cases()

    def _load_test_cases(self):
        """加载测试用例"""
        # 这里定义一些示例测试用例
        # 实际应用中应该从文件加载
        self.test_cases = [
            TestCase(
                id="TC001",
                query="清蒸鲈鱼怎么做？",
                expected_answer="清蒸鲈鱼是一道经典的粤菜...",
                expected_recipes=["清蒸鲈鱼"],
                route_type="step_lookup",
                difficulty="easy",
                description="基础步骤查询"
            ),
            TestCase(
                id="TC002",
                query="宫保鸡丁需要哪些食材？",
                expected_answer="宫保鸡丁的主要食材包括...",
                expected_recipes=["宫保鸡丁"],
                route_type="ingredient_search",
                difficulty="easy",
                description="食材查询"
            ),
            TestCase(
                id="TC003",
                query="没有蚝油可以用什么代替？",
                expected_answer="蚝油可以用...",
                expected_recipes=[],
                route_type="substitution",
                difficulty="medium",
                description="食材替换查询"
            ),
            TestCase(
                id="TC004",
                query="红烧肉有什么小技巧？",
                expected_answer="红烧肉的烹饪技巧包括...",
                expected_recipes=["红烧肉"],
                route_type="tip",
                difficulty="easy",
                description="技巧查询"
            ),
            TestCase(
                id="TC005",
                query="如何做一道简单的家常菜？",
                expected_answer="",
                expected_recipes=[],
                route_type="generate_recipe",
                difficulty="hard",
                description="模糊查询"
            ),
            TestCase(
                id="TC006",
                query="外星人怎么做？",
                expected_answer="根据现有食谱库，无法回答此问题。",
                expected_recipes=[],
                route_type="unknown",
                difficulty="medium",
                description="无答案查询"
            ),
        ]

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
        import re
        cited_ids = set(re.findall(r'\[(\d+)\]', answer))

        if not cited_ids:
            return 1.0  # 无引用时，认为准确率为100%

        valid_citations = sum(1 for cid in cited_ids if cid in citation_map)
        return valid_citations / len(cited_ids)

    def calculate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        计算忠实度（简化版本）

        PRD规范：使用RAGAS框架的Faithfulness指标

        Args:
            answer: 生成的回答
            context: 上下文

        Returns:
            忠实度分数
        """
        # 简化实现：检查回答中的关键信息是否在上下文中出现
        # 实际应用中应该使用RAGAS框架

        if not answer or not context:
            return 0.0

        # 提取回答中的关键句子
        sentences = answer.split('。')
        if not sentences:
            return 0.0

        # 检查每个句子是否与上下文相关
        relevant_count = 0
        for sentence in sentences:
            if len(sentence.strip()) < 5:
                continue
            # 简单检查：句子中的关键词是否在上下文中出现
            keywords = [kw for kw in sentence if len(kw) >= 2]
            if any(kw in context for kw in keywords):
                relevant_count += 1

        return relevant_count / len(sentences) if sentences else 0.0

    def calculate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        计算答案相关性

        Args:
            query: 用户查询
            answer: 生成的回答

        Returns:
            答案相关性分数
        """
        if not query or not answer:
            return 0.0

        # 简化实现：检查查询关键词是否在回答中出现
        query_keywords = [kw for kw in query if len(kw) >= 2]
        if not query_keywords:
            return 0.5

        matches = sum(1 for kw in query_keywords if kw in answer)
        return matches / len(query_keywords)

    def evaluate_single(
        self,
        test_case: TestCase,
        generated_answer: str,
        retrieved_recipes: List[str],
        citation_map: Optional[Dict[str, Any]] = None,
        refused: bool = False
    ) -> EvaluationResult:
        """
        评估单个测试用例

        Args:
            test_case: 测试用例
            generated_answer: 生成的回答
            retrieved_recipes: 检索到的菜谱
            citation_map: 引用映射
            refused: 是否拒答

        Returns:
            评估结果
        """
        errors = []

        # 计算指标
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

        # 忠实度
        faithfulness = self.calculate_faithfulness(generated_answer, "")
        metrics["faithfulness"] = faithfulness

        # 答案相关性
        relevance = self.calculate_answer_relevance(test_case.query, generated_answer)
        metrics["answer_relevance"] = relevance

        # 检查拒答准确性
        if test_case.expected_answer == "根据现有食谱库，无法回答此问题。":
            # 期望拒答
            if refused:
                metrics["refuse_accuracy"] = 1.0
            else:
                metrics["refuse_accuracy"] = 0.0
                errors.append("Expected refusal but got answer")
        else:
            # 期望有回答
            if not refused:
                metrics["refuse_accuracy"] = 1.0
            else:
                metrics["refuse_accuracy"] = 0.0
                errors.append("Expected answer but got refusal")

        # 判断是否通过
        passed = all([
            metrics["recall_at_5"] >= 0.5,
            metrics["citation_accuracy"] >= 0.8,
            metrics["refuse_accuracy"] >= 0.8,
        ])

        return EvaluationResult(
            test_case_id=test_case.id,
            query=test_case.query,
            generated_answer=generated_answer,
            expected_answer=test_case.expected_answer,
            retrieved_recipes=retrieved_recipes,
            expected_recipes=test_case.expected_recipes,
            metrics=metrics,
            passed=passed,
            errors=errors
        )

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        获取评估摘要

        Returns:
            评估摘要
        """
        if not self.results:
            return {"message": "No evaluation results available"}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        # 计算平均指标
        avg_metrics = {}
        for metric_name in ["recall_at_5", "citation_accuracy", "faithfulness", "answer_relevance", "refuse_accuracy"]:
            values = [r.metrics.get(metric_name, 0) for r in self.results]
            avg_metrics[metric_name] = sum(values) / len(values) if values else 0

        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "average_metrics": avg_metrics,
            "test_results": [
                {
                    "id": r.test_case_id,
                    "query": r.query,
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "errors": r.errors
                }
                for r in self.results
            ]
        }

    def export_results(self, file_path: str):
        """
        导出评估结果

        Args:
            file_path: 文件路径
        """
        summary = self.get_evaluation_summary()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported evaluation results to {file_path}")


# 全局实例
evaluation_service = EvaluationService()
