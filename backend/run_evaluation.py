"""
独立评估脚本
运行完整RAGAS评估，输出结果到 evaluation_results/ 目录

使用方法:
    cd backend
    python run_evaluation.py

可选参数:
    --no-ragas    跳过RAGAS评估（仅运行自定义指标）
    --case TC001  仅评估指定的测试用例
    --output DIR  指定输出目录
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("evaluation")


def _setup_evaluation_environment():
    """
    设置评估环境，替换掉依赖外部服务的组件：
    - Redis (对话记忆) -> 内存版
    - Database (对话持久化) -> no-op
    - Redis cache (缓存服务) -> no-op
    """
    # 1. 替换对话记忆为内存版
    from app.services import conversation_memory as cm_module
    from app.services.conversation_memory import InMemoryConversationMemory
    cm_module.conversation_memory = InMemoryConversationMemory(max_turns=5)

    # 2. 替换对话持久化服务为 no-op
    from app.services import conversation_service as cs_module
    mock_service = AsyncMock()
    mock_service.save_message = AsyncMock()
    mock_service.get_messages = AsyncMock(return_value=[])
    mock_service.get_conversation_list = AsyncMock(return_value=[])
    mock_service.delete_conversation = AsyncMock(return_value=True)
    cs_module.conversation_service = mock_service

    # 3. 替换缓存服务为 no-op（无Redis时缓存直接跳过）
    from app.services import cache_service as cache_module

    class NoOpCache:
        """无操作缓存，用于评估时跳过Redis依赖"""
        async def get(self, *args, **kwargs):
            return None
        async def set(self, *args, **kwargs):
            pass

    cache_module.query_rewrite_cache = NoOpCache()
    cache_module.retrieval_cache = NoOpCache()

    logger.info("评估环境初始化完成（内存模式，无需Redis/数据库）")


async def run_evaluation(
    case_id: str = None,
    skip_ragas: bool = False,
    output_dir: str = None,
):
    """
    运行评估

    Args:
        case_id: 指定测试用例ID（可选）
        skip_ragas: 是否跳过RAGAS评估
        output_dir: 输出目录
    """
    # 先设置环境，再导入依赖
    _setup_evaluation_environment()

    from app.services.evaluation_service import EvaluationService
    from app.services.rag_pipeline import RAGPipeline

    # 初始化
    service = EvaluationService()

    # 使用内存版记忆创建管道
    from app.services.conversation_memory import InMemoryConversationMemory
    pipeline = RAGPipeline(memory=InMemoryConversationMemory(max_turns=5))

    if not service.test_cases:
        logger.error("没有加载到测试用例，请检查 evaluation_dataset.json")
        return

    # 过滤指定用例
    if case_id:
        service.test_cases = [tc for tc in service.test_cases if tc.id == case_id]
        if not service.test_cases:
            logger.error(f"未找到测试用例: {case_id}")
            return

    logger.info(f"{'=' * 60}")
    logger.info(f"ChefMind RAG 评估")
    logger.info(f"测试用例数: {len(service.test_cases)}")
    logger.info(f"RAGAS评估: {'跳过' if skip_ragas else '启用'}")
    logger.info(f"{'=' * 60}")

    # 运行评估
    if skip_ragas:
        await _run_custom_only(service, pipeline)
    else:
        await service.run_full_evaluation(pipeline)

    # 输出结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path(__file__).parent.parent / "evaluation_results" / timestamp

    out_path.mkdir(parents=True, exist_ok=True)

    # 导出完整结果
    summary_path = out_path / "summary.json"
    service.export_results(str(summary_path))

    # 打印摘要
    _print_summary(service)

    logger.info(f"\n结果已保存到: {out_path}")


async def _run_custom_only(service, pipeline):
    """仅运行自定义指标（跳过RAGAS）"""
    for i, tc in enumerate(service.test_cases):
        logger.info(f"[{i+1}/{len(service.test_cases)}] {tc.id}: {tc.query}")

        try:
            answer_result, debug_info = await pipeline.run(tc.query)

            retrieved_recipes = list(set([
                r.get("recipe_title", "")
                for r in debug_info.get("rerank_results", [])
                if r.get("recipe_title")
            ]))

            retrieved_contexts = [
                r.get("content_preview", "")
                for r in debug_info.get("rerank_results", [])
            ]

            citation_map = {}
            for j, c in enumerate(answer_result.citations):
                citation_map[str(j + 1)] = {"citation_id": c.citation_id}

            result = service.evaluate_single(
                test_case=tc,
                generated_answer=answer_result.answer,
                retrieved_contexts=retrieved_contexts,
                retrieved_recipes=retrieved_recipes,
                citation_map=citation_map,
                refused=answer_result.refused,
            )
            service.results.append(result)

            status = "PASS" if result.passed else "FAIL"
            logger.info(f"  [{status}] Recall@5={result.metrics['recall_at_5']:.2f} "
                        f"Citation={result.metrics['citation_accuracy']:.2f} "
                        f"Refuse={result.metrics['refuse_accuracy']:.2f}")

        except Exception as e:
            logger.error(f"  评估失败: {e}", exc_info=True)


def _print_summary(service):
    """打印评估摘要"""
    summary = service.get_evaluation_summary()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"评估摘要")
    logger.info(f"{'=' * 60}")
    logger.info(f"总测试数: {summary['total_tests']}")
    logger.info(f"通过: {summary['passed']}")
    logger.info(f"失败: {summary['failed']}")
    logger.info(f"通过率: {summary['pass_rate']:.1%}")

    logger.info(f"\n自定义指标:")
    for name, value in summary.get("custom_metrics", {}).items():
        logger.info(f"  {name}: {value:.4f}")

    ragas = summary.get("ragas_metrics", {})
    if ragas:
        logger.info(f"\nRAGAS指标:")
        for name, value in ragas.items():
            logger.info(f"  {name}: {value:.4f}")

    # 按路由类型
    by_type = summary.get("by_route_type", {})
    if by_type:
        logger.info(f"\n按路由类型:")
        for route_type, stats in by_type.items():
            rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  {route_type}: {stats['passed']}/{stats['total']} ({rate:.0%})")

    # 按难度
    by_diff = summary.get("by_difficulty", {})
    if by_diff:
        logger.info(f"\n按难度:")
        for diff, stats in by_diff.items():
            rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  {diff}: {stats['passed']}/{stats['total']} ({rate:.0%})")

    # 失败的用例
    failed = [r for r in service.results if not r.passed]
    if failed:
        logger.info(f"\n失败的测试用例:")
        for r in failed:
            logger.info(f"  {r.test_case_id}: {r.query}")
            for err in r.errors:
                logger.info(f"    - {err}")


def main():
    parser = argparse.ArgumentParser(description="ChefMind RAG 评估脚本")
    parser.add_argument("--case", type=str, help="指定测试用例ID（如 TC001）")
    parser.add_argument("--no-ragas", action="store_true", help="跳过RAGAS评估")
    parser.add_argument("--output", type=str, help="输出目录")

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        case_id=args.case,
        skip_ragas=args.no_ragas,
        output_dir=args.output,
    ))


if __name__ == "__main__":
    main()
